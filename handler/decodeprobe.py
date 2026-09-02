"""Where `decode_s` actually goes — `docs/archive/instrumentation-archive.md` §11.

**This is §8a, §9 and §10 for the FOURTH time, and by now the only new information is which
number it is.** `decode_s` was 1.5 s at 4K and nobody looked; at 8K it is **106.8 s, 18.70% of
`compute_s`, the second-largest stage after the model and larger than the entire conversion
pipeline two waves were spent deleting**.

It covers at least three activities and the split decides what to build: h264 entropy decoding,
the `yuv420p -> BGR` colour conversion `cv2.VideoCapture.read()` performs, and whatever the
capture backend spends on threading it. **`bgr_s - yuv_s` IS the colour-conversion term**, and it
is the number that decides whether the yuv-to-GPU plan is worth the quality contract it would
need. **`entropy_s` is the floor nothing goes below without changing codec or hardware.**

**THREE PASSES, EACH THE PREVIOUS PLUS ONE STEP**, which is what makes §11c's ordering check
meaningful rather than decorative:

    entropy_s   -f null       decode and discard, no scaler, no output buffer
    yuv_s       rawvideo yuv420p to /dev/null   adds the output write, no colour conversion
    bgr_s       rawvideo bgr24 to /dev/null     adds swscale's yuv->BGR

**IT IS NOT THE FIX, AND THE VARIABLE IT USED TO NAME WAS NEVER THE LEVER.**
`OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;8"` is a Dockerfile `ENV`, and `F-2026-09-02-3`
established from OpenCV 5.x source that it goes to `av_dict_parse_string` and then
`avformat_open_input` — **the DEMUXER**. *The variable that sets decoder threads is
`OPENCV_FFMPEG_THREADS`, and this image has never set it.* **So every run banked on this image
decoded at whatever the auto path chose, not at 8**, and the paragraph that used to stand here
said the probe could not tell whether the backend had honoured a setting that was never reaching
the decoder at all.

**WHAT ANSWERS IT IS A READ AND NOT A PASS.** `decode._decoder_threads` asks the capture for
`CAP_PROP_N_THREADS` after open — `thread_count` off the codec context, the codec's own state
rather than an echo of a variable. *A `0` is a result about the instrument, not a defect to
chase; `None` is this build having no such property, which is a different fact and is filed as
one.*

**AND THE `decode_s` AGAINST `bgr_s` COMPARISON IS REFUTED — `F-2026-09-01-5`, STRUCTURALLY.**
*This docstring used to sell it as "cv2 against ffmpeg on identical work, inside one record, with
no A/B and no second job", and drew two branches from the size of the gap.* **The probe runs
AFTER the pipeline, so a CONTENDED read is being compared against an UNCONTENDED one** — the
retime's decode competes with the model and the encoder for the same cores, and these three
passes have the box to themselves. *No arming will ever make it one comparison.* **The two errors
were independent and correcting only the environment variable would have left the larger one
standing.**

*One asymmetry, small and stated rather than discovered: `decode_s` also covers the surplus
`grab()` sweep at `routec.py:847`, which is bounded at two frames by `SURPLUS_TOLERANCE_FRAMES`.*

**IT DOES NOT TOUCH `decode_s`.** The stage clock is unchanged, so a probed run's `decode_s`
stays comparable to every unprobed one — and the probe re-decodes the source three times, so a
run carrying it must not be used to bank a performance number. Gated by `params.decode_probe`,
per CF's rule that a per-job control belongs in the request.
"""

import subprocess
import time

#: §11a, verbatim — the five fields the kit grades. **`run` asserts the block carries all of
#: them rather than this sitting here unread**: the first cut declared this tuple, built the
#: block from a dict literal, and nothing ever compared the two. A list named as the single
#: source of truth that nothing consults is worse than no list, because the next person trusts it.
#:
#: The block also carries `elapsed_s`, which §11a does not name and the kit ignores. This tuple
#: is the REQUIRED set, not the exact one.
FIELDS = ("entropy_s", "yuv_s", "bgr_s", "frames", "threads")

#: Per pass. A 600-frame 8K decode is minutes; three of them plus the retime must still fit
#: inside a job, and a pass that hangs must not take the master with it.
TIMEOUT_S = 1800


#: **`-map 0:v:0` and `-an` on EVERY pass, and their absence was a real defect.** `-f null -`
#: maps every stream, so the entropy pass decoded the source's AUDIO and re-encoded it to
#: pcm_s16le while the two rawvideo passes mapped video only. `entropy_s` then carried work the
#: other two did not — and that `cv2.VideoCapture.read()`, the thing `decode_s` measures, never
#: does either.
#:
#: **Three consequences, and the third is the one that would have wasted somebody's day.**
#: `entropy_s` was not the floor it is named as; `yuv_s - entropy_s` was contaminated and could
#: come out NEGATIVE; and §11c's ordering check failed on a perfectly healthy audio-bearing
#: source, under a log line reading *"this cannot happen unless the probe is measuring something
#: other than what it names"*. Measured on two clips identical but for the audio stream: 4 of 4
#: out of order with audio, 4 of 4 ordered without.
#:
#: **The end-to-end exercise that passed was on a clip with no audio track**, which is exactly
#: the case where this is invisible.
_VIDEO_ONLY = ("-map", "0:v:0", "-an")

#: **Threading is stated rather than inherited.** *These passes would otherwise run at ffmpeg's
#: own `-threads 0` auto default and the block would carry a number nobody could reproduce.*
#: **A number measured under one configuration and labelled with another is the shape this
#: project keeps finding**, so the passes name their own. *The block used to report an OpenCV
#: environment variable beside it as "the capture backend's setting"; that field is deleted —
#: the variable reached the demuxer, not the decoder.*
PROBE_THREADS = "8"


def _pass(args, path, label, log):
    """One ffmpeg decode, wall-clocked. Returns seconds, or `None` if it did not complete."""
    command = (["ffmpeg", "-v", "error", "-nostdin", "-threads", PROBE_THREADS, "-i", path]
               + list(_VIDEO_ONLY) + args)
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, timeout=TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — a probe must never cost a delivered master
        log("[decode-probe] {} pass did not run ({}: {})".format(
            label, type(exc).__name__, str(exc)[:120]))
        return None
    elapsed = round(time.perf_counter() - started, 3)
    if completed.returncode != 0:
        log("[decode-probe] {} pass exited {}: {}".format(
            label, completed.returncode, completed.stderr.decode("utf-8", "replace")[-200:]))
        return None
    return elapsed


def run(path, frames, log=print):
    """§11a's five fields for the source at `path`. **Never raises. `None` if it did not run.**

    `frames` is the DECODER'S count, handed in rather than recounted here — `routec.DecodeCount`
    owns it and §11c grades this block against `retime.n_in`. **Two counters that agree because
    one was passed to the other is not evidence**, so this one does not count anything: it
    reports the number the retime actually decoded, and the ordering check below is what says
    the probe measured what it names.
    """
    started = time.perf_counter()
    try:
        # **Checked BEFORE the first subprocess, not after the third.** The first cut built the
        # block at the end with `int(frames)` in it, so a `None` threw after up to 5400 s of
        # completed measurement and discarded all of it. A guard on a value known before any
        # work starts belongs before the work.
        frames = int(frames)
    except Exception:  # noqa: BLE001
        log("[decode-probe] no decoded frame count to price the passes against; not run")
        return None
    try:
        # **`-f null` first, and it is the only pass with no output buffer at all.** Entropy
        # decoding and nothing else, which is the floor.
        entropy = _pass(["-f", "null", "-"], path, "entropy", log)
        # **`-pix_fmt yuv420p` is the codec's OWN layout**, so no scaler runs; what this adds
        # over the first pass is writing the frames out.
        yuv = _pass(["-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], path, "yuv", log)
        # **`bgr24` is what `cv2.VideoCapture.read()` hands back**, so this pass is the one whose
        # total the stage clock is measuring. `bgr_s - yuv_s` is swscale.
        bgr = _pass(["-pix_fmt", "bgr24", "-f", "rawvideo", "-"], path, "bgr", log)
        if None in (entropy, yuv, bgr):
            log("[decode-probe] a pass did not complete; the block is not filed rather than "
                "filed with a hole, because §11c grades an ORDERING and two of three numbers "
                "cannot be ordered")
            return None
        block = {"entropy_s": entropy, "yuv_s": yuv, "bgr_s": bgr, "frames": frames,
                 # **`threads` is what THESE PASSES ran at**, which is what the three seconds
                 # beside it were measured under. The capture backend's setting is a different
                 # fact about a different decoder and gets its own key rather than borrowing this
                 # one — the probe shells out to the ffmpeg CLI and never goes through OpenCV, so
                 # it cannot report on the backend by measuring itself.
                 "threads": PROBE_THREADS,
                 # **`capture_options` IS DELETED FROM THIS BLOCK — CF, 2026-09-02.** *It filed
                 # `OPENCV_FFMPEG_CAPTURE_OPTIONS` as the capture backend's configuration, and
                 # `F-2026-09-02-3` established that the variable reaches `avformat_open_input`
                 # — the DEMUXER. The decoder's thread count is `OPENCV_FFMPEG_THREADS`, which
                 # this image has never set.* **A record field that names one decoder and reports
                 # another is worse than an absent one, because it is indistinguishable from a
                 # true reading** — and it was read as one for as long as it existed. *What
                 # answers the question it was pretending to answer is `decode._decoder_threads`,
                 # which asks the codec context rather than the environment.*
                 "elapsed_s": round(time.perf_counter() - started, 3)}
        missing = [f for f in FIELDS if f not in block]
        if missing:
            log("[decode-probe] block is missing {} — not filed".format(", ".join(missing)))
            return None
        # **ONE THREAD FACT NOW, AND IT IS THE ONE THESE PASSES RAN AT.** *The line used to
        # print the capture backend's environment variable beside it, after an earlier draft had
        # printed this number under that variable's label — two facts confused, then split, and
        # now one of them deleted because it was never about the decoder.*
        log("[decode-probe] entropy {:.1f}s, +yuv {:.1f}s, +bgr {:.1f}s over {} frames "
            "(these passes: -threads {})".format(
                entropy, yuv - entropy, bgr - yuv, block["frames"], block["threads"]))
        # **Reported, not enforced.** §11c is the kit's check and this is the worker's own
        # reading of it — a probe that refused to file an out-of-order result would delete the
        # evidence that it measured something other than what it names.
        if not entropy <= yuv <= bgr:
            log("[decode-probe] PASSES ARE OUT OF ORDER and each pass is the previous plus one "
                "step, so this cannot happen unless the probe is measuring something other than "
                "what it names. Filed as measured; the kit fails it.")
        return block
    except Exception as exc:  # noqa: BLE001 — a probe must never cost a delivered master
        log("[decode-probe] did not complete ({}: {})".format(
            type(exc).__name__, str(exc)[:160]))
        return None
