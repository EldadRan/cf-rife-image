"""Where `decode_s` actually goes — `docs/instrumentation.md` §11.

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

**IT IS NOT THE FIX.** `OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;8"` is a Dockerfile `ENV` and
costs no code; this probe is what says whether it worked and what is left. **`decode_s` unchanged
after that lands is itself the answer**, not a null result.

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
#: The block also carries `capture_options` and `elapsed_s`, which §11a does not name and the kit
#: ignores. This tuple is the REQUIRED set, not the exact one.
FIELDS = ("entropy_s", "yuv_s", "bgr_s", "frames", "threads")

#: The env var item 2 sets on the image. Reported so a reading can be attributed to a
#: configuration rather than to a guess about one — a probe that cannot say what the backend was
#: told is a probe whose number nobody can reproduce.
CAPTURE_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"

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

#: **Threading is stated rather than inherited.** The ffmpeg CLI does not read
#: `OPENCV_FFMPEG_CAPTURE_OPTIONS` — that variable is OpenCV's — so these passes would otherwise
#: run at ffmpeg's own `-threads 0` auto default while the block reported OpenCV's setting beside
#: them. **A number measured under one configuration and labelled with another is the shape this
#: project keeps finding**, so the passes name their own and the block reports both.
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
    import os  # noqa: PLC0415 — trivial, but this module is imported only when asked

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
                 "capture_options": os.environ.get(CAPTURE_OPTIONS_ENV) or "unset",
                 "elapsed_s": round(time.perf_counter() - started, 3)}
        missing = [f for f in FIELDS if f not in block]
        if missing:
            log("[decode-probe] block is missing {} — not filed".format(", ".join(missing)))
            return None
        # **The two thread facts are printed as two facts.** The first draft of this line
        # formatted `block["threads"]` — which is what THESE passes ran at — under the
        # `OPENCV_FFMPEG_CAPTURE_OPTIONS` label, so the log said `OPENCV_FFMPEG_CAPTURE_
        # OPTIONS=8` on a box where that variable was unset. The field split of F5 was
        # pointless if the log put them back together.
        log("[decode-probe] entropy {:.1f}s, +yuv {:.1f}s, +bgr {:.1f}s over {} frames "
            "(these passes: -threads {}; capture backend: {}={})".format(
                entropy, yuv - entropy, bgr - yuv, block["frames"], block["threads"],
                CAPTURE_OPTIONS_ENV, block["capture_options"]))
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
