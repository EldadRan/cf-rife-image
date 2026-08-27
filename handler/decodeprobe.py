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

#: §11a, verbatim. Named here so the record, this module and the kit read one list.
FIELDS = ("entropy_s", "yuv_s", "bgr_s", "frames", "threads")

#: The env var item 2 sets on the image. Reported so a reading can be attributed to a
#: configuration rather than to a guess about one — a probe that cannot say what the backend was
#: told is a probe whose number nobody can reproduce.
CAPTURE_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"

#: Per pass. A 600-frame 8K decode is minutes; three of them plus the retime must still fit
#: inside a job, and a pass that hangs must not take the master with it.
TIMEOUT_S = 1800


def _pass(args, path, label, log):
    """One ffmpeg decode, wall-clocked. Returns seconds, or `None` if it did not complete."""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", path] + args
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
        block = {"entropy_s": entropy, "yuv_s": yuv, "bgr_s": bgr, "frames": int(frames),
                 "threads": os.environ.get(CAPTURE_OPTIONS_ENV) or "unset",
                 "elapsed_s": round(time.perf_counter() - started, 3)}
        log("[decode-probe] entropy {:.1f}s, +yuv {:.1f}s, +bgr {:.1f}s over {} frames "
            "({}={})".format(entropy, yuv - entropy, bgr - yuv, block["frames"],
                             CAPTURE_OPTIONS_ENV, block["threads"]))
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
