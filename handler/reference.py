"""The raw reference and the scores taken against it — contract §6g, `docs/archive/instrumentation-archive.md` §17.

**Every quality figure this project held before this module was measured against a DELIVERED h264
master**, so x264 was re-encoding its own artefacts and x265 was not. The frames handed to the
encoder were made by neither codec, and **the unbiased reference did not need to be found; it
needed to be kept.**

**THE FRAMES DO NOT LEAVE THE WORKER.** 8K at 8-bit is 23.89 GB against 29.98 GB free and uploading
it is about half an hour at the transfer rate the 8K runs show. So the scoring happens where the
frames already are, and what travels is the scores plus the worst frames as pictures.

**THE REFERENCE IS ffmpeg's OWN SECOND OUTPUT AND NOT A COPY OF WHAT CROSSED THE PIPE** (§6g, as
amended). The pipe is fed `rgb24`; ffmpeg converts to the encode's pixel format before a single
pixel reaches x264 or x265. *Retaining the rgb24 bytes would be byte-exact with the pipe and would
not be what either encoder saw, and converting them ourselves would put a second conversion path
beside ffmpeg's and require the two to agree — which this project has been bitten by twice.* **A
second output on the same command is the same binary, the same swscale, the same flags.**

**NOTHING HERE MAY COST A DELIVERED MASTER.** The reference is written by the encode itself, so a
failure in this module happens after the master exists; every entry point below is written so the
worst outcome is a missing `reference` block on the record.
"""

import json
import os
import shutil
import subprocess

from errors import INVALID_FIELD_VALUE, WorkerError

#: **Bytes per pixel of the retained reference, keyed on the encode's OWN pixel format** (§6g).
#: A named table rather than an expression, so a future format is one entry: the arithmetic that
#: decides whether a job is retainable must not be something an edit can get subtly wrong.
BYTES_PER_PIXEL = {"yuv420p": 1.5, "yuv420p10le": 3.0}

#: **Headroom left free after the reference and the master** — the master is written to the same
#: volume, and §6g's own figure is *"24.9 GB of 29.98 leaves under 5 GB"*. This is the margin the
#: refusal keeps rather than a prediction of the master's size: a master is a small fraction of
#: its own raw, and the number that matters is that the disk does not reach zero mid-encode.
DISK_MARGIN_BYTES = 3 * 1000 ** 3

#: How many worst frames come back as pictures. **`docs/archive/gate-findings-archive.md` §6g rules the decision on
#: minima and percentiles and the artefacts on a human opening them**, so this is small on
#: purpose: the PNGs are evidence for a claim the aggregates already made.
WORST_N = 3

#: The libvmaf features requested, and the per-frame metric keys each one produces.
#: **`psnr_y` is the LUMA plane and that is a choice, not the only number available.** Chroma
#: PSNR is reported by the same feature; luma is what quality comparisons are conventionally
#: stated in and what the banding question lives in.
FEATURES = "cambi|name=psnr|name=float_ssim"
METRIC_KEYS = {"psnr": "psnr_y", "ssim": "float_ssim", "cambi": "cambi"}

#: **CAMBI RISES WHEN QUALITY FALLS AND THE AGGREGATE NAMES SAY SO** (§17). PSNR and SSIM are
#: better when high, so they report `min`/`p1`; CAMBI is a banding index, so it reports
#: `max`/`p99`. *A record where one of three metrics runs the other way and nothing says so is a
#: reader's error waiting to happen.*
FALLING = {"psnr": True, "ssim": True, "cambi": False}


def required_bytes(width, height, frames, pixel_format):
    """Disk the reference will occupy. **Integers in, exact bytes out.**

    Raises rather than guessing on a format it does not know, because the alternative is a bound
    computed from a default that happens to be smaller than the truth.
    """
    if pixel_format not in BYTES_PER_PIXEL:
        raise WorkerError(INVALID_FIELD_VALUE, (
            "the reference bound has no bytes-per-pixel entry for {!r}; it knows {}. A bound "
            "guessed for an unknown format is worse than no bound.").format(
                pixel_format, ", ".join(sorted(BYTES_PER_PIXEL))))
    return int(int(width) * int(height) * BYTES_PER_PIXEL[pixel_format] * int(frames))


def refuse_if_it_will_not_fit(directory, width, height, frames, pixel_format):
    """**§6g's disk bound, and it is a REFUSAL rather than a hope.**

    *A job that fills the disk at frame 400 has spent the whole model cost to produce nothing,
    which is the most expensive failure this worker can have.* So the requirement is computed from
    the delivered frame size and the planned count **before the first frame**, and the refusal
    names both numbers so a caller can act on it.

    **8K at 10-bit is the case this exists to catch**: 47.78 GB against 29.98 GB free. It refuses
    on the same arithmetic that lets 8-bit through at 23.89 GB.
    """
    # **A COUNT THAT IS ABSENT OR ZERO IS REFUSED RATHER THAN BOUNDED**, and the reason is that
    # the bound would otherwise pass trivially on exactly the input that makes it meaningless:
    # `0` bytes required fits inside any disk, while the encode still writes a second output for
    # however many frames actually stream. *That is a check exercised from the direction the code
    # takes — it cannot fail on the case it exists for — which is `F-2026-08-25-2`'s class, in a
    # bound.* Found in review.
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise WorkerError(INVALID_FIELD_VALUE, (
            "'reference_score' cannot be bounded for a planned frame count of {!r}: the disk "
            "requirement is computed from it, and a count of none reserves nothing while the "
            "encode still writes every frame it produces.").format(frames))
    need = required_bytes(width, height, frames, pixel_format)
    free = shutil.disk_usage(directory).free
    if need + DISK_MARGIN_BYTES > free:
        raise WorkerError(INVALID_FIELD_VALUE, (
            "'reference_score' needs {:.2f} GB to retain {} frames of {}x{} as {}, and this "
            "worker has {:.2f} GB free — with the master written to the same volume, that does "
            "not fit. Refused before the first frame rather than at the last: the frames are "
            "retained for the whole encode, so a disk that fills mid-run costs the entire model "
            "pass and delivers nothing. Ask for it at a smaller size, or at 8-bit, where the "
            "same arithmetic fits.").format(
                need / 1e9, int(frames), int(width), int(height), pixel_format, free / 1e9))
    return need


def score_command(master_path, reference_path, width, height, fps, pixel_format, log_path):
    """The one ffmpeg invocation that produces all three metrics.

    **`libvmaf` WITH CAMBI AS A FEATURE, not the standalone `cambi` filter** (contract §6g). It is
    the form `--enable-libvmaf` guarantees — the pinned build's configuration line carries that
    flag, which is why no dependency was added for this — and it yields all three metrics in ONE
    pass over the frames rather than three. *Whether `vf_cambi` also exists in this build was
    probed and not established, and a spelling that depends on an unproven filter is a spelling
    that fails on a delivered job.*

    **INPUT ORDER IS LOAD-BEARING: distorted first, reference second.** `libvmaf` reads `[0]` as
    the distorted stream and `[1]` as the reference, and swapping them produces plausible numbers
    for the wrong comparison — the class of defect this project keeps recording, wearing a metric.

    **`-f null` because the FILTER is the output.** Nothing is muxed; the scores leave through
    `log_path`, and writing a file nobody reads would cost the disk this section is bounded by.
    """
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # [0] the DELIVERED master — what is being judged.
        "-i", master_path,
        # [1] the reference. rawvideo carries no shape, so its geometry and rate are declared —
        # the same promise `encoder` makes on the way in, and wrong here would silently compare
        # misaligned rows exactly as it would there.
        "-f", "rawvideo", "-pix_fmt", pixel_format,
        "-s", "{}x{}".format(int(width), int(height)), "-r", str(fps),
        "-i", reference_path,
        "-lavfi", "[0:v][1:v]libvmaf=feature=name={}:log_path={}:log_fmt=json".format(
            FEATURES, log_path),
        "-f", "null", "-",
    ]


def _percentile(values, fraction):
    """Nearest-rank percentile on an ascending list. **No interpolation, deliberately.**

    An interpolated percentile reports a number no frame had, and every figure in this block is
    meant to be traceable to a frame someone can open. `fraction` is 0.01 for p1, 0.99 for p99.
    """
    if not values:
        return None
    rank = int(round(fraction * (len(values) - 1)))
    return values[max(0, min(len(values) - 1, rank))]


def _aggregate(per_frame, falling):
    """§17's block for one metric. **`worst_frame` is an INDEX INTO THE DELIVERED MASTER**, so a
    claim about a defect names the frame it is in and a person can go and look at it.

    **The mean is present and is not the verdict** — §6g rules the decision on minima and
    percentiles, and the mean is carried because a corpus wants one comparable number per row.
    """
    ordered = sorted(per_frame)
    total = float(len(per_frame))
    block = {"p50": _percentile(ordered, 0.5),
             "mean": round(sum(per_frame) / total, 6) if total else None}
    if falling:
        block["min"] = ordered[0]
        block["p1"] = _percentile(ordered, 0.01)
        block["worst_frame"] = int(per_frame.index(ordered[0]))
    else:
        block["max"] = ordered[-1]
        block["p99"] = _percentile(ordered, 0.99)
        block["worst_frame"] = int(per_frame.index(ordered[-1]))
    return block


def parse_log(log_path, expected_frames):
    """libvmaf's JSON into §17's three blocks. **A MISSING KEY RAISES RATHER THAN SCORING ZERO.**

    *A scorer that quietly reports `0.0` because a metric name changed produces a record that is
    wrong in the direction nobody checks* — and `0.0` is a plausible CAMBI and a catastrophic
    PSNR, so the same silence reads as "excellent" on one metric and "broken" on another. **Every
    key this function expects is named in the error when it is absent.**

    **`frames` IS RETURNED AND NOT ASSUMED** (§17a). The caller grades it against `retime.n_out`:
    a scorer that covered fewer frames than were delivered reports honest aggregates over the
    wrong population, and the frames it skipped are exactly where a truncated tail would be.
    """
    with open(log_path, encoding="utf-8") as handle:
        parsed = json.load(handle)
    frames = parsed.get("frames")
    if not isinstance(frames, list) or not frames:
        raise WorkerError(INVALID_FIELD_VALUE, (
            "libvmaf wrote no per-frame scores to {} — the log has {!r} where a non-empty "
            "'frames' list was expected, so nothing can be aggregated and an aggregate over "
            "nothing would be a number about no frames.").format(
                log_path, type(frames).__name__))
    # **THE POSITION IN THIS LIST IS USED AS A FRAME INDEX, SO IT IS CHECKED AND NOT ASSUMED.**
    # `_aggregate` reports `worst_frame` as a position in `frames`, and `cut_png` feeds it to
    # `select=eq(n,N)`, which counts DECODED FRAMES OF THE MASTER. Those are the same number only
    # while libvmaf's log is zero-based and contiguous. *If it ever is not, every aggregate stays
    # correct, the frame count still matches, and the PNG is of a different frame — the exact
    # defect this codebase keeps producing, with every visible signal agreeing.* Found in review.
    for position, entry in enumerate(frames):
        number = entry.get("frameNum")
        if number is not None and number != position:
            raise WorkerError(INVALID_FIELD_VALUE, (
                "libvmaf's log has frameNum {!r} at position {} — this parser reports the worst "
                "frame as a POSITION and the PNG is cut by decoded-frame index, so a log that is "
                "not zero-based and contiguous would name a frame and show a different one."
                ).format(number, position))
    series = {}
    for name, key in sorted(METRIC_KEYS.items()):
        values = []
        for entry in frames:
            metrics = entry.get("metrics") or {}
            if key not in metrics:
                raise WorkerError(INVALID_FIELD_VALUE, (
                    "libvmaf's log has no {!r} on frame {} — this build reports {}. The score is "
                    "refused rather than defaulted, because a missing metric scored as 0.0 reads "
                    "as excellent CAMBI and as a broken PSNR from the same silence.").format(
                        key, entry.get("frameNum"), ", ".join(sorted(metrics)) or "nothing"))
            values.append(float(metrics[key]))
        series[name] = values
    counted = len(frames)
    if expected_frames is not None and counted != int(expected_frames):
        raise WorkerError(INVALID_FIELD_VALUE, (
            "libvmaf scored {} frames against {} delivered — §17a: aggregates over a subset are "
            "honest about the wrong population, and the frames it skipped are where a bad tail "
            "lives.").format(counted, int(expected_frames)))
    block = {"frames": counted}
    for name, values in series.items():
        block[name] = _aggregate(values, FALLING[name])
    return block


def worst_frames(block):
    """The frame indices the PNGs are cut at — **deduplicated, in order, at most `WORST_N`.**

    All three metrics often accuse the SAME frame, which is a signal rather than a coincidence:
    it is usually the frame with the hardest content. *Cutting it three times would spend the
    upload on one picture wearing three names.*
    """
    seen, out = set(), []
    for name in ("cambi", "psnr", "ssim"):
        index = (block.get(name) or {}).get("worst_frame")
        if isinstance(index, int) and index not in seen:
            seen.add(index)
            out.append(index)
    return out[:WORST_N]


def cut_png(master_path, index, fps, out_path):
    """One frame of the DELIVERED master as a PNG. **Cut from the master and not the reference**,
    because the artefact being looked for is the encoder's and the reference does not have it.

    **Selected by frame INDEX rather than by timestamp.** `select='eq(n,N)'` counts decoded frames,
    which is the same index `worst_frame` names; a `-ss` seek lands on a keyframe boundary and
    would return a neighbouring picture with every number still looking right.
    """
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", master_path,
            "-vf", "select=eq(n\\,{})".format(int(index)),
            # **`-fps_mode passthrough` and NOT `-vsync 0`.** `-vsync` was removed in ffmpeg 7 and
            # this image pins n8.1.2, where it is not a deprecation warning but
            # *"Unrecognized option 'vsync'"* and exit 8 — **caught by running the command rather
            # than by reading it**, which is the only reason it is not in the built image.
            "-fps_mode", "passthrough", "-frames:v", "1", out_path]


def score(master_path, reference_path, width, height, fps, pixel_format, n_out, workdir,
          run=None):
    """Score the master against the retained reference. **Returns §17's block, or None.**

    **THE REFERENCE IS DELETED WHATEVER HAPPENS**, in a `finally` — it is tens of GB on a volume
    the next job needs, and a run that failed to score has no more right to leave it behind than
    one that succeeded.

    `run` is the subprocess runner, injected so the command construction can be tested without
    an ffmpeg that has libvmaf. **The default is the real one.**
    """
    runner = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
    log_path = os.path.join(workdir, "reference-scores.json")
    try:
        raw_bytes = os.path.getsize(reference_path)
        completed = runner(score_command(master_path, reference_path, width, height, fps,
                                         pixel_format, log_path))
        if completed.returncode != 0:
            raise WorkerError(INVALID_FIELD_VALUE, (
                "libvmaf exited {} scoring the master against the reference: {}").format(
                    completed.returncode, (completed.stderr or "")[-400:]))
        block = parse_log(log_path, n_out)
        block["raw_bytes"] = int(raw_bytes)
        block["worst_pngs"] = []
        for index in worst_frames(block):
            png = os.path.join(workdir, "worst-frame-{:06d}.png".format(index))
            cut = runner(cut_png(master_path, index, fps, png))
            # **A PNG that did not cut is omitted rather than named.** §17b wants a picture that
            # OPENS; listing a key for a file that is not there would make the record claim
            # evidence it does not have.
            if cut.returncode == 0 and os.path.exists(png):
                block["worst_pngs"].append(png)
        return block
    finally:
        # **The disk is released on every path**, including the ones that raised above.
        try:
            if os.path.exists(reference_path):
                os.remove(reference_path)
        except OSError as exc:  # noqa: BLE001 — a cleanup failure must not displace a real error
            print("[reference] could not remove {}: {}".format(reference_path, exc), flush=True)
