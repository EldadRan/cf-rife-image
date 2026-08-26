"""Route C: `capture → INTERP → encoder`. No upscaler, no SeedVR2, no plan-and-retry ladder.

**The route with no model of ours in it.** Contract §6b: route C never loads SeedVR2, so this
path shares nothing with `_upscale_once` beyond the decoder and the writer — no rung ladder, no
residency schedule, no host guard promotion, none of which describe anything about a job whose
working set is one frame pair.

The pieces are already built and this is the wiring between them:

    decode.open_source   →   interpolate.Interpolator(rife.Rife)   →   encoder.MasterWriter

**cv2 arrived through the vendored CLI and now does not** (excision Wave 1). The reason it
did was the one-cv2-per-process property, which is a statement about a process that imported
SeedVR2 — this one has no vendored tree to import and no second cv2 to be one of. `decode` owns
the import and `open_source` no longer takes a CLI it never used for anything else.
"""
import numpy as np

from errors import INVALID_SOURCE, WorkerError

#: BGR uint8 out of cv2, RGB float in [0, 1] for RIFE, rgb24 bytes for the writer. Stated once
#: here because a channel order that is wrong is a picture that still plays.
_CHANNELS = 3

#: **Slack on the derived count, in source frames.** Zero was wrong: the count comes from a
#: duration stored to the millisecond times a rate, and edit-list trims are named in
#: `source_frame_count` as the reason a container and a decode disagree on real input. §2's
#: duration bound is +-2 output frames and this is its counterpart on the input side.
SURPLUS_TOLERANCE_FRAMES = 2

#: **The frozen encode settings are `encoder`'s now, as named fields** (contract §6a). This was
#: `FRUGAL_X264`, one string chosen to make the 8K run fit: the run was reaped in x264 at ~46 GiB
#: while this side held one frame and a cached pair — the pipe is backpressure and it was working,
#: and what filled memory was the encoder's own working set, one frame in flight per encoding
#: thread plus `medium`'s 40-frame lookahead plus references, dozens of 50 MiB frames at once. At
#: 4K the same arithmetic fits, which is why five 4K runs showed nothing.
#:
#: **The string is gone and the values are not.** `encoder.DEFAULT_THREADS`,
#: `DEFAULT_SLICED_THREADS` and `DEFAULT_RC_LOOKAHEAD` hold exactly what it held, and
#: `encoder.x264_params()` with no arguments reproduces it byte for byte — so the corpus taken
#: under the constant and the corpus taken under the fields are one corpus. What changed is that
#: a caller can now move them, which is what §6a's campaign needs and what a frozen string
#: refused by construction.


def source_frame_count(source):
    """How many frames the plan is sized from, and why it is not read from the container.

    `probe.probe_source` refuses to report a frame count on arbitrary input for a documented
    reason: an upstream trim can bound a video with an MP4 edit list, leaving frames in the
    stream that the container still counts, so a probe and a decode disagree on precisely the
    files this worker is sent. That argument holds here.

    So the count is **derived from two measured quantities** — the stream's own duration and its
    rate — and then checked against the decode. `stream()` already refuses a source shorter than
    the count it was given; `retime` below refuses one longer. Between them the derivation is
    verified in both directions rather than trusted.
    """
    fps = source.get("fps")
    duration = source.get("video_duration_s") or source.get("duration_s")
    if not fps or not duration:
        raise WorkerError(
            INVALID_SOURCE,
            "a retime needs the source's rate and duration and this container reports "
            "fps={!r} duration={!r}. Neither is guessed: the frame plan is sized from them and a "
            "wrong size is a wrong output length.".format(fps, duration))
    count = int(round(float(duration) * float(fps)))
    if count < 2:
        # `build_plan` refuses this too, but with a bare `ValueError`; every other refusal on
        # this path is a `WorkerError` the caller can read.
        raise WorkerError(
            INVALID_SOURCE,
            "a retime needs at least two source frames and this container's duration ({}s) at "
            "{} fps implies {}".format(duration, fps, count))
    return count


def _to_tensor(frame_bgr, torch):
    """cv2's BGR uint8 `H×W×3` to RIFE's RGB float `1×3×H×W` in [0, 1]."""
    rgb = frame_bgr[:, :, ::-1]
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _to_rgb24(tensor):
    """Back to the writer's contract: `rgb24`, `width × height × 3` bytes.

    Clamped before the cast because a synthesis is a model output and can land a hair outside
    [0, 1]; `uint8` would wrap that into a black pixel in a white region, which is the 16-bit
    downconvert defect in miniature.
    """
    # **`clamp`, not `clamp_`.** A copy or a hold is yielded as it arrived — the caller's own
    # tensor, deliberately never cast — and `.to("cpu", copy=False).float()` on a CPU float32
    # tensor hands back that same object, so an in-place clamp would write through to the source
    # frame. Harmless here, since a decoded frame is already in [0, 1] by construction; wrong as
    # a mechanism, and under route A or B the frames entering the shim are model output.
    array = tensor.detach().to("cpu", copy=False).float().clamp(0.0, 1.0)[0]
    array = (array.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array).tobytes()


def frames_from(capture, expect=None, clock=None):
    """Decode the source into cv2 frames, in order, until it is exhausted.

    `expect` is the `(height, width)` the writer was told, checked once on the first frame.
    **The writer is sized from the container and the bytes come from the decoder**, and nothing
    else compares the two: a rotation matrix or any decoder-side adjustment would produce a byte
    count ffmpeg does not expect, and rawvideo carries no shape — so the master shears from that
    frame on while the process exits 0. `MasterWriter.write` catches a wrong LENGTH; a transposed
    frame of the same length it cannot.
    """
    checked = False
    while True:
        # **`decode_s` is the read and nothing else** (`docs/instrumentation.md` §9a). The shape
        # check below is ours and costs nothing; charging it to the decoder would be charging
        # cv2 for our own assertion.
        if clock is None:
            ok, frame = capture.read()
        else:
            with clock.timing("decode_s"):
                ok, frame = capture.read()
        if not ok or frame is None:
            return
        if not checked and expect is not None:
            checked = True
            if tuple(frame.shape[:2]) != tuple(expect):
                raise WorkerError(
                    INVALID_SOURCE,
                    "the decoder returns {}x{} frames but the container reports {}x{}; the "
                    "encode is sized from the container and would shear."
                    .format(frame.shape[1], frame.shape[0], expect[1], expect[0]))
        yield frame


def retime(source, source_path, master_path, interpolator, target_fps, identity,
           snap_tolerance=None, crf=None, audio_source=None, progress=None,
           variant="direct", scale=None, preset=None, threads=None,
           sliced_threads=None, rc_lookahead=None, clock=None):
    """Decode, interpolate, encode. Returns the stats the plan produced.

    `snap_tolerance` is passed through as given and **is not defaulted here** (contract §5c): a
    tolerance defaulted to zero would ship the unsnapped plan as the ruled answer before the
    benchmark that decides it has run. `None` means unruled, and the shim reads it as zero for
    arithmetic while the request records that nobody chose.
    """
    import encoder  # noqa: PLC0415 — imported here so this module stays importable without one
    import variants  # noqa: PLC0415

    from decode import open_source  # noqa: PLC0415 — cv2 is a GPU-box import, like the rest

    capture, shape = open_source(source_path)
    try:
        width, height = shape["width"], shape["height"]
        n_in = source_frame_count(source)
        # **Peak VRAM measured rather than reported from a panel.** Route C has no plan, so
        # nothing else on this path produces one — and §8b's `--scale` axis exists to falsify
        # `w_scaling: FLAT`, which IS this reading. Reset before and read after, so the number is
        # this job's high-water mark rather than the process's history.
        peak_reset = _reset_peak()
        stream, stats = variants.run(
            variant, interpolator,
            _tensors(frames_from(capture, expect=(height, width), clock=clock), clock=clock),
            # **The declared cadence, from the same object `n_in` was derived from.** These two
            # lines used to disagree: `source_frame_count` reads `source["fps"]` — the container's
            # `r_frame_rate` — while this read cv2's `CAP_PROP_FPS`, which is `avg_frame_rate`. A
            # count derived at one rate, divided against another, nine lines apart. On a 23.976
            # source the plan came out 18/455/2 where the contract's arithmetic says 16/458/1, and
            # the response reported the rate the plan had not used.
            n_in=n_in, src_fps=source["fps"], dst_fps=target_fps,
            tol=snap_tolerance or 0.0, clock=clock)

        # **Frame-level, because decode, RIFE and encode are ONE streaming loop** (contract §1).
        # There are no phases to report the completion of — the writer pulls each frame through
        # the whole chain — so "decode complete" is never true and the only quantity that is true
        # per frame is the frame count. Both halves were already in hand and neither was used:
        # `n_out` in the stats returned before the loop begins, and `frames_written` on the writer.
        estimate = None
        if progress is not None:
            progress.plan_frames(stats.get("n_out"))
            # **The ETA gets something to plan from at last** (`docs/instrumentation.md` §8d,
            # contract §9b). `Progress.expect` has existed since it was written and route C had
            # nothing to seed it with, so the first ETA came from `observed` — elapsed against a
            # work fraction of nearly nothing — and at 8K opened at 11,553 s against an outturn
            # of 1,733 s.
            #
            # **Seeded HERE, and not earlier, because the plan is what the estimate is a
            # function of.** `n_synth` is what distinguishes a 20->60 retime from a 23.976->60
            # one at the same output count, and it does not exist until `variants.run` has
            # returned. Everything above this line is the fetch, the probe and the model load,
            # which is what `begin_phase` below excludes from the rate rather than what the ETA
            # is priced from.
            estimate = _seed_estimate(progress, source, stats, scale)

        writer_cm = encoder.MasterWriter(
            master_path, width, height, float(target_fps), identity,
            audio_source=audio_source, audio_codec=source.get("audio_codec"),
            audio_limit_s=source.get("video_duration_s"),
            crf=crf if crf is not None else encoder.DEFAULT_CRF,
            # **`None` means "the caller did not choose", so the default is applied HERE and in
            # one place.** `validation` already defaults every one of these, so nothing reaching
            # this line through `handle` is None — these guards are for the direct callers a test
            # makes, and they resolve to the same constants `validation` would have used.
            preset=preset if preset is not None else encoder.DEFAULT_PRESET,
            threads=threads if threads is not None else encoder.DEFAULT_THREADS,
            sliced_threads=(sliced_threads if sliced_threads is not None
                            else encoder.DEFAULT_SLICED_THREADS),
            rc_lookahead=(rc_lookahead if rc_lookahead is not None
                          else encoder.DEFAULT_RC_LOOKAHEAD))
        # **The rate clock starts where the frames do** (§8d). `_phase_started` was set when
        # `Progress` was constructed — before the fetch, before the probe, before the model
        # load — so the first measured seconds-per-frame amortised all of that into the per-frame
        # figure and flattered the ETA. `begin_phase` has had zero callers since the day it was
        # written and its own docstring says exactly this.
        #
        # **It cannot exclude ffmpeg's start-up and does not claim to.** `MasterWriter.__enter__`
        # is `return self` (`encoder.py:238-240`) and the process is spawned lazily inside the
        # first `write` (`encoder.py:262-263`), so ffmpeg's start-up is inside frame 1 wherever
        # this line goes. That is one more reason the FIRST ETA is the estimator's and not this
        # clock's: `_seconds_per_frame` is a cumulative average and amortises the warm-up away
        # over a few dozen frames, but the first reading is the warm-up.
        if progress is not None:
            progress.begin_phase()
        # **The peak is read on the FAILURE path too, and that is the path it exists for.** It
        # was read only after the `with` — so an encoder reaped by the kernel, which is the exact
        # event this instrumentation was added for, propagated past the read and took the sampled
        # maximum with it. A second fifty-minute run would have had its ceiling inferred from a
        # kill again, which is what the measurement was meant to end. The number now rides on the
        # refusal's own message, where the diagnostics bundle and the run-record both carry it.
        try:
            with writer_cm as writer:
                for frame in stream:
                    # **`convert_s` and `write_wait_s` split what used to be one expression**
                    # (§9a). `write_wait_s` is the producer BLOCKED in `write` — the encoder's
                    # share seen from the only side that can measure it without instrumenting
                    # ffmpeg, because `write` returns when the pipe accepts the frame
                    # (`encoder.py:270`).
                    if clock is None:
                        writer.write(_to_rgb24(frame))
                    else:
                        with clock.timing("convert_s"):
                            payload = _to_rgb24(frame)
                        with clock.timing("write_wait_s"):
                            writer.write(payload)
                    # **Every frame, and the rate limiter decides what is SENT.** progress._emit
                    # drops anything inside MIN_INTERVAL_S, so calling per frame costs a
                    # comparison and publishes at the module's own cadence rather than at one this
                    # loop would have to invent. `boundary=True` because in a one-frame-at-a-time
                    # stream every written frame IS a completed unit of work — there are no chunks
                    # whose mid-flight count would make `elapsed/done` lie.
                    if progress is not None:
                        progress.frames(writer.frames_written, phase="interpolate")
        except WorkerError as exc:
            peak = writer_cm.encoder_peak_rss_gb
            if peak is None:
                raise
            raise WorkerError(
                exc.code,
                "{} — ffmpeg reached {} GiB RSS before it stopped, over {} frame(s) written "
                "with x264-params {!r}".format(
                    exc.message, peak, writer_cm.frames_written, writer_cm.x264_params),
                remedy=exc.remedy, shortfall=exc.shortfall) from exc
        finally:
            # Said out loud whichever way the encode ended, because a log line survives a bundle
            # that was never written.
            print("[encode] ffmpeg peak RSS {} GiB over {} frame(s)".format(
                writer_cm.encoder_peak_rss_gb, writer_cm.frames_written), flush=True)
        # **The other half of the derived count.** `stream()` refuses a source shorter than the
        # plan; this refuses one longer. A container whose duration and rate imply fewer frames
        # than it holds would otherwise deliver a silently truncated retime.
        # `grab()` rather than `read()`: this counts, it does not look. Decoding the remainder
        # of a long file to produce a number we immediately refuse on is work nobody asked for.
        # **Charged to `decode_s`, because it is the decoder doing work** (§9a). It is `grab()`
        # rather than `read()` deliberately — this counts, it does not look — but a sweep over
        # whatever the source holds beyond the plan is decoder time that scales with the file,
        # and leaving it in the residual made the residual grow with clip length while it was
        # documented as fixed cost.
        surplus = 0
        if clock is None:
            while capture.grab():
                surplus += 1
        else:
            with clock.timing("decode_s"):
                while capture.grab():
                    surplus += 1
        # **Two frames of slack, not zero, and the docstring above says why it is needed.** A
        # count derived from a duration stored to the millisecond drifts by a frame over a long
        # clip, and an edit-list trim is exactly the case where a container's numbers and a
        # decode disagree by a little. §2's bound is +-2 output frames; the same tolerance in
        # source frames is the smallest one that does not refuse arithmetic noise. Beyond it the
        # disagreement is structural rather than rounding, and a retime would be truncated.
        if surplus > SURPLUS_TOLERANCE_FRAMES:
            raise WorkerError(
                INVALID_SOURCE,
                "the source holds {} frame(s) beyond the {} its duration and rate imply, which "
                "is past the {}-frame tolerance for rounding, so the retime would have been "
                "truncated. The container's own numbers disagree with its content."
                .format(surplus, n_in, SURPLUS_TOLERANCE_FRAMES))
        # **Always present, None when nothing measured it.** Setting the key only on success
        # makes "no GPU here" and "nobody thought to ask" reach a ledger row identically — as an
        # absent key and a `KeyError` — which is the distinction `build_identity`'s docstring
        # already argues for every field it reports.
        return dict(stats, scale=scale, peak_vram_gb=_read_peak(peak_reset),
                    # **The estimate rides out with the stats because this is its only channel**
                    # — `retime` is handed no `trace` and should not be. `handler._retime` lifts
                    # it into the record's `estimate` block and files the REST under `retime`, so
                    # the document has one home for a prediction and one for an outturn. §9b
                    # requires every time answer to carry its corpus, its reading count and its
                    # spread, so what travels is the whole answer and not the per-frame figure.
                    # Null where nothing could be priced, which is the honest shape of an
                    # unquotable.
                    estimate=estimate,
                    # ffmpeg's own high-water mark, beside the GPU's. The 8K ceiling was in the
                    # encoder rather than the model, and neither number alone would have said so.
                    encoder_peak_rss_gb=writer.encoder_peak_rss_gb,
                    # **What RAN, read off the writer** (§6c). A campaign attributing a
                    # difference between two arms to the settings that differed needs the settings
                    # that were used and not the module's defaults — and now that they can move,
                    # those are two different things.
                    x264_params=writer.x264_params,
                    # **ALL FIVE, `crf` INCLUDED, AND THE RECORD IS WHY.** `crf` and `preset` used
                    # to be added to the ENVELOPE by `handler._retime` and never reached `stats` —
                    # so `trace["retime"]`, which is built from `stats`, filed both as null on
                    # every run ever written, while the envelope beside it said `crf: 12`.
                    # `instrumentation.md` §2 names `crf, preset` as the two missing settings and
                    # says why: *a corpus recording three of five cannot attribute a difference*.
                    # The corpus is the RECORD. Reading all five off the writer that ran is what
                    # puts them in both artefacts and makes them impossible to disagree.
                    crf=writer.crf,
                    preset=writer.preset,
                    threads=writer.threads,
                    sliced_threads=writer.sliced_threads,
                    rc_lookahead=writer.rc_lookahead,
                    # **Same defect, same fix, one line further.** §2 lists `snap_tolerance` among
                    # the axes a run must carry and the record filed it null for the same reason.
                    # `target_fps` is what the whole job was FOR. Both were envelope-only.
                    target_fps=float(target_fps),
                    snap_tolerance=snap_tolerance)
    finally:
        if capture is not None:
            capture.release()


def _seed_estimate(progress, source, stats, scale):
    """Price the job, seed the ETA with it, and hand the whole answer back. **Never raises.**

    Returns the §9b estimate — point, band, basis and corpus — or None where it could not be
    priced. **A refusal to quote is not a failure of the job**: the estimator exists to make the
    ETA better than `observed`, and a run that cannot be priced falls back to exactly the
    behaviour it had before this line existed.
    """
    import estimator  # noqa: PLC0415 — pure-python, imported here like everything else on this path

    try:
        n_in = source_frame_count(source)
        per_frame, estimate = estimator.seconds_per_frame(
            source["width"], source["height"], n_in,
            stats.get("n_out"), stats.get("n_synth"), scale=scale or 1)
        # **Labelled `predicted_<basis>` by `expect` itself**, so the payload distinguishes a
        # planned estimate from a measured one — which is the whole reason `eta_basis` exists and
        # the reason §8g grades `eta.first_basis` as well as `eta.first_s`.
        progress.expect(per_frame, basis=estimator.BASIS)
        # **PUBLISHED HERE, and without this line the seed is unreachable.** `eta_s()` answers
        # from `_seconds_per_frame` whenever it exists, and route C sets it on the FIRST written
        # frame — every frame is a boundary on a one-frame-at-a-time stream — before that frame's
        # payload is built. So the first payload that could carry an ETA already had a measured
        # one, taken from a single frame that also paid for ffmpeg's start-up and the model's
        # first pass. The record would have filed `eta.first_basis: "measured"` on every run,
        # with a first ETA of `(n_out - 1) x (the cost of frame 1)` — §8d's 11,553-second failure
        # rebuilt exactly, wearing the label that says it was measured.
        #
        # The two `phase()` calls in `handler._retime` cannot do this job: both run before
        # `plan_frames`, so `_estimated_frames` is None and `eta_s()` returns None with no key.
        # **Forced past the rate limiter**, because §8f wants the ETA as PUBLISHED and a payload
        # the limiter dropped was never published.
        progress.phase("interpolate", pct=10, force=True,
                       frames_expected=stats.get("n_out"))
        print("[eta] estimator {}: {:.0f}s for {} frame(s) ({:.0f}-{:.0f}s, {} readings, "
              "spread {:.0%})".format(
                  estimate["basis"], estimate["point_s"], stats.get("n_out"),
                  estimate["low_s"], estimate["high_s"], estimate["corpus"]["readings"],
                  estimate["band_frac"]), flush=True)
        return estimate
    except Exception as exc:  # noqa: BLE001 — an estimate must never cost a delivered master
        print("[eta] not priced ({}: {}); the ETA falls back to what this run measures".format(
            type(exc).__name__, str(exc)[:200]), flush=True)
        return None


def _reset_peak():
    """Zero CUDA's high-water mark, returning whether it could be. `False` on CPU or no torch."""
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return False
        torch.cuda.reset_peak_memory_stats()
        return True if torch.cuda.memory_allocated() >= 0 else False
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return False


def _read_peak(was_reset):
    """The job's peak allocation in GiB, or **None where nothing measured it**.

    None rather than zero, and rather than a figure from a telemetry panel: a panel reading is
    not something a ledger row can cite, and a fabricated number is indistinguishable from a
    measurement — which is the rule four other places in this release already follow.
    """
    if not was_reset:
        return None
    try:
        import torch  # noqa: PLC0415
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001
        return None


def _tensors(frames, clock=None):
    """cv2 frames to tensors, lazily, one at a time — never a list, whatever the clip length.

    **Both halves of `convert_s` are here and in `_to_rgb24`** (§9a): BGR uint8 to RGB float on
    the way in, and the reverse on the way out. One field rather than two because they are one
    activity — the single-threaded float work between the decoder and the model — and §9's whole
    complaint is fields whose boundary is fictional.
    """
    import torch  # noqa: PLC0415 — the interpolator has already imported it by the time we run

    for frame in frames:
        if clock is None:
            yield _to_tensor(frame, torch)
        else:
            with clock.timing("convert_s"):
                tensor = _to_tensor(frame, torch)
            yield tensor
