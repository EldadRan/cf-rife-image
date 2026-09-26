"""`frame_repair` on the GPU box — `decisions.md` §19g. **One decode, RIFE on the replaced frames
only, one encode.**

    decode.open_source → routec._tensors → repair_plan.emit → routec._to_rgb24_device
                                              ↑ Interpolator.between       → encoder.MasterWriter

**EVERY PIECE IS RETIME'S, AND THAT IS THE RULING RATHER THAN A CONVENIENCE.** *§19g: every
frame, patched or not, takes the same conversion into the same encoder, so a seam cannot come
from colour handling.* An untouched frame goes through exactly the arithmetic a retime's COPY goes
through — `uint8 → float/255 → ×255, round → uint8`, which is exact — and is encoded beside its
patched neighbours by one writer at one setting. **What this module adds is only the order**,
and the order is `repair_plan.emit`'s.

**`routec.retime` IS NOT TOUCHED AND ITS LOOP IS NOT SHARED.** *The loop below repeats that
function's write loop — the checker, the staging buffer, the two clocks, the progress call —
line for line, because §19a rules retime UNCHANGED and a refactor that moved its loop into a
helper would be a change to the one path this wave must not change.* **A fix to one of the two
loops is owed to the other**; they are named here so the next reader looks.

**The heavy imports are inside the functions**, as everywhere on this path, so `handler` imports
this module on a box with no torch and `verify_master` runs on the tests tree.
"""
import os
from fractions import Fraction

import probe
import repair_plan
from errors import CAPACITY_EXCEEDED, INTERNAL, INVALID_SOURCE, WorkerError


def _drain(capture, clock):
    """Frames left in a capture after the plan stopped reading, counted and not decoded."""
    surplus = 0
    if clock is None:
        while capture.grab():
            surplus += 1
    else:
        with clock.timing("decode_s"):
            while capture.grab():
                surplus += 1
    return surplus


def run(source, source_path, frame_count, mapping, anchors, segments, master_path,
        interpolator, fps, identity, crf=None, preset=None, threads=None, sliced_threads=None,
        rc_lookahead=None, codec=None, bit_depth=None, frame_threads=None, pools=None,
        convert_check=None, input_check=None, reference_score=False, encode_defaults=None,
        progress=None, audio_source=None, scale=None, clock=None, armed=None, cap_note=None,
        tensors=None, to_bytes=None):
    """Write the repaired master. Returns the stats the record's `frame_repair` block carries.

    `frame_count` is the source's COUNTED packets, which the plan was built from; `mapping` is
    `repair_plan.plan`'s; `anchors` `{range_id: (a, b)}`; `segments` `{segment_id: {"path",
    "m_file"}}`. **`fps` is the source's `r_frame_rate` STRING**, handed to the writer as it came,
    because §19g holds the output to it as a rational and a float cannot carry `30000/1001`.

    `tensors` and `to_bytes` default to retime's own two conversions (`routec._tensors`,
    `routec._to_rgb24_device`). **They are parameters so this function runs on a box with no torch**
    — the builder's end-to-end test drives it with numpy stand-ins and a stand-in model, and reads
    the master back with the kit's witness. *Production passes neither.*
    """
    import encoder  # noqa: PLC0415 — GPU-box imports, like the rest of this path
    import ladder  # noqa: PLC0415
    import reference  # noqa: PLC0415
    import routec  # noqa: PLC0415
    from decode import open_source  # noqa: PLC0415

    tensors = tensors or routec._tensors  # noqa: SLF001 — retime's own conversion, by design
    to_bytes = to_bytes or routec._to_rgb24_device  # noqa: SLF001
    codec = encoder.resolve_codec(codec)
    reference_path = None
    captures = []
    try:
        capture, shape = open_source(source_path)
        captures.append(capture)
        width, height = shape["width"], shape["height"]
        # **THE CAP'S SECOND TEST, ON THE WRITER'S OWN DIMENSIONS** — retime's rule (§6d-1): the
        # handler capped on the prober's answer, and this is the decoder's.
        delivered_pixels = int(width) * int(height)
        cap_here = ladder.max_delivered_frames(delivered_pixels)
        if cap_note is not None:
            cap_note.update(delivered_height=int(height), delivered_width=int(width),
                            delivered_step=ladder.step_for(delivered_pixels),
                            delivered_pixels=delivered_pixels, frame_cap_delivered=cap_here,
                            n_out=frame_count)
        if frame_count > cap_here:
            raise WorkerError(
                CAPACITY_EXCEEDED,
                "this repair delivers {} frames and the limit for a {} frame is {}. This is the "
                "SECOND cap test, on the {}x{} frame the decoder returns; reaching it means the "
                "prober and the decoder disagree about the source's size.".format(
                    frame_count, ladder.step_for(delivered_pixels), cap_here, width, height))

        peak_reset = routec._reset_peak()  # noqa: SLF001
        input_checker = input_check if isinstance(input_check, routec.InputCheck) else (
            routec.InputCheck() if input_check else None)
        decoded = routec.DecodeCount()
        staging = routec.PinnedStaging()
        source_frames = tensors(
            routec.frames_from(capture, expect=(height, width), clock=clock, count=decoded),
            interpolator.device, clock=clock, checker=input_checker)
        seg_captures, seg_counts, seg_frames = {}, {}, {}
        for sid, seg in segments.items():
            seg_capture, _seg_shape = open_source(seg["path"])
            captures.append(seg_capture)
            seg_captures[sid] = seg_capture
            seg_counts[sid] = routec.DecodeCount()
            # **The same conversion as the source's and no input check**: `InputCheck` keys its
            # comparisons on the SOURCE's frame index, and a segment frame there would be graded
            # against the wrong one.
            seg_frames[sid] = tensors(
                routec.frames_from(seg_capture, expect=(height, width), clock=clock,
                                   count=seg_counts[sid]),
                interpolator.device, clock=clock)

        cache = {}
        stream = repair_plan.emit(
            mapping, anchors, source_frames, seg_frames,
            lambda key, frame_a, frame_b, t: interpolator.between(cache, key, frame_a, frame_b,
                                                                  t, clock))
        n_synth = repair_plan.synthesised(mapping)

        # ── the encoder, resolved exactly as `routec.retime` resolves it ────────────────────────
        encode_settings, provenance = encoder.resolve_defaults(
            delivered_pixels, codec=codec, threads=threads, sliced_threads=sliced_threads,
            rc_lookahead=rc_lookahead)
        if encode_defaults is not None:
            encode_defaults.update(provenance)
        print("[encode] defaults {} at {} delivered pixels ({}x{}, boundary {}): {}".format(
            provenance["basis"], delivered_pixels, width, height, provenance["boundary"],
            " ".join("{}={}".format(name, encode_settings[name])
                     for name in encoder.AREA_FIELDS if name in encode_settings)
            or "no x264 thread settings — this codec has no such table"), flush=True)
        encode_crf = crf if crf is not None else encoder.DEFAULT_CRF
        encode_preset = preset if preset is not None else encoder.DEFAULT_PRESET
        substituted = [name for name, value in (("crf", crf), ("preset", preset))
                       if value is None]
        encode_arm = dict(encode_settings, crf=encode_crf, preset=encode_preset)
        if codec == "h265":
            threading, threading_basis = encoder.x265_threading(
                frame_threads, pools, delivered_height=int(height), delivered_frames=frame_count)
            print("[encode] x265 threading: {}".format(" ".join(
                "{}={} ({})".format(name, threading[name], threading_basis[name + "_basis"])
                for name in ("frame_threads", "pools"))), flush=True)
            encode_arm.update(threading)

        reference_block = None
        reference_format = None
        if reference_score:
            reference_path = os.path.join(os.path.dirname(master_path) or ".", "reference.raw")
            reference_format = encoder.pixel_format(bit_depth)
            reference.refuse_if_it_will_not_fit(os.path.dirname(reference_path) or ".", width,
                                                height, frame_count, reference_format)

        # **§19g: the ladder and the table, counting the SOURCE's frames as delivered.** *`n_synth`
        # is what the model actually runs on — a handful where a retime runs it on most — and the
        # ruled seed does not read it; the fit beside it does, and says it is outside its corpus.*
        estimate = None
        if progress is not None:
            progress.plan_frames(frame_count)
            estimate = routec._seed_estimate(  # noqa: SLF001 — retime's own seeding, by design
                progress, source, {"n_out": frame_count, "n_synth": n_synth}, scale,
                encode_arm, armed, substituted, delivered_pixels=delivered_pixels, codec=codec)

        writer_cm = encoder.MasterWriter(
            master_path, width, height, fps, identity,
            audio_source=audio_source, audio_codec=source.get("audio_codec"),
            audio_limit_s=source.get("video_duration_s"), codec=codec, bit_depth=bit_depth,
            frame_threads=frame_threads, pools=pools, delivered_frames=frame_count,
            reference_path=reference_path, crf=encode_crf, preset=encode_preset,
            **encode_settings)
        if progress is not None:
            progress.begin_phase()
        checker = convert_check if isinstance(convert_check, routec.ConvertCheck) else (
            routec.ConvertCheck() if convert_check else None)
        try:
            # ── RETIME'S WRITE LOOP (`routec.retime`), REPEATED — see the module docstring ───
            with routec._held_alive(progress), writer_cm as writer:  # noqa: SLF001
                for frame in stream:
                    before = checker.snapshot(frame) if checker is not None else None
                    if clock is None:
                        payload = to_bytes(frame, staging)
                    else:
                        with clock.timing("convert_out_s"):
                            payload = to_bytes(frame, staging)
                    if checker is not None:
                        checker.compare(writer.frames_written, before, frame, payload)
                    if clock is None:
                        writer.write(payload)
                    else:
                        with clock.timing("write_wait_s"):
                            writer.write(payload)
                    staging.released()
                    if progress is not None:
                        try:
                            progress.frames(writer.frames_written, phase="interpolate")
                        except Exception as exc:  # noqa: BLE001 — never at the cost of a master
                            print("[progress] frame emit failed at {} ({}: {})".format(
                                writer.frames_written, type(exc).__name__, exc), flush=True)
                if progress is not None:
                    try:
                        progress.phase("draining", force=True)
                    except Exception as exc:  # noqa: BLE001 — never at the cost of a master
                        print("[progress] draining phase not emitted ({}: {})".format(
                            type(exc).__name__, exc), flush=True)
        except WorkerError as exc:
            peak = writer_cm.encoder_peak_rss_gb
            if peak is None:
                raise
            raise WorkerError(
                exc.code, "{} — ffmpeg reached {} GiB RSS before it stopped, over {} frame(s) "
                "written".format(exc.message, peak, writer_cm.frames_written),
                remedy=exc.remedy, shortfall=exc.shortfall) from exc
        finally:
            if clock is not None:
                clock.drain_s = writer_cm.drain_s
            print("[encode] ffmpeg peak RSS {} GiB over {} frame(s)".format(
                writer_cm.encoder_peak_rss_gb, writer_cm.frames_written), flush=True)
            routec._print_write_distribution(writer_cm)  # noqa: SLF001

        # ── §19d: THE COUNT AGREEMENT, CHECKED ON THE ONE DECODE ───────────────────────────
        #
        # **The plan was built from the counted PACKETS; this is what the decoder handed over.**
        # *Retime tolerates two frames here (`routec.SURPLUS_TOLERANCE_FRAMES`); §19d rules that
        # tolerance does NOT apply — "indices are exact or they are wrong".* A decoder that runs
        # short raised inside `emit` already; this catches one that runs LONG.
        source_decoded = decoded.decoded + _drain(capture, clock)
        if source_decoded != frame_count:
            raise WorkerError(
                INVALID_SOURCE,
                "the source decodes to {} frames and its container holds {} video packets. "
                "Frame indices are exact or they are wrong, so a repair addressed by index is "
                "refused rather than delivered shifted (retime's two-frame tolerance does not "
                "apply to this operation).".format(source_decoded, frame_count))
        for sid, seg in segments.items():
            seg_decoded = seg_counts[sid].decoded + _drain(seg_captures[sid], clock)
            if seg_decoded != seg["m_file"]:
                raise repair_plan.Refused(
                    INVALID_SOURCE,
                    "segment '{}' decodes to {} frames and its container holds {} video packets, "
                    "and its M was counted from the packets — so the frames it contributed are not "
                    "the ones the plan named.".format(sid, seg_decoded, seg["m_file"]), sid)

        if reference_path is not None:
            if progress is not None:
                try:
                    progress.phase("scoring", force=True)
                except Exception as exc:  # noqa: BLE001 — never at the cost of a master
                    print("[progress] scoring phase not emitted ({}: {})".format(
                        type(exc).__name__, exc), flush=True)
            try:
                with routec._held_alive(progress):  # noqa: SLF001
                    reference_block = reference.score(
                        master_path, reference_path, width, height,
                        float(Fraction(fps)), reference_format, frame_count,
                        os.path.dirname(master_path) or ".")
            except Exception as exc:  # noqa: BLE001 — a score must never displace a master
                print("[reference] not scored ({}: {})".format(type(exc).__name__, exc),
                      flush=True)
                reference_block = None

        return dict(
            n_in=source_decoded,
            n_out=writer.frames_written,
            n_synth=n_synth,
            n_replaced=sum(1 for entry in mapping if entry[0] != repair_plan.SRC),
            scale=scale,
            peak_vram_gb=routec._read_peak(peak_reset),  # noqa: SLF001
            decoder_threads=shape.get("n_threads"),
            convert_check=(checker.block() if checker is not None else None),
            input_check=(input_checker.block() if input_checker is not None else None),
            estimate=estimate,
            encoder_peak_rss_gb=writer.encoder_peak_rss_gb,
            reference=reference_block,
            **routec._encode_fields(writer))  # noqa: SLF001
    finally:
        # Retime's order and retime's reason: the reference's disk first, then the captures, and a
        # release that raises must not displace a delivery or the failure being unwound.
        if reference_path is not None:
            try:
                if os.path.exists(reference_path):
                    os.remove(reference_path)
            except OSError as exc:
                print("[reference] could not remove {}: {}".format(reference_path, exc),
                      flush=True)
        for capture in captures:
            try:
                capture.release()
            except Exception as exc:  # noqa: BLE001 — a release must never displace a delivery
                print("[decode] capture.release() failed ({}: {})".format(
                    type(exc).__name__, exc), flush=True)


def verify_master(master_path, source_counted):
    """§19g's three checks on the written master, **before upload, or the job fails**.

    `source_counted` is `probe.counted_frames` of the source. **One reading of the master, three
    comparisons, all exact where the ruling says exact:**

        frames        == the source's counted frames
        r_frame_rate  == the source's, as RATIONALS
        duration      within one frame of the source's video-stream duration

    *The duration bound is computed on the exact decimal ffprobe printed, never a float*: a drift
    of exactly one frame sits ON the bound, and float rounding decides that by noise.

    Returns the master's reading; raises `WorkerError(INTERNAL)` — the worker wrote the file, so
    a master that fails its own check is this worker's fault and not the caller's.
    """
    master = probe.counted_frames(master_path, INTERNAL, "counting the master's packets")
    failures = []
    if master["frames"] != source_counted["frames"]:
        failures.append("the master holds {} frames and the source {}".format(
            master["frames"], source_counted["frames"]))
    try:
        rates_equal = Fraction(master["r_frame_rate"]) == Fraction(source_counted["r_frame_rate"])
    except (TypeError, ValueError, ZeroDivisionError):
        rates_equal = False
    if not rates_equal:
        failures.append("the master's r_frame_rate is {} and the source's {}".format(
            master["r_frame_rate"], source_counted["r_frame_rate"]))
    try:
        drift = abs(Fraction(repr(master["stream_duration_s"]))
                    - Fraction(repr(source_counted["stream_duration_s"]))) \
            * Fraction(source_counted["r_frame_rate"])
        if drift > 1:
            failures.append("the master's video stream lasts {} s against the source's {} s, "
                            "{:.3f} frames apart; the bound is one".format(
                                master["stream_duration_s"],
                                source_counted["stream_duration_s"], float(drift)))
    except (TypeError, ValueError, ZeroDivisionError):
        failures.append("the duration could not be compared: master {!r}, source {!r}".format(
            master["stream_duration_s"], source_counted["stream_duration_s"]))
    if failures:
        raise WorkerError(
            INTERNAL,
            "the repaired master failed §19g's pre-upload check and was NOT delivered: {}. A "
            "repair changes frames and nothing else, so any of these is the worker's defect."
            .format("; ".join(failures)))
    return master
