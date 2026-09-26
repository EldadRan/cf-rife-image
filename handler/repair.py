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
import subprocess
import tempfile
import time
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
        tensors=None, to_bytes=None, source_format=False):
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
            source_format=source_format, **encode_settings)
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
                # **Added to, not replaced**: a repair whose copy fell back (§20e) already
                # drained its span encoders into this clock, and that time belongs in the
                # identity rather than in the residual. On every other run it starts at None.
                banked = [d for d in (clock.drain_s, writer_cm.drain_s) if d is not None]
                clock.drain_s = sum(banked) if banked else None
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


def _frames_of(argv, width, height, count=None, clock=None, what="a decode"):
    """Raw BGR frames off an ffmpeg pipe, as cv2-layout `(height, width, 3)` uint8 arrays —
    the shape `routec._tensors` takes from `decode.open_source`. A generator; the process is
    closed when the generator is. `count` is a `routec.DecodeCount`-like object or None."""
    import numpy as np  # noqa: PLC0415 — GPU-box import, as everywhere on this path

    size = width * height * 3
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            if clock is None:
                raw = _read_exactly(proc.stdout, size)
            else:
                with clock.timing("decode_s"):
                    raw = _read_exactly(proc.stdout, size)
            if len(raw) != size:
                return
            if count is not None:
                count.decoded += 1
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _read_exactly(stream, size):
    chunks, got = [], 0
    while got < size:
        chunk = stream.read(size - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


class _Count:
    def __init__(self):
        self.decoded = 0


def _died(proc, what, err_file):
    """A writer closed its input before it was done: wait for it and raise `CheckFailed` with
    its own words, so the fallback files ffmpeg's reason rather than a bare BrokenPipeError.
    Found in review."""
    import splice  # noqa: PLC0415

    proc.wait()
    err_file.seek(0)
    raise splice.CheckFailed("{} stopped reading (exit {}): {}".format(
        what, proc.returncode, err_file.read().decode(errors="replace").strip()[-300:]))


def _finish(proc, what, err_file):
    """Close a writer's stdin, wait, and raise `splice.CheckFailed` on a non-zero exit. Returns
    the seconds the close and the wait took: the encoder's drain (`stages.DRAIN`)."""
    import splice  # noqa: PLC0415

    started = time.perf_counter()
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    drained = time.perf_counter() - started
    if proc.returncode != 0:
        err_file.seek(0)
        raise splice.CheckFailed("{} exited {}: {}".format(
            what, proc.returncode, err_file.read().decode(errors="replace").strip()[-300:]))
    return drained


def run_copy(pmap, spans, delay, mapping, anchors, segments, master_path, interpolator,
             workdir, crf, preset, audio_source=None, progress=None, clock=None,
             tensors=None, to_bytes=None, threading=None):
    """`frame_repair`'s copy path — `decisions.md` §20. Writes the spliced master; returns the
    stats the record's `frame_repair` block carries. Raises `splice.CheckFailed` for anything
    the fallback should catch, `WorkerError` for what the full path would refuse too.

    **PER SPAN, TWO PASSES, AND NEITHER HOLDS THE SPAN** (the gate's order: stream everything at
    8K):

      1. The window [start-1, end] is decoded as BGR — both anchors of every item in the span
         lie inside it — and walked by `repair_plan.emit`, the full path's own order, over a
         window of the mapping. Each REPLACED frame goes through the model and `to_bytes`, then
         to one ffmpeg that turns rgb24 into the source's pixel format with the source's matrix
         and range stated (§20c), writing a spool file. *Held at once: `emit`'s two frames.*
      2. The span is decoded again in the source's own pixel format and piped to the span
         encoder, a replaced frame read off the spool in its place. **Untouched frames never go
         through RGB.** *Held at once: one frame.*

    `tensors` and `to_bytes` default to retime's conversions, and are parameters so this runs
    with the stand-in model on a box without torch, as `run` does.
    """
    import routec  # noqa: PLC0415
    import splice  # noqa: PLC0415

    tensors = tensors or routec._tensors  # noqa: SLF001 — the full path's conversion, by design
    to_bytes = to_bytes or routec._to_rgb24_device  # noqa: SLF001
    if len(mapping) != pmap.count:
        # The plan counted the packets one way and the splice another; indices would not name
        # the same frames. Found in review.
        raise splice.NotEligible("the plan holds {} frames and the packet map {}".format(
            len(mapping), pmap.count))
    matrix, rng = pmap.matrix()
    runs = splice.layout(pmap.count, spans)
    parts_dir = os.path.join(workdir, "splice")
    os.makedirs(parts_dir, exist_ok=True)
    parts, copy_argv = splice.copy_parts(pmap, runs, parts_dir)
    recipe = {"copy": copy_argv, "spans": []}
    size = pmap.frame_bytes
    staging = routec.PinnedStaging() if hasattr(routec, "PinnedStaging") else None
    frames_encoded = sum(end - start for start, end, _ in spans)
    replaced_total = 0
    written = 0
    drain_s = 0.0
    span_sets = []
    files = []
    if progress is not None:
        progress.begin_phase()
    for kind, start, end in runs:
        if kind == "copy":
            files.append((parts[start], end - start))
            # **Progress counts the source's frames, copied ones included** — the ETA was
            # planned on all of them (§20g), and a copied run is done the moment it is cut.
            written += end - start
            if progress is not None:
                try:
                    # **`boundary=False`: counted, never timed.** A copied frame costs nothing,
                    # so a rate taken over it would publish a "measured" ETA short by the share
                    # copied. The seed (§20g: the full path's, conservative) stands. Found in
                    # review.
                    progress.frames(written, phase="interpolate", boundary=False)
                except Exception as exc:  # noqa: BLE001 — never at the cost of a master
                    print("[progress] frame emit failed at {} ({}: {})".format(
                        written, type(exc).__name__, exc), flush=True)
            continue
        lo, hi = max(0, start - 1), min(pmap.count - 1, end)
        window = []
        for n in range(lo, hi + 1):
            entry = mapping[n]
            window.append((repair_plan.SRC, n - lo) if entry[0] == repair_plan.SRC else entry)
        replaced = [n for n in range(start, end) if mapping[n][0] != repair_plan.SRC]
        replaced_total += len(replaced)
        window_anchors = {rid: (a - lo, b - lo) for rid, (a, b) in anchors.items()
                          if start <= a + 1 and b - 1 < end}
        # ── pass 1: the replaced frames, through the model, to the spool ────────────────────
        spool = os.path.join(parts_dir, "span{:05d}.yuv".format(start))
        if replaced:
            inputs, select = pmap.from_frame(lo)
            source_rgb = _frames_of(splice.rgb_decoder(pmap.path, inputs, select, hi - lo + 1,
                                                       matrix, rng),
                                    pmap.width, pmap.height, clock=clock)
            seg_frames, seg_counts, seg_gens = {}, {}, []
            used = sorted({entry[1] for entry in window if entry[0] == repair_plan.SEG})
            for sid in used:
                seg = segments[sid]
                seg_counts[sid] = _Count()
                # **The segment's OWN matrix and range**, not the source's: its frames go to RGB
                # by its tags and come back by the source's, which is the conversion a caller's
                # clip needs to sit beside the source's frames.
                gen = _frames_of(splice.rgb_decoder(seg["path"], ["-i", seg["path"]], None,
                                                    seg["m_file"] + 1,
                                                    *splice.colour_of(seg["path"])),
                                 pmap.width, pmap.height, count=seg_counts[sid], clock=clock)
                seg_gens.append(gen)
                seg_frames[sid] = tensors(gen, interpolator.device, clock=clock)
            cache = {}
            stream = repair_plan.emit(
                window, window_anchors, tensors(source_rgb, interpolator.device, clock=clock),
                seg_frames,
                lambda key, frame_a, frame_b, t: interpolator.between(cache, key, frame_a,
                                                                      frame_b, t, clock))
            with tempfile.TemporaryFile() as err:
                converter = subprocess.Popen(
                    splice.rgb_to_source(pmap, len(replaced)) + [spool],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
                failed = True
                try:
                    for offset, frame in enumerate(stream):
                        n = lo + offset
                        if n >= end:
                            break
                        if mapping[n][0] == repair_plan.SRC:
                            continue
                        if clock is None:
                            payload = to_bytes(frame, staging)
                        else:
                            with clock.timing("convert_out_s"):
                                payload = to_bytes(frame, staging)
                        try:
                            converter.stdin.write(payload)
                        except BrokenPipeError:
                            _died(converter, "the RGB-to-source conversion", err)
                        if staging is not None:
                            staging.released()
                    # **§19d's count agreement, on each segment this span read**: the plan's M
                    # was counted from its packets, so the rest of its frames are drained and
                    # counted — a segment is at most a range's length.
                    for sid, gen in zip(used, seg_gens):
                        for _ in gen:
                            pass
                        if seg_counts[sid].decoded != segments[sid]["m_file"]:
                            raise repair_plan.Refused(
                                INVALID_SOURCE,
                                "segment '{}' decodes to {} frames and its container holds {} "
                                "video packets, and its M was counted from the packets — so the "
                                "frames it contributed are not the ones the plan named.".format(
                                    sid, seg_counts[sid].decoded, segments[sid]["m_file"]), sid)
                    failed = False
                except repair_plan.Refused as exc:
                    # **The source window running short is the splice's seek, not the caller's
                    # source**: the plan's count was checked against the packets already. A
                    # segment's own count (item set) stays a refusal. Found in review.
                    if exc.item is not None:
                        raise
                    raise splice.CheckFailed("the span window at [{}, {}] decoded short: {}"
                                             .format(lo, hi, exc.message))
                finally:
                    stream.close()
                    source_rgb.close()
                    for gen in seg_gens:
                        gen.close()
                    if failed:
                        converter.kill()
                        converter.wait()
                if not failed:
                    _finish(converter, "the RGB-to-source conversion", err)
            if os.path.getsize(spool) != size * len(replaced):
                raise splice.CheckFailed("the spool for the span at {} holds {} bytes, not {}"
                                         .format(start, os.path.getsize(spool),
                                                 size * len(replaced)))
        # ── pass 2: the span, in the source's own YUV, to its encoder ───────────────────────
        span_ts = os.path.join(parts_dir, "span{:05d}.ts".format(start))
        argv = splice.span_encoder(pmap, end - start, span_ts, delay, crf, preset, threading)
        decode_argv = splice.span_decoder(pmap, start, end - start)
        recipe["spans"].append({"span": [start, end], "decode": decode_argv, "encode": argv})
        with tempfile.TemporaryFile() as enc_err:
            decoder = subprocess.Popen(decode_argv, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL)
            encoder_proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL, stderr=enc_err)
            spool_file = open(spool, "rb") if replaced else None
            failed = True
            try:
                for n in range(start, end):
                    if clock is None:
                        frame = _read_exactly(decoder.stdout, size)
                    else:
                        with clock.timing("decode_s"):
                            frame = _read_exactly(decoder.stdout, size)
                    if len(frame) != size:
                        raise splice.CheckFailed(
                            "the span decode at [{}, {}) ended at frame {}".format(start, end, n))
                    if mapping[n][0] != repair_plan.SRC:
                        frame = spool_file.read(size)
                    try:
                        if clock is None:
                            encoder_proc.stdin.write(frame)
                        else:
                            with clock.timing("write_wait_s"):
                                encoder_proc.stdin.write(frame)
                    except BrokenPipeError:
                        _died(encoder_proc, "the span encoder at [{}, {})".format(start, end),
                              enc_err)
                    written += 1
                    if progress is not None:
                        try:
                            progress.frames(written, phase="interpolate", boundary=False)
                        except Exception as exc:  # noqa: BLE001 — never at the cost of a master
                            print("[progress] frame emit failed at {} ({}: {})".format(
                                written, type(exc).__name__, exc), flush=True)
                failed = False
            finally:
                if spool_file is not None:
                    spool_file.close()
                decoder.stdout.close()
                decoder.kill()
                decoder.wait()
                if failed:
                    encoder_proc.kill()
                    encoder_proc.wait()
            drained = _finish(encoder_proc, "the span encoder at [{}, {})".format(start, end),
                              enc_err)
            drain_s += drained
            if clock is not None:
                # **Banked per span**, so a copy that fails at a later span, the join or the
                # avcC still files the drains it paid for. Found in review.
                clock.drain_s = round((clock.drain_s or 0.0) + drained, 3)
        if replaced:
            os.remove(spool)
        span_sets.append(splice.span_param_sets(span_ts))
        files.append((span_ts, end - start))
    if progress is not None:
        try:
            # The join, the mux, the avcC rewrite and the check run after the last frame; the
            # full path's name for "frames done, file not yet" is this one.
            progress.phase("draining", force=True)
        except Exception as exc:  # noqa: BLE001 — never at the cost of a master
            print("[progress] draining phase not emitted ({}: {})".format(
                type(exc).__name__, exc), flush=True)
    joined, join_argv = splice.join(pmap, files, parts_dir)
    mux_argv = splice.mux(pmap, joined, master_path, audio_source)
    head, source_sps, source_pps, _ = splice.read_avcc(pmap.path)
    sps, pps = splice.avcc_sets(source_sps, source_pps, span_sets)
    delta = splice.rewrite_avcc(master_path, sps, pps, head)
    recipe.update(join=join_argv, mux=mux_argv,
                  avcc={"sps_ids": [splice.param_id(u) for u in sps], "delta_bytes": delta})
    return dict(
        n_in=pmap.count,
        n_out=pmap.count,
        n_synth=repair_plan.synthesised(mapping),
        n_replaced=replaced_total,
        frames_copied=pmap.count - frames_encoded,
        frames_encoded=frames_encoded,
        codec="h264",
        bit_depth=pmap.depth,
        x264_params=splice.x264_params(threading),
        x265_params=None,
        crf=crf,
        preset=preset,
        recipe=recipe,
    )


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
