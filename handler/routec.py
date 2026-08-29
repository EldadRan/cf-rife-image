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
# **`os` IS USED BY §6g's REFERENCE PATHS AND WAS NOT IMPORTED**, which made every armed
# `reference_score` run raise `NameError` at the disk-bound site before a frame was read
# (`F-2026-08-29-4`). **Nothing in the wave could have caught it**: `py_compile` does not resolve
# names, the image's import assertion imports the MODULE and a NameError is a run-time event, the
# reviewer was given a diff and imports are not in a hunk, and this module needs `numpy` so it
# cannot be imported on the tree where the rest of §6g was exercised. *Stdlib, and the first
# import in the file for that reason: it is the one this module cannot be read without.*
import os

import numpy as np

import stages

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


def _to_tensor_device(frame_bgr, torch, device):
    """§3b — the SAME contract as `_to_tensor`, with the arithmetic on `device`.

    **`convert_in_s` is 21.37% of `compute_s` at 4K and it is the stage the ENCODER IS STARVING.**
    `docs/test-plan.md` §14 is why this is worth more than its own seconds: `--threads 16
    --no-sliced-threads` made the encoder faster — `write_wait_s` 50.2 -> 14.5 — and the whole job
    four times slower, because `convert_in_s` went 32.7 -> 575.5 s. Not the neighbours; `model_s`
    moved 0.7%. Not core starvation; 150-350% of one core against 96 usable. **x264's frame
    threads and a single-threaded strided gather are both spending memory bandwidth and taking it
    from each other**, so removing this stage is what makes the encoder lever pullable at all.

    **The upload carries ONE byte per channel instead of four**, which is the same saving the
    outbound change makes in the other direction: `uint8` goes up, and the channel swap, the
    transpose, the widen and the divide all happen on the card in one pass.

    **OUT-OF-PLACE, AND HERE THE HAZARD IS SHARPER THAN OUTBOUND'S** (§3b-1). `torch.from_numpy`
    SHARES MEMORY with the decoder's array — no copy — so an in-place op on a tensor built that
    way writes into a frame the decoder still owns, **and each source frame is consumed by TWO
    pairs.** Every operation below allocates: `[..., [2, 1, 0]]` is an advanced index (a gather,
    always a copy), `.permute` is a view of that copy, `.float()` materialises it contiguous, and
    `.div` is the out-of-place spelling. Nothing writes back.

    **`np.ascontiguousarray` before `from_numpy` is not optional.** `from_numpy` refuses a
    negative-stride array outright, and cv2 hands back exactly that shape once anything has sliced
    it; the copy is on the `uint8`, which is a quarter the bytes the old path copied as float32.
    """
    up = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device)
    wide = up[..., [2, 1, 0]].permute(2, 0, 1).float()
    # **A DEVICE-SIDE SCALAR, NOT A PYTHON FLOAT, AND §3b'S SNIPPET SAYS `.div(255.0)`.**
    # PyTorch's CUDA true-division kernel special-cases a CPU-scalar divisor by computing
    # `a * (1/b)` rather than `a / b`, and its own source says that "may lose one bit of
    # precision". `.div(255.0)` wraps the Python float into exactly that scalar; numpy's float32
    # divide takes no such path. **In float32, `a/255` and `a*(1/255)` disagree on 126 of the 256
    # uint8 values, first at 3, worst 5.96e-08 at 192** — reproduced from IEEE alone, since there
    # is no torch on the machine this was written on.
    #
    # **Whether torch actually takes that path is NOT established here and must not be read as
    # if it were**; it falsifies in one line on the card. A device tensor is not a CPU scalar, so
    # the special case cannot apply either way: if the path was never taken this costs one
    # negligible allocation, and if it was, it is the difference between a wave that lands and a
    # wave whose own gate reports it broken on every frame.
    #
    # **The sharp edge is that a CPU-device exercise would not have shown it.** The CPU kernel's
    # reciprocal shortcut is gated to reduced floating types, not float32 — so the gate passes on
    # a CPU interpolator and fails only on the card, which is the check exercised from the one
    # direction that hides the case it exists for.
    return wide.div(torch.as_tensor(255.0, dtype=wide.dtype, device=wide.device)).unsqueeze(0)


class InputCheck:
    """§3b-1's in-run dual-path comparison, on the MODEL'S INPUT. **Never raises.**

    **THIS IS NOT `ConvertCheck` ONE STAGE OVER AND THE DIFFERENCE IS THE WHOLE REASON IT
    EXISTS.** §5-0's byte comparison sits at the last step before the file, so a difference there
    has one candidate cause. **Inbound changes what the MODEL IS FED, and 382 of 480 frames are
    syntheses** — one differing LSB propagates through a network nonlinearly, so a byte gate at
    the output would see the difference smeared across frames with no way to attribute it. The
    comparison has to sit at the boundary the change is at: **the float32 model-input tensor,
    old host path against new device path, per SOURCE frame.**

    **`frames` IS GRADED AGAINST `retime.n_in`, NOT `n_out`.** The inbound conversion runs once
    per source frame; grading it against the delivered count would pass a comparison covering 40%
    of the work at 24->60.

    **EXACT EQUALITY, NO TOLERANCE, AND `max_abs_delta` IS A FLOAT.** Both paths are `uint8` to
    `float32` — exact — then one divide by 255.0, a single correctly-rounded IEEE operation. They
    should be bit-identical. **§2g's sweep does not cover this**: that swept `clamp/mul/round/
    uint8` and this is a DIVIDE, and a neighbouring proof is not this proof. The delta is a float
    because a difference here is not bounded to +/-1 the way a `uint8` rounding difference is.
    """

    LIMIT = 16

    def __init__(self):
        self.frames = 0
        self.mismatches = 0
        self.max_abs_delta = 0.0
        self.max_abs_delta_at = None
        self.first = []
        #: Frames the device path wrote through to. **Graded == 0**, and the hazard is not
        #: symmetry with §5-0a: the snapshot here protects the DECODER'S OWN ARRAY, which
        #: `torch.from_numpy` aliases and which two pairs still read.
        self.mutated_frames = 0
        self.errors = 0

    def snapshot(self, frame_bgr):
        """A copy of the decoder's array, taken BEFORE the device path runs. **Never raises.**

        A numpy copy rather than a tensor clone, because the thing at risk is the numpy buffer:
        `torch.from_numpy` does not copy, so the aliasing runs the other way here than it does
        outbound.
        """
        try:
            return np.array(frame_bgr, copy=True)
        except Exception:  # noqa: BLE001 — a gate must never displace a delivered master
            return None

    def compare(self, index, before, frame_bgr, new_tensor, torch):
        """One source frame, both ways. **Swallows everything, reference arm included.**"""
        try:
            if before is None:
                raise ValueError("no snapshot was taken for source frame {}".format(index))
            old_tensor = _to_tensor(before, torch)
            mutated = not np.array_equal(before, frame_bgr)
            new_host = new_tensor.detach().to("cpu")
            entry = peak_entry = None
            worst = 0.0
            # **`torch.equal` first, and unlike `ConvertCheck`'s memcmp this one has no cheap
            # path behind it.** On a frame that differs, the numpy work below allocates five
            # full-resolution host buffers — ~400 MB at 4K — and if the divisor claim above holds
            # that is EVERY frame rather than none. `argmax` over `flatnonzero` is inherited from
            # `ConvertCheck` and saves the index array; nothing saves the subtraction, because
            # `max_abs_delta` is the field §3b-1 grades and it needs the whole difference.
            # **Stated rather than discovered on a box already running x264 at tens of GiB.**
            if not torch.equal(old_tensor, new_host):
                old = old_tensor.reshape(-1).numpy()
                new = new_host.reshape(-1).numpy()
                delta = np.abs(old - new)
                at = int((old != new).argmax())
                peak = int(delta.argmax())
                worst = float(delta[peak])
                entry = {"frame": int(index), "index": at,
                         "old": float(old[at]), "new": float(new[at])}
                peak_entry = {"frame": int(index), "index": peak,
                              "old": float(old[peak]), "new": float(new[peak])}
        except Exception:  # noqa: BLE001 — a gate must never displace a delivered master
            self.errors += 1
            return
        self.frames += 1
        if mutated:
            self.mutated_frames += 1
        if entry is not None:
            self.mismatches += 1
            if len(self.first) < self.LIMIT:
                self.first.append(entry)
            if worst > self.max_abs_delta:
                self.max_abs_delta = worst
                self.max_abs_delta_at = peak_entry

    def block(self):
        """§3b-1's five fields, plus the two `ConvertCheck` also carries."""
        return {"frames": self.frames, "mismatches": self.mismatches,
                "first": self.first, "max_abs_delta": self.max_abs_delta,
                "max_abs_delta_at": self.max_abs_delta_at,
                "mutated_frames": self.mutated_frames,
                "errors": self.errors}


class ConvertCheck:
    """§5-0's in-run dual-path comparison. **Never raises; a gate must not cost a master.**

    **THE CROSS-RUN GATE §5 ASKED FOR CANNOT WORK, and this exists because it was tried.**
    `docs/gate-findings.md` `F-2026-08-27-4`: two runs on the SAME image deliver files 3,809 bytes
    apart and every decoded frame differs, with x264 tested and exonerated. **So comparing an old
    master against a new one measures the platform rather than the change** — and `rc_lookahead`
    couples every frame's quantisation to its neighbours, which is why even frame 0, a COPIED
    frame with no model anywhere in its path, differs.

    **Both arms here see the same tensor in the same process, so a difference can only be the
    conversion.** No model non-determinism, no host contention, no encoder state, no request
    identity. It compares BEFORE the encoder, which §5 itself calls the stronger form, and it
    covers every delivered frame rather than a hash of an aggregate.

    **`frames` is graded against the plan's `n_out`** for the reason `tie_check.swept` is graded
    against the domain: a comparison that ran on a tenth of the frames and found nothing reads
    exactly like one that ran on all of them.

    **The instrument's own cost is deliberately OUTSIDE `convert_out_s`.** The old arm is not the
    shipped conversion, and timing it into the field whose share this wave exists to move would
    corrupt the number the wave is judged by. It lands in `stage_residual_s`, which
    `stages.RESIDUAL` reports rather than absorbs.
    """

    #: §5-0. Bounded for the reason `tie_check.first_mismatches` is: an unbounded list of every
    #: differing pixel in a 4K clip is a record nobody can fetch, and the COUNT decides the
    #: verdict.
    LIMIT = 16

    def __init__(self):
        self.frames = 0
        self.mismatches = 0
        self.max_abs_delta = 0
        self.max_abs_delta_at = None
        self.first = []
        #: Frames where the SHIPPED arm mutated the tensor it was given. **Zero is part of the
        #: acceptance**, because `mismatches == 0` does not cover out-of-placeness on its own —
        #: see `compare`. A non-zero count here is §2f's forbidden refinement having reached the
        #: code, and it is the failure that delivers a near-white master silently.
        self.mutated_frames = 0
        #: Frames the comparison itself failed on. **Not in §5-0's field list and reported
        #: anyway**, because the alternative is the failure this project keeps meeting: an
        #: instrument that breaks and leaves a record indistinguishable from one that worked.
        #: A frame counted here is NOT counted in `frames`, so `frames` falls short of `n_out`
        #: and the kit fails the run on the honest ground — a comparison that did not compare.
        self.errors = 0

    def snapshot(self, frame):
        """A pristine copy of `frame`, taken BEFORE the shipped arm touches it. **Never raises.**

        **THE ORDER IS THE WHOLE MECHANISM AND THE FIRST FIX FOR IT GOT THE ORDER WRONG.** The
        clone was taken inside `compare`, which the loop calls AFTER `_to_rgb24_device` has
        already returned — so on an in-place shipped arm it cloned an already-corrupted tensor,
        the reference arm read the corruption, and the two agreed exactly as they did before the
        fix. **Caught by executing the case rather than by reading the code**, which is the only
        way this class of bug is ever caught.

        `None` on failure, which `compare` counts as an error rather than as a pass: a gate that
        could not take its own reference has not compared anything.
        """
        try:
            return frame.clone()
        except Exception:  # noqa: BLE001 — a gate must never displace a delivered master
            return None

    def compare(self, index, before, frame, new_bytes):
        """One frame, both ways. **Swallows everything, including the reference arm.**

        **THE REFERENCE ARM IS EVALUATED IN HERE AND NOT AT THE CALL SITE**, which is what makes
        the class docstring's promise true. It was `compare(index, _to_rgb24(frame), payload)` —
        so the reference conversion ran OUTSIDE this try, and at 4K it allocates several
        full-resolution host float32 buffers per frame. A `MemoryError` there is not a
        `WorkerError`, so it would have escaped `retime` entirely: **arming a diagnostic could
        fail a job that unarmed delivers.**

        **THE SNAPSHOT IS THE OTHER HALF, AND IT IS WHY THE GATE CAN SEE AN IN-PLACE OP AT ALL.**
        It is taken by `snapshot`, at the call site, BEFORE the shipped arm runs — see there for
        why the ordering is not a detail.
        §5-0 argued that both arms seeing the same tensor means a difference can only be the
        conversion. **The first half is true and it breaks the second half**: the same tensor is
        a MUTABLE tensor, the device arm runs first, and an in-place device arm would corrupt it
        before the reference arm ever read it — so the two would agree and the gate would report
        zero on a run delivering near-white frames. **Two arms reading one object can prove
        arithmetic equivalence and cannot prove out-of-placeness; they are different properties.**
        So the reference arm reads a pristine copy, and `mutated` names the cause directly rather
        than leaving a reader to infer it from a wall of differing pixels.

        **NOTHING IS COMMITTED TO `self` UNTIL THE WORK THAT WOULD JUSTIFY IT HAS SUCCEEDED.**
        An earlier draft incremented `frames` and `mismatches` at the top and then did the numpy
        work — so an exception between them left `mismatches: 1, max_abs_delta: 0`, a shape the
        kit reads as *"a difference beyond +/-1, so NOT a rounding difference"*. **A broken
        instrument would have been graded as a broken conversion.**
        """
        try:
            if before is None:
                raise ValueError("no pristine snapshot was taken for frame {}".format(index))
            old_bytes = self._convert(before)
            mutated = not self._same(before, frame)
            # **The equality test first, and it is what makes this affordable.** `bytes.__eq__` is
            # a C memcmp; the numpy work below runs only on a frame that actually differed, which
            # on a passing run is never.
            if old_bytes == new_bytes and not mutated:
                self.frames += 1
                return
            entry = peak_entry = None
            worst = 0
            if old_bytes != new_bytes:
                old = np.frombuffer(old_bytes, dtype=np.uint8)
                new = np.frombuffer(new_bytes, dtype=np.uint8)
                # **int16 before subtracting.** uint8 arithmetic wraps, so a difference in the
                # other direction reports 255 and `max_abs_delta` would say the change is not a
                # rounding difference when it is exactly that. **§5's fallback turns entirely on
                # this number being 1**, and CF's ruling is asked for on its strength.
                delta = np.abs(old.astype(np.int16) - new.astype(np.int16))
                # **`argmax`, not `flatnonzero`.** The latter materialises EVERY differing index
                # as int64 — ~199 MiB at 4K when every byte differs, which is exactly the run
                # this gate exists to catch. **An instrument must not be most expensive on its
                # own worst case**, and `argmax` on a bool array stops at the first True.
                at = int((old != new).argmax())
                peak = int(delta.argmax())
                worst = int(delta[peak])
                entry = {"frame": int(index), "index": at,
                         "old": int(old[at]), "new": int(new[at])}
                # **The worst pixel is named, and it is NOT the same pixel as `first`.** `first`
                # holds the FIRST differing byte of a frame; `max_abs_delta` is the largest
                # difference anywhere across every frame. A reader checking the headline against
                # the evidence would otherwise compute `new - old` from `first[0]` and get a
                # number belonging to a different pixel and usually a different frame, with
                # nothing in the block saying so.
                peak_entry = {"frame": int(index), "index": peak,
                              "old": int(old[peak]), "new": int(new[peak])}
        except Exception:  # noqa: BLE001 — a gate must never displace a delivered master
            self.errors += 1
            return
        self.frames += 1
        if mutated:
            self.mutated_frames += 1
        if entry is not None:
            self.mismatches += 1
            if len(self.first) < self.LIMIT:
                self.first.append(entry)
            if worst > self.max_abs_delta:
                self.max_abs_delta = worst
                self.max_abs_delta_at = peak_entry

    @staticmethod
    def _convert(tensor):
        """The reference arm. Named so the class, not the loop, owns which arm is the reference."""
        return _to_rgb24(tensor)

    @staticmethod
    def _same(a, b):
        """`torch.equal`, imported where it is used. **A failure here is a failure of the
        comparison**, so it raises into `compare`'s own handler rather than returning a guess."""
        import torch  # noqa: PLC0415 — torch is a GPU-box import on this module

        return bool(torch.equal(a, b))

    def block(self):
        """§5-0's four fields, plus three this build adds — see `__init__` and `compare`."""
        return {"frames": self.frames, "mismatches": self.mismatches,
                "first": self.first, "max_abs_delta": self.max_abs_delta,
                "max_abs_delta_at": self.max_abs_delta_at,
                "mutated_frames": self.mutated_frames,
                "errors": self.errors}


def _to_rgb24_device(tensor, staging=None):
    """§3a — the SAME contract as `_to_rgb24`, with the arithmetic on whichever device the tensor
    is already on. **This is the conversion wave.**

    `convert_out_s` was 67.08% of `compute_s` at 4K (`docs/test-plan.md` §12) and it is one
    single-threaded pass of host float work per frame. The operations are identical; only where
    they run changes. **The `.to("cpu")` moves to the END and carries one byte per channel instead
    of four**, which is the whole of the saving — the device-to-host copy shrinks by 4x and the
    arithmetic stops being a host bottleneck.

    **OUT-OF-PLACE, AND IT IS A REQUIREMENT RATHER THAN A STYLE NOTE** (`docs/conversion-wave.md`
    §2f, §3a). `clamp_`, `mul_` or `round_` would write through to the caller's own tensor, and
    `interpolate.stream` ends both its copy and its hold branch with `yield held[i]` — **the same
    tensor object, emitted twice.** A held frame converted in place would be CLAMPED back into
    [0, 1] on its second pass, saturating every pixel at or above 1, and the multiply would then
    take it to 255: **a near-white frame**, passing every frame-count and cadence check there is.
    The refinement that would remove two full-size intermediates is safe and is deliberately NOT
    taken — §2f holds it, and a mechanically checkable rule beats a subtle one when the failure
    mode is a delivered master that is garbage.

    **`.float()` is here and §3a's snippet omits it.** It matches `_to_rgb24`'s own cast rather
    than the document's shorthand: `Interpolator` takes `dtype` as a constructor argument and its
    docstring states that *"the stream is not uniform in device or dtype, deliberately"*. Nothing
    reaches it with a half-precision dtype today — `handler` constructs it without one — so this
    changes no result now and stops the two arms diverging on dtype if it ever does. On an
    already-float32 tensor it returns that same tensor, so it costs nothing and allocates nothing.

    **`.tobytes()` STAYS, and removing it is not a free micro-optimisation** (§3a). The one guard
    between a wrong-sized frame and a master that shears while ffmpeg exits 0 is
    `encoder.MasterWriter.write`'s `len(frame_bytes) != width * height * 3` — and **`len()` of a
    3-D ndarray is its FIRST DIMENSION, not its byte count.** Handing the array in directly would
    defeat that check while appearing to work. Changing the guard is a change to the writer's
    contract and is explicitly not in this wave.
    """
    frame = (tensor.detach()[0].float()
             .clamp(0.0, 1.0).mul(255.0).round().to(_uint8())
             .permute(1, 2, 0).contiguous())
    if staging is None:
        return frame.cpu().numpy().tobytes()
    return staging.fetch(frame)


class StagingInvariantViolated(Exception):
    """§3c-1's invariant broke. **Its own type, and the reason is that `RuntimeError` is not one.**

    The first cut raised `RuntimeError` here and re-raised it past the fallback with
    `except RuntimeError: raise`. **Torch reports essentially every failure as `RuntimeError`** —
    a failed `cudaHostAlloc` behind `pin_memory=True`, `torch.cuda.OutOfMemoryError` (a subclass),
    a device fault in `copy_` — so that clause did not distinguish this invariant from an
    allocation failure. **A pinned-allocation failure under host-memory pressure would have
    killed a job that delivered a master before this wave**, which is the opposite of what a
    speed-up may cost.
    """


class PinnedStaging:
    """A reused page-locked destination for the device-to-host copy — §3c-1. **Never raises.**

    **A DEFECT, NOT AN OPTIMISATION.** `docs/test-plan.md` §22d: `convert_out_s` is **7.4x over
    its physical floor at 8K** — 65.4 s against ~8.8 s from the measured 7.90 GB/s bus plus one
    host memcpy — and 3.1x over at 4K. *The overshoot grows with frame size, so it is per-frame
    cost rather than fixed inefficiency*, and the suspect is a fresh PAGEABLE destination
    allocated for every frame. A copy into pageable memory cannot go straight from the device:
    the driver stages it through its own pinned bounce buffer, in chunks. ~56 s of an 8K job.

    **THIS IS NOT THE ASYNC CHANGE.** §3c deferred pinned buffers and `non_blocking=True`
    together and **they are separable; only the first is taken.** The copy stays synchronous, so
    `convert_out_s` still ends at a real synchronisation and §3c's second invariant — *the
    conversion clock must not become an enqueue clock* — is untouched. That is §9b's trap, and
    it would land on the very number this change is justified by.

    **THE INVARIANT, AND IT IS THE WHOLE RISK OF THE WAVE:**

        A REUSED BUFFER MAY ONLY BE REFILLED AFTER THE PREVIOUS FRAME'S `write` HAS RETURNED.

    Today `.tobytes()` mints a fresh object per frame, so nothing can alias. **A reused staging
    buffer removes that guarantee**, and what restores it is that `MasterWriter.write` blocks
    until the pipe accepts the frame — so on this path the invariant holds by construction.
    **Which is exactly what a future write-behind queue would break**, silently, on a buffer
    whose previous contents are still in flight.

    **WHAT `_armed` ACTUALLY CATCHES, STATED PRECISELY, BECAUSE THE FIRST DRAFT OF THIS
    DOCSTRING OVERSTATED IT IN BOTH DIRECTIONS.**

    - **It is not what protects the present.** `.tobytes()` always returns a fresh immutable
      copy, so the payload handed to `writer.write` never aliases this buffer no matter what the
      writer does. **`.tobytes()` is today's guarantee; `write` blocking is not the operative
      one**, and the invariant cannot be violated on this path at all.
    - **It does not catch the write-behind queue either.** `released()` is called by the write
      loop on the line after `write` returns, unconditionally — so a queue that returns on ENQUEUE
      clears the flag just the same and the next frame refills happily. **Making that case real
      needs `released()` called by whoever finished with the BYTES**, not by the loop.
    - **What it does catch is a restructured loop**: a second `fetch` with no intervening
      `released()` at all — a frame converted twice, a `released()` dropped in a refactor, a
      second producer sharing one instance. That is worth a boolean and it is not nothing. It is
      also not the invariant §3c-1 names, and a reader who believed it was would stop looking.
    """

    def __init__(self):
        self._buffer = None
        self._shape = None
        #: True between handing a buffer out and being told its write returned. **The invariant
        #: in one flag** — see the class docstring.
        self._armed = False

    def fetch(self, frame):
        """Copy `frame` to the host through the reused pinned buffer and return its bytes.

        Falls back to the unpinned path on ANY failure, because a staging buffer is a
        performance change and must never be the reason a master is not delivered.
        """
        import torch  # noqa: PLC0415 — a GPU-box import, like every other torch touch

        try:
            if self._armed:
                raise StagingInvariantViolated(
                    "the pinned staging buffer was refilled before the previous frame's write "
                    "returned. §3c-1's invariant is violated and the previous frame's bytes may "
                    "still be in flight to ffmpeg; this refuses rather than delivering a master "
                    "whose frames are a copy of each other")
            # **Keyed on `(shape, dtype)` and not on shape alone.** `Tensor.copy_` CONVERTS
            # rather than refusing, so a buffer allocated for one dtype silently truncates a
            # frame of another — producing a payload of the correct LENGTH, which passes
            # `MasterWriter.write`'s `len()` guard and every frame-count check there is.
            # Unreachable today: `_to_rgb24_device` ends `.to(_uint8())` invariantly. One tuple
            # compare against a silent-corruption class is not a trade worth thinking about.
            shape = (tuple(frame.shape), frame.dtype)
            if self._buffer is None or self._shape != shape:
                # **Allocated once per shape, which on this path means once.** A retime that
                # changed frame size mid-clip is already refused upstream by `_load_pair`.
                self._buffer = torch.empty(shape[0], dtype=frame.dtype,
                                           device="cpu", pin_memory=True)
                self._shape = shape
            self._buffer.copy_(frame)
            self._armed = True
            # **`.tobytes()` STAYS** (§3a, unchanged). This buffer is the DESTINATION of the
            # device-to-host copy, not a replacement for the bytes handed to the writer:
            # `MasterWriter.write`'s guard is `len(frame_bytes)`, and `len()` of a 3-D ndarray is
            # its first dimension. The copy `tobytes` makes is ~3.2 s of the 65.4 at 8K; the
            # allocation and the un-pinned transfer are the rest, and they are what this removes.
            return self._buffer.numpy().tobytes()
        except StagingInvariantViolated:
            # **The one thing that must escape.** Everything else falls back; this is a
            # correctness violation and delivering past it would deliver duplicated frames.
            raise
        except Exception:  # noqa: BLE001 — a speed-up must never cost a delivered master
            self._armed = False
            return frame.cpu().numpy().tobytes()

    def released(self):
        """The previous frame's `write` has returned; the buffer may be refilled."""
        self._armed = False


def _uint8():
    """`torch.uint8`, fetched where it is used. **torch is a GPU-box import on this module.**"""
    import torch  # noqa: PLC0415 — as everywhere else here; the CPU test tree imports this file

    return torch.uint8


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


class DecodeCount:
    """How many frames the DECODER actually handed on. **One number, one owner.**

    **`retime.n_in` is frames DECODED and never the source file's implied length** (ruled
    2026-08-27). Derived from duration times rate it would be a description of the container;
    counted here it is a description of what happened, and the identity
    `input_check.frames == retime.n_in` then holds at every ratio — including a DECIMATING one,
    where `_emit.advance_to` pulls one frame fewer than the file holds and a derived `n_in` would
    have failed a clean run.

    **Two counters in two modules, and neither reads the other.** This one is incremented in the
    decoder; `InputCheck.frames` is incremented in the converter. That is what keeps the identity
    evidence rather than a tautology — a single counter read twice would agree with itself no
    matter what happened between them.
    """

    def __init__(self):
        self.decoded = 0


def frames_from(capture, expect=None, clock=None, count=None):
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
        if count is not None:
            count.decoded += 1
        yield frame


def retime(source, source_path, master_path, interpolator, target_fps, identity,
           snap_tolerance=None, crf=None, audio_source=None, progress=None,
           variant="direct", scale=None, preset=None, threads=None,
           sliced_threads=None, rc_lookahead=None, clock=None, convert_check=False,
           input_check=False, armed=None, encode_defaults=None, codec=None,
           bit_depth=None, reference_score=False):
    """Decode, interpolate, encode. Returns the stats the plan produced.

    `snap_tolerance` is passed through as given and **is not defaulted here** (contract §5c): a
    tolerance defaulted to zero would ship the unsnapped plan as the ruled answer before the
    benchmark that decides it has run. `None` means unruled, and the shim reads it as zero for
    arithmetic while the request records that nobody chose.
    """
    import encoder  # noqa: PLC0415 — imported here so this module stays importable without one
    import reference  # noqa: PLC0415 — contract §6g, and stdlib-only like the rest of this pair
    import variants  # noqa: PLC0415

    from decode import open_source  # noqa: PLC0415 — cv2 is a GPU-box import, like the rest

    # **Bound BEFORE the `try`, because the `finally` that releases it is at THIS level.** A name
    # first assigned inside the block is unbound on every path that raises before reaching it, and
    # the cleanup would then raise `NameError` and mask the failure it was running after —
    # `capture` above is bound before the `try` for the same reason.
    reference_path = None
    capture, shape = open_source(source_path)
    try:
        width, height = shape["width"], shape["height"]
        n_in = source_frame_count(source)
        # **Peak VRAM measured rather than reported from a panel.** Route C has no plan, so
        # nothing else on this path produces one — and §8b's `--scale` axis exists to falsify
        # `w_scaling: FLAT`, which IS this reading. Reset before and read after, so the number is
        # this job's high-water mark rather than the process's history.
        peak_reset = _reset_peak()
        # **Created before the stream, because `_tensors` is consumed lazily inside it.** Handed
        # in like `ConvertCheck` and for the same §4b reason: a per-call result published on an
        # object that outlives the call is read by somebody with no way to know it is stale.
        input_checker = input_check if isinstance(input_check, InputCheck) else (
            InputCheck() if input_check else None)
        # **Counted at the decoder, not derived from the container** — see `DecodeCount`. The
        # plan above is still sized from `source_frame_count`, because a plan has to exist before
        # anything is decoded; this is what the record reports afterwards.
        decoded = DecodeCount()
        # **§3c-1. One buffer for the whole retime, handed down as an argument.** Contract §4b
        # and the same reason `StageClock` is a parameter: an accumulator on an object that
        # outlives the call is read by somebody with no way to know it is stale, and a cascade
        # variant calls `stream()` twice.
        staging = PinnedStaging()
        stream, stats = variants.run(
            variant, interpolator,
            # **The interpolator's device, read off the interpolator.** §3b does the inbound
            # arithmetic where the model lives, so the producer has to know that before the
            # interpolator ever sees a tensor.
            _tensors(frames_from(capture, expect=(height, width), clock=clock,
                                 count=decoded),
                     interpolator.device, clock=clock, checker=input_checker),
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
        # ── §6d's BRANCH, AND THIS IS THE ONLY SITE THAT CHOOSES AN ENCODER SETTING ──────────
        #
        # **HERE, because `width` and `height` are the DELIVERED frame and this is where they
        # exist.** §6d-1 pins the key to the size `MasterWriter` is constructed with, and those
        # are the two integers below — literally the same locals, so the branch and the encode
        # cannot be deciding against different answers to how big the frame is. They come from
        # `decode.open_source` (`:655`), not from `probe.probe_source`; the two readers agree on
        # ordinary sources and are two readers, and this is not the place to find out.
        #
        # **THIS REPLACED THE OLD `None`-to-constant GUARD RATHER THAN SITTING BESIDE IT.** §6d
        # forbids a second site that picks encoder settings, *"because a second answer to the
        # question of what a job encoded at"* — the guard resolved exactly these three to the
        # pre-§6d constants, so leaving it in place under a branch would have been that second
        # answer, reachable by any caller that did not go through `handler`. **One site now, and
        # a direct caller gets the area row the same as a request does.**
        #
        # **`None` STILL MEANS THE CALLER SENT NOTHING, AND THAT IS THE WHOLE MECHANISM.**
        # `validation` deliberately no longer defaults these three, precisely so that this line
        # can tell an absence from a value; §6b's surviving clause is that an explicitly-sent
        # field is obeyed and never silently overridden.
        #
        # **AND UNDER h265 THERE IS NO BRANCH TO TAKE** (contract §6e, `docs/instrumentation.md`
        # §15a). §6d's table is x264's vocabulary and cannot have resolved an h265 job, so
        # `resolve_defaults` returns NO settings and a basis that says the table was skipped —
        # and everything below reads off `encode_settings` rather than assuming three keys.
        encode_settings, encode_provenance = encoder.resolve_defaults(
            int(width) * int(height), codec=codec,
            threads=threads, sliced_threads=sliced_threads, rc_lookahead=rc_lookahead)
        # **Filled IN PLACE into the caller's dict rather than returned in `stats`.** `handler`
        # banks this object in `trace` before calling us, so a run reaped in ffmpeg still files
        # what its settings were resolved from — the same argument `ConvertCheck` is handed in
        # for, one wave earlier. A provenance that travelled out in `stats` would be null on
        # exactly the runs worth recording.
        if encode_defaults is not None:
            encode_defaults.update(encode_provenance)
        print("[encode] defaults {} at {} delivered pixels ({}x{}, boundary {}): {}".format(
            encode_provenance["basis"], int(width) * int(height), width, height,
            encode_provenance["boundary"],
            # **The settings that were resolved, or the fact that none were.** Formatted from
            # the dict rather than from three named locals, so the h265 line says what is true
            # about h265 instead of printing x264's vocabulary with placeholder values in it.
            " ".join("{}={}".format(name, encode_settings[name])
                     for name in encoder.AREA_FIELDS if name in encode_settings)
            or "no x264 thread settings — this codec has no such table"), flush=True)

        # **`crf` and `preset` are NOT §6d's and still resolve to constants here.** The table
        # decides three fields and says nothing about either. **Their substitution is REPORTED**
        # (`substituted` below) rather than silent: a direct caller that sent no `crf` would
        # otherwise have the estimate declare `crf=12` and stamp `estimator_v3/crf12` onto the
        # basis §8g grades, saying which CRF the job was computed for when nobody declared one.
        encode_crf = crf if crf is not None else encoder.DEFAULT_CRF
        encode_preset = preset if preset is not None else encoder.DEFAULT_PRESET
        substituted = [name for name, value in (("crf", crf), ("preset", preset))
                       if value is None]
        # **`encode_settings` is spread rather than restated**, so the arm carries three keys
        # under h264 and none of them under h265 — the same absence §15a requires of the record,
        # arriving from the same dict rather than from a second decision about what h265 has.
        encode_arm = dict(encode_settings, crf=encode_crf, preset=encode_preset)

        # ── contract §6g's DISK BOUND, AND IT FIRES BEFORE THE FIRST FRAME ──────────────────
        #
        # **HERE, because this is the first line at which all three inputs exist**: the delivered
        # frame size, the planned output count, and the pixel format the encode will use. It is
        # ALSO before `MasterWriter` is constructed, which is the requirement — §6g rules the
        # refusal *"before the first frame"* and gives the reason in full: *a job that fills the
        # disk at frame 400 has spent the whole model cost to produce nothing, which is the most
        # expensive failure this worker can have.*
        #
        # **The bound is computed against the DIRECTORY THE REFERENCE WILL BE WRITTEN TO**, not
        # against a hard-coded figure — the free space is this worker's, now, and a constant here
        # would be the parent project's number wearing this project's units.
        # **Bound before the branch so every path to the stats has it**, including the unarmed
        # one — a name that exists only inside an `if` is a `NameError` on the path that skipped.
        # *`reference_path` is bound above the `try` instead, because its cleanup is at function
        # level and must be able to read it on a path that never reached this line.*
        reference_block = None
        reference_format = None
        if reference_score:
            reference_path = os.path.join(os.path.dirname(master_path) or ".",
                                          "reference.raw")
            # **The format is resolved through `encoder.pixel_format`, which is the same call
            # `MasterWriter` makes** — so the bytes the bound reserves and the bytes ffmpeg
            # writes cannot be computed from two different answers to "what format is this".
            reference_format = encoder.pixel_format(bit_depth)
            reference.refuse_if_it_will_not_fit(
                os.path.dirname(reference_path) or ".", width, height,
                # **Passed AS IT IS, with no `or 0`.** The callee refuses a non-positive
                # count; coercing an absent one to zero here would produce a refusal naming a
                # planned count of 0 that the request never had. *`n_out` is a plain int by
                # construction — `interpolate.target_count` returns `int(...)` — so the callee's
                # type check refuses a shape this path cannot produce, which is what an internal
                # guard is for.*
                stats.get("n_out"), reference_format)

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
            estimate = _seed_estimate(progress, source, stats, scale, encode_arm, armed,
                                      substituted)

        writer_cm = encoder.MasterWriter(
            master_path, width, height, float(target_fps), identity,
            audio_source=audio_source, audio_codec=source.get("audio_codec"),
            audio_limit_s=source.get("video_duration_s"),
            # **Read off `encode_arm`'s own names**, which is what makes the estimate's declared
            # arm and the encode's actual settings impossible to disagree — the same argument the
            # record already makes by reading all five off the writer that ran, one step earlier.
            # **The codec goes to the ONE place that maps a name to a library**, and the
            # writer is then the single object that knows what ran — which is what the record
            # reads it off, exactly as it already does for the five settings.
            codec=codec,
            # §6f. Passed by name beside the codec, and the writer refuses the pair `validation`
            # already refused — a second guard on a path a request cannot reach.
            bit_depth=bit_depth,
            # §6g. None on an unarmed run, which is what leaves the command a single output.
            reference_path=reference_path,
            crf=encode_crf, preset=encode_preset, **encode_settings)
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
        # **The checker is HANDED IN, not created here, and that is F2's fix.** It used to be
        # built in this function and published only in the success `return` below — so a run
        # reaped in ffmpeg at frame 900 of 1400 discarded nine hundred frames of comparison,
        # including any mismatch already found, and filed a record indistinguishable from one
        # that never armed the gate. **`handler` owns the object now**, banks it in `trace`
        # before this call, and reads it in the `finally` that writes the record on every exit.
        # `retime` still publishes it in `stats` for a direct caller that passed nothing.
        checker = convert_check if isinstance(convert_check, ConvertCheck) else (
            ConvertCheck() if convert_check else None)
        try:
            with writer_cm as writer:
                for frame in stream:
                    # **`convert_out_s` and `write_wait_s` split what used to be one
                    # expression** (§9a). `write_wait_s` is the producer BLOCKED in `write` — the
                    # encoder's share seen from the only side that can measure it without
                    # instrumenting ffmpeg, because `write` returns when the pipe accepts the
                    # frame (`encoder.py:270`).
                    #
                    # **`convert_out_s` is §10's whole subject.** It was `convert_s` until §10a
                    # split that name three ways, and it is the OUTBOUND step: the device-to-host
                    # copy as float32 inside `_to_rgb24`, then the single-threaded host
                    # arithmetic after it. `docs/conversion-wave.md` is sized entirely against
                    # this field's share and that share has never been measured — the 61-74% was
                    # attributed here in a draft and the attribution was assumption. This is the
                    # number that arms that wave or retires it.
                    #
                    # **THE SHIPPED CONVERSION IS `_to_rgb24_device` FROM THIS WAVE ON.**
                    # `_to_rgb24` survives as the reference arm of §5-0's gate and as the thing
                    # that runs when nobody asked for the gate — it is not dead code and it is
                    # not a fallback: it is the definition the new path is being held to.
                    # **Taken BEFORE the shipped conversion and outside its clock.** This is
                    # the gate's reference and the only thing standing between an in-place
                    # shipped arm and a silent near-white master; see `ConvertCheck.snapshot`.
                    before = checker.snapshot(frame) if checker is not None else None
                    if clock is None:
                        payload = _to_rgb24_device(frame, staging)
                    else:
                        with clock.timing("convert_out_s"):
                            payload = _to_rgb24_device(frame, staging)
                    # **Outside the timed block, on purpose.** The reference arm is the
                    # instrument, not the shipped conversion, and charging it to `convert_out_s`
                    # would corrupt the one number this wave is judged by. It lands in
                    # `stage_residual_s`, reported and not absorbed.
                    # **Compared against the snapshot taken above, not against `frame`.** The
                    # shipped arm has already run by this line; on an in-place arm `frame` is
                    # whatever that arm left behind, and comparing against it would prove the
                    # corruption consistent with itself. `compare` swallows everything, reference
                    # arm included — see `ConvertCheck.compare`.
                    if checker is not None:
                        checker.compare(writer.frames_written, before, frame, payload)
                    if clock is None:
                        writer.write(payload)
                    else:
                        with clock.timing("write_wait_s"):
                            writer.write(payload)
                    # **§3c-1's invariant, released only after `write` RETURNED.** Not in a
                    # `finally`: a write that raised is a run that is ending, and re-arming the
                    # buffer on the way out would say the frame was accepted when it was not.
                    staging.released()
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
                # **The bound that was actually in force, named by the codec that had it.**
                # Reporting `x264-params` on an h265 encode would put the one string that was
                # null onto the one message written when the encoder ran out of memory — the
                # reading a person does under time pressure, with nothing else left to read.
                "{} — ffmpeg reached {} GiB RSS before it stopped, over {} frame(s) written "
                "with {}-params {!r}".format(
                    exc.message, peak, writer_cm.frames_written,
                    "x265" if writer_cm.codec == "h265" else "x264",
                    writer_cm.x265_params if writer_cm.codec == "h265"
                    else writer_cm.x264_params),
                remedy=exc.remedy, shortfall=exc.shortfall) from exc
        finally:
            # **§16: the drain reaches the RECORD, and this is the line that carries it there.**
            # It has been measured on every run this project has ever done and recorded on none —
            # `encoder.py` computes it, the block below PRINTS it, and no record has ever held
            # it. **Banked on the clock because that is the object `handler._timings` already
            # reads**, and inside the `finally` for the reason the two prints are: a run reaped
            # mid-encode is a run whose drain is the most interesting number it has.
            #
            # *`writer_cm.drain_s` is None until `__exit__` has run, and a null term contributes
            # zero to §16c's identity — so a run that died before the drain reports exactly what
            # it used to.*
            if clock is not None:
                clock.drain_s = writer_cm.drain_s
            # Said out loud whichever way the encode ended, because a log line survives a bundle
            # that was never written.
            print("[encode] ffmpeg peak RSS {} GiB over {} frame(s)".format(
                writer_cm.encoder_peak_rss_gb, writer_cm.frames_written), flush=True)
            # **`docs/test-plan.md` §18c's diagnostic, printed here for the same reason the line
            # above is: a run reaped mid-encode is a run whose write distribution is the most
            # interesting thing it has, and a `finally` is the only place that survives it.**
            #
            # **It is a DIAGNOSTIC AND NOT A QUEUE.** §18c is explicit that this is one small
            # experiment to decide whether the write-behind queue is worth building at all —
            # uniform says a small writer buffer hides the cost, bursty says buffering only
            # smooths it — and its readings are CRF- and codec-specific and expire at x265.
            # **Nothing is banked on it and it files no record field.**
            _print_write_distribution(writer_cm)
        # ── contract §6g's SCORING, and it runs only after the master exists ──────────────────
        #
        # **OUTSIDE the `try/finally` above and after the writer has closed**, because the
        # reference is not complete until ffmpeg has exited: it is that process's second output,
        # and reading it while the process still holds it would score a truncated file against a
        # finished master and report the difference as quality.
        #
        # **A failure here must never cost a delivered master.** The master is written, verified
        # and about to be uploaded by the time this runs; §6g's instrument is a diagnostic and
        # `reference` is absent from the record when it does not produce one — which §17a already
        # rules is the honest shape. *The exception is swallowed with its text printed, exactly as
        # `_print_write_distribution` is, and for the same reason.*
        if reference_path is not None:
            try:
                reference_block = reference.score(
                    master_path, reference_path, width, height, float(target_fps),
                    reference_format, stats.get("n_out"), os.path.dirname(master_path) or ".")
            except Exception as exc:  # noqa: BLE001 — a score must never displace a master
                print("[reference] not scored ({}: {})".format(
                    type(exc).__name__, exc), flush=True)
                reference_block = None
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
                    # **`docs/conversion-wave.md` §3b-0 item 2, and it is FRAMES DECODED.**
                    # The record carried `n_out`, `n_copy`, `n_hold` and `n_synth` and never the
                    # count they were all derived from — invisible until the inbound gate needed
                    # grading against it, since that conversion runs once per SOURCE frame and
                    # `n_out` would pass a comparison covering 40% of the work at 24->60.
                    #
                    # **Ruled as a definition rather than a tolerance** (2026-08-27): the
                    # derived count describes the container, and at a DECIMATING ratio the plan
                    # pulls one frame fewer than the file holds — `envelope` bounds `target_fps`
                    # only as positive, so nothing refuses one. A derived `n_in` would have
                    # failed a clean run on the identity the gate is graded by.
                    n_in=decoded.decoded,
                    # **§5-0, and it travels beside `estimate` for the same reason** — `retime`
                    # is handed no `trace`. `handler._retime` lifts it to the record's TOP level
                    # rather than filing it under `retime`, because the kit grades
                    # `convert_check.frames` AGAINST `retime.n_out` and a block nested inside the
                    # thing it is checked against reads as part of the measurement rather than as
                    # the check on it. Null when nobody asked for the gate.
                    convert_check=(checker.block() if checker is not None else None),
                    # §3b-1, and it travels the same way for the same reason.
                    input_check=(input_checker.block()
                                 if input_checker is not None else None),
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
                    # **What RAN, read off the writer** — the codec, both parameter strings
                    # and the settings that codec has. See `_encode_fields`.
                    **_encode_fields(writer),
                    # **§17, and it rides the stats because `retime` is handed no `trace`** —
                    # the same channel `estimate` and `convert_check` use. `handler` lifts it to
                    # the record's TOP level and uploads its PNGs. **None on an unarmed run and
                    # on one whose scoring failed**, which §17a rules is ABSENT rather than null.
                    reference=reference_block,
                    # **Same defect, same fix, one line further.** §2 lists `snap_tolerance` among
                    # the axes a run must carry and the record filed it null for the same reason.
                    # `target_fps` is what the whole job was FOR. Both were envelope-only.
                    target_fps=float(target_fps),
                    snap_tolerance=snap_tolerance)
    finally:
        # **THE REFERENCE GOES FIRST, AND THE ORDER IS THE FIX.** `capture.release()` below is an
        # unguarded cv2 call; if it raises on a wedged `VideoCapture` the rest of this block never
        # runs, and a cleanup sequenced behind it would leak exactly on the paths it exists for.
        # *Ordering rather than wrapping, because changing how a pre-existing release reports its
        # own failure is a different decision from making this cleanup reliable.* Found in review.
        #
        # **§6g's disk is released on EVERY exit from this function, not only the scored one.**
        # `reference.score` deletes the file in its own `finally`, but it is reached only when
        # the encode SUCCEEDED — an ffmpeg failure, an OOM, a broken pipe or the re-raised writer
        # error all propagate past the scoring block, and tens of GB of `reference.raw` would
        # survive in the workdir. **On a warm worker that is the next job's disk**: it computes
        # its own bound against a `free` that is 20 GB short and refuses a request that would
        # have fit, with a message blaming the caller's frame size. *Found in review; the module
        # docstring promised "deleted whatever happens" and delivered it only for one path.*
        #
        # **Belt and braces rather than a move.** `score`'s own cleanup stays where it is,
        # because it releases the disk BEFORE the PNG uploads rather than after this function
        # returns, and on the 8K path those are different amounts of time.
        if reference_path is not None:
            try:
                if os.path.exists(reference_path):
                    os.remove(reference_path)
            except OSError as exc:  # noqa: BLE001 — cleanup must not displace a real failure
                print("[reference] could not remove {}: {}".format(reference_path, exc),
                      flush=True)
        if capture is not None:
            capture.release()


def _encode_fields(writer):
    """What the encode RAN AT, read off the writer that ran it (§6c, §15a).

    **ALL FIVE SETTINGS, `crf` INCLUDED, AND THE RECORD IS WHY.** `crf` and `preset` used to be
    added to the ENVELOPE by `handler._retime` and never reached `stats` — so `trace["retime"]`,
    which is built from `stats`, filed both as null on every run ever written while the envelope
    beside it said `crf: 12`. `instrumentation.md` §2 names them as the two missing settings and
    says why: *a corpus recording three of five cannot attribute a difference*. **The corpus is
    the RECORD.** Reading them off the writer is what puts them in both artefacts and makes them
    impossible to disagree — the module's defaults are not the same thing as what ran, now that
    the settings can move.

    **THE THREE x264 FIELDS ARE OMITTED ENTIRELY UNDER h265, WHICH IS NOT THE SAME AS NULL**
    (`docs/instrumentation.md` §15a). *A row reading `sliced_threads: false` beside
    `codec: "h265"` would assert that a parameter x265 does not have took a value* — the corpus
    telling a reader something untrue in a field they have no reason to doubt. With §6e ruling
    2's refusal at the door and this absence here, **`sliced_threads` present on a row implies
    h264 by construction**, which is the self-keying property §14a already relies on.

    **The two parameter strings are both always present, each null under the other's codec.** The
    column exists on every row either way, so a reader can ask what bounded an encode without
    first knowing which codec answered — §15a's own argument for shipping `x265_params` as a null
    column even if no bound had shipped at all.

    **`codec` is here rather than assembled in `handler` from the request.** §15b holds that this
    field records WHAT RAN and not what was asked for; the two are the same thing today only
    because §6e leaves `codec: "source"` refused, and reading it off the writer is what keeps the
    field true on the day they stop being the same. *`handler` lifts it out of `retime` to the
    record's top level, where §15c's inference rule reads it.*
    """
    fields = {"codec": writer.codec,
              # **§6f, and carried EXPLICITLY at 8 rather than omitted.** §15a rules absence to
              # mean 8, so both spellings are legal — and one rule for `codec` and `bit_depth`
              # together is easier to hold than a field that appears only at one of its values.
              # *Absence stays meaningful for the 45 rows that predate the field.*
              "bit_depth": writer.bit_depth,
              "x264_params": writer.x264_params,
              "x265_params": writer.x265_params,
              "crf": writer.crf,
              "preset": writer.preset}
    if writer.codec != "h265":
        fields.update(threads=writer.threads,
                      sliced_threads=writer.sliced_threads,
                      rc_lookahead=writer.rc_lookahead)
    return fields


def _print_write_distribution(writer):
    """§18c's distribution as one log block. **Never raises and never delays a delivery.**

    Order statistics rather than a histogram, because a histogram's buckets would have to be
    chosen before anyone knows the scale — which is the thing this measurement exists to
    discover. `p99_over_p50` and `slowest_n_share` carry the verdict, `drain_s` carries the half
    no per-write sample can see, and **the two must be read together**: a flat distribution with
    a large drain is backpressure, not pipe transfer.

    **The conclusion is the gate's to draw from a run, not this function's to print.**
    """
    try:
        distribution = writer.write_distribution()
        # **THE DRAIN IS PRINTED FIRST AND UNCONDITIONALLY, AND THAT IS THE FIX.** It used to sit
        # after an early return taken whenever there were no samples — so the one run where the
        # drain is the interesting number, a first `write` that broke the pipe after ffmpeg was
        # already spawned, printed nothing about the seconds `__exit__` then spent waiting for it.
        # **A fallback that skips the case it exists for.**
        print("[write-dist] drain {} s closing the pipe and waiting for ffmpeg | docs/test-plan.md "
              "18c: flat writes AND a small drain is pipe transfer; a large drain is encoder "
              "backpressure however flat the writes look".format(
                  "not measured" if writer.drain_s is None else round(writer.drain_s, 3)),
              flush=True)
        if not distribution:
            print("[write-dist] no writes were sampled; nothing to distribute", flush=True)
            return
        if not distribution.get("samples"):
            print("[write-dist] {}; first write {} ms".format(
                distribution.get("note", "nothing to distribute"),
                distribution.get("first_ms")), flush=True)
            return
        print("[write-dist] {samples} steady-state write(s), {total_s}s total, mean {mean_ms}ms | "
              "min {min_ms} p50 {p50_ms} p90 {p90_ms} p99 {p99_ms} max {max_ms} ms".format(
                  **distribution), flush=True)
        print("[write-dist] p99/p50 {} | slowest {} write(s) hold {} of the total | first write "
              "{} ms, EXCLUDED from every figure above because it races ffmpeg's start-up and a "
              "queue cannot remove a process spawn".format(
                  distribution["p99_over_p50"], distribution["slowest_n"],
                  distribution["slowest_n_share"], distribution["first_ms"]), flush=True)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never displace a real failure
        print("[write-dist] NOT reported ({}: {}). The job is unaffected.".format(
            type(exc).__name__, str(exc)[:120]), flush=True)


def _seed_estimate(progress, source, stats, scale, encode_arm=None, armed=None,
                   substituted=None):
    """Price the job, seed the ETA with it, and hand the whole answer back. **Never raises.**

    Returns the §9b estimate — point, band, basis and corpus — or None where it could not be
    priced. **A refusal to quote is not a failure of the job**: the estimator exists to make the
    ETA better than `observed`, and a run that cannot be priced falls back to exactly the
    behaviour it had before this line existed.
    """
    import estimator  # noqa: PLC0415 — pure-python, imported here like everything else on this path

    # **Bound BEFORE the `try`, so the `finally` below reads a name that always exists.** The
    # alternative was fishing it out of `locals()`, which works and is unreadable — and a
    # measurement whose control flow needs explaining is one nobody will touch correctly later.
    estimate = None
    try:
        n_in = source_frame_count(source)
        # **§9e's two declared inputs.** `crf` because every row in the corpus is CRF 12 and a
        # coefficient cannot be estimated for an axis with one value — so the axis is named and
        # an estimate outside it says so rather than extrapolating. The arm for the sharper
        # reason: it moves `compute_s` by up to 1.72x on identical work, so an estimate quoted
        # against a different arm is outside its population and not merely at its edge.
        per_frame, estimate = estimator.seconds_per_frame(
            source["width"], source["height"], n_in,
            stats.get("n_out"), stats.get("n_synth"), scale=scale or 1,
            crf=(encode_arm or {}).get("crf"), encoder_arm=encode_arm, armed=armed,
            substituted=substituted)
        # **Labelled `predicted_<basis>` by `expect` itself**, so the payload distinguishes a
        # planned estimate from a measured one — which is the whole reason `eta_basis` exists and
        # the reason §8g grades `eta.first_basis` as well as `eta.first_s`.
        # **`basis_for(crf)` and not the bare `BASIS`** — contract §9e requires the declared CRF
        # in the basis string, and THIS is the channel that carries it to the caller and to
        # `eta.first_basis`, which is what §8g grades. Read off the estimate rather than
        # recomputed, so the seed and the answer cannot name different bases.
        # **`.get` with a fallback, NOT `estimate["basis"]`.** Subscripting made the ETA SEED
        # depend on the estimate's shape: a usable `per_frame` alongside an estimate missing that
        # key would raise here, `expect` would never be called, and the run would fall back to
        # `observed` — rebuilding §8d's 11,553-second failure that this very line exists to
        # prevent. **The seed is the valuable half and it must survive a malformed label.**
        progress.expect(per_frame, basis=(estimate or {}).get("basis") or estimator.BASIS)
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
        # **CLEARED, AND WITHOUT THIS THE `finally` CONTRADICTS THE RECORD.** `estimate` is
        # assigned before `progress.expect` and the priced-print, so a failure in either left it
        # bound while this branch returned None — the record would file `estimate: null` while
        # the `finally` below had already printed OUTSIDE THE CORPUS sentences for that same
        # estimate. **Two artefacts of one run disagreeing about whether the job was priced**,
        # which is the exact defect moving the print into a `finally` was meant to remove; it had
        # only been inverted. One name decides both.
        estimate = None
        print("[eta] not priced ({}: {}); the ETA falls back to what this run measures".format(
            type(exc).__name__, str(exc)[:200]), flush=True)
        return None
    finally:
        # **OUTSIDE THE `try` THAT DECIDES WHETHER THE ESTIMATE EXISTS, AND THAT IS DELIBERATE.**
        # §9e requires the caveat said out loud on the run itself: an estimate that has left its
        # corpus and says so only in a record is a caveat discovered after the money is spent.
        # **But printing it inside the try put it between `progress.expect` and the `return`** —
        # so a failure in the print would be caught above and return None, filing `estimate: null`
        # in the record while every progress payload the caller already received carried an
        # `eta_basis` derived from that same estimate. *Two artefacts of one run disagreeing about
        # whether the job was priced.* Nothing here can change what is returned.
        try:
            for sentence in (estimate or {}).get("outside_corpus") or []:
                print("[eta] OUTSIDE THE CORPUS: {}".format(sentence), flush=True)
        except Exception:  # noqa: BLE001
            pass


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


def _tensors(frames, device, clock=None, checker=None):
    """cv2 frames to tensors, lazily, one at a time — never a list, whatever the clip length.

    **This is `convert_in_s`, the INBOUND step** (§10a): the strided gather over the decoder's
    negative-stride BGR view, on the host. §9a had it and `_to_rgb24` sharing one `convert_s` on
    the grounds that they were one activity; **§10a overruled that**, because the two sit on
    opposite sides of the model and only one of them is what
    `docs/conversion-wave.md` proposes to change. One field covering both is the defect §9
    complains about — a boundary nobody can read — wearing the opposite sign.
    """
    import torch  # noqa: PLC0415 — the interpolator has already imported it by the time we run

    # **`device` is positional and REQUIRED, and it used to default to `None`.** `Tensor.to(None)`
    # matches the `to(device=None, dtype=None)` overload and returns `self` — so a caller who
    # forgot it got a correct-valued tensor produced entirely on the host, `convert_in_s` back at
    # its pre-wave value, and no error anywhere. **A wave that silently does not happen is worse
    # than one that fails**, and a keyword with a benign default is how that gets shipped.
    if device is None:
        raise ValueError("_tensors needs the device the model lives on; None would convert on "
                         "the host and report the wave as having no effect")

    for index, frame in enumerate(frames):
        # **Taken BEFORE the shipped conversion and outside its clock** — §5-0a's ordering rule,
        # and the first fix for that rule got the ordering wrong one wave ago by cloning after
        # the shipped arm had already run. What is protected here is the DECODER'S array, which
        # `torch.from_numpy` aliases without copying.
        before = checker.snapshot(frame) if checker is not None else None
        if clock is None:
            tensor = _to_tensor_device(frame, torch, device)
        else:
            with clock.timing("convert_in_s"):
                tensor = _to_tensor_device(frame, torch, device)
                # **§9b'S TRAP, ONE STAGE OVER, FOR THE THIRD TIME** — §9b at `_synthesise`,
                # §10b at `_load_pair`, and now here. The `.to(device)` inside blocks, because a
                # copy from pageable host memory synchronises; **everything after it does not.**
                # The gather, the widen and the divide are enqueued and return, and their real
                # cost lands at the next synchronisation — `stages.synchronise` inside
                # `_load_pair`, charged to `convert_dev_s`. Unsynchronised, `convert_in_s` would
                # be an upload clock wearing a conversion's name, **on the one number this wave
                # is judged by.**
                #
                # Unlike §10b's, this wait is genuinely moved rather than added: the pad in
                # `_load_pair` consumes this tensor, so the sync there was already waiting on
                # this work. What changes is which field is charged.
                stages.synchronise(tensor)
        # **Outside the timed block**, because the reference arm is the instrument and not the
        # shipped conversion; charging it to `convert_in_s` would corrupt the one number this
        # wave is judged by. It lands in `stage_residual_s`, reported and not absorbed.
        if checker is not None:
            checker.compare(index, before, frame, tensor, torch)
        yield tensor
