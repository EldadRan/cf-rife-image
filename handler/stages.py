"""Where `compute_s` actually goes — `docs/instrumentation.md` §9.

**§8a split `wall_s` into transfer and compute and the 44% fell out in one run. This is the same
split one layer down**, and the campaign has already spent an arm on not having it: sixteen
encoder threads at 8K used the same 4.4 cores as four, which falsifies the encoder as the
constraint and leaves decode, the model pass and the RGB conversion holding the time with one
number covering all three.

**THE CLOCK IS A PARAMETER AND NEVER AN ATTRIBUTE, AND CONTRACT §4b IS WHY.** The obvious home for
an accumulator is the `Interpolator`, which lives for the whole job — and a cascade variant calls
`interpolator.stream()` TWICE. Two live streams sharing one accumulator is *"a per-call result
published on an object that survives the call, read by somebody who had no way to know it was
stale"*, which is the defect §4a and §4b exist to refuse, in the one module where §4b's own
example was found. So a `StageClock` is created per retime, handed down as an argument, and
returned as data.

**NOTHING HERE MAY RAISE AND NOTHING HERE MAY BE EXPENSIVE.** It runs per frame — five timers
across 480 frames at 8K — so it is a `perf_counter` and a dict add, and every entry point
swallows. A measurement that can fail a delivered master is not a measurement worth taking.
"""

import contextlib
import time

#: §9a as amended by §10a, verbatim. Named here so the record, the worker and anyone diffing this
#: against the document read one list rather than three copies of it.
#:
#: **§10a REPLACED `convert_s` with the three activities it covered**, and replaced rather than
#: supplemented: a record carrying the old total beside its own parts would still close the
#: identity, because the old name would fall outside this tuple and be swallowed by `RESIDUAL` —
#: which is exactly the number `RESIDUAL` warns says the wrong thing with confidence. The three
#: name the three call sites: `convert_out_s` is `routec._to_rgb24` (the device-to-host copy and
#: the host arithmetic after it), `convert_in_s` is `routec._to_tensor` (the strided gather over
#: the decoder's negative-stride BGR view), and `convert_dev_s` is `interpolate._load_pair` (the
#: device-side cast, upload and pad, once per PAIR and cached).
STAGES = ("load_s", "decode_s", "model_s",
          "convert_out_s", "convert_in_s", "convert_dev_s", "write_wait_s")

#: What the stages did not account for. **Reported, never absorbed** (§9a): `compute_s` also covers
#: the probe, validation, the plan, the output probe and the frame count — and a residual folded
#: into whichever stage happens to be adjacent is a number that says the wrong thing with
#: confidence. A residual named is a residual someone can chase.
#:
#: **It is NOT purely fixed cost and must not be read as one.** The surplus-frame `grab()` sweep
#: at the end of a retime is decoder work that is deliberately not a `read()`, so it scales with
#: whatever the source holds beyond the plan. Anything else that grows with frame count belongs
#: in a stage and is a defect in the split rather than a property of the residual — which is what
#: makes a residual that grows with resolution worth chasing rather than shrugging at.
RESIDUAL = "stage_residual_s"

#: **The encoder's end-of-stream drain, a TERM IN THE IDENTITY AND NOT AN EIGHTH STAGE**
#: (`docs/instrumentation.md` §16, §16c). `stdin.close()` then `wait()`, after the last frame has
#: been handed over — the encoder's queued backlog, which at 8K with a lookahead can be tens of
#: 50 MiB frames and settles entirely in that call.
#:
#: **IT IS MEASURED BY `encoder.MasterWriter` AND NOT BY THIS CLOCK**, which is half the reason it
#: is not a stage: making it one would feed one instrument's output through another's timing
#: context for no gain. **The other half is backward compatibility** — `STAGES` keeps its seven
#: members, so the 45 records banked before this field close the identity unchanged, because a
#: missing term contributes zero to a sum. *An eighth stage would have needed `record_version`
#: keying in the kit to avoid retroactively un-certifying them.*
DRAIN = "drain_s"


class StageClock:
    """Per-retime accumulator for §9a's stages as amended by §10a. **Never raises.**

    Handed down through `variants.run` into `Interpolator.stream`, which is the only way a
    measurement of the model pass can reach the caller without the interpolator publishing it.
    """

    def __init__(self):
        self._totals = dict.fromkeys(STAGES, 0.0)
        #: **PRESENT AND NULL on a run that never opened a writer** (§16), which is the honest
        #: reading of a job refused before the encode or reaped in the model. Set from
        #: `MasterWriter.drain_s` by whoever ran the encode; `None` until then, and `totals`
        #: reports it as such rather than as a measured zero.
        self.drain_s = None
        #: Names banked against that are not in `STAGES`. **Kept so a typo is visible rather than
        #: absorbed** — see `_bank`.
        self._unknown = {}

    @contextlib.contextmanager
    def timing(self, stage):
        """Time a block into `stage`. **The clock stops even when the block raises**, because a
        run that died in the model is a run whose model time is the most interesting number it
        has."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self._bank(stage, time.perf_counter() - started)

    def add(self, stage, seconds):
        """Bank a span measured elsewhere. For a stage whose boundaries are not a block."""
        self._bank(stage, seconds)

    def _bank(self, stage, seconds):
        """Add to one stage. **A name that is not in `STAGES` is quarantined, not created.**

        The first version wrote `self._totals[stage] = self._totals.get(stage, 0.0) + …`, which
        CREATES an unknown key rather than raising — and `totals` reported only `STAGES` while
        subtracting every value. So a typo put real seconds into the record attributed to
        nothing, silently, with the residual absorbing them: **every visible signal correct and
        the number belonging to something else**, which is the shape this project keeps finding.
        The `except` that looked like it guarded this was inert, because a dict assignment with a
        default cannot raise.

        Quarantined rather than dropped, and said out loud: a stage nobody can see is how this
        started, and losing the seconds silently would be the same defect wearing the opposite
        sign.
        """
        try:
            seconds = float(seconds)
            if stage in self._totals:
                self._totals[stage] += seconds
                return
            if stage not in self._unknown:
                print("[stages] {!r} is not one of {}; its {:.3f}s is quarantined and will not "
                      "reach the record's stage fields".format(stage, STAGES, seconds), flush=True)
            self._unknown[stage] = self._unknown.get(stage, 0.0) + seconds
        except Exception:  # noqa: BLE001 — a measurement must never displace a real failure
            pass

    def totals(self, compute_s=None):
        """Every name in `STAGES` rounded, plus the residual.

        **The residual is only computed when `compute_s` is known**, and it is `compute_s` less
        the stages rather than the stages less anything: the stages are measured and `compute_s`
        is measured, and which of them is the remainder is a statement about what the fields mean.
        A negative residual is reported as it comes out — it would mean the stages overlap or
        that a clock ran outside `compute_s`, and rounding that to zero would hide exactly the
        defect it is evidence of.
        """
        out = {name: round(self._totals.get(name, 0.0), 3) for name in STAGES}
        # **§16c's term. Rounded like the stages and for the identical reason** — the remainder
        # below is taken from the ROUNDED parts, so a term left unrounded here would leave its
        # rounding error outside the identity and the check grades that identity exactly.
        # **INSIDE the guard, because this function promises never to raise and `drain_s` is a
        # PUBLIC attribute assigned from outside this class.** A non-numeric value there would
        # otherwise turn a never-raising accumulator into a raise, and the run would lose all
        # SEVEN stage timings rather than just the drain. Found in review.
        try:
            drain = None if self.drain_s is None else round(float(self.drain_s), 3)
            # **NaN and the infinities convert WITHOUT raising**, and a NaN here would propagate
            # into the residual and out through `json.dumps` as a bare `NaN` token, which strict
            # JSON readers reject — a record the corpus cannot load, from a diagnostic. *`!=`
            # against itself is the NaN test that does not need a float comparison.*
            if drain is not None and (drain != drain or drain in (float("inf"), float("-inf"))):
                raise ValueError("drain_s is not finite: {!r}".format(self.drain_s))
        except Exception:  # noqa: BLE001 — this function must never raise; see the docstring
            print("[stages] drain_s is {!r}, which is not seconds; reported as null".format(
                self.drain_s), flush=True)
            drain = None
        out[DRAIN] = drain
        if compute_s is not None:
            try:
                # **The remainder is taken from the ROUNDED stages, not the raw ones.**
                # Rounding each stage for the report and subtracting the unrounded sum leaves
                # each stage's rounding error outside the identity, so the stages and the
                # residual add to `compute_s` only to within a couple of milliseconds.
                # `handler._timings` already
                # carries this reasoning for §8f's three-way split — *"rounded BEFORE the
                # remainder is taken, which is what makes the identity exact rather than
                # exact-to-a-tolerance"* — and this is the same identity one layer down. The
                # quarantine above is what keeps an unknown name out of the arithmetic entirely.
                # **§16c: `compute_s` less the stages AND less the drain.** Before this the
                # drain landed here unnamed — 8K h264 7.231 s against 8K h265 38.029 s, a 5x
                # move in a bucket nobody reads, on the wave that introduced it. *A null drain
                # contributes zero, so a run with no writer reports exactly what it used to.*
                out[RESIDUAL] = round(
                    float(compute_s) - sum(out[n] for n in STAGES) - (drain or 0.0), 3)
            except Exception:  # noqa: BLE001
                pass
        return out


def synchronise(tensor=None):
    """Block until the GPU has finished what was enqueued. **This is §9b's ruling, in code.**

    **Torch is asynchronous, so a wall clock around the model call times the ENQUEUE.** The real
    cost lands at the first synchronisation, which on this path is the `.to("cpu")` inside
    `routec._to_rgb24` — so a naive `model_s` reads near zero and `convert_out_s` silently
    swallows the model. §9b names that and forbids it.

    **Ruled: end the model clock at an explicit synchronisation.** §10b then ruled the same for
    `_load_pair`'s clock, which ends at an enqueued pad. **BOTH CALLERS MOVE A WAIT RATHER THAN
    ADDING ONE, AND FOR THE SAME REASON: what this call waits on is already a dependency of
    something the path synchronises on a moment later.**

    - **`_synthesise` (§9b).** Every synthesised frame goes through `.to("cpu")` one line later
      in `_to_rgb24`, so the wait was going to happen microseconds afterwards either way.
    - **`_load_pair` (§10b).** The model consumes the very tensor this pads, so
      `_synthesise`'s own `synchronise(out)` was ALREADY waiting on that pad — **the pad's
      execution has been landing in `model_s` all along, and now lands in `convert_dev_s`.**
      **What is new is the CALL, not the wait.**

    **The instrument does not move the total it is splitting**, which is the property that makes
    both splits worth trusting.

    **The one genuinely added term, and it is at `_load_pair` only:** the overlap lost between the
    pad executing and the CPU enqueuing the model call — an enqueue window, **191 times on a
    480-frame job** (191 pairs from 192 source frames), nothing between the two on this strictly
    serial path but a `yield`. **Bounded by argument and not by measurement**, which §10b says out
    loud rather than hiding: `F-2026-08-25-10` measures identical work varying 18.8-49% on one
    worker, so a term that size cannot be resolved against a corpus at all and a close condition
    asking for that comparison would have graded whichever host answered.

    **§10e checks the separable half instead, and reads it INSIDE one run so host variance cannot
    drive it:** the move shows up as `model_s` falling by roughly what `convert_dev_s` gains, so
    the sum-share `(model_s + convert_dev_s) / compute_s` is compared against `sha-c8a2f63`'s
    `model_s / compute_s`. **If that sum-share moves materially, the sync did something other
    than move a wait.**

    **`_load_pair`'s price is bounded by its cache** — one call per PAIR, not per frame — and
    the first cut of §10b's code got that wrong by putting the call outside the cache rather
    than under it (`F-2026-08-26-2`). The alternative, leaving it async and saying so in the field
    name, is rejected by §10b: a field that must be read with a footnote is the one that gets
    quoted without one.

    **Device-wide, not stream-scoped, and the DEVICE IS TAKEN FROM THE TENSOR.** `tensor` selects
    both whether to synchronise and which device to wait on; `torch.cuda.synchronize(device)` then
    waits on every stream on that one. Waiting on all of a device's streams is correct for both
    callers — each wants everything it enqueued to be done — but it means a caller cannot use
    this to wait for its own work alone.

    **The device argument is not cosmetic.** Bare `torch.cuda.synchronize()` waits on torch's
    CURRENT device, which is not necessarily the tensor's: `Interpolator` takes its device as a
    constructor argument, and the guard below admits `cuda:1` as readily as `cuda:0`. An
    interpolator on the second GPU would have passed the guard, synchronised the first, and left
    `convert_dev_s` reading the enqueue while the pad drained into `model_s` — **§10b's defect
    restored exactly, wearing the field name that says it was fixed.** Today `handler` builds the
    interpolator with the bare `"cuda"` default so nothing reaches that state; a latent instrument
    failure is still an instrument failure, and this one costs an argument to close.

    **CUDA events were considered and rejected**, and the reason is not cost. `Event.elapsed_time`
    measures GPU-BUSY time, and §9a asks the stages to account for `compute_s`, which is wall.
    On a pipeline where the CPU blocks waiting for the GPU those are different quantities, so
    events would produce numbers that do not add up to the thing they are splitting — a better
    instrument answering a different question.

    Cheap where there is nothing to wait for: no torch, no CUDA, or a CPU tensor and this returns
    without touching the device. Never raises.
    """
    try:
        device = getattr(tensor, "device", None) if tensor is not None else None
        if tensor is not None and getattr(device, "type", "") != "cuda":
            return False
        import torch  # noqa: PLC0415 — a GPU-box import, like every other torch touch

        if not torch.cuda.is_available():
            return False
        # `None` keeps the historical meaning — no tensor offered, so wait on the current device.
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)
        return True
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return False
