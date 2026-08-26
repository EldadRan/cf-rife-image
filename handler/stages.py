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

#: §9a, verbatim. Named here so the record, the worker and anyone diffing this against the
#: document read one list rather than three copies of it.
STAGES = ("load_s", "decode_s", "model_s", "convert_s", "write_wait_s")

#: What the five did not account for. **Reported, never absorbed** (§9a): `compute_s` also covers
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


class StageClock:
    """Per-retime accumulator for §9a's five stages. **Never raises.**

    Handed down through `variants.run` into `Interpolator.stream`, which is the only way a
    measurement of the model pass can reach the caller without the interpolator publishing it.
    """

    def __init__(self):
        self._totals = dict.fromkeys(STAGES, 0.0)
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
        """`{load_s, decode_s, model_s, convert_s, write_wait_s}` rounded, plus the residual.

        **The residual is only computed when `compute_s` is known**, and it is `compute_s` less
        the five rather than the five less anything: the five are measured and `compute_s` is
        measured, and which of them is the remainder is a statement about what the fields mean.
        A negative residual is reported as it comes out — it would mean the stages overlap or
        that a clock ran outside `compute_s`, and rounding that to zero would hide exactly the
        defect it is evidence of.
        """
        out = {name: round(self._totals.get(name, 0.0), 3) for name in STAGES}
        if compute_s is not None:
            try:
                # **The remainder is taken from the ROUNDED five, not the raw ones.** Rounding
                # each stage for the report and subtracting the unrounded sum leaves each stage's
                # rounding error outside the identity, so the five and the residual add to
                # `compute_s` only to within a couple of milliseconds. `handler._timings` already
                # carries this reasoning for §8f's three-way split — *"rounded BEFORE the
                # remainder is taken, which is what makes the identity exact rather than
                # exact-to-a-tolerance"* — and this is the same identity one layer down. The
                # quarantine above is what keeps an unknown name out of the arithmetic entirely.
                out[RESIDUAL] = round(float(compute_s) - sum(out[n] for n in STAGES), 3)
            except Exception:  # noqa: BLE001
                pass
        return out


def synchronise(tensor=None):
    """Block until the GPU has finished what was enqueued. **This is §9b's ruling, in code.**

    **Torch is asynchronous, so a wall clock around the model call times the ENQUEUE.** The real
    cost lands at the first synchronisation, which on this path is the `.to("cpu")` inside
    `routec._to_rgb24` — so a naive `model_s` reads near zero and `convert_s` silently swallows
    the model. §9b names that and forbids it.

    **Ruled: end the model clock at an explicit synchronisation. The reason it is safe is that
    this path ALREADY synchronises once per synthesis** — every synthesised frame goes through
    `.to("cpu")` on the very next line of `_to_rgb24`. So this call does not ADD a wait, it moves
    one that was going to happen microseconds later. **The instrument does not move the total it
    is splitting**, which is the property that makes the split worth trusting; a sync inserted
    into a pipeline that was genuinely overlapping would have bought a number by changing the
    thing measured.

    **CUDA events were considered and rejected**, and the reason is not cost. `Event.elapsed_time`
    measures GPU-BUSY time, and §9a asks the five stages to account for `compute_s`, which is wall.
    On a pipeline where the CPU blocks waiting for the GPU those are different quantities, so
    events would produce five numbers that do not add up to the thing they are splitting — a
    better instrument answering a different question.

    Cheap where there is nothing to wait for: no torch, no CUDA, or a CPU tensor and this returns
    without touching the device. Never raises.
    """
    try:
        if tensor is not None and not getattr(getattr(tensor, "device", None), "type",
                                              "") == "cuda":
            return False
        import torch  # noqa: PLC0415 — a GPU-box import, like every other torch touch

        if not torch.cuda.is_available():
            return False
        torch.cuda.synchronize()
        return True
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return False
