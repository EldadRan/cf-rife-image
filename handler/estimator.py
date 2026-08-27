"""What a job will cost before it is submitted — memory certainly, time provisionally.

**Contract §9.** Two questions that are not the same question. *Will it fit?* is answerable
before anything runs and its answer is certified from three resolutions across a 15x span.
*How long?* is answerable only against a corpus, and this project's corpus is clean at last in
the one way that mattered — `compute_s` separated from transfer — and narrow in a new way that
`CORPUS` states rather than hides.
CF ruled both are built now, and §9b's three requirements are what make that ruling safe:

  the estimate carries its basis; the spread is published, not hidden; **and the coefficients
  live in one named place and are refittable without touching logic.**

The third is the one this module is shaped around. Every number is a module constant with a
comment saying where it came from — `FIT_*`, `TIME_*`, `CORPUS` — and nothing below the
constants block holds a literal.

**THE REFIT §9b PRE-ORDERED HAS LANDED (2026-08-26), AND IT CHANGED CODE AS WELL AS CONSTANTS.**
Worth stating plainly rather than leaving the original promise standing, because the promise was
*"when it lands it is a change of these constants and not a change of code"* and this is its
counterexample. The requirement bought what it was for — the LEVEL moved by editing four numbers
— but the SHAPE moved too, and could not have been foreseen: the old model carried a fixed term
and no transfer term because the corpus it was fitted from could not tell the two apart, every
wall figure having the download and the upload inside it. `docs/instrumentation.md` §8a split
them, and **a quantity that becomes visible is a term that becomes fittable.** A model whose
shape may never change in the light of a measurement it could not previously take is not a
model; it is a constant with arithmetic attached.

**`interp_plan` is not this module and does not become it.** That file states the FORM of route
C's fit predicate and refuses to quote, on the standing rule that a placeholder is
indistinguishable from a measurement. It is still right about that: `interp_plan.peak_vram_gb`
prices from a *registry line* that does not exist, and none is invented here. What this module
holds is a fit against readings this project took, cited to them, which is a different kind of
answer from an unmeasured coefficient and is named separately so the two cannot be confused.
`padded_megapixels` is imported from `interp_plan` rather than restated, because the padding
rule lives there and two copies of a rule are two rules.
"""

import interp_plan

# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE COEFFICIENTS. One named place, refittable without touching anything below.
# ─────────────────────────────────────────────────────────────────────────────────────────────

#: `peak_vram_gb = FIT_VRAM_SLOPE x padded_megapixels + FIT_VRAM_INTERCEPT` — contract §9a,
#: certified: three resolutions, a 15x span, residuals under half a percent.
#:
#: **RE-CERTIFIED 2026-08-27 AFTER THE CONVERSION WAVES, AND THE OLD PAIR WAS WRONG IN THE
#: PERMISSIVE DIRECTION.** `0.805 / 0.016` was fitted when the outbound conversion ran on the
#: host; moving it onto the card allocates full-resolution float32 intermediates there that were
#: nowhere near CUDA before, and once the frame is big enough they raise
#: `torch.cuda.max_memory_allocated()`:
#:
#:      2.21184 MP    measured 1.86    old fit 1.80    under by 0.063
#:      8.35584 MP    measured 6.95    old fit 6.74    under by 0.208
#:     33.42336 MP    measured 27.72   old fit 26.92   under by 0.798   <- 8K
#:
#: **The estimator REFUSES JOBS against this line**, so under-predicting is the one direction
#: that fails silently — and it under-predicted worst at the resolution with the least headroom.
#: `docs/conversion-wave.md` §2f required this re-certification in advance rather than leaving it
#: to be noticed.
#:
#: **FITTED ON THE WORST DEVICE-ERA READING AT EACH AREA, NOT THE MEAN, AND THAT IS DELIBERATE.**
#: The increment is not deterministic — 4K device runs read 6.75, 6.75, 6.84, 6.94 and 6.95,
#: because whether a transient raises the peak depends on what torch's caching allocator has
#: already reserved. **A refusal predicate fitted to the mean would be wrong half the time, in
#: the permissive direction.** Residuals against the worst-case fit are within 0.0003 GiB across
#: the 15x span, over 28 readings in `records/`, 8 of them device-era.
#: **The exact line rounded UP at the last digit, and four decimals were not enough.** The first
#: cut of this re-certification shipped `0.8285 / 0.0271`, which under-predicts all three worst
#: readings — 0.0016 GiB at 8K — so **the constants did not have the property the section above
#: asserts for them**, in the one direction that fails silently. The geometry was never the
#: problem: the exact line through the 1080p and 8K worst readings already passes through-or-above
#: the 4K one. **Truncation was.** A predicate is exactly as pessimistic as its evidence and no
#: more, which is why this is the exact line nudged rather than a rounder, safer-looking pair.
FIT_VRAM_SLOPE_GB_PER_MP = 0.82855
FIT_VRAM_INTERCEPT_GB = 0.02741

#: What the predicate holds back from `vram_total_gb`. Allocator fragmentation, the CUDA context
#: and cuDNN's workspaces are all real and none of them is in the fit.
VRAM_RESERVE_GB = 2.0

#: **`compute_s = (TIME_PER_IN x n_in + TIME_PER_SYNTH x n_synth) x mp^TIME_EXPONENT`**
#: **`transfer_s = TIME_TRANSFER_PER_MP x mp`**, and the answer is their sum, because the ETA is
#: a promise to a caller who waits through the transfer too.
#:
#: **REFITTED 2026-08-26 from six instrumented runs on `sha-9672d42`** — the first corpus in this
#: project's history where `compute_s` is separable from transfer, which is exactly what §9b was
#: waiting for and what `docs/instrumentation.md` §8a was built to produce.
#:
#: **The shape changed as well as the numbers, and §9b's requirement is what made that visible
#: rather than what it violated.** The old model had a `TIME_FIXED_S` term and no transfer term,
#: because the corpus it was fitted from could not tell the two apart — every wall figure had the
#: download and the upload inside it, so a fixed cost and a transfer cost were one unknown. §8a
#: split them. The fixed term is not separable from THIS corpus either (see `resolutions` below)
#: and has been dropped rather than carried at a value nothing measures.
TIME_PER_IN = 0.05525
TIME_PER_SYNTH = 0.02721
TIME_EXPONENT = 1.1053

#: **Transfer is linear in padded area because the BYTES are.** 73 MB in and 193 MB out at 4K
#: against 580 and 743 at 8K, over one link: 2.15 s/MP implied by the 4K runs and 2.02 by the 8K
#: ones, which is as close to one constant as a six-reading corpus can show. **This is a property
#: of this link and these sources, not of the worker** — a caller on a slower connection has a
#: different constant and nothing here would know.
TIME_TRANSFER_PER_MP = 2.0849

#: **§9b's first requirement, as data rather than as prose.** Every time answer carries this, so
#: a number that cannot say where it came from cannot leave this module.
CORPUS = {
    "name": "cf-rife-project records/, six instrumented runs on sha-9672d42, 2026-08-26",
    "readings": 6,
    # **TWO distinct padded areas and ONE frame plan across all six**, and this is the fit's
    # binding limitation rather than a footnote. Every run was 192 source frames producing 382
    # syntheses, so `n_in` and `n_synth` are perfectly collinear here and this corpus cannot
    # separate them; the ratio between them is HELD from the previous corpus, whose 24->12
    # decimation run is the only reading this project has ever taken that told them apart.
    #
    # **And with two distinct areas, two free parameters fit them exactly.** `TIME_EXPONENT` and
    # the overall level are determined by the two cluster means, so the fit passes through both by
    # construction — **its residual at those points is arithmetic, not evidence.** What the corpus
    # does measure well is the SPREAD, three readings at each point, and that is what the
    # published band is built from.
    "resolutions": ("3840x2160", "7680x4320"),
    "distinct_padded_areas": 2,
    "frame_plans": 1,
    "gpu": "NVIDIA A40",
    # **The compute half is fitted against `compute_s` and excludes transfer**, which the previous
    # corpus could not do. Transfer is its own term, fitted separately against the same six runs.
    "fitted_against": "timings.compute_s, with transfer fitted separately from timings.fetch_s "
                      "+ timings.upload_s",
    # **Named, not merely acknowledged.** §9b's first requirement is that an estimate say where it
    # came from, and a held constant whose source the answer cannot name is the half of this model
    # that could not be recovered from a record found later.
    "held_from_previous_corpus": "the TIME_PER_IN : TIME_PER_SYNTH ratio, 2.03:1, from the seven "
                                 "delivered runs on sha-f7cbf7d (2026-08-19..25) whose 24->12 "
                                 "decimation run is the only reading this project has ever taken "
                                 "that separated a copied frame from a synthesised one",
    # **The band is the LARGER of these two, and today the observed spread wins** — 42% at 4K
    # across three runs on one worker, one of which lost its cores mid-encode for 100 s, against
    # a 21.6% worst residual. §9b's second requirement is that the band never be tighter than the
    # noise the model was fitted through, and `max` is what enforces that whichever is larger.
    #
    # **`max_residual_frac` is a FLOOR here rather than a measure of shape**, and the distinction
    # matters to whoever refits next: with two distinct areas and two free parameters the fit
    # passes through both cluster means by construction, so the residual is the scatter of
    # individual runs about a curve that cannot miss — arithmetic, not evidence that the shape is
    # right. It becomes evidence the day the corpus holds more distinct areas than the fit holds
    # free parameters.
    "spread_frac": 0.42,
    "max_residual_frac": 0.216,
    "caveat": "n_in and n_synth are collinear in this corpus and the exponent rests on two "
              "cluster means; a third resolution is the cheapest thing that would falsify it",
}

#: What `eta_basis` carries into the progress payload and the run record. **Bumped with the
#: refit**, so a record cannot say `estimator_v1` and mean either set of constants — a corpus in
#: which one label covers two models is one nobody can sort.
BASIS = "estimator_v2"


class Unpriceable(Exception):
    """Raised where an estimate would have to be invented. Carries what is missing."""


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE FIT PREDICATE — §9a, certified, and the half that answers before a job is submitted
# ─────────────────────────────────────────────────────────────────────────────────────────────


def peak_vram_gb(width, height, scale=1):
    """What one padded frame pair will hold at the peak, in GiB.

    **Client-side by construction**: every input is knowable before a worker is touched, which
    is the entire point of having a fit predicate at all.
    """
    return (FIT_VRAM_SLOPE_GB_PER_MP * interp_plan.padded_megapixels(width, height, scale)
            + FIT_VRAM_INTERCEPT_GB)


def usable_vram_gb(hardware_snapshot, reserve_gb=VRAM_RESERVE_GB):
    """What the predicate may price against: **`vram_total_gb` less a reserve.**

    **NEVER `vram_free_gb`** (`F-2026-08-25-6`). A warm worker reports a fraction of the memory
    it can actually use, because torch's caching allocator holds its pool across jobs — 13.31 GiB
    reported free on a container holding roughly 31 GiB of reusable pool. An estimator pricing
    against free memory refuses work it could do, and refuses it more often the busier the
    endpoint is, which is the worst possible direction for that error to run.

    **There is no second function of this name to confuse this one with, and there was.**
    `hardware.usable_vram_gb` priced off the free figure — the same name in the same package,
    answering the opposite way, with nothing at a call site to distinguish them. It had no callers
    and was deleted under claim C-1 rather than left as a trap wearing the right name.
    """
    total = (hardware_snapshot or {}).get("vram_total_gb")
    if total is None:
        return None
    return max(0.0, float(total) - reserve_gb)


def fits(width, height, hardware_snapshot, scale=1, reserve_gb=VRAM_RESERVE_GB):
    """`(ok, detail)` — whether one padded pair fits the card, and the arithmetic either way.

    **A single comparison rather than a search**, for `interp_plan.fits`'s reason: RIFE holds one
    frame pair whatever the clip length, so there is no rung to step down to. Either it fits or
    the job does not run.

    **Unknown is not treated as plentiful and is not treated as a refusal either.** A snapshot
    with no `vram_total_gb` returns `(None, detail)` — three states, because "could not price"
    and "priced and refused" are different answers and a caller that collapses them either
    refuses work it could do or spends a job to find out.
    """
    needed = peak_vram_gb(width, height, scale)
    usable = usable_vram_gb(hardware_snapshot, reserve_gb)
    detail = {
        "needed_gb": round(needed, 2),
        "usable_gb": None if usable is None else round(usable, 2),
        "vram_total_gb": (hardware_snapshot or {}).get("vram_total_gb"),
        "reserve_gb": reserve_gb,
        "padded_megapixels": round(interp_plan.padded_megapixels(width, height, scale), 5),
        "basis": "contract 9a: {} x padded_megapixels + {}, priced against vram_total_gb less "
                 "a {} GiB reserve".format(FIT_VRAM_SLOPE_GB_PER_MP, FIT_VRAM_INTERCEPT_GB,
                                           reserve_gb),
    }
    if usable is None:
        return None, detail
    detail["shortfall_gb"] = round(max(0.0, needed - usable), 2)
    return needed <= usable, detail


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE TIME ESTIMATE — §9b, provisional by construction, and it says so in every answer
# ─────────────────────────────────────────────────────────────────────────────────────────────


def estimate_seconds(width, height, n_in, n_synth, scale=1):
    """How long, as **a point and a spread** — never a bare point (§9b, second requirement).

    Returns `{"point_s", "compute_s", "transfer_s", "low_s", "high_s", "band_frac", "basis",
    "corpus"}` — **the full key list, because the run record files this dict whole** and a caller
    sizing a schema from a partial one writes a partial schema. `compute_s` and `transfer_s` are
    published apart as well as summed, so a reader can grade each against the field
    `docs/instrumentation.md` §8a created for it.

    The band is the larger of the corpus's measured repeatability and this fit's worst residual,
    so it can never claim to be tighter than the noise it was fitted through.

    **Refuses rather than guessing where the plan is not in hand**: a copied frame and a
    synthesised one cost differently, so an estimate priced off output frames alone misprices a
    decimation. **That split is HELD from the previous corpus and this one cannot see it** — all
    six of its runs share one frame plan, so `n_in` and `n_synth` are collinear here. The
    refusal is right and its old justification was not: the corpus this module ships against
    cannot measure that factor, it inherits it, and `CORPUS["held_from_previous_corpus"]` names
    where from.

    **Refuses on the geometry too, in this module's own vocabulary.** Non-positive dimensions and
    a non-positive scale raise `ValueError` out of `interp_plan` (`interp_plan.py:65`,
    `interpolate.pad_multiple`), which is correct and is not a word any caller of THIS module is
    watching for — `_seed_estimate` catches broadly, `handler._retime` does not. They are
    unreachable today because `decode.open_source` refuses a source with no dimensions before
    anything here is called, and the padding floor of 128 px means a priced job can never be
    smaller than 0.016 MP. **The guard exists because the fixed term is gone**: `point_s` used to
    be bounded below by 16.6 s whatever arrived, and both terms are now proportional to area, so
    a degenerate geometry that once produced a harmless floor would now produce a fully-formed
    estimate of zero seconds — an answer, not a refusal.
    """
    if not n_in or n_in <= 0 or n_synth is None or n_synth < 0:
        raise Unpriceable(
            "a time estimate needs the source frame count and the synthesis count and got "
            "n_in={!r} n_synth={!r}. Neither is guessed: a copied frame and a synthesised one "
            "cost differently and the corpus can tell them apart.".format(n_in, n_synth))
    try:
        megapixels = interp_plan.padded_megapixels(width, height, scale)
    except ValueError as exc:
        raise Unpriceable("the geometry cannot be priced: {}".format(exc))
    if megapixels <= 0:
        raise Unpriceable(
            "padded area came back as {!r} for {}x{} at scale {}, and both terms of this model "
            "are proportional to it — a zero would publish an estimate of zero seconds rather "
            "than refuse".format(megapixels, width, height, scale))
    compute = (TIME_PER_IN * n_in + TIME_PER_SYNTH * n_synth) * megapixels ** TIME_EXPONENT
    transfer = TIME_TRANSFER_PER_MP * megapixels
    point = compute + transfer
    band = max(CORPUS["spread_frac"], CORPUS["max_residual_frac"])
    return {
        "point_s": round(point, 1),
        # **Published apart as well as together**, because they are now separately fitted and a
        # reader grading the estimate against a record can compare each against the field §8a
        # created for it. The old model could not offer this: it had one term and the record had
        # one number.
        "compute_s": round(compute, 1),
        "transfer_s": round(transfer, 1),
        "low_s": round(point * (1.0 - band), 1),
        "high_s": round(point * (1.0 + band), 1),
        "band_frac": band,
        "basis": BASIS,
        # **The whole account, carried rather than cited.** A record that outlives this session
        # has to be able to say which corpus its estimate came from without the reader going and
        # finding out which constants were in the image that day.
        "corpus": dict(CORPUS),
    }


def seconds_per_frame(width, height, n_in, n_out, n_synth, scale=1):
    """The estimate expressed as `Progress.expect` wants it — **seconds per OUTPUT frame.**

    `Progress` prices the whole clip as `n_out x this` and decays it against the work fraction,
    so dividing the total by `n_out` here reproduces the total there exactly. The conversion is
    in this module rather than in `progress` because it is arithmetic about the estimate, and
    `progress` is deliberately ignorant of where its seed came from beyond the basis label.

    **THE TOTAL IS EXACT AND THE SHAPE IS NOT, AND THE ERROR NOW GROWS WITH AREA.** `transfer_s`
    is spent entirely before the first frame and after the last, so amortising it across frames
    describes work no frame performs. The old model had the same flaw in its fixed term, where it
    was a constant 16.6 s; transfer is 17 s at 4K and 70 s at 8K. On this corpus's own plan that
    is ~6% of the point and invisible, but on a SHORT clip at high resolution it dominates —
    three output frames at 8K is ~8 s of compute against ~70 s of transfer, so nearly all of the
    published per-frame rate would describe the transfer.
    **`eta.first_s` is unaffected, which is why no check can see this**: the graded number is the
    total, and the total is right. What is wrong is the mid-run decay, until the first written
    frame replaces the seed with a measured rate. Filed to the gate rather than fixed here — the
    fix is `Progress` learning that an estimate has a non-per-frame component, which is that
    module's model and not this one's.

    Returns `(seconds_per_frame, estimate)`; the estimate is the full §9b answer, for the record.
    """
    if not n_out or n_out <= 0:
        raise Unpriceable("seconds per frame needs a positive output count, got {!r}".format(
            n_out))
    estimate = estimate_seconds(width, height, n_in, n_synth, scale=scale)
    return estimate["point_s"] / float(n_out), estimate
