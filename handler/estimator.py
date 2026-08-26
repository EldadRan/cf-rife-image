"""What a job will cost before it is submitted — memory certainly, time provisionally.

**Contract §9.** Two questions that are not the same question. *Will it fit?* is answerable
before anything runs and its answer is certified from three resolutions across a 15x span.
*How long?* is answerable only against a corpus, and this project's corpus is not yet clean.
CF ruled both are built now, and §9b's three requirements are what make that ruling safe:

  the estimate carries its basis; the spread is published, not hidden; **and the coefficients
  live in one named place and are refittable without touching logic.**

The third is the one this module is shaped around. Every number is a module constant with a
comment saying where it came from — `FIT_*`, `TIME_*`, `CORPUS` — and nothing below the
constants block holds a literal. **The refit is already ordered** for the moment
`docs/instrumentation.md` §8f's close condition is met and `compute_s` is separable from
transfer: when it lands it is a change of these constants and not a change of code.

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
FIT_VRAM_SLOPE_GB_PER_MP = 0.805
FIT_VRAM_INTERCEPT_GB = 0.016

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
    "held_from_previous_corpus": "the TIME_PER_IN : TIME_PER_SYNTH ratio, 2.03:1",
    # **The band is the OBSERVED spread and not the fit's residual**, because the residual is
    # exact-by-construction and the spread is real: 42% at 4K across three runs on one worker,
    # one of which lost its cores mid-encode for 100 s. A model reporting a tighter band than the
    # noise it was fitted through misrepresents its own quality, which §9b's second requirement
    # forbids by name.
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

    Returns `{"point_s", "low_s", "high_s", "basis", "corpus"}`. The band is the larger of the
    corpus's measured repeatability and this fit's worst residual, so it can never claim to be
    tighter than the noise it was fitted through.

    Refuses rather than guessing where the plan is not in hand: `n_in` and `n_synth` are the two
    quantities the corpus is keyed on, and an estimate priced off output frames alone misprices a
    decimation by a factor this corpus can measure.
    """
    if not n_in or n_in <= 0 or n_synth is None or n_synth < 0:
        raise Unpriceable(
            "a time estimate needs the source frame count and the synthesis count and got "
            "n_in={!r} n_synth={!r}. Neither is guessed: a copied frame and a synthesised one "
            "cost differently and the corpus can tell them apart.".format(n_in, n_synth))
    megapixels = interp_plan.padded_megapixels(width, height, scale)
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

    Returns `(seconds_per_frame, estimate)`; the estimate is the full §9b answer, for the record.
    """
    if not n_out or n_out <= 0:
        raise Unpriceable("seconds per frame needs a positive output count, got {!r}".format(
            n_out))
    estimate = estimate_seconds(width, height, n_in, n_synth, scale=scale)
    return estimate["point_s"] / float(n_out), estimate
