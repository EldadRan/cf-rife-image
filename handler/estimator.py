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
#: **REFITTED 2026-08-28 — contract §9e, and the trigger was `docs/instrumentation.md` §8g.**
#: The factor-of-two ETA gate had been failing at 2.3-2.7x, which is a LEVEL error and a refit
#: trigger rather than a reason to widen the gate. `estimator_v2` was fitted on `sha-9672d42`,
#: before both conversion waves and before the producer wave; on the three runs this refit is
#: fitted to it reads 1.37x, 1.64x and 1.73x high **against `wall_s`, which is what §8g grades
#: — point against outturn, transfer included, and not against `compute_s`** — and on the arm §6d
#: now makes the default it reached 2.64x.
#:
#: **THE SHAPE DID NOT CHANGE AND ONLY THE LEVEL DID**, which is what §9b's third requirement was
#: written for and is the first time this project has been able to say it: `TIME_EXPONENT`
#: 1.1053 -> 0.9868, `TIME_PER_IN` 0.05525 -> 0.04730, `TIME_PER_SYNTH` 0.02721 -> 0.02330, and
#: not one line below the constants block moved for it.
#:
#: **THE POPULATION IS THREE RUNS AND THE FILTER THAT PRODUCED IT IS PART OF THE FIT.** Every row
#: satisfying all of: status `ok`; `timings.compute_s` present; **all four request gates false**;
#: the arm read FROM THE RECORD as `crf=12 threads=4 sliced_threads=true rc_lookahead=10`; and
#: image in `{705fba4, 36fa477}`.
#:
#:      705fba4  3840x2160   8.35584 MP   n_in 192  n_synth 382   compute_s 146.1
#:      705fba4  7680x4320  33.42336 MP   n_in 192  n_synth 382   compute_s 571.2
#:      36fa477  7680x4320  33.42336 MP   n_in 192  n_synth 382   compute_s 576.4
#:
#: **THE TWO IMAGES ARE ONE POPULATION AND THAT IS MEASURED RATHER THAN ASSUMED.**
#: `docs/test-plan.md` §24 puts the producer wave's before and after on this arm at 8K at 571.2
#: and 576.4 — +0.9%, while 84 s moved BETWEEN stages. **The wave did not move the quantity being
#: fitted.** Nothing certifies that across the earlier image boundaries, where the same arm at 4K
#: reads 655.7, 265.1, 155.6 and 146.1, so `sha-9672d42` and everything before it stays out — and
#: the second reason is stronger than the provenance one: those rows describe a worker before
#: both conversion waves.
#:
#: **AND TWO 1080p ROWS WERE EXCLUDED FOR A REASON THAT IS NOT NOISE.** Both carry request gates —
#: `convert_check` + `input_check` + `tie_check` on one, `convert_check` + `input_check` on the
#: other — and `docs/instrumentation.md` §12 rules that `compute_s` and the shares never survive
#: an armed run, with the pollution not recoverable by subtraction. Their 43.8 against 26.1 on
#: identical work is three instruments against two, and it is evidence FOR §12 rather than a
#: spread this model may carry.
TIME_PER_IN = 0.04730
TIME_PER_SYNTH = 0.02330
TIME_EXPONENT = 0.9868

#: **Transfer is linear in padded area because the BYTES are.** 73 MB in and 193 MB out at 4K
#: against 580 and 743 at 8K, over one link: 2.15 s/MP implied by the 4K runs and 2.02 by the 8K
#: ones, which is as close to one constant as a six-reading corpus can show. **This is a property
#: of this link and these sources, not of the worker** — a caller on a slower connection has a
#: different constant and nothing here would know.
#:
#: **NOT REFITTED IN 2026-08-28's REFIT, DELIBERATELY, AND THE NUMBERS ARE WHY.** The three rows
#: above imply 3.183, 2.672 and 1.526 s/MP — a 2.1x span over three readings — against a constant
#: fitted from six. **A three-reading refit of this term would be a worse number wearing a newer
#: date.** And the reason it CAN be left alone while the compute half could not is the sentence
#: above it: transfer is a property of the link, and the refit is scoped to an encoder arm, which
#: is a property of the worker. §9e scopes the refit to what it can honestly do, and this term is
#: outside that scope.
TIME_TRANSFER_PER_MP = 2.0849

#: **§9b's first requirement, as data rather than as prose.** Every time answer carries this, so
#: a number that cannot say where it came from cannot leave this module.
CORPUS = {
    "name": "cf-rife-project records/, three unarmed single-arm runs on sha-705fba4 and "
            "sha-36fa477, 2026-08-27..28",
    "readings": 3,
    "records": ("rife-f8754e26265b-8292e77f", "rife-eea67de71d4a-786f9015",
                "rife-1580676755e0-ceee7a33"),
    # **TWO distinct padded areas and ONE frame plan across all three**, and this is the fit's
    # binding limitation rather than a footnote. Every run was 192 source frames producing 382
    # syntheses, so `n_in` and `n_synth` are perfectly collinear here and this corpus cannot
    # separate them; the ratio between them is HELD from the previous corpus, whose 24->12
    # decimation run is the only reading this project has ever taken that told them apart.
    #
    # **And with two distinct areas, two free parameters fit them exactly.** `TIME_EXPONENT` and
    # the overall level are determined by the two cluster means, so the fit passes through both by
    # construction — **its residual at those points is arithmetic, not evidence.** The refit of
    # 2026-08-28 did NOT change this: it moved the level, which is what §8g's failure was, and it
    # could not improve the shape, which is what this caveat has always said.
    "resolutions": ("3840x2160", "7680x4320"),
    "distinct_padded_areas": 2,
    "frame_plans": 1,
    "gpu": "NVIDIA A40",
    "images": ("sha-705fba4", "sha-36fa477"),
    # **§9e: `crf` IS A DECLARED INPUT AND NOT A FITTED ONE, AND THE CORPUS IS WHY.** Every row in
    # `records/` is CRF 12. A coefficient cannot be estimated for an axis with one value, and a
    # refit that produced one would have invented it. So the axis is NAMED and its range is
    # published, and an estimate outside it says so rather than extrapolating — which is the whole
    # of what `docs/test-plan.md` §23c was protecting: not accuracy at other CRFs, which nobody
    # has measured, but the ESTIMATE'S ABILITY TO SAY IT IS OUT OF ITS DEPTH.
    "crf_range": (12, 12),
    # **THE ENCODER ARM IS DECLARED THE SAME WAY AND FOR A SHARPER REASON**, and this entry is
    # uncomfortable to read on purpose. The arm moves `compute_s` by up to 1.72x on identical work
    # (`docs/test-plan.md` §25), so the corpus is not one population unless the arm is fixed —
    # **and the arm it is fixed at is NOT the arm the worker now defaults to.** §6d makes an unset
    # request resolve to `threads=16`, at which the settings this fit was taken on hold three rows
    # in the whole corpus. A refit across all arms would average a 1.7x spread and fit nothing; a
    # refit restricted to the new default would have three points. **Neither is a fit**, so the
    # refit is scoped to the population that exists and the gap is stated rather than papered
    # over.
    #
    # **Declared as the three x264 FIELD NAMES and not as §6d's row name**, because §6d ruled the
    # table bare — there is no codec key — and because the small row is `threads=16` while this
    # corpus is `threads=4`. Naming the row would say the opposite of the truth.
    #
    # **The calibration against the new default is re-read once the wave has banked runs on it**,
    # which certification does as a side effect rather than as extra work.
    "encoder_arm": {"crf": 12, "threads": 4, "sliced_threads": True, "rc_lookahead": 10,
                    "preset": "medium"},
    # **§12: EVERY ROW IS UNARMED, AND THIS IS A FILTER AND NOT A FOOTNOTE.** An armed run does
    # not yield a usable `compute_s` and the pollution is not recoverable by subtraction, so a
    # corpus that did not exclude them would be fitting instrument cost.
    "request_gates": "all four false on every row",
    # **Named so `outside_corpus` reads the list rather than restating it.** `docs/instrumentation`
    # §12 owns these four; a second hand-written copy here would be the fact in two places, and
    # the copy that rots is the one nobody re-reads.
    "request_gate_fields": ("convert_check", "input_check", "tie_check", "decode_probe"),
    # **The compute half is fitted against `compute_s` and excludes transfer**, which the corpus
    # before `estimator_v2`'s could not do. **Transfer is NOT refitted here** — see
    # `TIME_TRANSFER_PER_MP`, whose constant is carried from the six-reading corpus because these
    # three rows imply 3.183, 2.672 and 1.526 s/MP and cannot improve on it.
    "fitted_against": "timings.compute_s only; the transfer term is HELD from the six-reading "
                      "sha-9672d42 corpus and was not refitted",
    # **Named, not merely acknowledged.** §9b's first requirement is that an estimate say where it
    # came from, and a held constant whose source the answer cannot name is the half of this model
    # that could not be recovered from a record found later.
    "held_from_previous_corpus": "the TIME_PER_IN : TIME_PER_SYNTH ratio, 2.03:1, from the seven "
                                 "delivered runs on sha-f7cbf7d (2026-08-19..25) whose 24->12 "
                                 "decimation run is the only reading this project has ever taken "
                                 "that separated a copied frame from a synthesised one; AND "
                                 "`spread_frac`, from the six-reading sha-9672d42 corpus, "
                                 "because this population has no repeat pair that could measure "
                                 "one",
    # **THE BAND IS HELD AND IT IS NOT MEASURED HERE, WHICH IS THE HONEST SHAPE OF THIS CORPUS.**
    # The three rows hold exactly one repeat pair — 571.2 against 576.4, 0.9% — and publishing
    # that as the band would claim a repeatability an order of magnitude tighter than
    # `F-2026-08-25-10`'s measured 18.8% on identical work on one worker. §9b's second requirement
    # is that the band never be tighter than the noise the model was fitted through, and that
    # floor is not decorative. **So `spread_frac` is carried from the previous corpus as a HELD
    # constant, named as such above.**
    #
    # **`max_residual_frac` is a FLOOR here rather than a measure of shape**, and the distinction
    # matters to whoever refits next: with two distinct areas and two free parameters the fit
    # passes through both cluster means by construction, so 0.005 is the scatter of the one repeat
    # pair about a curve that cannot miss — arithmetic, not evidence that the shape is right. It
    # becomes evidence the day the corpus holds more distinct areas than the fit holds free
    # parameters, and **one unarmed 1080p run at the new default is what would buy that.**
    "spread_frac": 0.42,
    "max_residual_frac": 0.005,
    "caveat": "n_in and n_synth are collinear in this corpus and the exponent rests on two "
              "cluster means; a third resolution is the cheapest thing that would falsify it",
}

#: What `eta_basis` carries into the progress payload and the run record. **Bumped with the
#: refit**, so a record cannot say `estimator_v2` and mean either set of constants — a corpus in
#: which one label covers two models is one nobody can sort.
BASIS = "estimator_v3"


def basis_for(crf=None):
    """The basis label with the declared CRF in it. **Contract §9e, first bullet, literally.**

    §9e: *"`crf` is a declared input of the estimator and appears in the BASIS STRING, so every
    estimate says which CRF it was computed for."* Putting `crf` on the answer's own key covers
    the record's `estimate` block and **does not cover the channel §8g actually grades** —
    `eta.first_basis` — nor the `eta_basis` in every published progress payload, both of which
    carry this string and nothing else. A caller watching progress would have seen
    `predicted_estimator_v3` and had no way to ask which CRF it was computed for.

    **The model label stays sortable.** `BASIS` alone still names the constants, so a corpus can
    still be split by model; the CRF is appended rather than substituted, so one label never
    covers two models and never hides which point on the declared axis it was quoted at.
    """
    return BASIS if crf is None else "{}/crf{}".format(BASIS, int(crf))


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


def outside_corpus(crf=None, encoder_arm=None, armed=None, substituted=None):
    """**§9e: which declared axes this job has left, each stated with the corpus's own range.**

    Returns a list of sentences, empty when the job sits inside every declared axis. **Never a
    refusal and never a silence.** §9e rules that an estimate outside the corpus *"is not silently
    extrapolated — it is returned with the corpus's range stated, so a CRF walk can tell it has
    left the corpus"*, and the same sentence governs the encoder arm, which is declared here for
    the sharper reason that it moves `compute_s` by up to 1.72x on identical work.

    **AN EMPTY LIST MEANS "NOTHING DECLARED WAS LEFT" AND NOT "EVERYTHING WAS CHECKED."** An axis
    the caller did not supply is not checked, so a caller that declares neither gets `[]` — the
    same value as a caller that declared both and sat inside them. That is a real ambiguity and it
    is bounded rather than papered over: `_seed_estimate` always supplies both, so the unchecked
    case is reachable only by a direct caller in a test. **The answer's own `crf` and
    `encoder_arm` keys are what distinguish the two**, and they are on the estimate for exactly
    this reason — a reader holding `outside_corpus: []` beside `crf: null` can see which it is.

    **THIS WILL FIRE ON ALMOST EVERY JOB THE DAY §6d SHIPS, AND THAT IS CORRECT.** An unset
    request now resolves to `threads=16` and this corpus is `threads=4`, so nearly every delivered
    run will carry a sentence saying the estimate was fitted at an arm the worker no longer
    defaults to. **That is true, it is the reason the arm is declared rather than baked, and a
    reader who cannot see it would fit a fourth thing on top of it.** It stops firing when the
    wave has banked runs on the new default and the corpus is re-read against them.
    """
    left = []
    try:
        # **THE ARMED INSTRUMENTS, AND THIS SENTENCE IS WHAT MAKES §8g-1 HONEST RATHER THAN
        # CONVENIENT.** `docs/instrumentation.md` §8g-1 lets the kit DECLINE TO GRADE an armed
        # run's first ETA, because the instruments land in `wall_s` and §12 rules that pollution
        # unrecoverable by subtraction. **A gate that looks away without the estimate saying
        # anything would leave the caller of an armed run with a silently wrong ETA** — a worse
        # arrangement than the failing row it replaces. So the decline follows from something the
        # answer says out loud, and it is said HERE rather than in the kit because the kit reads
        # records and the caller reads the estimate.
        #
        # **The four are request fields, known before the job starts**, so this is decidable at
        # the moment the estimate is made and needs nothing measured.
        # **A SETTING NOBODY DECLARED IS NOT A SETTING INSIDE THE CORPUS, EVEN WHEN THE VALUE
        # MATCHES.** `routec` resolves `crf` and `preset` to module constants when the caller
        # sends neither — and `crf`'s constant is 12, which is exactly this corpus's declared
        # range. Without this the estimate would report no departure, stamp `estimator_v3/crf12`
        # onto the basis §8g grades, and claim to say which CRF the job was computed for when
        # nobody declared one. **The value being right is what makes it invisible**, which is why
        # the provenance is reported rather than the value.
        if substituted:
            left.append(
                "{} was not declared by the caller and was filled in from the worker's own "
                "constant, so the basis names a value nobody chose; the estimate may match this "
                "corpus without the run having declared it".format(
                    ", ".join(sorted(substituted))))
        if armed:
            left.append(
                "this run arms {} and the estimate does not model {} cost; the instruments land "
                "in wall_s and instrumentation 12 rules that pollution not recoverable by "
                "subtraction, so the outturn is not comparable to this number".format(
                    ", ".join(sorted(armed)), "their" if len(armed) > 1 else "its"))
        low, high = CORPUS["crf_range"]
        if crf is not None and not low <= int(crf) <= high:
            left.append(
                "crf {} is outside the range this corpus was fitted over ({}-{}); the estimate "
                "is not extrapolated to it".format(int(crf), low, high))
        if encoder_arm:
            fitted = CORPUS["encoder_arm"]
            # **`crf` IS SKIPPED HERE BECAUSE IT HAS ITS OWN DECLARED AXIS ABOVE.** It appears in
            # `CORPUS["encoder_arm"]` too, so that a record carrying the arm carries the whole
            # arm — but comparing it in both places emitted two sentences about one fact, which
            # is the duplication the shared law's ONE FACT, ONE HOME clause is about, in
            # miniature: a reader would count two departures where the job made one.
            differs = {name: value for name, value in encoder_arm.items()
                       if value is not None and name != "crf" and name in fitted
                       and fitted[name] != value}
            if differs:
                left.append(
                    "the encoder arm differs from the one this corpus was fitted at ({}): {}. "
                    "The arm moves compute_s by up to 1.72x on identical work, so this estimate "
                    "is outside its population and not merely at its edge".format(
                        ", ".join("{}={}".format(k, fitted[k]) for k in sorted(differs)),
                        ", ".join("{}={}".format(k, differs[k]) for k in sorted(differs))))
    except Exception as exc:  # noqa: BLE001 — saying less must never cost a delivered master
        # **THE FAILURE SAYS SO RATHER THAN RETURNING SILENCE, AND THAT IS THE WHOLE POINT OF
        # THIS FUNCTION.** It used to `pass`, so a check that crashed returned `[]` — the same
        # value as a check that ran and found the job inside every declared axis. **A function
        # whose entire purpose is the estimate's ability to say it is out of its depth cannot
        # have silence as its failure mode**, and the partial case was worse than the total one:
        # the `crf` sentence appends, the arm block raises, and the answer carries one sentence
        # while never having checked the axis that moves `compute_s` by 1.72x.
        #
        # **Appended rather than raising**, because an estimate that could not check its corpus
        # is still a better ETA than none — and appended LAST so whatever was checked before the
        # failure is kept rather than discarded.
        left.append(
            "the corpus check itself failed ({}: {}), so this estimate has NOT been compared "
            "against the range and arm it was fitted over — treat it as unverified rather than "
            "as inside".format(type(exc).__name__, str(exc)[:120]))
    return left


def estimate_seconds(width, height, n_in, n_synth, scale=1, crf=None, encoder_arm=None,
                     armed=None, substituted=None):
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
        "basis": basis_for(crf),
        # **§9e's declared inputs, ON THE ANSWER and not only in the corpus block.** The corpus
        # says which range was fitted; these say which point was asked for. A record carrying only
        # the first cannot tell a reader whether the estimate it is sitting beside was inside it.
        "crf": crf,
        "encoder_arm": dict(encoder_arm) if encoder_arm else None,
        # **The armed list ON THE ANSWER too**, for `encoder_arm`'s reason: `outside_corpus` says
        # a departure happened and this says which instruments caused it, without a reader parsing
        # a sentence back apart.
        "armed": sorted(armed) if armed else [],
        # **Empty list means inside every declared axis; it is never None on a checkable answer.**
        # An absent key and an empty one would be the same bytes to a reader and different facts.
        "outside_corpus": outside_corpus(crf=crf, encoder_arm=encoder_arm, armed=armed,
                                         substituted=substituted),
        # **The whole account, carried rather than cited.** A record that outlives this session
        # has to be able to say which corpus its estimate came from without the reader going and
        # finding out which constants were in the image that day.
        "corpus": dict(CORPUS),
    }


def seconds_per_frame(width, height, n_in, n_out, n_synth, scale=1, crf=None,
                      encoder_arm=None, armed=None, substituted=None):
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
    estimate = estimate_seconds(width, height, n_in, n_synth, scale=scale, crf=crf,
                                encoder_arm=encoder_arm, armed=armed,
                                substituted=substituted)
    return estimate["point_s"] / float(n_out), estimate
