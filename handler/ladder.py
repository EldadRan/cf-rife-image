"""CF's ruled table — `docs/decisions.md` §11 — and it is ONE table with two readers.

**THE SETTINGS AND THE SEED COME FROM THE SAME ROWS, ON A 16:9 FRAME.** *`docs/current.md` prints them as one table for a
reason: an area's `frame_threads / pools` and its `s/frame` were measured together, on the same
runs, and a job that takes one row's settings is priced by that row's seconds.* **Split across two
modules they would be two things to keep equal, and the copy that rotted would be
indistinguishable from the live one** — which is this project's central hazard stated in
`CLAUDE.md` about four repositories, arriving inside one.

So `encoder` reads the levers from here and the ETA seed comes from here, and neither owns the
numbers.

**TWO KEYINGS, DELIBERATELY, BECAUSE THE TABLE ANSWERS TWO QUESTIONS.**

    the price and the cap    DELIVERED PIXELS — they are questions about WORK, and a frame's
                             work is its area. `step_for` and everything that reads it.
    the threading levers     HEIGHT — WPP counts CTU rows and a CTU-64 frame has `height / 64`
                             of them, so pools is a question about rows. `levers` alone.

**ONE FUNCTION ANSWERED BOTH UNTIL 2026-09-03 AND KEYED ON HEIGHT**, which is a frame's pixel
count only when it is 16:9. *A 2160x3840 portrait 4K clip took 8K's row — six times the cap and
six times the price of the identical pixel count in landscape — and a 2560x1080 ultrawide took
1080p EXACTLY, under-priced for a third more work and over-capped, with no `:rounded` suffix to
make it findable.* **The second is the direction the cap exists to prevent.**

**`s/frame` IS `wall_s` OVER DELIVERED FRAMES, POOLED ACROSS THE BAND** — *not `calc_t`, not
`frame_s`* — **so it already contains fetch, decode and upload.** *A prediction must contain the
terms that overrun a cap, because those are exactly the ones nobody can isolate.*

**THE POOL IS THE BAND, NOT THE CELL**, so the low-volume figures carry the deliberately bad `ft=1`
and `ft=2` runs. *That is the conservative direction and it is deliberate.*
"""

#: HEVC's coding-tree-unit size, which is what makes the row count `height / 64`.
CTU_SIZE = 64

#: **THE THREE STEPS, IN DELIVERED PIXELS, AND THEY ARE THE TEST FILES' OWN PIXEL COUNTS.**
#: *`1920x1080`, `3840x2160`, `7680x4320` — CF, 2026-09-03, asked directly: the names are those
#: files and nothing else.*
#:
#: **THIS WAS KEYED ON HEIGHT UNTIL 2026-09-03 AND A HEIGHT IS ONLY THESE COUNTS ON A 16:9
#: FRAME.** *`docs/current.md` has said "keyed on DELIVERED pixels" since the ladder was written,
#: and gave the portrait case as its worked example; the code and the document disagreed from the
#: first commit and nothing caught it, because while only an ETA rode on it the error was
#: conservative.* **The cap made it functional**: it decides whether a job is refused.
#:
#:     2160x3840 portrait 4K   8,294,400 px   by height it took 8K's row — priced 0.96 against
#:                                            0.157 and capped 3,000 against 20,000, which
#:                                            limits a vertical 4K clip to 50 seconds at 60 fps
#:     2560x1080 ultrawide     2,764,800 px   by height it took `1080p` EXACTLY, with no
#:                                            `:rounded` — under-priced at 0.115 for a third
#:                                            more work, over-capped at 30,000, and not
#:                                            findable afterwards as anything unusual
#:
#: **THE SECOND IS THE DIRECTION THE CAP EXISTS TO PREVENT**, and it was the silent one.
STEPS = {2_073_600: "1080p", 8_294_400: "4K", 33_177_600: "8K"}

#: **`levers` READS THIS AND NOT `STEPS`, AND THAT SEPARATION IS THE WHOLE FIX.** *WPP counts CTU
#: ROWS, so the thread settings are a question about HEIGHT — `height / 64` — while the price and
#: the cap are questions about WORK, which is pixels.* **One function was answering both.**
LEVER_HEIGHTS = {1080: "1080p", 2160: "4K", 4320: "8K"}

#: **The largest step, in pixels, and what an above-8K frame is priced from.** *There is no row
#: ABOVE it to round up to, so `step_for` answers `OUTSIDE` — and `seconds_per_frame` then borrows
#: this step's seconds rather than publishing nothing, which is CF's ruling of 2026-09-02.*
LARGEST = 33_177_600

#: **CF's rows, `docs/decisions.md` §11.** `switch` is the DELIVERED FRAME COUNT the row changes
#: at: below it takes `below`, at it and above takes `at`. **Both switches sit on the safe side of
#: a gap rather than at a measured boundary** — nothing ran between 480 and 1500 delivered frames
#: at either area — *so the more expensive setting starts earlier than the curves were seen to
#: cross.*
#:
#: **`levers` IS `None` FOR h264 AND THAT IS NOT AN OMISSION.** *x264 has no `pools` and no
#: `frame-threads`; it parallelises by slices and by its own `threads`, which §6d's area table
#: decides.* **A `None` here is the table saying this codec has no such setting**, which is a
#: different fact from a setting whose value is unknown.
#:
#: **1080p AND 4K h264 ARE CARRIED FORWARD RATHER THAN RE-MEASURED.** *1080p has no h264 run on
#: the deployed image at all; that 0.115 is the pooled h265 figure standing in, and CF ruled
#: 2026-09-02 that nothing between images moves it enough to justify a re-run.*
ROWS = {
    "1080p": {
        "h264": {"switch": None, "below": (None, 0.115), "at": (None, 0.115)},
        "h265": {"switch": None, "below": ((16, 16), 0.115), "at": ((16, 16), 0.115)},
    },
    "4K": {
        "h264": {"switch": None, "below": (None, 0.135), "at": (None, 0.135)},
        "h265": {"switch": 1400, "below": ((8, 32), 0.236), "at": ((16, 32), 0.157)},
    },
    "8K": {
        "h264": {"switch": None, "below": (None, 1.0), "at": (None, 1.0)},
        "h265": {"switch": 1200, "below": ((16, 64), 0.75), "at": ((16, 32), 0.96)},
    },
}

#: **The corpus transfer rates, in bytes per second, over 58 runs.** *`docs/decisions.md` §11.*
#:
#: **THE FETCH SPREAD IS WHY ITS SEED IS A PLACEHOLDER AND THE UPLOAD'S IS A NUMBER.** *Same
#: mechanism, honestly different confidence — upload 3.3x across the corpus, fetch 35.5x — and the
#: band is what says which is which.* **The caller sizing a timeout is the one who pays for a
#: false precision.**
UPLOAD_BYTES_PER_S = 13.1 * 1000 * 1000
FETCH_BYTES_PER_S = 54.5 * 1000 * 1000

#: **TWO BAND FRACTIONS STOOD HERE AND THEY ARE DELETED. THE COMMENT CALLED THEM DERIVED FROM
#: THE RANGES ABOVE AND THEY WERE NOT** — and nothing read them, so the file carried a false
#: claim about a number that did no work. *Found by the gate, which asked for the arithmetic and
#: could not reproduce either value.*
#:
#: **THE ARITHMETIC, IN TIME, WHICH IS WHAT A CALLER SIZES A TIMEOUT IN.** *The payload's band is
#: symmetric — `progress.py` publishes `point x (1 +/- f)`:*
#:
#:     upload   13.1 / 7.7   = 1.70x the point at the slow end
#:              13.1 / 25.5  = 0.51x at the fast end        -> f = 0.70 covers the slow side
#:                                                             exactly, overstates the fast by 21
#:     fetch    54.5 / 3.9   = 13.97x the point at the slow end
#:              54.5 / 138.2 = 0.39x at the fast end
#:
#: **A SYMMETRIC FRACTION CANNOT EXPRESS THE FETCH SPREAD AT ALL.** *It would need `f = 12.97`,
#: and `expect()` drops any band outside `[0, 1)` because at `f >= 1` the low edge is zero or
#: negative.* **So `0.95` was a number chosen to satisfy the validator, wearing a derivation's
#: clothes** — which is the shape §11's own clause warns about, since the caller sizing a timeout
#: is the one who pays for false precision.
#:
#: **AND THE GAP IS REAL RATHER THAN CLOSED BY THIS DELETION.** *§11 requires a spread beside
#: every point and the transfer phases publish `phase_expected_s` bare.* **The band the payload
#: has belongs to the FRAME ETA and is symmetric; a transfer spread of 0.39x to 13.97x is neither
#: symmetric nor inside `[0, 1)`, so it needs a shape nobody has ruled.** *Filed to the gate; not
#: built, because inventing the spelling here would be the second time this pair of numbers was
#: asserted rather than derived.*

#: **WHAT PRICED THE RUN, FOR THE BASIS STRING — and it is not the estimator.** *`eta_basis` read
#: `predicted_estimator_v3/crf12` on both of the first two delivered runs, for numbers this table
#: produced: 26 s at 1080p is 225 x 0.115 and 1440 s at 8K is 1500 x 0.96.* **`estimate.time.basis`
#: is a corpus key**, so every row said `estimator_v3` about a number `estimator_v3` did not
#: produce, and a query separating table-priced rows from fit-priced ones separated nothing.
#:
#: **VERSIONED THE WAY `estimator.BASIS` IS, AND FOR THE SAME REASON.** *A re-ruled table is a
#: different pricer; a row priced by today's numbers and a row priced by tomorrow's must not
#: share a label.*
#:
#: **AND BUMPING IT IS A CONVENTION WITH NOTHING BEHIND IT, WHICH IS WORTH SAYING PLAINLY.** *The
#: import guard below compares the tables' KEY SETS and cannot see a value change, so re-ruling
#: 4K's 0.157 without touching this line pools two pricers under one label — the condition
#: `estimator.BASIS`'s own comment calls a corpus nobody can sort.* **`estimator.BASIS` carries
#: the identical exposure**, so this is parity rather than a new hazard; it is named here because
#: the first draft's comment read as though the rule were enforced. Found in review.
#:
#: *It does not carry the step: `eta_ladder` does, and one fact has one home.*
BASIS = "ruled_table_v1"


def basis_for(crf):
    """`ruled_table_v1/crfNN`, or the bare string where no CRF was declared.

    **§9e REQUIRES THE DECLARED CRF ON THE ANSWER'S OWN KEY AND THE FIRST DRAFT DROPPED IT.** *The
    fit's `basis_for` puts it there and says why — `eta.first_basis` is the channel §8g grades,
    and it is not covered by the record's `estimate` block.* **These rows are CRF-12
    measurements**, so a CRF-20 job is priced off-axis; a basis that does not say so loses the
    declaration on nearly all traffic, since the table prices almost every job. *Found in review:
    the first version traded one mislabelling for the loss of a declared input on the same field.*
    """
    return BASIS if crf is None else "{}/crf{}".format(BASIS, crf)

#: What `step_for` says about a frame the table does not name.
#:
#: **`ROUNDED` IS A SUFFIX ON A STEP AND NOT A STEP OF ITS OWN, AND THE FIRST DRAFT HAD A THIRD
#: BARE VALUE THAT NO INPUT COULD PRODUCE.** *`step_for` rounds every unnamed size UP to the
#: step above it, so a `derived` return was unreachable — and a 1440p job then published
#: `eta_ladder: "4K"` while its `pools_basis` said `derived`: two fields on one run disagreeing
#: about whether CF's table names the frame, and the one a caller reads said it does.* **The seed
#: really did come from 4K's row, so the step is 4K — what was missing is that the frame is not
#: 4K.** *§11: a rule that is silent about being outside its population reads as being inside
#: it.* Found in review.
ROUNDED = ":rounded"
OUTSIDE = "outside"


#: **THE MAXIMUM A JOB MAY DELIVER, PER STEP — CF, 2026-09-02, `docs/decisions.md` §11.**
#:
#: **KEYED BY `step_for`, SO THE CAP AND THE SEED TAKE ONE BOUNDARY FROM ONE FUNCTION.** *A cap
#: keyed on its own sizes would be free to disagree with the ladder about where 4K ends, and the
#: disagreement would be invisible until a job landed between them.*
#:
#: **`LEVER_HEIGHTS` IS A SECOND TABLE AND THIS SENTENCE USED TO FORBID ONE.** *It is not the
#: thing that was forbidden: it answers a DIFFERENT question — CTU rows — on a different
#: quantity, and the three tables are checked against each other below rather than trusted to
#: stay aligned.* Found in review.
#:
#: **ABOVE 8K IS CAPPED HARDER THAN 8K AND THAT IS THE POINT OF THE ROW.** *A frame larger than 8K
#: is served on 8K's timings, which under-predict it by construction — nothing has been measured
#: up there — so the smaller cap is the margin that covers the difference.*
#:
#: **WHAT EACH CAP ACTUALLY BUYS, AGAINST THE 3,600 s KILL, FROM THE ROWS IN THIS FILE.** *The
#: first version of this comment called the above-8K row "the tightest of the four" and it is the
#: LOOSEST — the phrase was lifted from §11, where it compares three s/frame BASES for that one
#: row, and re-scoping it to the four rows inverts it.* **A constant defended by arithmetic that
#: does not reproduce is the offence this file deletes two other constants for**, seventy lines
#: up, so here is the arithmetic:
#:
#:     1080p   30,000 x 0.115 = 3,450 s    1.04x   <- the tightest by a wide margin
#:     4K      20,000 x 0.157 = 3,140 s    1.15x
#:     8K       3,000 x 0.96  = 2,880 s    1.25x
#:     >8K      1,800 x 1.0   = 1,800 s    2.00x   <- the loosest
#:
#: **SO THE BAND WITH FOUR PER CENT OF MARGIN IS 1080p, WHICH IS THE ONE NOBODY WORRIED ABOUT.**
#: *A 30,000-frame 1080p job is predicted to finish 150 seconds inside a wall that kills it with
#: nothing delivered, and the fleet's own host spread is 1.7x.* **Reported to the gate rather than
#: adjusted here: the caps are CF's numbers and this file does not get to pick a different one.**
MAX_DELIVERED_FRAMES = {
    "1080p": 30_000,
    "4K": 20_000,
    "8K": 3_000,
    OUTSIDE: 1_800,
}


def max_delivered_frames(delivered_pixels):
    """The cap for this frame. **A rounded step takes the step it rounds to.**

    *`step_for` rounds an unmeasured size UP — 1440p is priced at 4K — and the cap follows the
    same step for the same reason: the seed says this job costs 4K seconds, so the limit that
    keeps it inside the 3,600 s kill is 4K's.* **Two answers from one step, which is what putting
    them in one module buys.**
    """
    step = step_for(delivered_pixels)
    return MAX_DELIVERED_FRAMES[step[:-len(ROUNDED)] if step.endswith(ROUNDED) else step]


#: **THE THREE TABLES NAME ONE SET OF STEPS, AND UNTIL THIS LINE NOTHING SAID SO.** *`STEPS` keys
#: on pixels, `LEVER_HEIGHTS` on height and `ROWS`/`MAX_DELIVERED_FRAMES` on the label both
#: produce — so a step added to one and not the others raises `KeyError` deep inside a job: in
#: `max_delivered_frames` that is uncaught inside `handler._retime`, and a 1440p job would crash
#: with a bookkeeping error instead of being capped.* **Checked at import, where it costs nothing
#: and cannot be reached by a request.** Found in review.
_LABELS = set(STEPS.values())
if set(LEVER_HEIGHTS.values()) != _LABELS or set(ROWS) != _LABELS \
        or set(MAX_DELIVERED_FRAMES) != _LABELS | {OUTSIDE}:
    raise ImportError(
        "the ladder's tables name different steps: STEPS {}, LEVER_HEIGHTS {}, ROWS {}, "
        "MAX_DELIVERED_FRAMES {}. They are four keyings of one set of frames and a step present "
        "in one and absent from another is a KeyError inside a job.".format(
            sorted(_LABELS), sorted(set(LEVER_HEIGHTS.values())), sorted(ROWS),
            sorted(set(MAX_DELIVERED_FRAMES))))


def step_for(delivered_pixels):
    """`"1080p"`, `"4K"`, `"8K"`, any of those three plus `ROUNDED`, or `OUTSIDE`.

    **EACH NAMED RESOLUTION IS THE UPPER LIMIT OF ITS STEP AND AN UNMEASURED ONE ROUNDS UP** — CF,
    2026-09-02. *1440p seeds from 4K, not from 1080p.* **The direction is the whole point**: the
    error is then an OVER-estimate against a platform that kills at 3600 s, and the failure this
    prevents published 3663.9 s against an actual 2803.3. *A ladder that rounded down would
    reproduce it in the other direction.*

    **A ROUNDED STEP AND `OUTSIDE` ARE DIFFERENT STATES.** *A rounded step is between or below
    the steps and has a row ABOVE it to round up to, and says so with the `:rounded` suffix;
    `OUTSIDE` is above the tallest step, so there is nothing above it and it BORROWS the tallest
    step's seconds instead.* **The label stays `outside` either way**: a borrowed price is not a
    measured one, and the field exists to say which.
    """
    pixels = int(delivered_pixels)
    if pixels in STEPS:
        return STEPS[pixels]
    if pixels > LARGEST:
        return OUTSIDE
    for boundary in sorted(STEPS):
        if pixels < boundary:
            return STEPS[boundary] + ROUNDED
    # **Unreachable while `LARGEST` is the largest key of `STEPS`, and it is not an assertion of
    # that.** *A step added above it without moving `LARGEST` would land here, and returning
    # `OUTSIDE` is the honest answer for a size no row was found for.*
    return OUTSIDE


def levers(delivered_height, delivered_frames, codec="h265"):
    """The ruled `(frame_threads, pools)` for this frame, or `None` if the table does not name it.

    **`None` IS A REAL ANSWER AND NOT A FAILURE** — it is every resolution CF did not measure,
    which includes 1440p, every portrait clip and everything above 8K, and it is also every h264
    job at every size, because x264 has no such settings.

    **A NAMED HEIGHT WITH AN UNKNOWN FRAME COUNT DOES NOT TAKE THE TABLE.** *Two of the three
    h265 rows switch on delivered frames, so a count of `None` cannot choose between them* — and
    picking either silently would put the job on a row nobody selected, which is the failure the
    `derived` basis exists to make visible.

    **THE LEVERS TAKE THE EXACT STEP AND NEVER THE ROUNDED ONE.** *`step_for` rounds an unmeasured
    size UP for the seed, which is the conservative direction for a TIME estimate — but a setting
    is not an estimate: handing a 1440p frame 4K's 32 pools would be a claim about its CTU rows,
    and it has 23.*

    **AND IT READS `LEVER_HEIGHTS`, NOT `STEPS`.** *`STEPS` moved to pixels on 2026-09-03 because
    the price and the cap are questions about work; this one did not, because CTU rows are a
    question about height.* **The two tables name the same three frames and are keyed on
    different quantities, which is why they are two tables.*
    """
    name = LEVER_HEIGHTS.get(int(delivered_height))
    if name is None:
        return None
    row = ROWS[name].get(codec or "h265")
    if row is None:
        return None
    if row["switch"] is None:
        return row["at"][0]
    if delivered_frames is None:
        return None
    return (row["at"] if int(delivered_frames) >= row["switch"] else row["below"])[0]


def seconds_per_frame(delivered_pixels, delivered_frames, codec="h265"):
    """`(s_per_frame, step)` for the ETA seed. **Above the ladder it seeds from 8K and says so.**

    **ABOVE 8K IS PRICED AS 8K — CF, 2026-09-02, ORDERED 2026-09-03.** *Priced, not executed: the
    SEED comes from 8K's row and the SETTINGS come from the CTU-row rule, because `levers`
    answers `None` for any height the table does not name.*

    **THE HEADER'S CLAIM — that a job taking one row's settings is priced by that row's seconds —
    HOLDS FOR A 16:9 FRAME AND FOR NOTHING ELSE, AND THAT IS A CONSEQUENCE OF THE SPLIT.** *The
    price keys on pixels and the settings key on height, so the two agree exactly when the frame's
    height implies its area.* **On any other aspect they name different rows by construction:**

        2560x1080 ultrawide     priced 0.157 from 4K's row      encoded 16/16, 1080p's settings
        2160x3840 portrait 4K   priced 0.157 from 4K's row      encoded 16/60, no row's settings

    *Neither is a wrong number — the price is the frame's work and the settings are its CTU rows,
    and both are right about what they measure.* **But a reader taking the header at its word
    would attribute an ultrawide job's outturn to a row it was not encoded on**, which is why the
    exception is written here rather than left to be discovered. Found in review.

    **AND THE ORIGINAL EXCEPTION SURVIVES INSIDE ABOVE-8K, WHERE IT IS NARROWER THAN IT LOOKS:**

        below 1,200 frames    seeded 0.75, measured at 16/64    encoded at 16/64    they AGREE
        1,200 and above       seeded 0.96, measured at 16/32    encoded at 16/64    they part

    **AND THEY PART ACROSS A PAIR THAT WAS MEASURED AND DID NOT SEPARATE.** *`docs/decisions.md`
    §11 on that very row: "32 and 64 measured indistinguishable at 8K and CF took the cheaper
    one."* **So the 16/32 the price was measured at is not the faster configuration — it is the
    cheaper of two that tied**, and the price is borrowed across a difference that was looked for
    and not found.

    **THE TIDY FIX WOULD HAVE BEEN THE WRONG WAY ROUND AND THE REASON IS ON RECORD.** *Extending
    `levers` so `OUTSIDE` takes 8K's settings puts 32 pools against 8640's 135 CTU rows, which
    starves WPP on the mechanism `F-2026-09-01-2` predicted before it was measured — a real cost
    paid to remove a difference that measured as zero.* **Gate-ruled 2026-09-03: leave it as
    built.** *This returned `None` until
    then, and the docstring argued for the absence: three readings were open, and a number shipped
    ahead of its ruling is indistinguishable afterwards from one CF chose.* **That reasoning was
    right and its premise is gone**, so the argument goes with the behaviour rather than being
    left to defend an absence beside a present seed.

    **THE SEED UNDER-PREDICTS A LARGER FRAME BY CONSTRUCTION AND THE CAP IS WHAT ABSORBS IT.**
    *8K's row is the slowest measured and a frame above 8K is slower still, so this is the
    optimistic direction the ladder otherwise exists to refuse.* **`MAX_DELIVERED_FRAMES[OUTSIDE]`
    is 1,800: at the cap the seed predicts 1,800 s against a 3,600 s kill, so the job may run
    twice as slow per frame as 8K did and still land.** *The two halves are one ruling and neither
    is safe alone — a seed with no cap is the 3,663-second failure pointing the other way.*

    **THE STEP STAYS `OUTSIDE` AND MUST NOT READ AS 8K.** *The seed comes from 8K's row; the frame
    is not 8K, and nothing measured says it costs what 8K costs.* **`eta_ladder` is the field that
    says the frame is off the table, and a seed arriving does not put it on** — which is §9e's
    symmetry: a rule silent about being outside its population reads as being inside it.
    """
    step = step_for(delivered_pixels)
    if step == OUTSIDE:
        # **8K's ROW, AND THE LABEL IS STILL `OUTSIDE`.** *The band is chosen by the frame count
        # exactly as an 8K job's is, because the seed IS 8K's row — reading it any other way
        # would be inventing a second rule for a size nobody has measured.*
        seed, _ = seconds_per_frame(LARGEST, delivered_frames, codec)
        return seed, OUTSIDE
    # **The suffix is a label for the caller, not a key** — the seed comes from the row it names.
    name = step[:-len(ROUNDED)] if step.endswith(ROUNDED) else step
    row = ROWS[name].get(codec or "h265")
    if row is None:
        # **An unimplemented codec is not an unmeasured frame.** The step is still named — the
        # frame is what it is — and the seed is absent because this table has no row for it.
        return None, step
    if row["switch"] is None:
        return row["at"][1], step
    if delivered_frames is None:
        # **AN UNKNOWN COUNT TAKES THE SLOWER ROW, WHICH IS NOT THE SAME CHOICE `levers` MAKES.**
        # *There the answer is `None` and the job falls to the CTU rule, because a setting is a
        # claim about the frame.* **Here both rows are legitimate prices for this frame and only
        # the length is unknown, so the ladder's own direction decides: over-estimate against a
        # platform that kills at 3600 s.** *Taking the cheaper row would be the optimism the whole
        # item exists to refuse, arriving through an absent input rather than a wrong one.*
        return max(row["at"][1], row["below"][1]), step
    return (row["at"] if int(delivered_frames) >= row["switch"] else row["below"])[1], step


def derived_pools(delivered_height):
    """`clamp(round(height / 64), 1, 64)` — the CTU-row rule, for a frame the table does not name.

    **HALF-UP, AND PYTHON'S `round` IS NOT.** *§11 writes the rule as `round(delivered_height / 64)`
    and then works the example: a 1440p job takes `pools 23`.* **1440/64 is exactly 22.5 and
    `round(22.5)` is 22 in Python** — banker's rounding, ties to even. *The gate ruled half-up
    2026-09-02: §11's own row counts are ceilings — 1080p 17, 4K 34, 8K 68 — so a 1440p frame is 23
    CTU rows with the last one partial, and 22 pools against 23 rows is the starvation
    `F-2026-09-01-2` predicted before it was measured.*

    **The clamp's ends are not decoration.** *Below 64 pixels of height there is less than one CTU
    row and `pools 0` is not a setting; above 4096 the row count passes what `envelope.POOLS_RANGE`
    already refuses a caller for asking.*
    """
    rows = int((float(delivered_height) / CTU_SIZE) + 0.5)
    return max(1, min(rows, 64))
