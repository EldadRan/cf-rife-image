"""CF's ruled table — `docs/decisions.md` §11 — and it is ONE table with two readers.

**THE SETTINGS AND THE SEED ARE THE SAME ROWS.** *`docs/current.md` prints them as one table for a
reason: an area's `frame_threads / pools` and its `s/frame` were measured together, on the same
runs, and a job that takes one row's settings is priced by that row's seconds.* **Split across two
modules they would be two things to keep equal, and the copy that rotted would be
indistinguishable from the live one** — which is this project's central hazard stated in
`CLAUDE.md` about four repositories, arriving inside one.

So `encoder` reads the levers from here and the ETA seed comes from here, and neither owns the
numbers.

**THE ROWS ARE KEYED ON DELIVERED HEIGHT AND NOT ON PIXELS.** *A CTU-64 frame has `height / 64`
rows; WPP cannot use more pools than it has rows.* **A portrait 1080x1920 clip has 1080p's pixel
count and 30 CTU rows**, so a bucket keyed on area starves it — and vertical video is the case a
three-bucket table has no answer for at all.

**`s/frame` IS `wall_s` OVER DELIVERED FRAMES, POOLED ACROSS THE BAND** — *not `calc_t`, not
`frame_s`* — **so it already contains fetch, decode and upload.** *A prediction must contain the
terms that overrun a cap, because those are exactly the ones nobody can isolate.*

**THE POOL IS THE BAND, NOT THE CELL**, so the low-volume figures carry the deliberately bad `ft=1`
and `ft=2` runs. *That is the conservative direction and it is deliberate.*
"""

#: HEVC's coding-tree-unit size, which is what makes the row count `height / 64`.
CTU_SIZE = 64

#: The three heights CF's table names, and nothing else is in the ladder.
STEPS = {1080: "1080p", 2160: "4K", 4320: "8K"}

#: **The tallest step, and what an above-8K frame is priced from.** *There is no row ABOVE it to
#: round up to, so `step_for` answers `OUTSIDE` — and `seconds_per_frame` then borrows this step's
#: seconds rather than publishing nothing, which is CF's ruling of 2026-09-02.*
TALLEST = 4320

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

#: What `step_for` says about a frame the table does not name.
#:
#: **`ROUNDED` IS A SUFFIX ON A STEP AND NOT A STEP OF ITS OWN, AND THE FIRST DRAFT HAD A THIRD
#: BARE VALUE THAT NO INPUT COULD PRODUCE.** *`step_for` rounds every unnamed height UP to the
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
#: **KEYED BY `step_for` AND NOT BY A SECOND TABLE OF SIZES.** *The cap and the seed take one step
#: boundary from one function; a second table would be free to disagree with the ladder about
#: where 4K ends, and that disagreement would be invisible until a job landed between them.*
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


def max_delivered_frames(delivered_height):
    """The cap for this frame. **A rounded step takes the step it rounds to.**

    *`step_for` rounds an unmeasured height UP — 1440p is priced at 4K — and the cap follows the
    same step for the same reason: the seed says this job costs 4K seconds, so the limit that
    keeps it inside the 3,600 s kill is 4K's.* **Two answers from one step, which is what putting
    them in one module buys.**
    """
    step = step_for(delivered_height)
    return MAX_DELIVERED_FRAMES[step[:-len(ROUNDED)] if step.endswith(ROUNDED) else step]


def step_for(delivered_height):
    """`"1080p"`, `"4K"`, `"8K"`, `DERIVED`, or `OUTSIDE`.

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
    height = int(delivered_height)
    if height in STEPS:
        return STEPS[height]
    if height > TALLEST:
        return OUTSIDE
    for boundary in sorted(STEPS):
        if height < boundary:
            return STEPS[boundary] + ROUNDED
    # **Unreachable while `TALLEST` is the largest key of `STEPS`, and it is not an assertion of
    # that.** *A step added above 4320 without moving `TALLEST` would land here, and returning
    # `OUTSIDE` is the honest answer for a height no row was found for.*
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
    height UP for the seed, which is the conservative direction for a TIME estimate — but a
    setting is not an estimate: handing a 1440p frame 4K's 32 pools would be a claim about its CTU
    rows, and it has 23.*
    """
    name = STEPS.get(int(delivered_height))
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


def seconds_per_frame(delivered_height, delivered_frames, codec="h265"):
    """`(s_per_frame, step)` for the ETA seed. **Above the ladder it seeds from 8K and says so.**

    **ABOVE 8K IS PRICED AS 8K — CF, 2026-09-02, ORDERED 2026-09-03.** *Priced, not executed: the
    SEED comes from 8K's row and the SETTINGS do not.* **`levers` still answers `None` for any
    height the table does not name, so an above-8K frame takes the CTU-row rule** — 64 pools at
    8640, against the 32 that 8K's row was measured at. *That is the module header's claim
    inverted for exactly one case: the price and the settings were measured together and here
    they part.* **Reported to the gate rather than reconciled here** — the row rule is right about
    a taller frame's CTU rows and the seed is CF's ruling, and choosing between them is not this
    file's to do. *This returned `None` until
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
    step = step_for(delivered_height)
    if step == OUTSIDE:
        # **8K's ROW, AND THE LABEL IS STILL `OUTSIDE`.** *The band is chosen by the frame count
        # exactly as an 8K job's is, because the seed IS 8K's row — reading it any other way
        # would be inventing a second rule for a size nobody has measured.*
        seed, _ = seconds_per_frame(TALLEST, delivered_frames, codec)
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
