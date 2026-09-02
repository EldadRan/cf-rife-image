"""Writing the master, once, with everything CF requires in the same mux.

**This module exists because the vendored writer cannot satisfy the contract** (see
`docs/decisions.md` 0.3). SeedVR2's `save_frames_to_video` offers `mp4v` through OpenCV or
`libx264 -crf 12` through an ffmpeg pipe, and neither writes faststart, `-metadata` tags or
audio. CF requires all three **in the mux already being done** and explicitly forbids a second
pass — a remux that exists only to add tags is a rewrite, and CF has measured a stream-copy
faststart pass take a 342-frame trim to 343.

So the worker owns the encode. Frames arrive as they are produced and go straight down a pipe;
nothing is staged on disk and no finished file is ever reopened to fix what the first pass should
have set.

**On `+faststart` being one pass and not two.** ffmpeg relocates the moov atom after writing the
last frame, inside the same invocation. That is not a remux of a finished file: no packet is
re-encoded, no container is reinterpreted, and no edit list exists to be moved — this worker's
output has none, because it writes a fresh timeline rather than bounding an existing one. The
distinction matters because the failure CF measured comes from *reinterpreting* a container that
carried an edit list, which cannot arise here.
"""

import math
import os
import subprocess
import time

import envelope
import ladder
import probe

from errors import INTERNAL, WorkerError

#: Audio codecs that mux into MP4 as-is. Anything else is re-encoded to AAC — the media worker's
#: rule, and the reason a copied track stays bit-exact where it can.
MP4_NATIVE_AUDIO = ("aac", "mp3", "alac", "ac3", "eac3")

#: The master's encoder settings. Not measured yet — the encode figures CF was owed were never
#: would justify them, and until those exist these are a starting point rather than a decision.
#: CRF 12 is the vendored writer's own choice, kept so the first measurements compare against
#: something rather than against a number invented here.
DEFAULT_CRF = 12
DEFAULT_PRESET = "medium"

#: **The three x264 settings that bound the encoder's MEMORY, as named fields with today's values
#: as the defaults** (contract §6a). They were one frozen string — `routec.FRUGAL_X264` — chosen
#: to make an 8K run fit on a 24-core host, and §6a rules all five settings changeable so the
#: campaign can price them. `crf` and `preset` were already fields; these three join them rather
#: than inventing a second mechanism.
#:
#: **`sliced_threads` is the large one.** Threads split ONE frame rather than each taking their
#: own, which cuts frames-in-flight from dozens to one at a compression and speed cost. `threads`
#: caps the frame-level parallelism that multiplies the working set. `rc_lookahead` shortens the
#: window `medium` sets to 40.
#:
#: **Every value is an ordering hint and not a prediction** — the gate modelled x264 at ~4 GiB
#: against an observed 40-plus — which is why §6c refuses to let an estimator throttle against
#: them until a campaign has fitted host RSS to them.
DEFAULT_THREADS = 4
DEFAULT_SLICED_THREADS = True
DEFAULT_RC_LOOKAHEAD = 10

#: x264's own preset ladder, slowest-to-fastest presets being a speed/compression trade. **The
#: enumeration IS the range** — a preset is a name x264 either knows or rejects, so nothing here
#: is a bound this project had to justify.
PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
           "slow", "slower", "veryslow", "placebo")

#: **x264's documented maximum thread count, and the floor is 1 rather than 0 ON PURPOSE.** x264
#: reads `threads=0` as *auto*, which is `1.5 x cores` capped at this number — on the 96-core host
#: this worker runs, auto is 128 frame-threads and precisely the configuration that filled 46 GiB
#: and got the first 8K run reaped. **§6a rules that the whole point of these settings is bounding
#: the encoder's memory and a pass-through would hand that bound to the caller**; accepting `0`
#: would hand it over in a different spelling, so `auto` is not reachable through this field.
THREADS_MIN, THREADS_MAX = 1, 128

#: x264's own range for `rc-lookahead`. 0 disables the lookahead entirely, which is a legitimate
#: setting and the cheapest one; 250 is x264's ceiling.
RC_LOOKAHEAD_MIN, RC_LOOKAHEAD_MAX = 0, 250


# ─────────────────────────────────────────────────────────────────────────────────────────────
# §6d — THE AREA TABLE. What a caller receives when it sends nothing.
# ─────────────────────────────────────────────────────────────────────────────────────────────

#: **The boundary, in INTEGER DELIVERED PIXELS** (contract §6d as amended by §6d-1). 3840x2160 is
#: exactly this, so the 4K measurement IS the boundary rather than a number the contract invented —
#: both rows are applied at a frame size where their arm was actually run.
#:
#: **IT IS THE DELIVERED FRAME AND NOT THE PADDED AREA, AND THAT WAS A CORRECTION.** The first
#: draft keyed on `interp_plan.padded_megapixels` — §9a's variable — on the argument that the table
#: should use the independent variable the project already had. **That was reasoning by adjacency.**
#: §9a is right for §9a because THE MODEL allocates the padded tensor; this table's consumer is the
#: ENCODER, which allocates against the delivered frame and never sees the padding at all. The
#: padding multiple is a function of `scale`, so `force_scale=0.5` moved a 4K job across the
#: boundary and handed it an arm measured only at 8K **while the frame ffmpeg encodes had not
#: changed by one pixel**. And `docs/test-plan.md` §25a's mechanism — the whole reason there are
#: two rows — is stated in delivered frame sizes: a slice frame ~12 MB at 4K and cache-resident,
#: ~50 MB at 8K and not. *The table had been keyed on a quantity its own justifying mechanism does
#: not mention.*
#:
#: **The two keys agree at `scale=1` and diverge only where padding and delivery diverge**, so
#: every measurement this project has taken agrees with both and no run in the corpus could have
#: told them apart. *Reuse looked like consistency and was an unexamined premise.*
#:
#: **NEVER COMPARE THIS IN MEGAPIXELS.** `8294400 / 1e6` against a float literal is an identity
#: comparison at the exact point a real job sits on, and the shared law forbids it. Width and
#: height are integers; multiply and compare them there.
#:
#: **Recorded on every row** (`docs/archive/instrumentation-archive.md` §13a) because it is a constant that will
#: move. **It moved within hours of that argument being written and before a single row was
#: banked**, which is the difference between a field argued for and a field guessed at.
AREA_BOUNDARY_DELIVERED_PIXELS = 8_294_400

AREA_ROW_SMALL = "area:small"
AREA_ROW_LARGE = "area:large"

AREA_DEFAULTS = {
    #                       threads  sliced_threads  rc_lookahead
    AREA_ROW_SMALL: {"threads": 16, "sliced_threads": False, "rc_lookahead": 10},
    AREA_ROW_LARGE: {"threads": 16, "sliced_threads": True, "rc_lookahead": 10},
}

#: The three fields the table decides. `crf` and `preset` are NOT among them — they are §6a
#: fields with their own defaults and the table says nothing about either, so a job that sends
#: neither still gets `DEFAULT_CRF` and `DEFAULT_PRESET`.
AREA_FIELDS = ("threads", "sliced_threads", "rc_lookahead")

#: What `encode_defaults.basis` may read (`docs/archive/instrumentation-archive.md` §13a). **One field and not
#: three**, and `"mixed"` is why it needs a name: §6d keeps every setting individually
#: overridable, so a caller may send `threads` and leave `sliced_threads` unset. A per-field basis
#: would be three columns answering one question; a single field that could not express the
#: partial case would file that job under whichever half it happened to check first.
BASIS_CALLER = "caller"

#: §6i: the other half of a per-field basis. **`caller` or `default`, per field, because the two
#: levers are independently optional** — one sent and one defaulted is a legal request that a
#: single basis cannot describe, which is `BASIS_MIXED`'s defect re-made one surface over.
BASIS_DEFAULT = "default"
BASIS_MIXED = "mixed"

#: **THE THIRD VALUE, AND WITHOUT IT THE RECORD CANNOT BE SEARCHED.** *`caller` and `default` were
#: enough while an absent field resolved to a constant.* **Now an absent field resolves either
#: from CF's ruled table or from the CTU-row rule, and a 1440p job at `pools 23` would be
#: indistinguishable in the record from a ruled one** — so nobody could find those jobs later to
#: check whether the rule held. *`default` keeps meaning what it meant on every banked row: the
#: worker chose, from what it had ruled. `derived` says the worker COMPUTED it, from a frame the
#: ruling does not name.* (`docs/decisions.md` §11, CF 2026-09-02.)
BASIS_DERIVED = "derived"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# §6e — THE CODEC IS THE CALLER'S CHOICE, and each library gets its bound assembled for it.
# ─────────────────────────────────────────────────────────────────────────────────────────────

#: **ffmpeg's encoder name for each codec this worker implements**, and the mapping exists so
#: that the library is chosen in ONE place from a name the contract enumerated.
#:
#: **`envelope.CODECS` carries a third name and it is deliberately absent from here.**
#: `"source"` is contract-legal and unimplemented — §6e leaves it refused at the door as a
#: CAPABILITY refusal — and a library for it here would be this worker claiming to resolve a
#: codec it never opens the file to read. *Deleting the name from `envelope` would turn a
#: forward-looking request into a schema error instead; leaving it here would turn it into a
#: silent encode. Neither is the true answer, and the refusal is.*
CODEC_LIBRARIES = {"h264": "libx264", "h265": "libx265"}

#: What `encode_defaults.basis` reads under h265 (`docs/archive/instrumentation-archive.md` §15a). **§6d's table
#: is x264's and cannot have resolved an h265 job**, so the honest reading is *"the table was not
#: consulted because this codec has no such table"* — a state, and it needs a name because null
#: was already taken: §13a reserves absent-or-null for a run that died before the branch, and
#: reusing it would make provenance inferable from silence.
BASIS_CODEC_H265 = "codec:h265"

#: **x265's settings, DECLARED rather than fitted** (contract §6e ruling 1, outcome B — CF,
#: 2026-08-28). *This project has zero x265 measurements and the reading arrives as a side effect
#: of §6e's certification.*
#:
#: **THEY ARE NO LONGER A BOUND "CAPPED AT THE DOOR", AND THIS BLOCK SAID THEY WERE.** *Under §6i
#: `frame_threads` and `pools` are optional request fields with ranges — 1..16 and 1..64 in
#: `envelope` — so nothing here is pinned and the cap is the caller's within those bounds. The
#: DEFAULTS are still this worker's, and a request naming neither field still receives them.*
#: Found in review.
#:
#: **WHAT `frame-threads` DOES, which is why it is the lever §57 sweeps:** it multiplies
#: frames-in-flight and therefore the working set — the role `threads` plays in x264 — and the
#: default of 1 takes parallelism from INSIDE the frame instead, through x265's wavefront (`wpp`),
#: which is on by default. `pools` sizes the worker pool; `rc-lookahead` mirrors §6a's value
#: rather than inventing a second one and is NOT caller-settable.
#:
#: **Declared as a BOUND and never claimed as an optimum.** §6c forbids an estimator throttling
#: against a host-RSS model nobody has built, and a fixed cap on what the encoder ASKS FOR is not
#: that model: it refuses nothing, downgrades nothing and predicts nothing.
#: **The output pixel format per bit depth** (contract §6f). **`-pix_fmt` is what selects the
#: PROFILE, and `-profile:v main10` is deliberately not sent beside it.** libx265 derives `main10`
#: from a 10-bit input format on its own; sending both would be two statements of one fact, and
#: the one that loses an argument with the other is the one nobody is looking at. *ONE FACT, ONE
#: HOME, at the command line.*
#:
#: **The INPUT stays `rgb24` at both depths.** The model produces 8-bit RGB and nothing upstream
#: of the encoder has more precision to give — so 10-bit here buys headroom in the prediction and
#: transform loops rather than in the source, which is exactly the mechanism §6f expects to
#: suppress BANDING. *A claim that this makes the master "10-bit content" would be false and the
#: record carries `bit_depth` rather than a quality assertion.*
PIXEL_FORMATS = {8: "yuv420p", 10: "yuv420p10le"}

#: **x265's threading, and until 2026-08-31 these three were bare integers with no comment in a
#: file where every other constant carries a paragraph** (contract §6h). *Nothing in the contract,
#: in `docs/gate-findings.md` or in any commit message said why `frame-threads` was 1, and the
#: 46 GiB incident everyone reaches for is x264's `threads=0` auto resolving to 128 at
#: `encoder.py:64-70` — a different codec and a different field.* **A value whose justification
#: exists only in someone's memory is indistinguishable from one that was never justified**, which
#: is `F-2026-08-29-7`'s remedy and `F-2026-08-29-13`'s, so the reason is written here now.
#:
#: **WHAT IS ACTUALLY KNOWN, and it is a measurement rather than a rationale**
#: (`docs/test-plan.md` §55): `write_wait_s` is 80% of an h265 wall at 1080p and 4% of an h264
#: one — 148.9 s against 5.2 s on comparable clips — while `model_s` is flat to 1.5% across all of
#: it. The handoff is a synchronous `stdin.write` of a 6-25 MB frame into a 64 KB pipe, so that
#: figure is the encoder's throughput seen through a small window and not a queueing artefact.
#: **x265 throughput varies 1.74x ACROSS hosts and reproduces to 1.1% WITHIN one**, which §6h
#: rules is host contention and an infrastructure ask rather than anything this file can fix.
#: *`X265_POOLS` and `X265_FRAME_THREADS` used to sit here, then `envelope.DEFAULT_POOLS` and
#: `envelope.DEFAULT_FRAME_THREADS` replaced them as the one home for a flat default. **Both are
#: gone: §11 rules the values from a TABLE keyed on delivered height, and a flat default is what
#: it refutes.*** Keeping a second copy with a comment asserting the two agreed is the drift this
#: file has been fixing all week, and a stale constant is the same drift with one copy.
X265_RC_LOOKAHEAD = 10

#: **§6i: THE x265 THREADING PATH IS NO LONGER KEYED ON AREA.** *`X265_FRAME_THREADS_BY_AREA`
#: lived here from §6h until CF ruled §6i the same day. It is gone rather than left holding one
#: value: `area` is not under the requester's control, so it is neither a field nor something a
#: caller sees, and it stops selecting x265 threading.* **`area_row` is untouched and still drives
#: the x264 defaults row**, which is where it always belonged.
#:
#: **§11: AND AS OF 2026-09-02 IT IS KEYED ON THE FRAME AGAIN — ON HEIGHT, WHICH IS NOT WHAT §6h
#: KEYED ON.** *§6h keyed on AREA and §6i struck it, on the ground that area is not the
#: requester's to set. That ground still holds and this is not its reversal: what a caller SENDS
#: is unchanged, both fields are still optional and still obeyed when sent, and this table only
#: decides what an ABSENT field resolves to.* **A flat default could not serve 1080p and 8K at
#: once** — measured, `docs/decisions.md` §11 — *and shipping one meant every unset job at one of
#: the two areas ran at a setting nothing measured.*
#:
#: **HEIGHT RATHER THAN PIXELS, AND THAT IS THE WHOLE MECHANISM.** *A CTU-64 frame has
#: `height / 64` rows; WPP cannot use more pools than it has rows, and pools below the row count
#: starves it.* **A portrait 1080x1920 clip has 1080p's pixel count and 30 CTU rows** — an area
#: bucket hands it 16 pools against 30 rows and starves the encoder, which is the case a
#: three-bucket table has no answer for at all.
#: **THE TABLE ITSELF LIVES IN `ladder`, WHICH IS THE ONE HOME FOR CF's §11 ROWS.** *The settings
#: and the ETA seed are the same rows — measured together, on the same runs — and `current.md`
#: prints them as one table for that reason.* **Two modules holding half a table each would be two
#: things to keep equal**, and the copy that rotted would be indistinguishable from the live one.
#: *That is this project's central hazard, stated in `CLAUDE.md` about four repositories, arriving
#: inside one.*
DERIVED_FRAME_THREADS = 16


def area_row(delivered_pixels):
    """Which of §6d's two rows a frame takes. **`<=` the boundary is the small row.**

    **`delivered_pixels` is `width * height` OF THE FRAME THE ENCODER IS HANDED**, which §6d-1
    pins to the size `MasterWriter` is constructed with — not the source's probed size and not the
    model's padded area. The caller passes the same two integers it builds the writer from, so the
    branch and the encode cannot be deciding against different answers to "how big is this frame".

    The comparison is on an integer and this function will not accept anything else — a float
    reaching here is a megapixel figure arriving where a pixel count was meant, which is precisely
    the comparison §6d spends a paragraph refusing.
    """
    if isinstance(delivered_pixels, bool) or not isinstance(delivered_pixels, int):
        raise WorkerError(INTERNAL, (
            "the encoder default is keyed on INTEGER delivered pixels and got {!r} ({}). "
            "It is width x height of the frame handed to the encoder; a megapixel figure is a "
            "float and is a different quantity.").format(
                delivered_pixels, type(delivered_pixels).__name__))
    return (AREA_ROW_SMALL if delivered_pixels <= AREA_BOUNDARY_DELIVERED_PIXELS
            else AREA_ROW_LARGE)


def x265_threading(frame_threads=None, pools=None, delivered_height=None,
                   delivered_frames=None):
    """§6i's two levers resolved: `(values, basis)`, each field independently.

    **`None` MEANS THE CALLER SENT NOTHING, AND THAT IS THE WHOLE MECHANISM** — §6d's rule, which
    `validation` preserves by handing these through as `None` when unset rather than resolving
    them early. A branch that cannot tell *"the caller asked for 1"* from *"the caller asked for
    nothing"* silently overwrites an explicit value, and `§6b`'s surviving clause is that a knob a
    caller sets and the worker quietly ignores is worse than no knob.

    **A BASIS PER FIELD, NEVER ONE FOR BOTH.** One sent and one defaulted is a legal request —
    §6i: *optional, and independently so* — and a single basis cannot describe it. That is
    `BASIS_MIXED`'s defect re-made: a row saying `caller` about a number the caller never sent.

    **THREE STATES NOW.** *`caller` — sent. `default` — CF's ruled table named this frame.
    `derived` — the table did not, and the CTU-row rule computed it.* **The third exists so the
    computed jobs can be FOUND** in a corpus, which is the only way anyone checks later whether
    the rule held.

    **AND THE DEFAULTS ARE NO LONGER TWO CONSTANTS.** *`envelope.DEFAULT_FRAME_THREADS` and
    `envelope.DEFAULT_POOLS` were the one home for a flat default, and a flat default is what
    §11 refutes: 1 and 16 served neither 1080p nor 8K, and every unset job ran at a setting
    nothing had measured.* **A table is the home now, and the constants are gone rather than left
    beside it saying something that is no longer true.**

    `delivered_height` and `delivered_frames` describe THE FRAME THE ENCODER IS HANDED, the same
    pair `MasterWriter` is constructed with — §6d-1's pin, one surface over.
    """
    ruled = (None if delivered_height is None
             else ladder.levers(delivered_height, delivered_frames, "h265"))
    if ruled is not None:
        default_ft, default_pools = ruled
        unsent_basis = BASIS_DEFAULT
    elif delivered_height is not None:
        default_ft, default_pools = (DERIVED_FRAME_THREADS,
                                     ladder.derived_pools(delivered_height))
        unsent_basis = BASIS_DERIVED
    else:
        # **A caller with no frame to key on gets the 1080p row and the record says `derived`.**
        # *There is no honest `default` here: `default` means CF ruled this frame's values, and
        # nothing ruled a frame nobody named.* **Reachable only by a direct caller** — `routec`
        # has the delivered size at the one site that builds the writer.
        default_ft, default_pools = (DERIVED_FRAME_THREADS,
                                     ladder.levers(1080, None, "h265")[1])
        unsent_basis = BASIS_DERIVED
    values = {
        "frame_threads": default_ft if frame_threads is None else int(frame_threads),
        "pools": default_pools if pools is None else int(pools),
    }
    basis = {
        "frame_threads_basis": unsent_basis if frame_threads is None else BASIS_CALLER,
        "pools_basis": unsent_basis if pools is None else BASIS_CALLER,
    }
    return values, basis


def resolve_codec(codec):
    """`None` -> the default, anything else unchanged. **The one place that answer is made.**

    *It was made in two places that never wrote back — `resolve_defaults` and
    `MasterWriter.__init__` each resolved internally — so a caller passing `None` held a name that
    was already false for the encode about to happen, and any test it did against `"h265"` took
    the wrong branch while both consumers took the right one.* Found in review.
    """
    return envelope.DEFAULT_CODEC if codec is None else codec


def resolve_defaults(delivered_pixels, codec=None, threads=None, sliced_threads=None,
                     rc_lookahead=None):
    """§6d's branch: **the row fills in what the caller did not send, and never what it did.**

    Returns `(settings, provenance)` — the three resolved values, and the
    `docs/archive/instrumentation-archive.md` §13a block that says who chose them.

    **`None` MEANS THE CALLER SENT NOTHING, AND THAT IS THE WHOLE MECHANISM.** §6d states the
    requirement as a property rather than as an implementation: *absence must remain
    distinguishable from a value all the way to the branch*, and *a design in which the two are
    indistinguishable at the branch site is refused on sight*. `validation` used to resolve all
    three to constants before the source was even probed, so this function would have been unable
    to tell *"the caller asked for threads=4"* from *"the caller asked for nothing and validation
    supplied 4"* — and a branch that cannot tell those apart silently overwrites an explicit
    caller value, which is the clause of §6b that survives §6d. **A knob a caller sets and the
    worker quietly ignores is worse than no knob.**

    So `validation` now hands these three through as `None` when unset, having range-checked
    whatever WAS sent, and this is the only place a default is chosen for them.

    **UNDER h265 THERE IS NO TABLE AND NO SETTINGS** (contract §6e, `docs/archive/instrumentation-archive.md`
    §15a). §6d's three fields are x264's vocabulary — `threads` and `sliced_threads` have no x265
    spelling at all, and `sliced-threads` is ABSENT rather than renamed — so this returns an EMPTY
    settings dict and a basis that says the table was skipped. *An empty dict rather than the row's
    values is what stops the caller writing x264's numbers onto an h265 row, which §15a calls the
    corpus telling a reader something untrue in a field they have no reason to doubt.*

    `codec` of `None` means the caller named none, which is `envelope.DEFAULT_CODEC` — resolved
    here rather than at the call site so the default has one home.
    """
    codec = resolve_codec(codec)
    if codec not in CODEC_LIBRARIES:
        raise WorkerError(INTERNAL, (
            "the encoder was asked to resolve defaults for codec {!r}, which this worker does "
            "not implement. `validation` refuses everything but {} at the door, so reaching here "
            "is a plumbing defect and not a request.").format(
                codec, ", ".join(sorted(CODEC_LIBRARIES))))
    if codec == "h265":
        # **§6e ruling 2 refuses these three at the door under h265, so a value here is a
        # PLUMBING defect and not a caller's.** Checked rather than ignored: dropping them
        # silently is the exact shape ruling 2 refused — a bound the caller believes is in force
        # and the encoder never saw — and it would arrive here wearing the same face.
        sent = {name: value for name, value in
                (("threads", threads), ("sliced_threads", sliced_threads),
                 ("rc_lookahead", rc_lookahead)) if value is not None}
        if sent:
            raise WorkerError(INTERNAL, (
                "codec h265 reached the encoder carrying {} — contract §6e ruling 2 refuses "
                "these at the door because x265 has no such parameters, so a value here would "
                "be recorded beside an encode that never saw it.").format(
                    ", ".join("{}={!r}".format(k, v) for k, v in sorted(sent.items()))))
        # **§6i: THE BASIS IS BARE AGAIN.** §6h had made it `"codec:h265/area:small"` so a row
        # could say which threading it ran at; with `frame_threads` and `pools` on the record AS
        # VALUES the basis does not need to carry a branch, and `area` is not the caller's to see.
        # *`sha-88fec73` shipped that suffix while `record_version` stayed 2, so its h265 rows
        # label the same encode differently from every row before them. Going back closes the seam
        # rather than papering over it with a version bump, and the discriminator for the rows that
        # image did write is `retime.frame_threads` being present at all.*
        return {}, {
            "basis": BASIS_CODEC_H265,
            # **Still carried, and it is not the boundary that decided anything here.** §13a
            # wants the integer the branch compared against as it stood in the image that ran;
            # under h265 nothing was compared, and omitting the field would make this block a
            # different shape from every other row for a reason no reader could see.
            "boundary": AREA_BOUNDARY_DELIVERED_PIXELS,
        }
    row = area_row(delivered_pixels)
    sent = {"threads": threads, "sliced_threads": sliced_threads, "rc_lookahead": rc_lookahead}
    chosen = {name: value for name, value in sent.items() if value is not None}
    settings = dict(AREA_DEFAULTS[row])
    settings.update(chosen)
    # **Three states, and the middle one is the reason `basis` is one field.** Nothing sent is the
    # row; everything sent is the caller; anything else is genuinely mixed and says so rather than
    # being filed under whichever half was checked first.
    if not chosen:
        basis = row
    elif len(chosen) == len(AREA_FIELDS):
        basis = BASIS_CALLER
    else:
        basis = BASIS_MIXED
    provenance = {
        "basis": basis,
        # **The boundary as it stood in the image that ran, NOT the job's own area** — §13a is
        # explicit that `padded_megapixels` already carries the second.
        "boundary": AREA_BOUNDARY_DELIVERED_PIXELS,
    }
    return settings, provenance


def x264_params(threads=DEFAULT_THREADS, sliced_threads=DEFAULT_SLICED_THREADS,
                rc_lookahead=DEFAULT_RC_LOOKAHEAD):
    """Build the `-x264-params` string from validated fields. **Assembled, never accepted.**

    **Contract §6a rules this shape and gives the reason.** The obvious implementation is a
    request field carrying the options string straight through, and it is a hole: the arguments
    are passed as a list rather than through a shell, so it is not a command-injection hazard —
    it is a RESOURCE one, and sharper than it looks. These settings exist to bound the encoder's
    memory, on a path §1 says has no host guard, so a pass-through would let a request restore
    exactly the configuration that killed the 8K run.

    **The default call reproduces the frozen string byte for byte**, key order included, so the
    corpus taken at `sliced-threads=1:threads=4:rc-lookahead=10` and the corpus taken at these
    defaults are the same corpus rather than two that have to be reconciled later.
    """
    return "sliced-threads={}:threads={}:rc-lookahead={}".format(
        1 if sliced_threads else 0, int(threads), int(rc_lookahead))


def pixel_format(bit_depth=None):
    """The encode's output pixel format for a bit depth. **`None` means the caller named none.**

    **ONE HOME for the resolution**, because two callers need it and they must not disagree:
    `MasterWriter` builds the command from it, and contract §6g's disk bound computes bytes per
    pixel from it before the writer exists. *A bound computed against a different format from the
    one that gets written is a bound about a different job.*
    """
    depth = envelope.DEFAULT_BIT_DEPTH if bit_depth is None else bit_depth
    if depth not in PIXEL_FORMATS:
        raise WorkerError(INTERNAL, (
            "no pixel format for bit depth {!r}; `envelope` enumerates {} and `validation` "
            "refuses the rest at the door.").format(
                depth, ", ".join(str(d) for d in sorted(PIXEL_FORMATS))))
    return PIXEL_FORMATS[depth]


def x265_params(frame_threads, pools, rc_lookahead=X265_RC_LOOKAHEAD):
    """Build the `-x265-params` string from named fields. **Assembled, never accepted** — and
    here the rule has teeth the x264 side does not.

    **x265 VALIDATES VALUES AND SILENTLY DISCARDS UNKNOWN NAMES** (`docs/gate-findings.md`
    F-2026-08-28-7, measured 2026-08-28). `frame-threads=999` refuses with exit 183 and no
    output; `frame-thread=1` — one character short — is accepted, thrown away, and the encode
    succeeds. **So "the job succeeded" is not evidence that a bound was applied**, and a typo
    disables the whole of this string with no error, no non-zero exit, a plausible byte count and
    a passing witness.

    *§6a already forbids a pass-through options string on the x264 side for a resource reason.
    This is that reason arriving on the x265 side with the error reporting removed*, which is why
    the names live here as identifiers rather than in a literal a request or an edit could reach:
    **a name this contract has not enumerated cannot get onto the command line.**

    **`frame_threads` AND `pools` ARE REQUIRED AND HAVE NO DEFAULTS, WHICH THEY USED TO.** Both
    defaulted to the module constants, and that agreed with what shipped for every input until §6i
    made them caller-settable — so a bare `x265_params()` returns the DEFAULT string and is
    silently wrong for any request naming either field. *A caller wanting "what this worker emits"
    must say for which request, because there is no longer one answer.* **Removing the defaults
    turns a wrong string into a `TypeError` at the call site**, which is the failure worth having:
    the silent version passes on exactly the requests that send nothing.

    **VERIFIED TO TAKE EFFECT AND NOT MERELY TO BE ACCEPTED**, which the paragraph above shows is
    a distinction with teeth: `pools=1:frame-threads=1` and `pools=16:frame-threads=4` produce
    `frame threads / pool features : 1 / wpp` and `: 4 / wpp` respectively in x265's own banner.
    """
    return "pools={}:frame-threads={}:rc-lookahead={}".format(
        int(pools), int(frame_threads), int(rc_lookahead))


def _identity_tags(identity):
    """`-metadata` arguments. **Identity only** — this file is delivered.

    Timings, hardware, tiling configuration, worker ids and anything resembling a credential stay
    in the manifest and the diagnostics bundle. What goes in the container is what the file needs
    to say what it is when it is found in R2 with no job and no manifest beside it.

    It is a recovery aid and never a source of truth: CF's standing rule is to read the worker's
    reported fields rather than re-probe the file, and these tags are what someone falls back to
    when the response and the manifest are both gone.
    """
    args = []
    for key, value in identity.items():
        if value is None:
            continue
        args += ["-metadata", "{}={}".format(key, value)]
    return args


def still_master_extension(width, height):
    """Always `.png`.

    **PNG rather than lossless WebP, chosen for the people who have to look at the output.** Both
    are lossless and WebP is materially smaller at these dimensions, which is what this returned
    before. But a master is the thing a person opens to check the work, hands to a customer, or
    drags into an editor, and PNG opens everywhere without a thought while WebP still meets tools
    that will not preview it. Paying storage for that is the right trade: the file is written once
    and looked at many times, and an artefact nobody can conveniently open is an artefact nobody
    checks.

    It also removes a ceiling. WebP is limited to 16383 pixels per side by the format itself,
    which sits inside the range this worker is aimed at — 12K fits, 16K does not — so the old
    two-format rule had a real edge in it. PNG has no practical limit, so there is one format, one
    path, and no dimension at which the master silently changes type.

    Lossless WebP is still used for the `crop` derives, where CF asked for it by name and the
    files are small evidence images rather than the deliverable.

    The arguments are kept so the signature does not change if a size-dependent rule ever comes
    back.
    """
    del width, height
    return ".png"


def _peak_rss_gb(pid):
    """The largest resident set this process reached, in GiB, or None where it cannot be read.

    **`/proc/<pid>/status`'s `VmHWM`, sampled while the process is alive.** `getrusage`'s
    `RUSAGE_CHILDREN` would be the easy answer and is the wrong one: it reports the maximum across
    every child this worker has ever reaped, so an ffprobe from a previous phase and the encode
    would be indistinguishable — a plausible number about a different process, which is the class
    this project keeps finding. `VmHWM` is that process's own high-water mark and it disappears
    when the process does, which is why it is sampled rather than read at the end.

    None on anything that is not Linux, which is honest: a figure that is absent says nothing and
    a figure that is zero says the encode used no memory.

    **It under-reads on a clip shorter than the encoder's buffering window, and there it under-
    reads totally.** The last sample is taken when the last `write()` returns; nothing samples
    while x264 drains its lookahead and flushes. On a long encode that phase is memory-
    non-increasing — frames are released and none admitted — so the shortfall is near zero. On a
    fixture of a few dozen frames the whole encode happens after the final write, and the number
    describes the buffering footprint rather than the encode. **A reassuringly low peak from a
    small fixture means nothing**, which matters because a small fixture is what somebody
    reaches for when checking that the measurement works.
    """
    try:
        with open("/proc/{}/status".format(pid), encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 2)
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return None
    return None


class MasterWriter:
    """A one-pass ffmpeg encode fed frame by frame.

    Used as a context manager so the pipe is closed and the process reaped on any path,
    including an OOM raised mid-generation by the model upstream of it.
    """

    def __init__(self, path, width, height, fps, identity,
                 audio_source=None, audio_codec=None, audio_limit_s=None,
                 crf=DEFAULT_CRF, preset=DEFAULT_PRESET, codec=None, bit_depth=None,
                 threads=None, sliced_threads=None, rc_lookahead=None,
                 reference_path=None, frame_threads=None, pools=None,
                 delivered_frames=None):
        self.path = path
        #: **Where ffmpeg writes contract §6g's raw reference, or None for an unarmed run.**
        #: A SECOND OUTPUT of this same command rather than a copy of what crossed the pipe: the
        #: pipe carries `rgb24` and ffmpeg converts to the encode's own format before either
        #: encoder sees a pixel, so this is the frames the codec actually encoded. *Same binary,
        #: same swscale, same flags — there is no agreement question between two conversion paths
        #: because there is only one.*
        self.reference_path = reference_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        #: **ffmpeg's own high-water mark, not ours.** The 8K run died at ~46 GiB in x264's
        #: working set while this side held one frame and a cached pair — and nothing reported
        #: it, so the ceiling had to be inferred from a kernel kill. A test path built to find a
        #: memory ceiling that does not report memory has to be run twice to learn anything, and
        #: each run is fifty minutes of A40. None where it cannot be measured.
        self.encoder_peak_rss_gb = None
        #: **`docs/test-plan.md` §18c's diagnostic, and it is a DIAGNOSTIC AND NOT A QUEUE.**
        #: §18c sized the write-behind queue's entire theoretical prize at `compute_s` less the
        #: producer — 18.7 s at 4K — and said whether it is capturable depends on what that 18.7 s
        #: IS. **Uniform means pipe-transfer cost**, which a writer thread with a two-or-three
        #: frame buffer hides behind GPU compute; **bursty means residual encoder backpressure**,
        #: which buffering only smooths. One number cannot tell those apart and a distribution can.
        #:
        #: **The samples are the `stdin.write` ITSELF and nothing around it** — not the length
        #: check, not the lazy start, not the `/proc` read — because §18c's question is about
        #: exactly that call: *"frame threading already took the encoder-side buffering; the
        #: queue's remaining job is only the stdin write itself."* This is deliberately NARROWER
        #: than `write_wait_s`, which times all of `write`.
        #:
        #: **A list rather than a histogram, and the cost is nothing.** Eight bytes a frame
        #: against 50 MiB a frame at 8K — 480 frames is under 4 KB — and order statistics need
        #: the samples. A histogram computed on the way in would have fixed its buckets before
        #: anyone knew the scale, which is the shape this measurement exists to discover.
        self._write_durations = []
        #: Seconds spent in `__exit__` closing the pipe and waiting for ffmpeg — **the encoder's
        #: queued backlog, which no per-write sample can see** (see `__exit__`). None until the
        #: encode has ended, which is also the honest reading for a run reaped before it did.
        self.drain_s = None
        self._proc = None
        self._identity = dict(identity or {})
        self._audio_source = audio_source
        self._audio_codec = audio_codec
        #: Seconds of audio to read, or None to read it all. Bounds the carried track to the
        #: picture without the muxer being allowed to bound the picture to the track.
        self._audio_limit_s = audio_limit_s
        #: Frames the container reports once ffmpeg has exited, or None where it does not say.
        #: The only frame count this class holds that was measured after the encode.
        self.verified_frames = None
        #: Public for the same reason the three below are: **the record has to carry what
        #: actually ran**, and §6a made all five of these caller-settable on the same day.
        self.crf = crf
        self.preset = preset
        #: **Assembled here from validated fields, and this class no longer accepts a string**
        #: (contract §6a). It used to take `x264_params` as an override that "the production path
        #: never passes" — reasoning about an upscale path that left with the excision, leaving
        #: one caller that always passed it. A parameter with one caller and one value is not an
        #: override; it was a frozen constant reached through an argument.
        #:
        #: **Public, because the record has to carry what actually ran.** §6c's campaign attributes
        #: a difference between two arms to the settings that differed, and a corpus that recorded
        #: the DEFAULTS while the run used something else would attribute it to the wrong thing.
        #: **The codec this master is encoded with** (contract §6e), and `None` means the caller
        #: named none — resolved through `envelope` so the default has one home.
        #:
        #: **Public for the same reason the five settings below are: the record has to carry what
        #: RAN.** `docs/archive/instrumentation-archive.md` §15 makes the codec a second vocabulary, and a corpus
        #: that cannot exclude a population includes it — `compute_s` per megapixel, the encoder
        #: arm and the estimator's corpus are each a fit over rows, and two encoders averaged
        #: together is not a noisier corpus but one wrong in a direction that changes with the job.
        #: **The master's bit depth** (contract §6f), and `None` means the caller named none —
        #: resolved through `envelope` so the default has one home, exactly as `codec` is.
        #:
        #: **Public because the record carries it** (`docs/archive/instrumentation-archive.md` §15a), and it
        #: carries the same self-keying invariant the three x264 fields do: `bit_depth: 10`
        #: implies `codec: "h265"` by construction, because §6f refuses the pair at the door.
        self.bit_depth = envelope.DEFAULT_BIT_DEPTH if bit_depth is None else bit_depth
        if self.bit_depth not in PIXEL_FORMATS:
            raise WorkerError(INTERNAL, (
                "the master writer was constructed for bit depth {!r}; `envelope` enumerates {} "
                "and `validation` refuses the rest at the door.").format(
                    self.bit_depth, ", ".join(str(d) for d in sorted(PIXEL_FORMATS))))
        self.codec = resolve_codec(codec)
        if self.codec not in CODEC_LIBRARIES:
            raise WorkerError(INTERNAL, (
                "the master writer was constructed for codec {!r}, which this worker does not "
                "implement. `validation` refuses everything but {} at the door.").format(
                    self.codec, ", ".join(sorted(CODEC_LIBRARIES))))
        #: **THE THREE ARE ABSENT UNDER h265, NOT ZERO AND NOT DEFAULTED** (§6e ruling 2, §15a).
        #: They are x264's vocabulary: `sliced-threads` has no x265 equivalent at all — it is
        #: absent rather than renamed — so a value here beside `codec: h265` would assert that a
        #: parameter x265 does not have took effect. **`None` is what the record then files as an
        #: absent field**, and with ruling 2's refusal at the door `sliced_threads` present on a
        #: row implies h264 by construction.
        #:
        #: **The two parameter strings are the pair, and each is null under the other's codec.**
        #: The column exists on every row either way, which is what lets a reader ask "what
        #: bounded this encode" without first knowing which codec answered.
        if self.bit_depth != envelope.DEFAULT_BIT_DEPTH and self.codec != "h265":
            raise WorkerError(INTERNAL, (
                "the master writer was constructed for {}-bit under codec {!r}; §6f refuses that "
                "pair at the door because h264 High10 is hardware-decoded almost nowhere, so "
                "reaching here is a plumbing defect and not a request.").format(
                    self.bit_depth, self.codec))
        if self.codec == "h265":
            if threads is not None or sliced_threads is not None or rc_lookahead is not None:
                raise WorkerError(INTERNAL, (
                    "codec h265 reached the master writer carrying x264's thread settings; §6e "
                    "ruling 2 refuses them at the door precisely so they cannot be recorded "
                    "beside an encode that never saw them."))
            self.x264_params = None
            #: **Assembled from named fields for a reason sharper than §6a's** — see
            #: `x265_params`: x265 discards an unknown NAME without a word, so a bound is not in
            #: force merely because the job succeeded.
            #:
            #: **§6i's two levers, resolved HERE and only here.** `None` from the caller means
            #: absent; `x265_threading` applies the default and says per field which it was.
            #:
            #: **THE VALUES DEPEND ON THE FRAME AGAIN, SO THE HOOK COMES BACK.** *§6h derived
            #: `frame-threads` from the delivered frame and had to re-derive whenever
            #: `set_frame_size` moved it; §6i struck the keying and the comment here recorded
            #: that a whole class of staleness had been removed rather than guarded.* **§11 keys
            #: on HEIGHT, so the dependency exists once more and `set_frame_size` re-runs this.**
            #: *A value computed from a size, kept across a change of that size, is the defect
            #: that comment was celebrating the absence of — it does not stop being one because
            #: nothing calls the setter today.*
            self._x265_caller = (frame_threads, pools)
            self._delivered_frames = delivered_frames
            self._resolve_x265()
            self.threads = None
            self.sliced_threads = None
            self.rc_lookahead = None
        else:
            #: The module constants stand in only where nothing was passed. **The production
            #: path never reaches them** — `routec` resolves all three through §6d's table before
            #: constructing this — so they are the answer for a direct caller and not a second
            #: site that chooses encoder settings.
            threads = DEFAULT_THREADS if threads is None else threads
            sliced_threads = (DEFAULT_SLICED_THREADS if sliced_threads is None
                              else sliced_threads)
            rc_lookahead = DEFAULT_RC_LOOKAHEAD if rc_lookahead is None else rc_lookahead
            self.x264_params = x264_params(threads=threads, sliced_threads=sliced_threads,
                                           rc_lookahead=rc_lookahead)
            self.x265_params = None
            #: **Null on the x264 path, and defined rather than absent.** The two parameter
            #: strings above are already a pair where each is null under the other's codec, for
            #: the stated reason that the column exists on every row either way. A `frame_threads`
            #: that existed only under h265 would make a reader of the attribute crash on an h264
            #: encode — the shape `band` failed at four hours ago, one file over.
            # **§6i's mirror of §6e ruling 2, and it was missing.** The h265 branch above
            # REFUSES x264's three fields reaching this writer, with the stated reason that
            # dropping them silently is the exact shape ruling 2 refused — a bound the caller
            # believes is in force and the encoder never saw. The h264 branch took x265's two and
            # discarded them without a word, so the guard existed in one direction only.
            # *`validation` closes the request path; this closes the direct-caller one, which is
            # where §6h's calibration was meant to live and where nothing else would notice.*
            # Found in review.
            crossed = {name: value for name, value in
                       (("frame_threads", frame_threads), ("pools", pools)) if value is not None}
            if crossed:
                raise WorkerError(INTERNAL, (
                    "codec {} reached the master writer carrying {} — contract §6i refuses "
                    "x265's threading levers under x264 at the door because x264 has no such "
                    "parameters, so a value here would be recorded beside an encode that never "
                    "saw it.").format(
                        self.codec, ", ".join("{}={!r}".format(k, v)
                                              for k, v in sorted(crossed.items()))))
            self.frame_threads = None
            self.frame_threads_basis = None
            self.pools = None
            self.pools_basis = None
            #: Kept individually as well as in the assembled string, so the record can report a
            #: field without anyone parsing the string back apart. **The string is what ran;
            #: these are what was asked for**, and they cannot disagree because one is built from
            #: the other.
            self.threads = int(threads)
            self.sliced_threads = bool(sliced_threads)
            self.rc_lookahead = int(rc_lookahead)

    def _resolve_x265(self):
        """§6i's two levers against THIS frame. **Called at construction and on every resize.**"""
        frame_threads, pools = self._x265_caller
        values, basis = x265_threading(frame_threads, pools, delivered_height=self.height,
                                       delivered_frames=self._delivered_frames)
        self.frame_threads = values["frame_threads"]
        self.pools = values["pools"]
        self.frame_threads_basis = basis["frame_threads_basis"]
        self.pools_basis = basis["pools_basis"]
        #: **Assembled from named fields for a reason sharper than §6a's** — see `x265_params`:
        #: x265 discards an unknown NAME without a word, so a bound is not in force merely
        #: because the job succeeded.
        self.x265_params = x265_params(frame_threads=self.frame_threads, pools=self.pools)

    def set_frame_size(self, width, height):
        """Adopt the size the model actually produced, before ffmpeg is started.

        **This is why ffmpeg is started lazily.** `-s` on a rawvideo input is a promise about
        bytes that carry no shape of their own: declare 8210×4320 and feed 8208×4320, and ffmpeg
        does not complain — it reads across frame boundaries and writes a master that shears
        progressively, exiting 0 with a plausible file size. The still path caught the same
        disagreement as a byte-count refusal; this path would not have caught it at all.
        """
        if self._proc is not None:
            raise WorkerError(INTERNAL, "the master's frame size was changed after ffmpeg started")
        self.width, self.height = int(width), int(height)
        self._identity["cf_output"] = "{}x{}".format(self.width, self.height)
        # **AND THE x265 SETTINGS ARE RE-DERIVED, because §11 keys them on height.** *Adopting a
        # new frame without re-running this would encode at the pools count of a frame that was
        # never written, and the record would carry it.*
        if self.codec == "h265":
            self._resolve_x265()

    def _build_command(self):
        width, height, fps = self.width, self.height, self.fps
        identity = self._identity
        audio_source, audio_codec = self._audio_source, self._audio_codec
        crf, preset = self.crf, self.preset
        path = self.path

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # Input 0: raw frames on stdin, exactly as the model produces them.
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(width, height), "-r", str(fps), "-i", "-",
        ]

        carry_audio = audio_source is not None
        if carry_audio:
            # **The trim goes on the audio input, never on the output.** `-t` before `-i` bounds
            # how much of *that* input is read and can only ever shorten the audio.
            if self._audio_limit_s:
                command += ["-t", "{:.6f}".format(float(self._audio_limit_s))]
            command += ["-i", audio_source]

        command += ["-map", "0:v:0", "-c:v", CODEC_LIBRARIES[self.codec],
                    "-preset", preset, "-crf", str(crf),
                    # §6f. `yuv420p` at 8 and `yuv420p10le` at 10, which is what makes the
                    # encode `main10` — see `PIXEL_FORMATS`.
                    "-pix_fmt", pixel_format(self.bit_depth)]
        # `-x26N-params` rather than more `-preset` flags: the preset stays the caller's and
        # these are the specific knobs §6a and §6e name, so a reader can see which of the two
        # moved.
        #
        # **CONDITIONAL, AND IT IS NOT PREVENTING A CRASH — IT IS PREVENTING A SILENT UNBOUNDED
        # ENCODE.** §6e originally held that `-x264-params` under `libx265` kills ffmpeg before
        # the first frame, so a naive port would fail every h265 job loudly. **Measured, and it
        # is false** (`docs/gate-findings.md` F-2026-08-28-7): ffmpeg emits *"Codec AVOption
        # x264-params ... has not been used for any stream"* at WARNING level and carries on,
        # exit 0, master written. **This worker encodes at `-loglevel error`, where that line
        # does not appear at all.**
        #
        # *So the failure this branch prevents is a delivered 8K master encoded with NO BOUND and
        # nothing anywhere saying so — no error, no non-zero exit, a plausible byte count, a
        # passing witness — on the one path §6e's ruling 1 exists to protect. A crash is loud,
        # cheap, and caught by the first h265 job anybody runs; this is not.*
        if self.codec == "h265":
            command += ["-x265-params", self.x265_params]
        else:
            command += ["-x264-params", self.x264_params]

        if carry_audio:
            # `?` makes the mapping optional, so a source whose audio stream vanished between the
            # probe and the mux does not fail the encode of an expensive master.
            command += ["-map", "1:a:0?"]
            command += ["-c:a", "copy"] if audio_codec in MP4_NATIVE_AUDIO else \
                       ["-c:a", "aac", "-b:a", "192k"]
            # **No `-shortest` here, and the reason is a delivered defect.** The flag was added
            # to stop a longer source track leaving an audio-only tail past the last frame, under
            # the comment "the video stream is authoritative". It is symmetric and does the
            # opposite: it ends the output when *any* input ends, so an audio track shorter than
            # the video truncates the video. On 2026-08-15 a 1.984 s AAC track against 2.000 s of
            # picture cost two frames of a delivered master; reproduced locally at 45 of 48 frames
            # with the flag and 48 of 48 without, from the same source and the same mux.
            #
            # AAC frames are 1024 samples, so a track almost never lands exactly on the video's
            # duration and is usually a fraction short. Every audio job was exposed. It had never
            # shown because every fixture that had been run at size was silent.
            #
            # The tail it was defending against is handled by `_audio_limit_s` above, which cannot
            # touch the picture. Where no limit is known the tail is accepted: audio playing past
            # the last frame is a cosmetic fault, and a master missing frames is not.

        command += _identity_tags(identity)
        # Two flags, and the second is not optional despite looking like a detail.
        #
        # `+faststart` puts the moov atom at the front, in this pass. Never a later one.
        #
        # `+use_metadata_tags` is what makes the identity tags above actually exist. **The MP4
        # muxer silently discards any metadata key it does not recognise** — `comment` and
        # `title` survive, `cf_request_id` does not — with a zero exit code and no warning.
        # Measured 2026-08-12 (`docs/decisions.md` 3.3). Without it the whole "a file found in
        # R2 with no job and no manifest still says what it is" mechanism is absent from every
        # file while every check around it passes.
        command += ["-movflags", "+faststart+use_metadata_tags", path]

        # ── contract §6g's reference, as a SECOND OUTPUT on this same command ────────────────
        #
        # **Appended after the master's filename, which is what makes it a second output rather
        # than more options on the first.** ffmpeg reads a filename as the end of an output spec,
        # so everything from here to the next filename applies only to the reference.
        #
        # **`-map 0:v:0` again, because mapping is per-output.** Without it this output would take
        # ffmpeg's default stream selection and, on a job carrying audio, would try to write an
        # audio stream into a rawvideo file.
        #
        # **The SAME `-pix_fmt` the master is encoded at**, which is the whole point: 10-bit
        # delivers a 10-bit reference and the scores are taken in the space the codec worked in.
        # *`PIXEL_FORMATS` is read once, above, so the two cannot drift apart.*
        if self.reference_path is not None:
            command += ["-map", "0:v:0", "-f", "rawvideo",
                        "-pix_fmt", pixel_format(self.bit_depth), self.reference_path]

        return command

    def __enter__(self):
        return self

    def _start(self):
        self.command = self._build_command()
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise WorkerError(INTERNAL, "could not start ffmpeg: {}".format(exc))

    def write(self, frame_bytes):
        """One frame, already `rgb24` and `width × height × 3` bytes."""
        # **The check the still path had and this one did not.** rawvideo carries no shape, so a
        # frame of the wrong length is not an error to ffmpeg — it is the first bytes of the next
        # frame, and the master shears from that point on while the process exits 0. Cheap to
        # check, and it turns the worst failure mode in this file into a refusal.
        expected = self.width * self.height * 3
        if len(frame_bytes) != expected:
            raise WorkerError(INTERNAL, "frame {} is {} bytes, expected {} for {}x{}".format(
                self.frames_written, len(frame_bytes), expected, self.width, self.height))
        if self._proc is None and not self.frames_written:
            self._start()
        if self._proc is None or self._proc.poll() is not None:
            raise WorkerError(INTERNAL, self._died("ffmpeg exited before the frames did"))
        started = time.perf_counter()
        try:
            self._proc.stdin.write(frame_bytes)
        except BrokenPipeError:
            raise WorkerError(INTERNAL, self._died("ffmpeg closed the pipe"))
        # **Banked after the write returned and not in a `finally`**, for `staging.released()`'s
        # reason one module over: a write that raised did not transfer a frame, and a duration
        # recorded for it would put the pipe's death in a distribution of the pipe's throughput.
        self._write_durations.append(time.perf_counter() - started)
        self.frames_written += 1
        # **Sampled here because this is where the loop already blocks.** `write` returns when
        # the pipe accepts the frame, which is exactly when the encoder is working — so the
        # samples land across the whole encode without a thread, and the maximum survives the
        # process that produced it. One `/proc` read per frame is noise against a 50 MiB write.
        peak = _peak_rss_gb(self._proc.pid)
        if peak is not None and peak > (self.encoder_peak_rss_gb or 0.0):
            self.encoder_peak_rss_gb = peak

    def write_distribution(self):
        """§18c's question as numbers: **is the write cost uniform or bursty?** Never raises.

        Returns None where nothing was written. Otherwise the order statistics, plus the two
        quantities that actually discriminate:

        - **`p99_over_p50`** — how much worse the tail is than the middle. A pipe moving a fixed
          number of bytes into a buffer that is always ready is flat; an encoder that stops to
          think is not.
        - **`slowest_n_share`, with `slowest_n` beside it** — the fraction of the TOTAL spent in
          the slowest `n` writes, where `n` is one percent of the count ROUNDED UP. This is the
          one that decides what a buffer would buy, and it is not the same question as the ratio
          above: a tail that is ten times the median but holds two percent of the time is a tail
          a queue cannot pay for. **The count is published rather than the word "1%"**, because
          at these frame counts the two are not the same thing and the difference biases the
          answer toward not building the queue.

        **`first_ms` is reported apart from every other sample and is excluded from none of
        them.** ffmpeg is spawned lazily inside the first `write` (`_start`, just above), so the
        first frame's write is the one racing the encoder's own start-up. It is a real cost the
        job pays; it is not a sample of steady-state throughput, and a reader who cannot see it
        separately would read a one-frame artefact as a tail.

        **THE READINGS EXPIRE AT x265 AND NOTHING MAY BE BANKED ON THEM.** §18c's own note: they
        are CRF- and codec-specific, and every value in the encoder's parameter string names a
        parameter the next codec does not have. This is why the distribution is PRINTED and files
        no record field — a log entry is read by whoever went looking for it, and a corpus column
        outlives its meaning and gets averaged in by someone who never read §18c.
        """
        # **NO `except` HERE, AND THAT IS THE FIX RATHER THAN AN OMISSION.** This used to swallow
        # and return None, which is also what it returns when nothing was written — so a failed
        # computation printed *"no frames were written"* through the caller, a positive false
        # statement about a run that wrote 480 frames, and the caller's own honest error branch
        # was unreachable because this could not raise. **One return value standing for two
        # states is the defect this record has been fixed for before.** The never-raises posture
        # is kept where it matters, at the call site in a `finally`:
        # `routec._print_write_distribution` catches and says which of the two happened.
        import math  # noqa: PLC0415 — one call, on a path that runs once per encode

        first = self._write_durations[0] if self._write_durations else None
        # **THE FIRST WRITE IS EXCLUDED FROM EVERY STEADY-STATE STATISTIC, AND IT HAD NOT BEEN.**
        # ffmpeg is spawned lazily inside the first `write` (`_start`, just above), so that sample
        # is racing the encoder's own start-up. It was reported apart AND left in the totals,
        # which made the numbers §18c decides on describe something a queue cannot remove:
        # `[2.0] + [0.04] * 479` reads `p99/p50 = 1.0` — perfectly flat — while
        # `slowest_n_share` reads 0.102, so ten percent of "the tail a buffer would buy back" is
        # one process spawn. **Every number correct, the ten percent belonging to something
        # else.** `first_ms` still carries it, which is the honest home for it.
        samples = sorted(self._write_durations[1:])
        count = len(samples)
        if not count:
            # **One write is not a distribution and this says so rather than describing itself.**
            # A single-frame encode has nothing left once the start-up sample is set aside.
            return None if first is None else {
                "samples": 0, "first_ms": round(1000.0 * first, 3),
                "note": "one write only; the start-up sample is excluded and nothing remains",
            }
        total = sum(samples)

        def at(fraction):
            # Nearest-rank, which is the only percentile definition that returns a sample that
            # was actually observed. Interpolating between two writes invents a duration, and
            # this measurement's whole purpose is telling a real tail from a smooth one.
            index = min(count - 1, max(0, int(round(fraction * (count - 1)))))
            return samples[index]

        p50, p90, p99 = at(0.50), at(0.90), at(0.99)
        # **CEILING, AND THE COUNT IS REPORTED BESIDE THE SHARE.** This was `count // 100`, a
        # floor, and the log line called the result "the slowest 1%" — at 480 writes it was 4
        # samples, 0.83%; at any count under 100 it was one sample and could be several percent.
        # **The bias ran toward a smaller share, which is toward "the queue is not worth
        # building"** — one of the two answers §18c exists to choose between, so a measurement
        # that leans that way by an arithmetic accident is the wrong one to leave in.
        slowest_n = max(1, int(math.ceil(count / 100.0)))
        top = samples[count - slowest_n:]
        return {
            # **The steady-state count, which is one fewer than the frames written.** Named so
            # nobody reconciles it against `frames_written` and finds an off-by-one.
            "samples": count,
            "total_s": round(total, 3),
            "mean_ms": round(1000.0 * total / count, 3),
            "min_ms": round(1000.0 * samples[0], 3),
            "p50_ms": round(1000.0 * p50, 3),
            "p90_ms": round(1000.0 * p90, 3),
            "p99_ms": round(1000.0 * p99, 3),
            "max_ms": round(1000.0 * samples[-1], 3),
            # **Outside every statistic above.** The one sample that paid for ffmpeg's start-up.
            "first_ms": round(1000.0 * first, 3),
            # None rather than a large number where the median is zero — a ratio against zero is
            # not a big ratio, it is an absent one, and `inf` in a log line reads as a
            # measurement.
            "p99_over_p50": None if p50 <= 0 else round(p99 / p50, 2),
            "slowest_n": slowest_n,
            "slowest_n_share": None if total <= 0 else round(sum(top) / total, 4),
        }

    def _died(self, why):
        stderr = b""
        if self._proc is not None:
            try:
                stderr = self._proc.stderr.read() or b""
            except Exception:  # noqa: BLE001 — we are already reporting a failure
                pass
        detail = stderr.decode(errors="replace")[-400:].strip()
        return "{}{}".format(why, ": " + detail if detail else "")

    def __exit__(self, exc_type, exc, traceback):
        if self._proc is None:
            return False
        # **THE DRAIN, AND WITHOUT IT §18c's DISTRIBUTION ANSWERS THE WRONG QUESTION.** Every
        # sample in `_write_durations` is a `stdin.write`, and the encoder's queued backlog is
        # not paid there — it is paid HERE, in `close()` then `wait()`, after the last frame has
        # been handed over. At 8K with a lookahead the encoder can be holding tens of 50 MiB
        # frames, and all of that settles in this call.
        #
        # **So a run with real backpressure can print a perfectly flat distribution**, and §18c's
        # decision rule reads flat as *"pipe-transfer cost, a small writer buffer hides it"* —
        # the wrong answer, produced by numbers every one of which is correct. *Right number,
        # different subject.*
        #
        # Measured rather than argued, and reported beside the distribution so the two are read
        # together. **Not banked to `write_wait_s`**: that field is `routec`'s loop clock and its
        # boundaries are the loop's, and moving a quantity into a stage after the corpus was
        # banked against it would make two populations wearing one field name.
        drain_started = time.perf_counter()
        try:
            self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        self._proc.wait()
        self.drain_s = time.perf_counter() - drain_started
        # An exception on the way in owns the failure; do not replace it with one about ffmpeg,
        # which most likely died *because* of it. The original diagnosis is the useful one —
        # especially for an OOM, where the exception carries the phase and the allocation that
        # failed and no log gives better.
        if exc_type is not None:
            return False
        if self._proc.returncode != 0:
            raise WorkerError(INTERNAL, self._died(
                "ffmpeg exited {}".format(self._proc.returncode)))
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            raise WorkerError(INTERNAL, "ffmpeg wrote no output to {}".format(self.path))

        # **The first count this worker takes on the far side of the encode**, and the reason it
        # exists is that every other one is on the near side. `decoded_in` and `written_out` are
        # both counters in this process, so `frames_match` compares the write loop to itself and
        # passes on a master the muxer silently truncated. It did exactly that on 2026-08-15.
        #
        # Refusing is the right end for it. A short master is not a degraded success — it is a
        # video that plays correctly and is missing frames, which is the one failure a caller
        # cannot detect downstream either. `internal` is honest: the request was fine and this
        # worker wrote the wrong file.
        self.verified_frames = probe.written_frame_count(self.path)
        if self.verified_frames is not None and self.verified_frames != self.frames_written:
            raise WorkerError(INTERNAL, (
                "the master was written with {} frames but the file holds {} — the encode lost "
                "{} frame(s) after the write loop. The file plays; it is short.").format(
                    self.frames_written, self.verified_frames,
                    self.frames_written - self.verified_frames))
        return False


def _run(command, what):
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise WorkerError(INTERNAL, "could not start ffmpeg while {}: {}".format(what, exc))
    if completed.returncode != 0:
        raise WorkerError(INTERNAL, "ffmpeg failed while {}: {}".format(
            what, completed.stderr.decode(errors="replace")[-400:].strip()))
    return completed
