"""The request surface: the codec, the bit depth, the retime's own fields.

**THE AUTHORITY IS `cf-rife-project/docs/api.md` §1a AND `docs/decisions.md` §12, AND THERE IS NO
ORACLE FOR THIS SURFACE.** *This docstring named `fable/envelope_oracle.py` until 2026-09-04. No
such file has ever existed — under that name or any other — and no `fable/` directory exists in
either rife repository; it is residue from the upscale project, carried across in the seed.*
**`interpolate.py`'s equivalent cite was a path error and this one was not**: that oracle is real
and only its address was wrong.

**SO THE "TWO INDEPENDENT STATEMENTS OF ONE RULE" PROPERTY IS NOT HELD FOR WHAT THIS FILE
COMPUTES, AND SAYING SO IS THE POINT OF THIS PARAGRAPH.** *Nothing states the derivation rules a
second time, so a defect in `derive` is caught by reading and by the delivered artefacts, and not
by disagreement with an independent statement.* **A file claiming an authority that does not exist
is the same defect as a check that cannot fail: it reads as covered.** The gap is named here rather
than papered over, because the day someone goes looking for the oracle is the day they discover
what this surface actually rests on, and they should not discover it then.

**What DOES exist is `gate_scripts/request_witness.py`, and it is narrower than an oracle in a way
that matters.** *Its field lists are TRANSCRIBED from `api.md` §1a rather than imported from this
module (`request_witness.py:35`), so it is genuinely independent* — but it grades the REFUSAL
surface, which names are accepted and which are declined and with what code. **It says nothing
about what `derive` returns for a request it accepts**, which is most of this file.

**Kept out of `validation.py` deliberately.** That module is release 2's surface and is large; a
release-3 block folded into it would be indistinguishable from the fields that have always been
there, and the one property protecting production — that a request carrying none of these fields
behaves exactly as it did — is easiest to keep true when the new surface is one file that can be
read end to end.
"""
from errors import (FIELD_NOT_SUPPORTED, INVALID_FIELD_VALUE, MISSING_REQUIRED_FIELD,
                    WorkerError)

#: **`source` means "match the input's codec"**, which is a release-3 field and not a default.
CODECS = ("h264", "h265", "source")

#: **Unchanged, so an omitted field cannot move anything.** Every release-2 caller encodes h264
#: today and must still encode h264 after this ships — `default_off_identity` is the assertion.
DEFAULT_CODEC = "h264"

#: **THERE IS NO FLAT DEFAULT FOR THE TWO THREADING LEVERS ANY MORE.** *`DEFAULT_FRAME_THREADS`
#: and `DEFAULT_POOLS` lived here — 1 and 16 — as the one home for a caller-facing default, and
#: `encoder.x265_threading` read them.* **CF's §11 ruling refutes a flat default outright**: 1 and
#: 16 served neither 1080p nor 8K, and every unset job at one of those areas ran at a setting
#: nothing had measured — *1080p h265 shipped at `frame-threads=1` and read as a 5x codec penalty
#: for a week.* **The home is now `encoder.X265_RULED` plus the CTU-row rule**, and the constants
#: are gone rather than left beside a table that overrules them. *A named default that is not the
#: default is worse than none: it is the first thing a reader believes.*
#: The ranges. **Bounded rather than open**, for §6a's reason one codec over: these settings decide
#: how much the encoder allocates on a path §1 says has no host guard, so an unbounded value is a
#: memory bound handed to the caller. *`frame-threads` above a small number multiplies the frames
#: in flight; `pools` above the visible core count buys nothing and costs scheduling.*
#: Every key `params.output` may carry, SPLIT BY WHICH LIST IT IS ON. **One home, and the
#: refusal in `derive` reads both** — a field added without joining one of them is refused by
#: name, which is the outcome a caller can act on.
#:
#: **THE TWO THREADING LEVERS ARE DEBUG NAMES AS OF THE STRICT SURFACE (CF, 2026-09-02).** *They
#: are encoder internals: a caller with no basis to choose between wavefront settings should not
#: be choosing, and the worker's own defaults govern a production job.* **Sending either without
#: `debug: true` is refused BY NAME rather than ignored**, which is the distinction the whole
#: strictness rule rests on.
OUTPUT_PRODUCTION_FIELDS = ("codec", "bit_depth")
OUTPUT_DEBUG_FIELDS = ("frame_threads", "pools")
#: *The union used to be the one list `derive` checked against. It is gone rather than left as a
#: third name for a fact the two lists above already hold* — nothing read it once the split
#: landed, and an unread union is the copy that goes stale without anyone noticing.


def refuse_field(name, where, gated):
    """The two refusals a strict request can produce, spelled once for every level.

    **THEY ARE DIFFERENT FACTS ABOUT THE REQUEST AND THE MESSAGES SAY WHICH.** *A DEBUG name
    arriving without `debug: true` is a name this contract DEFINES, arriving in the wrong state;
    an unlisted name is one the contract does not define at all.* **Calling a defined field
    unknown would be a false statement in an error message** — and the caller's next action
    differs: one adds a flag, the other fixes a typo or stops sending the field.

    **CF RULED BOTH THE BEHAVIOUR AND THE STRING, 2026-09-02, AND THESE TWO MESSAGES ARE WHAT WAS
    RULED.** *An earlier version of this paragraph said the string was still open and that what
    stood here was the gate's reading — true when it was written and false from the moment the
    ruling landed.* **The reader most likely to open this docstring is the one deciding whether a
    message may change**, and a sentence telling them the question is still open is the worst
    place in the file for that to be wrong.
    """
    if gated:
        raise WorkerError(
            FIELD_NOT_SUPPORTED,
            "field '{}' {} is a debug field and this request did not set 'debug: true'. It is "
            "defined by this contract and refused in this state rather than ignored: send "
            "'debug: true' at the top level to use it, or drop it.".format(name, where))
    raise WorkerError(
        FIELD_NOT_SUPPORTED,
        "field '{}' {} is not a field this worker accepts. Every field in a request is named by "
        "the contract or the request is refused — nothing is silently ignored.".format(
            name, where))


def refuse_unlisted(present, production, debug, where, debug_on):
    """Every name in `present` is production, or debug with the flag, or REFUSED.

    **The order matters: a debug name is checked before it is called unknown**, so the caller who
    sent a real field in the wrong state is never told their field does not exist.
    """
    for name in sorted(present):
        if name in production:
            continue
        if name in debug:
            if debug_on:
                continue
            refuse_field(name, where, gated=True)
        refuse_field(name, where, gated=False)

FRAME_THREADS_RANGE = (1, 16)
POOLS_RANGE = (1, 64)

#: **`main10` IS A PROFILE AND NOT A CODEC NAME** (contract §6f). CF's ruling was *"we can switch
#: main10 on and off as needed"*, and a switch is a field — folding it into `CODECS` as a third
#: value would make one enumeration answer two questions, which encoder and at what precision.
#: *§6d's lesson one surface over: a value that quietly stands for two states is the defect this
#: project has been fixed for twice.*
#:
#: **10 is REFUSED UNDER h264 and the refusal lives in `validation`, not here.** This module owns
#: the enumeration — is it a bit depth at all — and the cross-field rule is a capability question
#: about the codec, which is the door's. Same split as `codec`: `source` is enumerable here and
#: refused there.
BIT_DEPTHS = (8, 10)

#: **Unchanged, so an omitted field cannot move anything** — the reason `DEFAULT_CODEC` gives.
#: Every record written before §6f was `yuv420p` from an unconditional literal, so absence is 8
#: by construction (`docs/archive/instrumentation-archive.md` §15a).
DEFAULT_BIT_DEPTH = 8

#: **`DEFAULT_TARGET_FPS` IS GONE AND WAS DELETED RATHER THAN LEFT UNREAD** (CF, 2026-09-04).
#: It was `60.0`, and it meant a request that named no output frame rate was SERVED at a rate
#: nobody asked for — on a worker whose entire capability is choosing that rate. *A default for
#: the one field the job is FOR is not a convenience; it is the worker guessing the answer.*
#:
#: **The deletion is half the ruling and not a tidy-up.** A constant left sitting beside a rule
#: that no longer reads it is the next reader's evidence that the rule does not apply — the same
#: hazard the `OUTPUT_*_FIELDS` split names two screens up, where an unread union was removed
#: rather than kept as a third name for a fact two lists already held.
#:
#: **IT IS A BREAKING CHANGE TO A CONTRACT A CLIENT IS HOLDING.** *`api.md` already says
#: REQUIRED and carries a transitional line naming the live default; that line comes out when
#: the image enforcing this ships.*

#: **`target_fps` AND `snap_tolerance` SIT IN `params` AND THE `interpolate` BLOCK IS GONE.**
#: *That block was the seeded worker's switch between upscaling and retiming; with one capability
#: it discriminates nothing, and it cost `target_fps` a refusal at the top level for want of it.*
#: **`route` went with it and was DELETED rather than moved** — it named where interpolation sat
#: RELATIVE TO THE UPSCALE, and the same argument that deletes the block deletes it. (CF,
#: 2026-09-02.)
RETIME_FIELDS = ("target_fps", "snap_tolerance")
#: **`diagnostics` READS THIS RATHER THAN KEEPING ITS OWN COPY.** *It had one — two spellings of
#: one tuple in two modules — which is the drift `ladder.py` argues against for the ruled table.*


def _threading_lever(output, name, bounds):
    """One of §6i's two x265 threading fields: `None` when absent, a bounded int when sent.

    **The type is checked before the range, and a range test alone would not do it.** `True == 1`
    and `2.0 == 2` in Python, so a bool or a float satisfies a numeric comparison and reaches the
    record wearing a type the field does not have — which is `bit_depth`'s reasoning three fields
    up, and the shared law's float-identity clause arriving where nobody would check.
    """
    if name not in output:
        return None
    value = output[name]
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'output.{}' must be an integer in {}..{}; got {!r}. It is an x265 threading "
            "lever and it is bounded because these settings decide what the encoder allocates on "
            "a path with no host guard.".format(name, low, high, value))
    return value


def derive(params, debug=False):
    """`params` in, a normalised config out, or `WorkerError`.

    **THE UPSCALE PATH IS GONE FROM HERE, AND WITH IT THE SHAPE OF THIS FUNCTION.** *It used to
    answer "is this a retime or an upscale" first and everything else after; `upscale: false` was
    the explicit retime spelling, an omitted `upscale` meant an upscale, and `validation` then
    refused that default request shape by name.* **One capability means the question no longer
    discriminates**, so `upscale`, the two sizing fields, the `interpolate` block and `route` are
    all DELETED rather than defaulted, and `target_fps` and `snap_tolerance` sit in `params`
    where every other per-job field already sat. (CF, 2026-09-02.)

    **`release_2_equivalent` AND `default_off_identity` WENT WITH IT.** *They asserted that a
    request carrying none of release 3's fields behaves as release 2 did — and a release-2
    request is an upscale request, which is now refused at the door.* **An assertion about a
    request shape the worker no longer accepts cannot fail and cannot pass**, which is worse than
    no assertion: it reads as protection.

    `debug` gates the DEBUG names in `params.output`; the levels above are `validation`'s, which
    is the module that reads the flag off the top level.
    """
    params = dict(params or {})
    # **TYPE-CHECKED BEFORE IT IS WALKED, AND IT WAS NOT.** *`dict(params["output"])` on an `int`
    # raises `TypeError` and on a string raises `ValueError` — neither is a `WorkerError`, so a
    # malformed encode block left `validate` as an unhandled exception and `handle` reported a
    # request-shape problem as a worker fault.* **This is the level the strict rule was strict
    # about NAMES and silent about TYPE**, while the destination block one module over has
    # type-checked itself all along. Found in review; pre-existing, and inside the item that
    # claims strictness at every level.
    raw_output = params.get("output")
    if raw_output is not None and not isinstance(raw_output, dict):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'params.output' must be an object holding the encode settings ({}); got "
            "{!r}. The top-level 'output' is the DESTINATION and is a different object.".format(
                ", ".join(OUTPUT_PRODUCTION_FIELDS), raw_output))
    output = dict(raw_output or {})
    # **UNKNOWN KEYS IN `params.output` ARE REFUSED BY NAME, AND SO ARE DEBUG KEYS WITHOUT THE
    # FLAG.** Nothing checked this sub-object until §6i, and the gap was invisible while its only
    # fields were enums a typo turned into a refusal anyway.
    #
    # **§6i PUT TWO NUMBERS BEHIND IT AND THAT CHANGES THE FAILURE.** The x265 spelling is
    # `frame-threads`, hyphenated, in every comment and document in this tree — so the natural
    # typo is `"frame-threads": 4`, which would have validated, encoded at the default, and filed
    # `frame_threads: 1, frame_threads_basis: "default"`. No error, a plausible byte count, a
    # passing witness, and a swept arm recorded as a production default. *That is verbatim the
    # failure `validation`'s h264 refusal says it exists to prevent — "a dropped bound leaves you
    # believing a bound is in force" — and `x265_params`' own F-2026-08-28-7 hazard, where x265
    # discards an unknown NAME without a word, re-made one layer up at the request.* Found in
    # review.
    refuse_unlisted(output, OUTPUT_PRODUCTION_FIELDS, OUTPUT_DEBUG_FIELDS,
                    "in 'params.output'", debug)

    codec = output.get("codec", DEFAULT_CODEC)
    if codec not in CODECS:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'output.codec' must be one of {}; got {!r}".format(CODECS, codec))

    # **§6i's two levers, parsed here beside `codec` and `bit_depth` and range-checked here too.**
    # *The CROSS-FIELD rule — both are h265-only — is `validation`'s, exactly as `bit_depth`'s
    # 10-is-h265-only is: this file enumerates and ranges, that one refuses combinations.*
    #
    # **`None` when absent, and the DEFAULT is applied in the encoder rather than here.** That is
    # §6d's rule and the one §6b's surviving clause rests on: absence must stay distinguishable
    # from a value all the way to the branch, or the branch cannot tell "the caller asked for 1"
    # from "the caller asked for nothing". *Resolving to 16 here would make `pools_basis` a
    # guess.*
    frame_threads = _threading_lever(output, "frame_threads", FRAME_THREADS_RANGE)
    pools = _threading_lever(output, "pools", POOLS_RANGE)

    # **The enumeration only. Contract §6f's cross-field rule — 10 is h265-only — is
    # `validation`'s**, exactly as `codec: source` is enumerable here and refused there.
    #
    # **THE TYPE IS CHECKED BEFORE THE MEMBERSHIP, AND A MEMBERSHIP TEST ALONE WOULD NOT DO IT.**
    # `8.0 in (8, 10)` is True and `True == 1` — Python compares across numeric types — so a
    # float or a bool can satisfy `in BIT_DEPTHS` and reach the record wearing a type the field
    # does not have. *`8.0` on a row is the shared law's float-identity clause arriving in a
    # field nobody would think to check, and it would compare equal in the kit too.*
    bit_depth = output.get("bit_depth", DEFAULT_BIT_DEPTH)
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) \
            or bit_depth not in BIT_DEPTHS:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'output.bit_depth' must be one of {}; got {!r}. It is a PROFILE switch and "
            "not a codec name — 10 selects HEVC main10 and is refused on h264.".format(
                BIT_DEPTHS, bit_depth))

    # **`params.target_fps`, and the block it used to live in is gone.** *`envelope` REFUSED this
    # name at the top level until today, for want of the `interpolate` object it belonged to —
    # so the field a caller most obviously wants to send was the one spelling the contract would
    # not take.*
    # **REQUIRED, and absent is its own refusal rather than a bad value** (CF, 2026-09-04). The
    # two codes are two different facts about the request and the caller's next action differs:
    # `missing_required_field` says send the field, `invalid_field_value` says the field you sent
    # is not a rate. *Collapsing them into the positive-number check below would tell a caller
    # who sent nothing that their nothing was not positive.* Same split `refuse_field` makes one
    # screen up between a gated name and an unknown one.
    #
    # **`is None` rather than `not in params`, which is `validation._require`'s test exactly**
    # (`validation.py:239`). An explicit `"target_fps": null` is a caller declining to choose,
    # which is the state this ruling refuses, and reading it as "present, therefore check the
    # type" would refuse it with the wrong code.
    target_fps = params.get("target_fps")
    if target_fps is None:
        raise WorkerError(
            MISSING_REQUIRED_FIELD,
            "field 'target_fps' is required in 'params'. It is the rate this worker exists to "
            "produce and there is no default: a job that does not name one has not said what to "
            "do.")
    # **Unchanged by the ruling above**, and it still rejects `bool` before the numeric test
    # because `True` is an `int` in Python and would otherwise retime a clip to 1 fps.
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) \
            or target_fps <= 0:
        raise WorkerError(
            INVALID_FIELD_VALUE, "field 'target_fps' must be a positive number")

    # **No default, and absent is not zero** (§5c). A `snap_tolerance` defaulted to 0 would ship
    # the unsnapped plan as the ruled answer before the benchmark that decides it has run.
    # Unruled must be visible as unruled, including in the code.
    tolerance = params.get("snap_tolerance")
    if tolerance is not None and (isinstance(tolerance, bool)
                                  or not isinstance(tolerance, (int, float))
                                  or not 0.0 <= tolerance < 0.5):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'snap_tolerance' is a fraction of one source interval, in [0, 0.5)")

    return {
        "codec": codec,
        "bit_depth": bit_depth,
        "frame_threads": frame_threads,
        "pools": pools,
        # **FLAT, where an `interpolate` sub-object used to sit.** *Keeping the nesting for
        # readers downstream would leave the record and the wire disagreeing about the shape of
        # the request, which is the one thing a normalised form exists to prevent.*
        "target_fps": float(target_fps),
        "snap_tolerance": tolerance,
    }
