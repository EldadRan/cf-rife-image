"""Release 3's request surface: the codec, the retime-only spelling, the interpolation block.

`fable/envelope_oracle.py` is this section of the contract executable and is the authority — if
this file and that one disagree, one of them is a bug and it gets a decision entry rather than a
patch. The rules are written from contract §5c rather than copied from the oracle, for the reason
the retime plan is: two independent statements of one rule is the point, and agreement reached by
sharing code proves nothing.

**Kept out of `validation.py` deliberately.** That module is release 2's surface and is large; a
release-3 block folded into it would be indistinguishable from the fields that have always been
there, and the one property protecting production — that a request carrying none of these fields
behaves exactly as it did — is easiest to keep true when the new surface is one file that can be
read end to end.
"""
from errors import INVALID_FIELD_VALUE, MISSING_REQUIRED_FIELD, WorkerError

#: **`source` means "match the input's codec"**, which is a release-3 field and not a default.
CODECS = ("h264", "h265", "source")

#: **Unchanged, so an omitted field cannot move anything.** Every release-2 caller encodes h264
#: today and must still encode h264 after this ships — `default_off_identity` is the assertion.
DEFAULT_CODEC = "h264"

#: **contract §6i: x265's two threading levers, optional and independently so.** *Either sent is
#: used, either absent takes its default, any combination is legal — and a request sending NEITHER
#: behaves exactly as `sha-88fec73` does, which is what makes the field shippable before any
#: calibration has happened.*
#:
#: **These are an OVERRIDE ON TOP OF A DEFAULT, not a decision moved to the caller**, which is why
#: §6i is not the request-field shape CF refused earlier the same day. *What was refused was
#: handing the caller a BOUND — a knob deciding what the worker does when the caller has no basis
#: to choose. Here the default still governs and a caller who says nothing gets the worker's
#: answer.*
#: **THE ONE HOME FOR BOTH NUMBERS.** `encoder.x265_threading` reads these rather than keeping
#: its own copies — the pattern `validation` already uses for `encoder.DEFAULT_CRF`, "imported
#: rather than repeated". *They lived in both files for one commit and nothing kept them equal;
#: a second home for one fact is how the copies drift and the comment claiming they agree stays.*
DEFAULT_FRAME_THREADS = 1
DEFAULT_POOLS = 16

#: The ranges. **Bounded rather than open**, for §6a's reason one codec over: these settings decide
#: how much the encoder allocates on a path §1 says has no host guard, so an unbounded value is a
#: memory bound handed to the caller. *`frame-threads` above a small number multiplies the frames
#: in flight; `pools` above the visible core count buys nothing and costs scheduling.*
#: Every key `params.output` may carry. **One home, and the refusal above reads it** — a field
#: added without joining this tuple is refused by name, which is the outcome a caller can act on.
OUTPUT_FIELDS = ("codec", "bit_depth", "frame_threads", "pools")

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

#: CF: request-carried, default 60.
DEFAULT_TARGET_FPS = 60.0

#: Relative to the upscale. **No default** — Phase 2's A/B rules it.
ROUTES = ("before", "after")

#: The two spellings of "what size do you want", either of which satisfies the release-2 rule.
SIZING_FIELDS = ("target_short_edge_px", "output_size")

INTERPOLATE_FIELDS = ("target_fps", "snap_tolerance", "route")


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


def derive(params):
    """`params` in, a normalised release-3 config out, or `WorkerError`.

    **Absent fields produce the release-2 answer exactly**, which is the property that lets the
    development tier run this code while other tiers serve customers.
    """
    params = dict(params or {})
    interpolate = params.get("interpolate")
    upscale = params.get("upscale")
    output = dict(params.get("output") or {})
    # **UNKNOWN KEYS IN `params.output` ARE REFUSED BY NAME**, exactly as `interpolate`'s are
    # below and as `validation._refuse_unknown` does for `params` and the destination block.
    # Nothing checked this sub-object until §6i, and the gap was invisible while its only fields
    # were enums a typo turned into a refusal anyway.
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
    unknown_output = sorted(set(output) - set(OUTPUT_FIELDS))
    if unknown_output:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "unknown field(s) in 'params.output': {}. Known: {}.".format(
                unknown_output, sorted(OUTPUT_FIELDS)))
    has_size = any(params.get(field) is not None for field in SIZING_FIELDS)

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

    # **`upscale: false` is explicit, and the tempting spelling was a trap.** Letting a missing
    # size field mean "no upscale" needs no new field and reuses the sizing refusal as the
    # discriminator — and a caller who wants interpolation AND an upscale but forgets the size
    # field would then silently receive a retime instead of an error. This project has paid for
    # the silent-reinterpretation class twice: an endpoint renamed by a defaulted `--name`, and
    # 16-bit sources downconverted without a word. A forgotten field must still refuse; a
    # deliberate retime says so in words.
    if upscale is not None:
        if upscale is not False:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'upscale' is not a toggle: the only legal value is false, which asks for "
                "a retime with no upscaling. Omit it to upscale.")
        if has_size:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "'upscale: false' contradicts a sizing field. A retime does not resize, so say "
                "one or the other.")
        if interpolate is None:
            raise WorkerError(
                MISSING_REQUIRED_FIELD,
                "'upscale: false' with no 'interpolate' asks the worker to do nothing.")
    elif not has_size:
        # Release-2 behaviour, restated only because release 3 must not weaken it.
        raise WorkerError(
            MISSING_REQUIRED_FIELD,
            "a request must say what size it wants: either 'target_short_edge_px' (one edge, "
            "aspect preserved) or 'output_size' as {'width': W, 'height': H} (an exact canvas). "
            "Neither was given in 'params'.")

    if interpolate is None:
        # **Named at the top level, they mean nothing and are refused rather than ignored.** A
        # caller who put `target_fps` beside `params` instead of inside `interpolate` has asked
        # for something; silence would deliver the opposite of it.
        for orphan in INTERPOLATE_FIELDS:
            if orphan in params:
                raise WorkerError(
                    INVALID_FIELD_VALUE,
                    "field '{}' has no meaning without 'interpolate'".format(orphan))
        # **Both §6i levers are on THIS return too, and that is not tidiness.** A config key that
        # exists on one return path and not the other makes every reader of it correct on one
        # path and a `KeyError` on the other — the shape that cost a live job earlier today, one
        # repository over. *They are `None` here for the same reason they are `None` anywhere the
        # caller said nothing: absence stays absence until the encoder resolves it.*
        return {"codec": codec, "bit_depth": bit_depth, "interpolate": None, "upscale": True,
                "frame_threads": frame_threads, "pools": pools,
                # **`bit_depth` joins the equivalence test rather than riding beside it.** A
                # 10-bit request is not what a release-2 caller sends, and `default_off_identity`
                # asserts that the DEFAULT path is unmoved — which stays true, because an omitted
                # field is 8.
                #
                # **And the two threading levers join it for the same reason**: a request naming
                # either is not a release-2 request, and an omitted field is `None`, so the
                # default path is still unmoved.
                "release_2_equivalent": (codec == DEFAULT_CODEC
                                         and bit_depth == DEFAULT_BIT_DEPTH
                                         and frame_threads is None and pools is None)}

    if not isinstance(interpolate, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'interpolate' must be an object")
    unknown = sorted(set(interpolate) - set(INTERPOLATE_FIELDS))
    if unknown:
        raise WorkerError(
            INVALID_FIELD_VALUE, "unknown field(s) in 'interpolate': {}".format(unknown))

    target_fps = interpolate.get("target_fps", DEFAULT_TARGET_FPS)
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) \
            or target_fps <= 0:
        raise WorkerError(
            INVALID_FIELD_VALUE, "field 'interpolate.target_fps' must be a positive number")

    # **No default, and absent is not zero** (§5c). A `snap_tolerance` defaulted to 0 would ship
    # the unsnapped plan as the ruled answer before the benchmark that decides it has run.
    # Unruled must be visible as unruled, including in the code.
    tolerance = interpolate.get("snap_tolerance")
    if tolerance is not None and (isinstance(tolerance, bool)
                                  or not isinstance(tolerance, (int, float))
                                  or not 0.0 <= tolerance < 0.5):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'interpolate.snap_tolerance' is a fraction of one source interval, in "
            "[0, 0.5)")

    # **No default either**, for the same reason: "before" by omission would settle Phase 2's A/B
    # without the A/B.
    route = interpolate.get("route")
    if route is not None and route not in ROUTES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'interpolate.route' must be one of {}".format(ROUTES))
    if route is not None and upscale is False:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'interpolate.route' names where interpolation sits relative to the upscale, and "
            "'upscale: false' has no upscale to sit beside.")

    return {
        "codec": codec,
        "bit_depth": bit_depth,
        "frame_threads": frame_threads,
        "pools": pools,
        "interpolate": {"target_fps": float(target_fps), "snap_tolerance": tolerance,
                        "route": route},
        "upscale": upscale is not False,
        "release_2_equivalent": False,
    }


def default_off_identity(release_2_params):
    """One assertion: a request carrying none of release 3's fields behaves as it always did.

    **This is what lets the development tier run release-3 code while other tiers serve
    customers** — h264, no interpolation, upscaling on. Enforced by a local run before a dispatch,
    never by CI, which no longer sees the suite.
    """
    config = derive(release_2_params)
    return (config["codec"] == DEFAULT_CODEC
            and config["interpolate"] is None
            and config["upscale"] is True
            and config["release_2_equivalent"] is True)
