"""Request validation.

Two rules that pull in opposite directions, and both matter:

  **Unrecognised fields at the top level of `input` are ignored, not refused.** That is where CF
  adds things, and a worker that refused every unfamiliar name would reject every job the moment
  CF sent a new one.

  **A name the contract defines, offered where it is not accepted, is refused by name.** An
  ignored field there changes the output silently, which is the failure the rule prevents.

**The safety argument is structural, and it was not always.** Everything affecting the output
lives in `params`, so an unknown name at the *top level* is metadata by construction and ignoring
it is provably safe rather than conventionally safe.

Until CF's answers of 2026-08-12 this contract had no `params` block: `target_short_edge_px` and
`allow_oom_retry` sat at the top level and did affect the output, so the leniency rested on the
handoff's wording — *anything that changes the output is a named field* — which holds only while
everyone remembers it. CF adopted the media worker's split instead, and the pair of failure
directions is what it buys:

  **unknown inside `params` → refused by name.** CF has sent something that changes the output
  and this worker does not implement it. Failing loudly is correct.

  **unknown at the top level → ignored.** That is where CF adds things, and a worker that refused
  every unfamiliar name would reject every job the moment CF sent a new one.

Inside a `derive` entry there is no leniency either: anything the role does not take is refused,
known or not.
"""

import encoder
import envelope
from errors import (
    FIELD_NOT_SUPPORTED,
    INVALID_FIELD_VALUE,
    MISSING_REQUIRED_FIELD,
    WorkerError,
)

# The envelope: identity, where the bytes come from, where they go, and what set to produce.
# None of these change *how* the output is made. `debug` is a worker-side testing facility rather
# than a contract field.
TOP_LEVEL_FIELDS = {
    "request_id",
    "source_url",
    "output",
    "diagnostics",
    # **A presigned PUT the worker keeps, for the failure with no job attached** (CF, 2026-08-15).
    # Every diagnostics URL until now arrived *with* a job, so the class CF cannot see at all is a
    # worker that dies before it can report: an init that fails, a weight that will not load, a
    # driver mismatch. This one is retained across jobs and used when nothing job-scoped exists.
    #
    # Top level rather than in `params`, for `execution_timeout_ms`' reason exactly: it changes
    # whether a failure can be *reported*, never what the output looks like.
    #
    # **Accepted before CF sends it**, deliberately, and this time the leniency rule is the whole
    # argument. An unknown top-level field is ignored rather than refused, so a worker that did not
    # know this name would discard it silently — no error, no diagnostic, and nothing to indicate
    # either. Which is also why the name was asked for rather than guessed.
    "diagnostics_reserve",
    # **Where this run's record goes** (CF, 2026-08-19, F-2026-08-19-36). A presigned PUT beside
    # `diagnostics` and minted the same way, because the two answer different questions about the
    # same job: the bundle says what went wrong and exists only when something did, this says what
    # happened and exists on every run.
    #
    # **The address arrives with the request rather than living on the endpoint.** A standing
    # credential provisioned as an endpoint secret was designed and rejected: the caller owns its
    # telemetry destination, and a worker holding a long-lived write credential to somebody's
    # bucket is a durable liability for an object written once per job.
    #
    # Absent is a supported state, not a degraded one — the record is skipped and says so. Same
    # leniency argument as `diagnostics_reserve`: an unknown top-level field is ignored, so the
    # name was ruled rather than guessed.
    "run_record",
    "derive",
    "params",
    "debug",
    # **Listed so it can be REFUSED, which is the only reason it is still here.** It pinned a rung
    # on the upscaler's measured ladder; that ladder left with the estimator and this worker has
    # no rungs. `KNOWN_FIELD_NAMES` is built from this tuple and the top level is lenient, so
    # deleting the name would not refuse it — it would ignore it silently, and a field silently
    # ignored reads as supported to every client. `_rung_name` refuses it by name instead.
    "force_rung",
    # **Route C's two test axes, top-level like every other testing switch.** §8b's variants and
    # the `--scale` reading are a benchmark's parameters, not the product's: the contract's
    # `interpolate` block is what a caller sends and these are what the wave sends. Keeping them
    # out of `params` is what stops a test axis becoming a promise — the same reason `force_rung`
    # lives here rather than beside `tile_quality`.
    "force_variant",
    "force_scale",
    # **The same calibration facility, one level finer.** `batch_size` is *frames per model
    # batch* — the model's temporal window, and on CF's account the dominant quality lever for
    # video: a bigger batch sees more of the motion at once. It only ever moved as part of a rung,
    # bundled with five other changes, so its effect has never been isolated from theirs. These
    # two let one knob move at a time. CF never sends them.
    "force_batch_size",
    "force_temporal_overlap",
    # **And the one that turned out to gate the other.** `chunk_size` is how many frames are
    # streamed to the model at a time, so the temporal window is `min(batch_size, chunk_size)` —
    # a batch larger than the chunk is silently the chunk. Measured the hard way: on `swapped`,
    # which chunks at 9, batch 21, 33, 49 and 65 all produced byte-identical masters at an
    # identical 23.15 GB peak. Three runs of an experiment that had already finished.
    #
    # Without this there is no way to raise the window on a memory-constrained rung at all: every
    # rung that can hold 4K in an A40 chunks below its own batch size, so the quality lever CF
    # names as dominant cannot be moved from outside.
    "force_chunk_size",
    # **The rest of the levers, so that "one knob at a time" is true rather than aspirational.**
    # Until these existed the only way to change tiling or block-swapping was to change rung —
    # which also changes the window, which is the confound that made every tiling question
    # unanswerable (`decisions.md` 4.41). Each of these moves exactly one thing.
    #
    # Encode and decode tiling are separate flags on separate *frames*: encode works on the
    # input, decode on the output. At 1080p in and 4K out the same tile size gives a 2x2 grid on
    # one side and 5x3 on the other, so a single number applied to both is two different
    # decisions wearing one name. Every calibration run so far tiled the encode at whatever the
    # rung set — 512 on `swapped`, which cross-fades 44% of every input frame before the model
    # sees it.
    "force_vae_encode_tiled",
    "force_vae_encode_tile_size",
    "force_vae_encode_tile_overlap",
    "force_vae_decode_tiled",
    "force_vae_decode_tile_size",
    "force_vae_decode_tile_overlap",
    # Block-swapping is the one memory lever with **no quality cost at all** — it trades VRAM for
    # time and nothing else — and it has never moved on its own.
    "force_blocks_to_swap",
    "force_swap_io_components",
    # **A pinned configuration must fail, not ratchet.** Without this every limit-finding run
    # silently becomes a run of something else: the ratchet steps the rung, the job succeeds, and
    # the row banked describes a configuration nobody asked for. That is how 68 of 70 peak
    # measurements ended up unattributable. `pin` says: run exactly this, and if it does not fit,
    # say so.
    "pin",
    # **Which upscaler handles the alpha channel, and it is a temporary field.** This worker takes
    # alpha out before the model and resizes it with Lanczos; the model can carry it itself and
    # interpolate it along the edges it just produced (`decisions.md` 4.9). The second is better
    # by every reading of the vendored source and is measured at nothing, so it is a flag until a
    # real cutout has gone through both. Whichever wins becomes the behaviour and this name goes
    # away. CF never sends it.
    "keep_alpha_in_model",
    # **The job's own deadline, set by CF at submit and sent to RunPod in the same breath.**
    # Spelled exactly as RunPod's execution policy spells it, so one integer reaches two
    # recipients with nothing translated between them.
    #
    # Top level rather than `params` because `params` is provably "everything that changes the
    # output"; a deadline changes *whether there is* an output, not what it looks like. It is
    # envelope, next to `request_id` and the URLs.
    #
    # **Accepted before CF sends it**, deliberately: unknown top-level fields are ignored under
    # the leniency rule, so a worker that did not know this name would silently discard it and
    # keep failing at the wall. Support ships first; CF starts sending second.
    "execution_timeout_ms",
    # **THE THIRTEEN BELOW AND ABOVE ARE LISTED SO THEY CAN BE REFUSED** (excision plan §7.6).
    # Every one validated and was then silently dropped: nine had no consumer at all and four were
    # echoed into the diagnostics bundle and nowhere else — which reads as supported HARDER than
    # being ignored, because the caller gets their value back. They were already dead when the
    # excision started, which is why they were not among the five Wave 3 closed.
    #
    # The names stay because the top level is LENIENT and `KNOWN_FIELD_NAMES` is built from this
    # tuple: deleting a name makes the field silently ignored rather than refused, which is the
    # defect being closed rather than a smaller version of it.
    #
    # **Listed so it can be REFUSED.** It priced a job and returned without touching the GPU, by
    # calling the planner; there is no planner and no plan. **This is the sharpest instance of the
    # silent-acceptance class in this file and it has a price attached**: until it was refused, a
    # caller sending `plan_only: true` validated, fell through to the retime, and was billed for
    # loading RIFE, interpolating, encoding and uploading the run they had explicitly asked not to
    # have. The name stays because the top level is lenient and `KNOWN_FIELD_NAMES` is built from
    # this tuple — deleting it would ignore the field rather than refuse it.
    "plan_only",
}

REQUIRED_TOP_LEVEL = ("request_id", "source_url", "output", "params")

# Everything that changes the output. Strict: a name here that this worker does not implement is
# refused by name rather than ignored.
PARAMS_FIELDS = {
    "target_short_edge_px",
    # **An exact canvas, for callers who have one.** `target_short_edge_px` fixes one edge and
    # derives the other from the source's aspect, which cannot express "land on this frame".
    # Measured need: CF separates an image into RGBA layers, the separator returns them at 864x480
    # against a 1376x768 canvas, and a short-edge target produces 1382x768 — six pixels that make
    # the composite wrong. The separator had rounded 860 up to 864 to reach its own 16 grid and
    # *stretched* the content 0.465% doing it, so cropping cannot repair it and a resize to the
    # canvas can (`decisions.md` 4.15).
    #
    # **An object, not two scalars** (CF, 2026-08-15). This is CF's existing vocabulary on its image
    # models, and it carries `dimension_bounds` — the only capability CF has that bounds *a pixel
    # budget* rather than an edge, which is the shape every measurement in this repo took. The
    # megapixel ceiling maps onto `max_pixels` with nothing invented. Shipped first as
    # `target_width` + `target_height`; that spelling never reached a caller and is gone rather
    # than aliased, because two names for one field is how a contract acquires a wrong one.
    "output_size",
    "keep_audio",
    # **The master's constant-rate factor** (CF, 2026-08-18, pulled forward from the parked
    # encoder track). Default 12, which is what `encoder.py` has silently baked since the first
    # commit — so the default is today's behaviour named rather than changed. Applies to the
    # master's encode only: codec, preset, pixel format and the derives' own settings stay with
    # the encoder track. Recorded in the manifest and the ledger, because a master's CRF is part
    # of what that master *is*.
    "crf",
    # **The other four of §6a's five encode settings** (CF, ruled 2026-08-25, built 2026-08-26).
    # `crf` above set the pattern and these join it rather than inventing a second mechanism: a
    # validated request field per setting, today's frozen value as the default, refused outside
    # its range with a message naming that default. **There is deliberately no field carrying an
    # x264 options string** — §6a rules that out and gives the reason: these settings exist to
    # bound the encoder's memory on a path with no host guard, so a pass-through would hand that
    # bound to the caller and let a request restore the configuration that killed the 8K run.
    "preset",
    "threads",
    "sliced_threads",
    "rc_lookahead",
    # ── release 3 ────────────────────────────────────────────────────────────────────────────
    # **Validated in `envelope.py`, not here.** Release 2's surface is large and a release-3
    # block folded into it would be indistinguishable from the fields that have always been
    # there — and the one property protecting production is that a request carrying none of
    # these behaves exactly as it did. Naming them here is what stops `_refuse_unknown` rejecting
    # them by name; the rules that govern them live in one file that can be read end to end.
    "upscale",
    "interpolate",
    # **`params.output` is the ENCODE, and the top-level `output` is the DESTINATION.** One word,
    # two objects, and the collision is the contract's spelling (§5c) rather than a choice made
    # here. Raised to the gate as a claim.
    "output",
}

#: **Nothing is unconditionally required, and that is the point.** `target_short_edge_px` is
#: required only when no `output_size` is given (CF, 2026-08-15). The alternative — keeping it
#: required always — would have had CF derive a short edge purely to satisfy a field it was
#: simultaneously overriding, putting *two* sizing forms on every exact-canvas request in order to
#: preserve a convention about there being one.
#:
#: CF guarantees exactly one form arrives, rejecting a caller who sends both before dispatch. The
#: precedence rule below is kept anyway, as a backstop against a CF bug: a job that completes and
#: warns beats one that dies on a technicality after the GPU is spent.
SIZING_FIELDS = ("target_short_edge_px", "output_size")

# **Default `true`, and it inverts the platform's `keep_audio: false` deliberately.** That default
# exists because several generators invent a soundtrack nobody asked for, so silence is the safe
# answer. Here the track is the caller's **own source audio**, and returning a customer's video
# muted is losing something they supplied rather than suppressing something a model made up — so
# the same rule applied to a different question gives the opposite answer.
#
# The model carries no audio at all (`docs/decisions.md` 0.4); this comes out of the worker's own
# mux as a stream copy. CF sends the value explicitly, as it does `color_correction`; this default
# covers a bare invocation.
DEFAULT_KEEP_AUDIO = True

# Read off the pinned SeedVR2 source's own argparse `choices`, not from a log or a README, so
# this set cannot drift from what the vendored code accepts (docs/decisions.md 0.5). The CLI's
# own default is `lab` — described upstream as "perceptual color matching, recommended" — and
# **not** `wavelet`, which is what the image worker hardcodes and what CF's specs discuss.
#
# CF decides what is advertised and what the default is. The worker decides nothing here; what
# it owes CF is what each value does observably on video, which is a measurement nobody has
# taken. Until then this worker sends the upstream default rather than inheriting the image
# worker's choice, because inheriting it would carry a decision nobody made for video.

#: x264's own range, and the whole of it. 0 is lossless and 51 is unwatchable; both are legal
#: things to ask an encoder for, and a worker inventing a narrower band would refuse work that
#: would have succeeded. The default is `encoder.DEFAULT_CRF`, imported rather than repeated.
CRF_MIN, CRF_MAX = 0, 51


DERIVE_FIELDS_BY_ROLE = {
    "poster": {"role", "at_fraction"},
    "proxy": {"role", "max_duration_s"},
    # `crop` takes `at_fraction` too — every crop comes from one frame and the set is only
    # comparable if they share it. Agreed with CF (bf4471c).
    "crop": {"role", "count", "select", "at_fraction"},
}

DERIVE_FIELDS = {f for fs in DERIVE_FIELDS_BY_ROLE.values() for f in fs}

OUTPUT_FIELDS_REQUIRED = ("endpoint", "bucket", "prefix", "access_key_id", "secret_access_key")
#: `name` is the caller's stem for the master (F-2026-08-19-38) — optional, and absent is a
#: supported state that delivers today's names byte-for-byte. It is validated only as a string
#: here; what makes it safe to use as a key segment is `keys.sanitize_stem`, which is where the
#: rule belongs because it is the module that owns what a name may be.
OUTPUT_FIELDS_OPTIONAL = ("session_token", "name")

#: **Names this worker used to accept and now refuses, kept so that refusing them stays
#: possible.** Removing a name from `PARAMS_FIELDS` refuses it inside `params` for free, because
#: `_refuse_unknown` is strict there — but the SAME tuple feeds `KNOWN_FIELD_NAMES` below, and
#: the top level is LENIENT: `_refuse_known_but_unaccepted` refuses only names in that set and
#: ignores everything else. So a name dropped from the tuple stops being refused at the top level
#: and starts being silently ignored, which is the exact defect refusing it was meant to close.
#:
#: **This is not hypothetical for these three.** The docstring above records that
#: `allow_oom_retry` and `target_short_edge_px` SAT AT THE TOP LEVEL until CF's `params` split of
#: 2026-08-12, so the legacy spelling is the one a stale client actually sends.
RETIRED_FIELD_NAMES = ("tile_quality", "color_correction", "allow_oom_retry", "schedule")

# Every field name the contract defines, anywhere, plus the ones it has retired. A name in here,
# offered somewhere it is not accepted, is refused; a name outside it is metadata at the top
# level and ignored.
KNOWN_FIELD_NAMES = set(TOP_LEVEL_FIELDS) | set(PARAMS_FIELDS) | set(DERIVE_FIELDS) \
    | set(OUTPUT_FIELDS_REQUIRED) | set(OUTPUT_FIELDS_OPTIONAL) | set(RETIRED_FIELD_NAMES)


def _variant_name(value):
    """`None`, or one of §8b's four variant codes. Anything else is refused by name."""
    if value is None:
        return None
    names = ("direct", "cas", "casdec", "pull")
    if value not in names:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "force_variant must be one of {}, got {!r}".format(", ".join(names), value))
    return value


def _positive_float_or_none(value, field):
    """A positive number, or None. **`scale` is RIFE's flow-pyramid resolution**, not an upscale
    factor: 1.0 is full resolution and 0.5 runs motion estimation at half, which finds large
    motion a full-resolution pass misses. There is no sensible default other than the model's, so
    absent stays absent."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise WorkerError(
            INVALID_FIELD_VALUE, "field '{}' must be a positive number, got {!r}".format(
                field, value))
    return float(value)


def _rung_name(value):
    """`None`, or REFUSED BY NAME. This worker has no rung ladder to pin.

    It used to validate against `estimator.RUNGS`, which was the right shape while a ladder
    existed: a rung renamed in one place could not go silently unreachable from the other. The
    estimator and its ladder leave with the upscale path, so there is nothing left to name.

    **The field keeps its entry in `TOP_LEVEL_FIELDS`, and that is the whole point.**
    `KNOWN_FIELD_NAMES` is built from that tuple, and the top level is policed by
    `_refuse_known_but_unaccepted` — lenient, because an unknown name up there is metadata by
    construction. Dropping `force_rung` from the tuple would therefore not refuse it; it would
    make it **silently ignored**, which reads as supported to every client and is exactly the
    failure a named refusal exists to prevent. A refusal by name is not a breach of that
    leniency: the leniency is for names the contract does not define, and this one it does.
    """
    if value is None:
        return None
    raise WorkerError(
        FIELD_NOT_SUPPORTED,
        "field 'force_rung' is not accepted by this worker: it pinned a rung on the upscaler's "
        "measured ladder, and this worker has no ladder and no rungs. Send it unset.",
    )


def _positive_int_or_none(value, field, maximum, minimum=0):
    """Absent, or a plain integer within reach.

    `minimum` is per field rather than fixed at zero: `force_temporal_overlap: 0` is meaningful
    (blend nothing), while `execution_timeout_ms: 0` is not — and zero would be read as *absent*
    by every falsy test downstream, so a caller sending it would silently get no deadline at all
    instead of a refusal. A value that means something different from what it says is worse than
    one that is rejected.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(INVALID_FIELD_VALUE,
                          "{} must be an integer, got {!r}".format(field, value))
    if value < minimum or value > maximum:
        raise WorkerError(INVALID_FIELD_VALUE,
                          "{} must be between {} and {}, got {}".format(
                              field, minimum, maximum, value))
    return value


def _refuse_unknown(present, allowed, where):
    """Strict: anything not accepted here is refused, known to the contract or not."""
    for field in sorted(set(present) - set(allowed)):
        raise WorkerError(
            FIELD_NOT_SUPPORTED, "field '{}' is not accepted {}".format(field, where)
        )


def _refuse_known_but_unaccepted(present, allowed, where):
    """Lenient: refuse names the contract defines, ignore names it does not."""
    for field in sorted(set(present) - set(allowed)):
        if field in KNOWN_FIELD_NAMES:
            raise WorkerError(
                FIELD_NOT_SUPPORTED, "field '{}' is not accepted {}".format(field, where)
            )


def _require(mapping, field, where):
    if mapping.get(field) is None:
        raise WorkerError(
            MISSING_REQUIRED_FIELD, "field '{}' is required {}".format(field, where)
        )
    return mapping[field]


def _as_int(value, field):
    # bool is an int subclass in Python; an explicit reject keeps `true` out of a pixel count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be an integer".format(field))
    return value


def _bool_or_none(value, field):
    """A forced boolean, where absent means "the configuration decides" rather than False.

    Distinct from `_as_bool` because these three states are all meaningful for a forcing field:
    force it on, force it off, or do not force it. Collapsing absent to False would make
    `force_vae_decode_tiled` unsettable to True by omission — and, worse, would silently turn
    tiling *off* on every job that never mentioned it.
    """
    return None if value is None else _as_bool(value, field)


def _as_bool(value, field):
    if not isinstance(value, bool):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be a boolean".format(field))
    return value


def _as_str(value, field):
    if not isinstance(value, str):
        raise WorkerError(INVALID_FIELD_VALUE, "field '{}' must be a string".format(field))
    return value


#: Both keys, and only these two. An `output_size` carrying one of them has said something the
#: contract cannot act on, and guessing which of the two readings it meant — "this width, aspect
#: free" or "this width, and derive the height" — is how a caller gets an output they did not ask
#: for and no message explaining why. The first reading already has a field.
OUTPUT_SIZE_FIELDS = ("width", "height")

#: Not a capacity limit — the capacity refusal is, from measured VRAM against the card in hand.
#: This only catches a value that cannot be a canvas at all, so a transposed field or a byte count
#: fails here rather than at the fit, after the GPU is spent.
OUTPUT_SIZE_MAX_EDGE = 65_536


def _validate_output_size(value):
    """`{'width': W, 'height': H}` on the wire, `(W, H)` out, None if absent."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "field 'output_size' must be an object like {'width': 1376, 'height': 768}, got "
            "{}. For one edge with the aspect left free, use 'target_short_edge_px'."
            .format(type(value).__name__),
        )
    _refuse_unknown(value, OUTPUT_SIZE_FIELDS, "in 'output_size'")
    for field in OUTPUT_SIZE_FIELDS:
        if value.get(field) is None:
            raise WorkerError(
                MISSING_REQUIRED_FIELD,
                "'output_size' needs both 'width' and 'height'; '{}' is missing. An exact canvas "
                "is two numbers, and one of them alone is ambiguous against "
                "'target_short_edge_px'.".format(field),
            )
    return tuple(
        _positive_int_or_none(value[field], "output_size." + field, OUTPUT_SIZE_MAX_EDGE,
                              minimum=1)
        for field in OUTPUT_SIZE_FIELDS
    )


def _validate_output(output):
    if not isinstance(output, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'output' must be an object")
    _refuse_unknown(output, OUTPUT_FIELDS_REQUIRED + OUTPUT_FIELDS_OPTIONAL, "in 'output'")
    for field in OUTPUT_FIELDS_REQUIRED:
        _as_str(_require(output, field, "in 'output'"), "output." + field)
    if output.get("session_token") is not None:
        _as_str(output["session_token"], "output.session_token")
    if output.get("name") is not None:
        _as_str(output["name"], "output.name")
    prefix = output["prefix"]
    # Every file goes under the prefix and the keys are the worker's within it. A prefix that
    # does not end in `/` would make `prefix + name` a sibling of the prefix rather than a child
    # of it, which is a write outside the scope the credential grants — so it fails at R2 rather
    # than here, with a message about credentials instead of about the request.
    if not prefix.endswith("/"):
        raise WorkerError(
            INVALID_FIELD_VALUE, "field 'output.prefix' must end with '/', got {!r}".format(prefix)
        )
    return output


def _refused_upscale_field(name, value, because):
    """`None` if the field is absent or empty, REFUSED BY NAME otherwise.

    **The shape `_rung_name` established, for every field whose consumer left with the upscale
    path.** A field that validates and then does nothing is worse than one refused by name,
    because the caller has evidence it was understood — and these are top-level fields, where
    `_refuse_known_but_unaccepted` is lenient and a name dropped from `TOP_LEVEL_FIELDS` becomes
    silently ignored rather than refused. So the names stay and this refuses their values.

    **Absent and empty are both accepted**, because neither asks for anything: a caller sending
    `derive: []` or `plan_only: false` has requested nothing this worker cannot do, and refusing
    them would be refusing the default.
    """
    # **Falsy, not a hand-written list of falsy things.** `plan_only` used to be normalised with
    # `bool(job_input.get(...))`, so a client serialising booleans as 0/1 and sending
    # `plan_only: 0` was asking for a NORMAL run. An identity check against `False` refuses that
    # 0 and would refuse the default in the spelling half of JSON's users write.
    if not value:
        return None
    raise WorkerError(
        FIELD_NOT_SUPPORTED,
        "field {!r} is not accepted by this worker: {} Refused rather than accepted and "
        "ignored, because a field that validates and does nothing reads as supported.".format(
            name, because))


def validate(job_input):
    """Return the request, normalised. Raises `WorkerError` on anything the contract refuses."""
    if not isinstance(job_input, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "'input' must be an object")

    # Lenient at the top level: unknown names are metadata by construction, since everything that
    # changes the output lives in `params`.
    _refuse_known_but_unaccepted(job_input, TOP_LEVEL_FIELDS, "at the top level of 'input'")

    for field in REQUIRED_TOP_LEVEL:
        _require(job_input, field, "at the top level of 'input'")

    request_id = _as_str(job_input["request_id"], "request_id")
    if not request_id.strip():
        raise WorkerError(INVALID_FIELD_VALUE, "field 'request_id' must not be empty")

    params = job_input["params"]
    if not isinstance(params, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'params' must be an object")
    # Strict inside `params`: a name CF sent here changes the output, and this worker not
    # implementing it must be loud rather than silent.
    _refuse_unknown(params, PARAMS_FIELDS, "in 'params'")

    output_size = _validate_output_size(params.get("output_size"))

    # **Release 3's surface, derived before the sizing rule because it can suspend it.**
    # `upscale: false` is the explicit retime spelling, and a retime does not resize — so the
    # requirement that a request say what size it wants is release 2's rule and stays exactly
    # that. `envelope.derive` refuses the contradiction (a size beside `upscale: false`) and the
    # emptiness (`upscale: false` with nothing to do) itself.
    release_3 = envelope.derive(params)

    # **What the contract accepts and this worker cannot yet serve is REFUSED, not ignored.**
    # `envelope.derive` is §5c complete and correct; the paths behind two of its answers are not
    # wired. Accepting them would deliver the opposite of what was asked — an `upscale: false`
    # request would be planned as an upscale and die on a null size, and an `h265` request would
    # return h264 without a word. That is the silent-reinterpretation class this whole section
    # exists to prevent, and a field that validates and then does nothing is worse than one
    # refused by name, because the caller has evidence it was understood.
    # `upscale: false` is served: `handle` branches to route C on it. `interpolate` BESIDE an
    # upscale is not — that is route A or B, which need the A/B `route` field Phase 2 rules and a
    # pipeline placement nothing has built.
    if release_3["interpolate"] is not None and release_3["upscale"]:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'interpolate' beside an upscale is route A or B and this worker serves neither yet; "
            "only 'upscale: false' — interpolation alone — is wired. Refused rather than "
            "silently upscaling without it.")
    # **AND AN UPSCALE ITSELF IS REFUSED, because this worker no longer has one.** The upscale
    # path, its estimator and its rung ladder left with the SeedVR2 excision; `handle` has one
    # route. **This is the DEFAULT request shape, not an exotic one** — `envelope.py:66` says
    # "Omit it to upscale", so every request that does not spell `upscale: false` resolves here.
    # Refusing it by name is the whole of §5c's argument: until this existed the request validated,
    # reached a branch calling a function that had been deleted, and came back `internal` — a
    # request-shape problem reported as a worker fault, which is the defect `handle`'s two error
    # tables exist to prevent.
    #
    # **NO SURFACE CHANGE.** `upscale: false` still means exactly what it meant. Making omission
    # mean "retime" would be redesigning the request surface, which is an entry condition and not
    # an excision's to answer.
    if release_3["upscale"]:
        raise WorkerError(
            FIELD_NOT_SUPPORTED,
            "this worker performs frame interpolation and nothing else: it has no upscaler. A "
            "request must ask for a retime explicitly with 'upscale: false' in 'params'. Refused "
            "rather than retimed without being asked, because a caller who asked to be upscaled "
            "has not asked for this.")
    if release_3["codec"] != envelope.DEFAULT_CODEC:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'output.codec: {}' is contract-legal and this worker cannot serve it yet; only {!r} "
            "is implemented. Refused rather than silently encoded as {}.".format(
                release_3["codec"], envelope.DEFAULT_CODEC, envelope.DEFAULT_CODEC))


    target = params.get("target_short_edge_px")
    if target is None and not release_3["upscale"]:
        pass
    elif target is None:
        # One of the two has to arrive. Naming both in the message matters: a caller who omitted
        # the short edge because they *meant* to send `output_size` and mistyped it has already
        # been told about the typo by `_refuse_unknown`, and a caller who sent neither is being
        # told the contract rather than scolded about one field.
        if output_size is None:
            raise WorkerError(
                MISSING_REQUIRED_FIELD,
                "a request must say what size it wants: either 'target_short_edge_px' (one edge, "
                "aspect preserved) or 'output_size' as {'width': W, 'height': H} (an exact "
                "canvas). Neither was given in 'params'.",
            )
    else:
        target = _as_int(target, "target_short_edge_px")
        # Type and positivity only. **No maximum, deliberately.** The bounds are CF's product
        # choice, and a bound the worker invents refuses work that would have succeeded — the
        # failure this model row has already produced once, on an input rule CF withdrew. A target
        # below the source's short edge is a downscale, which the model supports: permitted, warned
        # in the response, never refused. What the worker owes CF is where quality and memory
        # actually fall off, which is a measurement, not a constant.
        if target <= 0:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'target_short_edge_px' must be positive, got {}".format(target),
            )

    keep_audio = params.get("keep_audio")
    keep_audio = DEFAULT_KEEP_AUDIO if keep_audio is None else _as_bool(
        keep_audio, "keep_audio")

    crf = params.get("crf")
    if crf is None:
        crf = encoder.DEFAULT_CRF
    else:
        crf = _as_int(crf, "crf")
        if not CRF_MIN <= crf <= CRF_MAX:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'crf' must be within x264's range {}-{}, got {}. Lower is better quality "
                "and a larger file; {} is this worker's default and what every measurement in "
                "its calibration was taken at.".format(
                    CRF_MIN, CRF_MAX, crf, encoder.DEFAULT_CRF),
            )

    preset = params.get("preset")
    if preset is None:
        preset = encoder.DEFAULT_PRESET
    else:
        preset = _as_str(preset, "preset")
        if preset not in encoder.PRESETS:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'preset' must be one of x264's presets ({}), got {!r}. {!r} is this "
                "worker's default and what every measurement in its calibration was taken "
                "at.".format(", ".join(encoder.PRESETS), preset, encoder.DEFAULT_PRESET),
            )

    threads = params.get("threads")
    if threads is None:
        threads = encoder.DEFAULT_THREADS
    else:
        threads = _as_int(threads, "threads")
        if not encoder.THREADS_MIN <= threads <= encoder.THREADS_MAX:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                # **The floor is 1 and the message says why**, because `0` is a value a caller
                # will try: it is x264's spelling of *auto*, and auto on this worker's 96-core
                # host is 128 frame-threads — the configuration that filled 46 GiB and got the
                # first 8K run reaped. Refusing it silently would look like an off-by-one.
                "field 'threads' must be within {}-{}, got {}. x264 reads 0 as 'auto', which on "
                "a large host means up to {} frame-threads and is the setting this worker exists "
                "to bound — so auto is not reachable through this field. {} is the default and "
                "what every measurement in its calibration was taken at.".format(
                    encoder.THREADS_MIN, encoder.THREADS_MAX, threads,
                    encoder.THREADS_MAX, encoder.DEFAULT_THREADS),
            )

    sliced_threads = params.get("sliced_threads")
    sliced_threads = (encoder.DEFAULT_SLICED_THREADS if sliced_threads is None
                      else _as_bool(sliced_threads, "sliced_threads"))

    rc_lookahead = params.get("rc_lookahead")
    if rc_lookahead is None:
        rc_lookahead = encoder.DEFAULT_RC_LOOKAHEAD
    else:
        rc_lookahead = _as_int(rc_lookahead, "rc_lookahead")
        if not encoder.RC_LOOKAHEAD_MIN <= rc_lookahead <= encoder.RC_LOOKAHEAD_MAX:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                "field 'rc_lookahead' must be within x264's range {}-{}, got {}. 0 disables the "
                "lookahead and is the cheapest setting; {} is this worker's default and what "
                "every measurement in its calibration was taken at.".format(
                    encoder.RC_LOOKAHEAD_MIN, encoder.RC_LOOKAHEAD_MAX, rc_lookahead,
                    encoder.DEFAULT_RC_LOOKAHEAD),
            )

    # **Refused rather than validated-then-dropped**, and this one read as supported harder than
    # any other: `_validate_derive` checked role uniqueness, per-role field strictness and
    # `at_fraction` bounds, so a client probing the surface got a detailed acknowledgement of a
    # feature that produced nothing. A caller asking for a poster and a proxy received
    # `status: DELIVERED`, one file, and no word about the other two.
    derive = _refused_upscale_field(
        "derive", job_input.get("derive"),
        "posters, proxies and crops were taken from an upscaled master by a module that left "
        "with the upscale path; this worker delivers the retimed master and nothing beside it.")

    diagnostics = job_input.get("diagnostics")
    if diagnostics is not None:
        diagnostics = _as_str(diagnostics, "diagnostics")

    reserve = job_input.get("diagnostics_reserve")
    if reserve is not None:
        reserve = _as_str(reserve, "diagnostics_reserve")

    run_record = job_input.get("run_record")
    if run_record is not None:
        run_record = _as_str(run_record, "run_record")

    # Flattened for the handler's use. The **wire** shape is nested; this is the normalised form
    # everything downstream reads, so the nesting exists exactly once — here — rather than being
    # threaded through every caller.
    return {
        "request_id": request_id,
        "source_url": _as_str(job_input["source_url"], "source_url"),
        "target_short_edge_px": target,
        # The normalised release-3 config, one object rather than four loose keys, so a caller
        # downstream cannot read `interpolate` without also having what decided it.
        "release_3": release_3,
        "keep_audio": keep_audio,
        "crf": crf,
        "preset": preset,
        "threads": threads,
        "sliced_threads": sliced_threads,
        "rc_lookahead": rc_lookahead,
        "derive": derive,
        "output": _validate_output(job_input["output"]),
        "diagnostics": diagnostics,
        "diagnostics_reserve": reserve,
        "run_record": run_record,
        "debug": bool(job_input.get("debug")),
        "force_rung": _rung_name(job_input.get("force_rung")),
        "force_variant": _variant_name(job_input.get("force_variant")),
        "force_scale": _positive_float_or_none(job_input.get("force_scale"), "force_scale"),
        # RunPod's own ceiling is 7 days. No lower bound beyond positive: a caller who sends a
        # deadline this worker cannot meet gets a refusal with the arithmetic, which is more
        # useful than an argument about whether the number was reasonable.
        "execution_timeout_ms": _positive_int_or_none(
            job_input.get("execution_timeout_ms"), "execution_timeout_ms", 604_800_000,
            minimum=1),
        # Snapped to the 4n+1 lattice by the pipeline, not refused here: a caller asking for 20
        # means "about twenty", and the nearest valid value is a better answer than an error.
        "output_size": output_size,
        "pin": _refused_upscale_field(
            "pin", job_input.get("pin"),
            "it pinned a configuration on the upscaler's rung ladder so a mid-run ratchet could not move it; there is no ladder and no ratchet, and `_rung_name` sixteen lines above refuses `force_rung` on exactly that ground"),
        "keep_alpha_in_model": _refused_upscale_field(
            "keep_alpha_in_model", job_input.get("keep_alpha_in_model"),
            "it chose which upscaler handled the alpha channel; there is no upscaler, and route C's writer is rgb24"),
        "force_batch_size": _refused_upscale_field(
            "force_batch_size", job_input.get("force_batch_size"),
            "it set the model's temporal window in frames per batch; RIFE takes a frame pair and has no batch"),
        "force_chunk_size": _refused_upscale_field(
            "force_chunk_size", job_input.get("force_chunk_size"),
            "it sized the vendored coder's streaming chunks; there are no chunks — decode, interpolation and encode are one streaming loop"),
        "force_temporal_overlap": _refused_upscale_field(
            "force_temporal_overlap", job_input.get("force_temporal_overlap"),
            "it set the overlap blended between chunks; there are no chunks to overlap"),
        "force_blocks_to_swap": _refused_upscale_field(
            "force_blocks_to_swap", job_input.get("force_blocks_to_swap"),
            "it tuned BlockSwap, the vendored model's VRAM-relief mechanism"),
        "force_swap_io_components": _refused_upscale_field(
            "force_swap_io_components", job_input.get("force_swap_io_components"),
            "it tuned the same mechanism's IO components"),
        "force_vae_encode_tiled": _refused_upscale_field(
            "force_vae_encode_tiled", job_input.get("force_vae_encode_tiled"),
            "it tiled the VAE encode; there is no VAE"),
        "force_vae_encode_tile_size": _refused_upscale_field(
            "force_vae_encode_tile_size", job_input.get("force_vae_encode_tile_size"),
            "it sized those tiles; there is no VAE"),
        "force_vae_encode_tile_overlap": _refused_upscale_field(
            "force_vae_encode_tile_overlap", job_input.get("force_vae_encode_tile_overlap"),
            "it overlapped those tiles; there is no VAE"),
        "force_vae_decode_tiled": _refused_upscale_field(
            "force_vae_decode_tiled", job_input.get("force_vae_decode_tiled"),
            "it tiled the VAE decode; there is no VAE"),
        "force_vae_decode_tile_size": _refused_upscale_field(
            "force_vae_decode_tile_size", job_input.get("force_vae_decode_tile_size"),
            "it sized those tiles; there is no VAE"),
        "force_vae_decode_tile_overlap": _refused_upscale_field(
            "force_vae_decode_tile_overlap", job_input.get("force_vae_decode_tile_overlap"),
            "it overlapped those tiles; there is no VAE"),
        "plan_only": _refused_upscale_field(
            "plan_only", job_input.get("plan_only"),
            "it priced a job through the planner and returned without touching the GPU; this "
            "worker has no planner and nothing to price, and every request it accepts runs."),
    }
