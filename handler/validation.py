"""Request validation.

**ONE RULE, AND IT REPLACED TWO THAT PULLED IN OPPOSITE DIRECTIONS (CF, 2026-09-02):**

  **Every field in a request is named in the PRODUCTION list or the DEBUG list, or the request is
  REFUSED.** At any level, whether or not the name once meant something. A DEBUG name is accepted
  only alongside `debug: true` and refused BY NAME without it — never ignored, and never called
  unknown, because those are different facts about the request and a caller acts on them
  differently.

**WHAT THIS ENDED, AND THE ARGUMENT IT ENDED IS WORTH KEEPING.** *The top level was LENIENT:
unknown names there were metadata by construction — everything that changes the output lives in
`params` — so ignoring them was provably safe rather than conventionally safe, and it meant a
worker did not reject every job the moment CF sent a new name.* **The cost is what CF ruled on:
an unknown top-level field returned 200 with the field silently discarded, which reads as
supported to every client, and a name DROPPED from a list was therefore not refused but
swallowed.** *So this module used to carry a long roll of dead upscaler names for the sole
purpose of refusing them by hand — `force_rung`, `plan_only`, `pin`, the six `force_vae_*` and
the rest. Strictness deletes the names and the machinery together.*

**THE PRICE IS FORWARD COMPATIBILITY, AND IT WAS BOUGHT DELIBERATELY ONCE.** *`run_record` and
`diagnostics_reserve` were both accepted before CF sent them, precisely because a worker that did
not know the name would otherwise discard it in silence.* **That is no longer possible: a new
field lands in the worker before a caller may send it, and a caller who sends it early gets a
refusal.** *A refusal is the better failure and it is still a cost — written here so nobody
re-derives leniency from the comments that argue for it.*

**THE UPSCALE PATH IS GONE FROM THE SURFACE ENTIRELY**: `upscale`, `target_short_edge_px`,
`output_size`, the `interpolate` block and `route`. *`target_fps` and `snap_tolerance` moved into
`params`, where every other per-job field already sat.*
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
#: **THE REQUEST IS STRICT AT EVERY LEVEL (CF, 2026-09-02), AND THAT IS A CHANGE OF KIND RATHER
#: THAN A LONGER LIST.** *Every field in a request is named on one of the two lists below, or the
#: request is REFUSED — at any level, whether or not the name once meant something.*
#:
#: **THE TOP LEVEL WAS LENIENT UNTIL TODAY AND THAT IS WHAT THIS DELETES.**
#: `_refuse_known_but_unaccepted` refused names the contract defined and SILENTLY IGNORED names
#: it did not, so an unknown top-level field returned 200 with the field discarded — which reads
#: as supported to every client.
#:
#: **AND IT IS WHAT MAKES A DELETION A DELETION.** *Under leniency, dropping a name from this set
#: did not refuse it; it made it silently ignored, which is the state the deletion was meant to
#: end.* **So the long roll of dead upscaler names is simply GONE** — `derive`,
#: `execution_timeout_ms`, `pin`, `plan_only`, `force_rung`, `keep_alpha_in_model`, the six
#: `force_vae_*`, `force_batch_size`, `force_chunk_size`, `force_temporal_overlap`,
#: `force_blocks_to_swap`, `force_swap_io_components` — together with `_rung_name`,
#: `_refused_upscale_field`, `RETIRED_FIELD_NAMES` and `KNOWN_FIELD_NAMES`, every one of which
#: existed only to refuse a name that leniency would otherwise have swallowed.
#:
#: **THE PRICE IS FORWARD COMPATIBILITY AND IT WAS DELIBERATELY BOUGHT ONCE.**
#: *`diagnostics_reserve` and `run_record` were both accepted BEFORE CF sent them, on the
#: argument that a worker which did not know the name would otherwise discard it in silence.*
#: **Under a strict request that is no longer possible**: a new field must land in the worker
#: before a caller may send it, and a caller who sends it early gets a refusal instead of
#: silence. *A refusal is the better failure and it is still a cost; written here so nobody
#: re-derives leniency later from the comments that argue for it.*
TOP_LEVEL_PRODUCTION = (
    "request_id",
    "source_url",
    "output",
    "params",
    # **A presigned PUT the worker keeps, for the failure with no job attached** (CF, 2026-08-15).
    # The class CF cannot see at all is a worker that dies before it can report: an init that
    # fails, a weight that will not load, a driver mismatch. This one is retained across jobs and
    # used when nothing job-scoped exists.
    "diagnostics",
    "diagnostics_reserve",
    # **Where this run's record goes** (CF, 2026-08-19, F-2026-08-19-36). A presigned PUT beside
    # `diagnostics` and minted the same way, because the two answer different questions about the
    # same job: the bundle says what went wrong and exists only when something did, this says
    # what happened and exists on every run. Absent is a supported state, not a degraded one.
    "run_record",
    # **THE GATE ITSELF IS A PRODUCTION FIELD, AND IT HAS TO BE.** *A strict rule with `debug`
    # outside both lists refuses every request that sets it — so no debug field would be
    # reachable on any request and the flag would refuse itself.*
    "debug",
)

#: **Accepted ONLY with `debug: true`, and refused BY NAME without it** — not ignored, and not
#: called unknown. *`envelope.refuse_field` carries the two messages and the reason they differ.*
#:
#: **THESE TWO STAY AT THE TOP LEVEL RATHER THAN MOVING INTO `params`.** *`params` is provably
#: "everything that changes the output"; these are a benchmark's parameters, and keeping them out
#: is what stopped a test axis becoming a promise back when nothing else could.* **The wave
#: changes what is GATED and not where anything is SENT** — moving them in the same change would
#: make a refusal ambiguous between the two causes. *With CF; if it rules them into `params` it
#: is a level move on two names.*
TOP_LEVEL_DEBUG = (
    "force_variant",
    "force_scale",
)

REQUIRED_TOP_LEVEL = ("request_id", "source_url", "output", "params")

# Everything that changes the output. Strict: a name here that this worker does not implement is
# refused by name rather than ignored.
#: **`params` — everything that changes the output.** *Strict since it was written; what changes
#: today is that the DELETED names are gone from it rather than listed for refusal, because the
#: top level is strict too and a dropped name is now refused everywhere.*
#:
#: **`target_fps` AND `snap_tolerance` ARRIVE HERE NOW.** *They lived in a nested `interpolate`
#: object that existed to distinguish a retime from an upscale; with one capability it
#: discriminates nothing, and `envelope` REFUSED `target_fps` at the top level for want of it —
#: so the field a caller most obviously wants to send was the one spelling the contract would not
#: take.*
#:
#: **`output` IS THE ENCODE AND THE TOP-LEVEL `output` IS THE DESTINATION.** One word, two
#: objects, and the collision is the contract's spelling (§5c) rather than a choice made here.
#: *Its own two lists live in `envelope`, which owns what may appear inside it.*
PARAMS_PRODUCTION = (
    "target_fps",
    "keep_audio",
    # **The master's constant-rate factor** (CF, 2026-08-18). Default 12, which is what
    # `encoder.py` has silently baked since the first commit — the default is today's behaviour
    # named rather than changed.
    "crf",
    "preset",
    "output",
)

#: **The debug half of `params`, refused by name without `debug: true`.**
#:
#: *The five instrumentation gates were already opt-in per job and each costs real time or disk;
#: the three x264 settings and `snap_tolerance` are encoder internals and an unruled test axis.*
#: **What the flag adds is that a production caller cannot reach any of them by accident** — a
#: misspelling is still refused by name, which is the property that made a per-job control safe
#: to add here in the first place.
PARAMS_DEBUG = (
    # **`docs/archive/conversion-wave-archive.md` §5-0.** Runs the whole outbound conversion a
    # SECOND time per frame.
    "convert_check",
    # **§3b-0 item 4: the tie check moved out of the environment.** `CF_RIFE_TIECHECK` armed it
    # until 2026-08-27 and was left set, taxing two jobs ~12.6 s each, invisible on the endpoint
    # page. **A per-job control belongs in the request.**
    "tie_check",
    # **§3b-1's gate. Its own field, not folded into `convert_check`** — they sit at different
    # boundaries and prove different things, and one flag arming both would make a failed run
    # ambiguous about which comparison failed.
    "input_check",
    # **§11's decode decomposition.** Re-decodes the source three times beside the retime.
    "decode_probe",
    # **§6g's fifth gate and the most expensive of the five**: it retains every frame handed to
    # the encoder on worker disk and scores the delivered master against them. *A run carrying it
    # must not bank a performance number.*
    "reference_score",
    "threads",
    "sliced_threads",
    "rc_lookahead",
    # **No default, and absent is not zero** — a `snap_tolerance` of 0 would ship the unsnapped
    # plan as the ruled answer before the benchmark that decides it has run.
    "snap_tolerance",
)

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

#: x264's own range, and the whole of it. 0 is lossless and 51 is unwatchable; both are legal
#: things to ask an encoder for, and a worker inventing a narrower band would refuse work that
#: would have succeeded. The default is `encoder.DEFAULT_CRF`, imported rather than repeated.
CRF_MIN, CRF_MAX = 0, 51


OUTPUT_FIELDS_REQUIRED = ("endpoint", "bucket", "prefix", "access_key_id", "secret_access_key")
#: `name` is the caller's stem for the master (F-2026-08-19-38) — optional, and absent is a
#: supported state that delivers today's names byte-for-byte. It is validated only as a string
#: here; what makes it safe to use as a key segment is `keys.sanitize_stem`, which is where the
#: rule belongs because it is the module that owns what a name may be.
OUTPUT_FIELDS_OPTIONAL = ("session_token", "name")

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


def _refuse_unknown(present, allowed, where):
    """Strict: anything not accepted here is refused, known to the contract or not."""
    for field in sorted(set(present) - set(allowed)):
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


def _english_list(items):
    """`a`, `a and b`, `a, b and c`. **A refusal a caller reads once has to parse on that read**,
    and three names joined by two `and`s is a sentence they have to re-read to count."""
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return "{} and {}".format(", ".join(items[:-1]), items[-1])


def validate(job_input):
    """Return the request, normalised. Raises `WorkerError` on anything the contract refuses."""
    if not isinstance(job_input, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "'input' must be an object")

    # **STRICT AT THE TOP LEVEL, WHICH IS THE CHANGE.** *Until today an unknown name up here was
    # metadata by construction and was discarded in silence — so a caller could send
    # `force_vae_decode_tile_size` and get a 200 with the field ignored.* **The flag is read
    # first, because it decides which of the two refusals every level below produces.**
    debug = job_input.get("debug")
    if debug is not None:
        debug = _as_bool(debug, "debug")
    debug = bool(debug)
    envelope.refuse_unlisted(job_input, TOP_LEVEL_PRODUCTION, TOP_LEVEL_DEBUG,
                             "at the top level of 'input'", debug)

    for field in REQUIRED_TOP_LEVEL:
        _require(job_input, field, "at the top level of 'input'")

    request_id = _as_str(job_input["request_id"], "request_id")
    if not request_id.strip():
        raise WorkerError(INVALID_FIELD_VALUE, "field 'request_id' must not be empty")

    params = job_input["params"]
    if not isinstance(params, dict):
        raise WorkerError(INVALID_FIELD_VALUE, "field 'params' must be an object")
    # Strict inside `params`, as it has always been — what is new is that a DEBUG name here is
    # refused for its state rather than accepted, and that the deleted names are gone from the
    # list rather than listed in order to be refused.
    envelope.refuse_unlisted(params, PARAMS_PRODUCTION, PARAMS_DEBUG, "in 'params'", debug)

    # **Release 3's surface, derived before the sizing rule because it can suspend it.**
    # `upscale: false` is the explicit retime spelling, and a retime does not resize — so the
    # requirement that a request say what size it wants is release 2's rule and stays exactly
    # that. `envelope.derive` refuses the contradiction (a size beside `upscale: false`) and the
    # emptiness (`upscale: false` with nothing to do) itself.
    # **The flag reaches `envelope` because `params.output`'s two lists live there** — that
    # module owns what may appear inside the encode block, and splitting the knowledge would put
    # a name's level in one file and its gating in another.
    release_3 = envelope.derive(params, debug)

    # **§6e UNSEALED THIS SURFACE RATHER THAN BUILDING IT.** `envelope.CODECS` has carried all
    # three names since `rife-seed`; what moved is this refusal, from *"only h264"* to the pair
    # the worker now implements. **`"source"` stays in `CODECS` and stays refused HERE**, which
    # is what keeps it a CAPABILITY refusal rather than a schema error: deleting the name would
    # turn a caller's forward-looking request into "no such value", and the capability answer is
    # the true one. *Implementing it needs the probe's codec resolved after the file is open and
    # a refusal for every third codec — a surface of its own, and CF did not ask for it.*
    #
    # **Read off `encoder.CODEC_LIBRARIES` rather than written out**, so the day a third library
    # lands this sentence is right without anyone remembering to come back — and so the door and
    # the encoder cannot disagree about what is implemented.
    if release_3["codec"] not in encoder.CODEC_LIBRARIES:
        raise WorkerError(
            INVALID_FIELD_VALUE,
            "'output.codec: {}' is contract-legal and this worker cannot serve it yet; only {} "
            "are implemented. Refused rather than silently encoded as {}.".format(
                release_3["codec"], " and ".join(sorted(encoder.CODEC_LIBRARIES)),
                envelope.DEFAULT_CODEC))


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

    # ── §6d's three fields. **UNSET STAYS `None` ALL THE WAY OUT OF HERE.** ────────────────────
    #
    # **This function used to resolve all three to constants, and §6d refuses that on sight.**
    # `validate` runs at `handler.py:200` and the source is not probed until `:344`, so the padded
    # area does not exist yet at this line and the branch cannot live here. It lives downstream,
    # at the call site that hands the settings to the encoder — and once a default has been filled
    # in HERE, that site cannot tell *"the caller asked for threads=4"* from *"the caller asked
    # for nothing and validation supplied 4"*. **A branch there would silently overwrite an
    # explicit caller value**, which is the clause of §6b that survives §6d.
    #
    # So the range checks stay exactly where they were — a sent value is still refused outside
    # its range, with the same coded error — and only the DEFAULTING moves. `None` reaching
    # `encoder.resolve_defaults` means the caller sent nothing, and that is the whole mechanism.
    #
    # **`crf` and `preset` above are untouched and still resolve here**, because §6d's table
    # decides three fields and says nothing about either of them.
    threads = params.get("threads")
    if threads is not None:
        threads = _as_int(threads, "threads")
        if not encoder.THREADS_MIN <= threads <= encoder.THREADS_MAX:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                # **The floor is 1 and the message says why**, because `0` is a value a caller
                # will try: it is x264's spelling of *auto*, and auto on this worker's 96-core
                # host is 128 frame-threads — the configuration that filled 46 GiB and got the
                # first 8K run reaped. Refusing it silently would look like an off-by-one.
                #
                # **The closing sentence no longer names a constant, because there no longer is
                # one.** It used to end *"{} is the default and what every measurement in its
                # calibration was taken at"*, naming `encoder.DEFAULT_THREADS`. Under §6d an
                # unset field is filled from the area table, so that sentence became false on the
                # wire the day the branch shipped — and a refusal message that misdescribes what
                # sending nothing would have done is worse than one that says less.
                # **The closing sentence names the MECHANISM and reads the values off the
                # table, and it must keep doing both.** Today both of §6d's rows say 16, so a
                # message that spelled the two values out would read "16 at or below the
                # boundary, 16 above it" — a sentence that tells a caller nothing while looking
                # like it told them something. It is `sliced_threads` that differs between the
                # rows, not this field. **Formatted from `AREA_DEFAULTS` rather than written
                # out**, so the day the rows disagree this says so without anyone remembering to
                # come back.
                "field 'threads' must be within {}-{}, got {}. x264 reads 0 as 'auto', which on "
                "a large host means up to {} frame-threads and is the setting this worker exists "
                "to bound — so auto is not reachable through this field. Send nothing and this "
                "field is not a constant: it is chosen by the DELIVERED frame size against a "
                "boundary of {} pixels, and the two rows say {} and {} respectively.".format(
                    encoder.THREADS_MIN, encoder.THREADS_MAX, threads,
                    encoder.THREADS_MAX,
                    encoder.AREA_BOUNDARY_DELIVERED_PIXELS,
                    encoder.AREA_DEFAULTS[encoder.AREA_ROW_SMALL]["threads"],
                    encoder.AREA_DEFAULTS[encoder.AREA_ROW_LARGE]["threads"]),
            )

    sliced_threads = params.get("sliced_threads")
    if sliced_threads is not None:
        sliced_threads = _as_bool(sliced_threads, "sliced_threads")

    # **`docs/archive/conversion-wave-archive.md` §5-0. Default FALSE and there is no other reachable default.**
    # The gate doubles the outbound conversion for every delivered frame, so it is opt-in per
    # job, asked for by the run that wants the evidence and by nothing else.
    convert_check = params.get("convert_check")
    convert_check = (False if convert_check is None
                     else _as_bool(convert_check, "convert_check"))

    # **§2g's fp32 sweep, and it costs minutes of GPU time**, so it is opt-in per job exactly as
    # `convert_check` is and for the same reason.
    tie_check = params.get("tie_check")
    tie_check = False if tie_check is None else _as_bool(tie_check, "tie_check")

    # §3b-1. Doubles the inbound conversion for every SOURCE frame, so opt-in per job.
    input_check = params.get("input_check")
    input_check = False if input_check is None else _as_bool(input_check, "input_check")

    # §11. Three extra decodes of the whole source; a run carrying it must not bank a number.
    decode_probe = params.get("decode_probe")
    decode_probe = False if decode_probe is None else _as_bool(decode_probe, "decode_probe")

    # **contract §6g's FIFTH GATE, and the most expensive of the five.** It retains every frame
    # handed to the encoder on worker disk, scores the delivered master against them, and uploads
    # the scores — so it writes tens of GB to the same volume the master is written to and runs
    # three filters over every delivered frame.
    #
    # **Opt-in per job for the reason the other four are, one order of magnitude louder.**
    # `docs/archive/instrumentation-archive.md` §12 rules that request-gated instruments do not compose and that
    # an armed run's `compute_s` is comparable to nothing; this one also makes the run's DISK
    # comparable to nothing. *A run carrying it must not bank a performance number.*
    reference_score = params.get("reference_score")
    reference_score = (False if reference_score is None
                       else _as_bool(reference_score, "reference_score"))

    rc_lookahead = params.get("rc_lookahead")
    if rc_lookahead is not None:
        rc_lookahead = _as_int(rc_lookahead, "rc_lookahead")
        if not encoder.RC_LOOKAHEAD_MIN <= rc_lookahead <= encoder.RC_LOOKAHEAD_MAX:
            raise WorkerError(
                INVALID_FIELD_VALUE,
                # **"this worker's default" is the phrase §6d made false**, and it was false
                # here in the same way it was false for `threads`: an unset field is filled from
                # the area table, so there is no constant to name. What is still true — and is
                # what a caller actually wants from this sentence — is what the rows say and what
                # the calibration was taken at, which today are the same number and need not stay
                # that way. Read off the table for that reason.
                # **THE CALIBRATION CLAUSE IS GONE, AND ITS SOURCE IS WHY.** It used to end
                # *"{} is this worker's default and what every measurement in its calibration was
                # taken at"*, formatted from `encoder.DEFAULT_RC_LOOKAHEAD`. Both halves are now
                # wrong from that constant: §6d means there is no default, and the fact "the
                # calibration was taken at rc_lookahead=10" now lives in
                # `estimator.CORPUS["encoder_arm"]`. They agree at 10 today and would diverge the
                # moment either moved, leaving this message asserting a fact about a corpus from
                # a constant that does not own it. **The `threads` message dropped the same
                # clause; this one now matches it.**
                "field 'rc_lookahead' must be within x264's range {}-{}, got {}. 0 disables the "
                "lookahead and is the cheapest setting. Send nothing and this field is not a "
                "constant: it is chosen by the DELIVERED frame size against a boundary of {} "
                "pixels, and the two rows say {} and {} respectively.".format(
                    encoder.RC_LOOKAHEAD_MIN, encoder.RC_LOOKAHEAD_MAX, rc_lookahead,
                    encoder.AREA_BOUNDARY_DELIVERED_PIXELS,
                    encoder.AREA_DEFAULTS[encoder.AREA_ROW_SMALL]["rc_lookahead"],
                    encoder.AREA_DEFAULTS[encoder.AREA_ROW_LARGE]["rc_lookahead"]),
            )

    # ── §6f — 10-BIT IS REFUSED UNDER h264, AND THE REASON IS DELIVERY ──────────────────
    #
    # **CF's ruling: `bit_depth` is a field and not a codec name**, so `main10` is switchable
    # without `output.codec` having to answer two questions at once. `envelope` has already
    # refused anything that is not 8 or 10; what is decided here is whether the codec can honour
    # the one that was sent.
    #
    # **THE REASON IS DELIVERY RATHER THAN TASTE, AND THE MESSAGE SAYS SO.** HEVC `main10` is
    # hardware-decoded by essentially every modern playback chain; h264 `High10` is decoded in
    # hardware by almost none. **A master this worker can produce and the wall cannot play is
    # worse than a refusal** — so the caller is told the FACT and not the rule, and can act on it
    # by moving to h265 rather than by learning that we said no.
    #
    # **Refused rather than dropped, which is §6e ruling 2's shape one field over**: a field that
    # cannot be honoured is refused at the door, because a dropped one leaves the caller believing
    # a 10-bit master exists and the record would carry `bit_depth: 10` beside an 8-bit encode.
    if release_3["bit_depth"] != envelope.DEFAULT_BIT_DEPTH and release_3["codec"] != "h265":
        raise WorkerError(
            FIELD_NOT_SUPPORTED,
            "'output.bit_depth: {}' was sent with 'output.codec: {}'. 10-bit is served on h265 "
            "only, and the reason is playback rather than capability: HEVC main10 is decoded in "
            "hardware by essentially every modern chain and h264 High10 by almost none, so a "
            "master this worker can produce and your screen cannot play is worse than this "
            "refusal. Send 'output.codec: h265' with it, or drop the field for 8-bit."
            .format(release_3["bit_depth"], release_3["codec"]))

    # ── §6i — x265's TWO FIELDS ARE REFUSED UNDER h264, THE MIRROR OF §6e RULING 2 ──────
    #
    # **`pools` and `frame-threads` have no x264 spelling**, exactly as `sliced-threads` has no
    # x265 one. x264 parallelises by slices and by its own `threads`; these two are x265's
    # wavefront controls and mean nothing to it.
    #
    # **Refused rather than dropped, and the reason is the one the block below already gives:** a
    # dropped bound leaves the caller believing a bound is in force and the record would carry
    # their value beside an encode that never saw it. *§6b's surviving clause is the same rule
    # one step out.*
    #
    # **AND IT KEEPS THE CORPUS SELF-KEYING.** With the refusal, `frame_threads` on a row implies
    # h265 by construction — the property `docs/archive/instrumentation-archive.md` §15a already relies on for
    # `sliced_threads` implying h264, now holding in both directions.
    if release_3["codec"] != "h265":
        crossed_265 = [name for name in ("frame_threads", "pools")
                       if release_3.get(name) is not None]
        if crossed_265:
            raise WorkerError(
                FIELD_NOT_SUPPORTED,
                "'output.codec: {}' was sent together with {}, which {} x265's wavefront "
                "threading control{} and {} no x264 spelling: x264 parallelises by slices and by "
                "its own 'threads'. Refused rather than accepted and dropped, because a dropped "
                "bound leaves you believing a bound is in force and the record would carry your "
                "value beside an encode that never saw it. Send these with 'output.codec: h265', "
                "or send h264 with 'threads' and 'sliced_threads' instead.".format(
                    release_3["codec"],
                    _english_list(["'{}'".format(name) for name in crossed_265]),
                    "is" if len(crossed_265) == 1 else "are",
                    "" if len(crossed_265) == 1 else "s",
                    "has" if len(crossed_265) == 1 else "have"))

    # ── §6e RULING 2 — x264's THREE FIELDS ARE REFUSED UNDER h265, NOT DROPPED ──────────
    #
    # **CF's ruling, 2026-08-28: *"It's the wrong shape."*** A request carrying `threads`,
    # `sliced_threads` or `rc_lookahead` alongside `codec: h265` is refused here, naming the
    # codec and the field.
    #
    # **THE FIELDS HAVE NO x265 SPELLING.** x265 parallelises by wavefront through `--pools` and
    # `--frame-threads`; **`sliced-threads` has no equivalent at all — it is ABSENT, not
    # renamed** — so there is no honest translation to make, and the only alternatives were
    # refusing and dropping.
    #
    # **DROPPING IS THE WRONG SHAPE BECAUSE OF WHAT THESE FIELDS ARE FOR.** §6a made them fields
    # rather than a pass-through options string *precisely so that a bound on the encoder could
    # not be handed to the caller*: they exist to bound memory on a path §1 says has no host
    # guard. **A dropped bound leaves the caller believing a bound is in force, and the record
    # would show their value beside an encode that never saw it** — a row wrong in the direction
    # nobody checks. *§6b's surviving clause is the same rule one step out: an explicitly-sent
    # field is obeyed and never silently overridden. A field that CANNOT be obeyed is not
    # overridden, it is dropped — the case §6b never had to consider, because until now every
    # field applied to every job.*
    #
    # **AND IT KEEPS THE CORPUS HONEST.** With the refusal, `sliced_threads` on a row implies
    # h264 by construction (`docs/archive/instrumentation-archive.md` §15a). Without it, the field could appear
    # on an h265 row having done nothing.
    #
    # **HERE, AFTER ALL THREE ARE PARSED, AND THAT ORDER IS DELIBERATE.** A caller who sent both
    # a malformed value and the wrong codec is told about the value first — the refusal they can
    # act on without knowing this rule exists.
    if release_3["codec"] == "h265":
        crossed = [name for name, value in
                   (("threads", threads), ("sliced_threads", sliced_threads),
                    ("rc_lookahead", rc_lookahead)) if value is not None]
        if crossed:
            raise WorkerError(
                FIELD_NOT_SUPPORTED,
                "'output.codec: h265' was sent together with {}, which {} x264's own "
                "setting{} and {} no x265 spelling: x265 parallelises by wavefront through "
                "pools and frame-threads, and 'sliced-threads' is absent from it rather than "
                "renamed. Refused rather than accepted and dropped, because a dropped bound "
                "leaves you believing a bound is in force and the record would carry your value "
                "beside an encode that never saw it. Send these with 'output.codec: h264', or "
                "send h265 without them and the encoder is bounded by this worker's own "
                "declared x265 settings.".format(
                    _english_list(["'{}'".format(name) for name in crossed]),
                    "is" if len(crossed) == 1 else "are",
                    "" if len(crossed) == 1 else "s",
                    "has" if len(crossed) == 1 else "have"))

    # **Refused rather than validated-then-dropped**, and this one read as supported harder than
    # any other: `_validate_derive` checked role uniqueness, per-role field strictness and
    # `at_fraction` bounds, so a client probing the surface got a detailed acknowledgement of a
    # feature that produced nothing. A caller asking for a poster and a proxy received
    # `status: DELIVERED`, one file, and no word about the other two.
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
        # The normalised config, one object rather than six loose keys, so a caller downstream
        # cannot read `target_fps` without also having what decided it.
        "release_3": release_3,
        "keep_audio": keep_audio,
        "crf": crf,
        "preset": preset,
        "threads": threads,
        "sliced_threads": sliced_threads,
        "rc_lookahead": rc_lookahead,
        "convert_check": convert_check,
        "tie_check": tie_check,
        "input_check": input_check,
        "decode_probe": decode_probe,
        "reference_score": reference_score,
        "output": _validate_output(job_input["output"]),
        "diagnostics": diagnostics,
        "diagnostics_reserve": reserve,
        "run_record": run_record,
        # **Read at the TOP of `validate` now, because it decides which refusal every level
        # below produces.** It used to be coerced here and read only by the diagnostics summary.
        "debug": debug,
        "force_variant": _variant_name(job_input.get("force_variant")),
        "force_scale": _positive_float_or_none(job_input.get("force_scale"), "force_scale"),
    }
