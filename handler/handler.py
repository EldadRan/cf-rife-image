"""RunPod entrypoint for CF's frame-interpolation worker.

    fetch → probe → load → retime → **write the master** → upload

**One capability and one route.** The upscale path, its estimator, its rung ladder and its
derive-and-manifest half left with the SeedVR2 excision; what remains is `handle` and the retime
it branches to. The order above is not a ladder of stages that can fail independently — decode,
interpolation and encode are ONE streaming loop, the writer pulling each frame through the whole
chain, which is why nothing here reports a phase completing.

**`handle`'s twelve steps are inherited whole and were not refactored while the branch beneath
them was pruned.** Three of them are rulings rather than style: the reserve is remembered off the
RAW input BEFORE validation, because a request that fails validation never produces a `request`;
validation sits INSIDE the `try`, because a `WorkerError` raised there once escaped and the outer
wrapper turned it into `internal`, reporting every bad field as a worker fault; and the run record
is written in the `finally`, because that is the only point a delivered, a refused and a crashed
run all pass through.
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import bake_weights
import diagnostics
import encoder
import errors
import estimator
import hardware
import interp_plan
import keys
import ladder
import phasewatch
import probe
import progress as progress_module
import runrecord
import stages
import storage
import validation
from errors import Remedy, WorkerError

WORKER_VERSION = os.environ.get("WORKER_VERSION", "0.1.0-dev")

#: **The `trace` keys the two gates' live objects hang on.** Constants rather than string
#: literals, because each is written in one function and read in another: a typo in either half
#: produced a record with no block, no exception and no warning — while `diagnostics`' request
#: echo still said the gate had been armed. **An absent block beside `request.input_check: true`
#: is exactly the shape a crashed run produces**, so the two would have been indistinguishable.
CONVERT_CHECK_LIVE = "convert_check_live"
INPUT_CHECK_LIVE = "input_check_live"

#: **§3b-0 item 4 retired `TIECHECK_ENV` and the predicate that read it.** The sweep is armed by
#: `params.tie_check` now, so the question *"is it on"* is answered by `validation` like every
#: other request field, and this module has nothing left to restate. **The env var that armed it
#: was left set, taxed two jobs, and was invisible on the endpoint page** — the failure §2g-1
#: granted its exception in spite of, which is why the exception did not carry.
#:
#: `tiecheck.ENV` still exists and now names the CHUNK-SIZE override alone, which is
#: deployment-scoped by nature: a stale one costs a differently-sized sweep, never a sweep nobody
#: asked for.


def build_identity():
    """What code produced this result, in a form that survives the run.

    **Every number measured before this existed rests on an assertion.** The ten-run calibration
    campaign of 2026-08-17 reports `"image": null` in every bundle, because `IMAGE_TAG` and
    `WORKER_IMAGE` were read here and set nowhere — so the campaign's claim that all ten runs
    share one image comes from "the endpoint was not re-created", not from anything the worker
    saw. A registry keyed on that is keyed on a memory.

    The image cannot know its own digest: the digest is computed when the image is pushed, which
    is after the last layer is sealed. What it can know is the immutable reference it was pushed
    under — `ghcr.io/owner/repo:sha-<commit>`, a tag CI never reuses — and the commit that reference
    names. A digest is one registry lookup away from either, and neither can drift.

    **Every key is always present, null when unknown.** A local build has no honest answer for
    most of these, and a placeholder string that looks like one is worse than a null; but an
    absent key is worse still, because "this build could not identify itself" and "nobody thought
    to ask" then read identically to whoever finds the bundle later.
    """
    return {
        # `IMAGE_TAG` first, then `WORKER_IMAGE`: the second is what the diagnostics path has
        # always looked for, and is kept so an endpoint that sets it by hand still works.
        "image": (os.environ.get("IMAGE_TAG") or os.environ.get("WORKER_IMAGE")),
        "commit": os.environ.get("BUILD_COMMIT"),
        "built_utc": os.environ.get("BUILD_UTC"),
        # **The weights this image actually baked, read from the pin rather than restated.**
        # `rife_weights_sha256` is the ARCHIVE's hash (ruled), not `flownet.pkl`'s — they pin
        # different objects, and the file's own hash is beside it in `bake_weights` if the two are
        # ever wanted to agree. `RIFE_SOURCE_COMMIT` pins the `model` package that loads the
        # weights and has no field here: two builds can report an identical revision and hash and
        # still differ in that code. Raised before the ruling, not blocking, no third field.
        "rife_weights_revision": bake_weights.RIFE_REVISION,
        "rife_weights_sha256": bake_weights.RIFE_ARCHIVE_SHA256,
        "worker_version": WORKER_VERSION,
        # **The weights file this worker loads, read from the pin.** It was `SEEDVR2_MODEL` with
        # a literal default, so removing that ENV would not have nulled it — it would have pinned
        # every envelope and every muxed tag to a SeedVR2 checkpoint this image does not contain.
        "model": bake_weights.WEIGHTS_FILE,
        # **Constant `true`, not an environment reading.** RIFE is baked unconditionally — the
        # `BAKE_WEIGHTS` flag only ever gated SeedVR2 — so there is no build of this repo where
        # this is false, and a flag that cannot vary is a question nobody is asking.
        "weights_baked": True,
        # **Present and null until Phase 2** (§7.1). The value it carried was the version of a
        # registry that cannot quote a route-C job; the key survives because it is the worker's
        # answer to "which version am I talking to". `interp_plan` owns the constant and gets a
        # value when Phase 2 measures route C's line. A pinned or minted string here would be a
        # number indistinguishable from a measurement, which `interp_plan` refuses by name.
        "registry_version": interp_plan.REGISTRY_VERSION,
    }


def identity_tags(request, width, height):
    """The tags muxed into the delivered file. **The model is named only if the image has one.**

    `build_identity()` reports `model: null` on a weightless build (contract §6b), and this is the
    same claim one layer out — a delivered file asserting a checkpoint its image does not contain
    is a falsehood about its own bytes.

    **The key is OMITTED rather than nulled** (CF, 2026-08-23). A null in a container tag becomes
    the string `"None"`, so nulling would have the file name a model called None — a worse
    falsehood than the one §6b prevents, because it asserts an identity rather than declining to
    state one. Absent means absent, which is the same rule as `snap_tolerance`'s missing default
    and route C's unmeasured coefficients. The manifest carries "no model" as a fact, which is
    something a tag key cannot do.
    """
    tags = {
        "cf_request_id": request["request_id"],
        "cf_worker_version": WORKER_VERSION,
        "cf_output": "{}x{}".format(width, height),
    }
    # **Unconditional now, and carrying the same value `build_identity` reports.** The gate here
    # was `WEIGHTS_BAKED`, which existed so a weightless route-C image could decline to name a
    # SeedVR2 checkpoint it did not contain. RIFE is baked on every build of this repo, so the
    # condition cannot be false and a tag key that is always written is not a choice. **The
    # default reading of §7.1**: the tag carries what `model` carries, the weights filename. If
    # CF wants the delivered file stamped with the revision instead, that is a separate word.
    tags["cf_model_build"] = bake_weights.WEIGHTS_FILE
    return tags


#: Printed once per container, on the first job it handles. **Once, not per job**: it describes
#: the machine rather than the work, and a line repeated on every request is a line people learn
#: to skip.
_SAID_BOOT = []

#: **There is no module-level banner list any more, and its removal is the point.**
#: `_HOST_BANNERS` was a job's readings held at module scope, cleared at the top of `handle`. That
#: was harmless while nothing ever appended to it — it was empty on every route-C run ever
#: written. **Giving it a writer (§8b) made it per-job state in a process-wide place**, and a
#: worker run with a concurrency modifier above 1 would then have job B's clear wipe job A's
#: banner: a machine description filed against the wrong run, which is the exact defect the clear
#: was there to prevent. The banner now lives in `trace`, which is created per call and cannot be
#: reached by another job. Nothing was gained by the global and one failure mode was closed by
#: dropping it.


#: **How often a transfer publishes, and it is a THROTTLE rather than a cadence.** *`storage`
#: reports every chunk because it should not be deciding a poll interval; this is where that
#: decision lives.* **Below the client's own poll floor on purpose** — a payload the client never
#: fetches costs nothing, and a transfer that publishes less often than the client polls is a
#: transfer the client watches go stale.
BYTE_REPORT_INTERVAL_S = 3.0


def _byte_reporter(progress, phase_name, pct=None, bytes_per_s=None):
    """A throttled `on_bytes(done, expected)` that publishes `phase_name` with the byte counts.

    **`bytes_expected` MAY BE UNKNOWABLE AND THE PHASE MUST STILL EMIT.** *A name and an elapsed
    time and no percentage is strictly better than silence, and it is the shape `draining`
    already ships.* So an absent expectation drops the key rather than filing a zero — *a zero
    there is a number, and it would read as a transfer of nothing.*

    **NOTHING IN HERE MAY RAISE INTO THE TRANSFER.** *The bytes are already moving; a progress
    payload that abandons a fetch or an upload has cost the delivery it was reporting on.* The
    callers treat a raising callback as a dead one and stop calling it, which is the same rule
    from the other side.
    """
    #: `started` is set on the first report rather than at construction: the transfer begins when
    #: bytes move, and a reporter built ahead of a connection would charge the handshake to the
    #: measured rate.
    state = {"last": 0.0, "started": None}

    def report(done, expected):
        now = time.time()
        if state["started"] is None:
            state["started"] = now
        if now - state["last"] < BYTE_REPORT_INTERVAL_S:
            return
        first = state["last"] == 0.0
        state["last"] = now
        facts = {"bytes_done": int(done)}
        expected_s = None
        if expected:
            facts["bytes_expected"] = int(expected)
            # ── §11: THE SEED SURVIVES EXACTLY ONE EMISSION ────────────────────────────────
            #
            # **CF, 2026-09-02: the first status on fetch and upload may come from previous
            # runs; the next one comes from real progress.** *No byte threshold* — a threshold
            # would be a second thing to choose and would keep the corpus rate alive on exactly
            # the transfers that are behaving unlike the corpus.
            #
            # **The measured rate needs both an elapsed time and bytes to divide**, and a
            # zero denominator is the nearly-right quantity §2b refuses — so a report that
            # arrives with neither falls back to the corpus rate rather than to a division.
            elapsed = now - state["started"]
            rate = None
            if not first and elapsed > 0 and done > 0:
                rate = float(done) / elapsed
                facts["bytes_basis"] = "measured"
            elif bytes_per_s:
                rate = float(bytes_per_s)
                facts["bytes_basis"] = "predicted"
            if rate:
                # **THE REMAINDER, NOT THE WHOLE.** *`expected_s` sets the poll interval and the
                # rule one level up is half the stretch — so a transfer that is nearly finished
                # must say so, or the client waits half of a transfer that has already
                # happened.*
                expected_s = max(0.0, (int(expected) - int(done))) / rate
        try:
            progress.phase(phase_name, pct=pct, force=True, expected_s=expected_s, **facts)
        except Exception as exc:  # noqa: BLE001 — never at the cost of the transfer
            print("[progress] {} bytes not emitted ({}: {})".format(
                phase_name, type(exc).__name__, exc), flush=True)

    return report


def handle(job_input, job=None):
    started = time.time()
    machine = hardware.read()

    # **How many cores this container may actually use, said out loud before anything else.**
    # The phase-4 tail runs at one core's worth while thirty sit idle, and the first question
    # that investigation has to answer is how many cores there were to be idle — a number that
    # was, until now, only obtainable by someone thinking to run `nproc` on a live worker during
    # a tail. Now every log carries it.
    #
    # **TAKEN EVERY JOB, PRINTED ONCE** (`docs/archive/instrumentation-archive.md` §8b). The banner was computed
    # inside the print-once guard, so wiring the append there would have banked a banner on a
    # container's FIRST job and an empty list on every job after it — a record whose `host` field
    # depends on how many jobs the worker happened to have served, which is worse than the empty
    # list it replaces. The print stays once because it describes the machine and a line repeated
    # every job is a line people learn to skip; the READING is per job, because a run record is
    # per run. **Stored as lines**, which is what §8f's `host` names and what a reader of the log
    # sees.
    banner_lines = phasewatch.boot_banner().splitlines()
    if not _SAID_BOOT:
        _SAID_BOOT.append(True)
        print("\n".join(banner_lines))

    # **Kept before the request is validated**, and read straight off the raw input rather than
    # off `request`. A request that fails validation never produces a `request`, and a job that
    # arrives while the reserve is stale is exactly the traffic that refreshes it — so taking it
    # only from validated jobs would drop the refresh on the requests most likely to precede
    # trouble.
    diagnostics.remember_reserve(
        job_input.get("diagnostics_reserve") if isinstance(job_input, dict) else None)

    # **Validation is inside the envelope, not in front of it.** It sat outside the try below
    # until rung 1 caught it: a `WorkerError` raised here escaped `handle`, and the outer
    # `handler()` turned it into `internal` — so every bad field was reported as a worker fault.
    # That is precisely the failure the two error tables exist to prevent, and it costs CF three
    # retries and a wrong diagnosis on a request that will fail identically forever.
    try:
        request = validation.validate(job_input)
    except WorkerError as exc:
        payload = exc.to_dict()
        payload["hardware"] = machine
        payload["execution_ms"] = int((time.time() - started) * 1000)
        return payload

    warnings = []
    attempts = []
    # **What the record needs, filled in as the run learns it.** Its three writers were inside
    # `_run`; `_retime` wrote none of it, so every run record filed `rationale`, `source`,
    # `output` and `load_strip` as null while `_retime` demonstrably had those numbers and
    # returned them in the envelope. `source`, `output` and `retime` were wired with the excision;
    # `load_strip`, the transfer split, the host banner and the ETA are this wave's, ruled by
    # `docs/archive/instrumentation-archive.md` §8. `rationale` stays null and is honest: route C has no planner
    # to have reasoned.
    trace = {}
    workdir = tempfile.mkdtemp(prefix="cf-upscale-")
    # **The host-load series, sampled on the cadence `progress` already publishes at**
    # (`docs/archive/instrumentation-archive.md` §8c). It is handed in as `Progress`'s sampler rather than given
    # a thread: *a measurement whose cost is a new thread is a measurement that changes what it
    # measures*. `trace` holds the live list by reference, so whatever accumulated by the
    # `finally` is what the record files — including on a run that died mid-encode, which is the
    # run whose load history is worth the most.
    load_strip = phasewatch.LoadStrip(started)
    progress = progress_module.Progress(job=job, sampler=load_strip.sample)
    # **Created HERE and not in `_retime`, so every run that files a record carries every
    # field in `stages.STAGES`.** It was created beside the model load — after the fetch, the
    # probe and the fit predicate's refusal — so a job refused for not fitting, or one that
    # died in the fetch,
    # filed `wall_s`/`fetch_s`/`upload_s`/`compute_s` and NONE of the stages. That is the
    # distinction `routec`'s `peak_vram_gb` comment already argues against: an absent key and a
    # measured zero must not read alike, and `_transfer` states the same rule as *"zero where
    # nothing moved, never absent"*. A refused run really did spend zero seconds in the model,
    # and its whole `compute_s` really is residual — both are facts, and both are now filed.
    clock = stages.StageClock()
    trace["clock"] = clock
    # **Per call, which is what makes the concurrency question go away.** `trace` is created here
    # and reaches nothing outside this invocation, so two jobs in one process cannot write each
    # other's host readings — the reason the module-level banner list was retired above.
    trace["host"] = banner_lines

    # **The outcome, recorded where all three exits that REACH IT can be reached.** The run-record
    # is written in the `finally` below because that is the only point every run that enters this
    # block passes through — delivered, refused and crashed alike — and "every run" is the whole
    # content of F-2026-08-19-36: a record written on the success path only rebuilds, in a new
    # place, the sampling bias the finding exists to remove.
    #
    # **AND A VALIDATION REFUSAL RETURNS ABOVE IT, so "every run" is not true of the job as a
    # whole.** That was already so for the route-A/B and codec refusals; refusing an upscale by
    # name moved the DEFAULT request shape into the same class, since a request that does not
    # spell `upscale: false` is now refused in `validation` rather than reaching here. The
    # evidence that this was not the intent is in `_write_run_record` itself, which guards
    # `(request or {})` against a `request` that validation never produced — a defensive branch
    # against a state this early return makes unreachable. **Filed as excision-plan §7.5**, beside
    # §1's response-shape item; not repaired here, because what the worker records is §1's.
    outcome = {"status": "internal", "error": None}
    with diagnostics.LogCapture() as captured:
        try:
            # **Route C branches here, and `handle` keeps its name** (CF, 2026-08-23). The
            # first shape proposed was `handle` -> `handle_org` with a new `handle` for the
            # test; `handler()` below calls `handle` by name, so a test holding that name
            # becomes the production entry point of the next FULL image built from `main` —
            # same tag, same everything, wrong function, and nothing today would show it
            # because low is on the route-C image while medium and high are elsewhere.
            #
            # **That reasoning outlived the branch it was about.** It was written when this was
            # one `if` with the upscale path falling through below it; there is no `if` now and
            # nothing to fall through to. What survives is the half that still binds: whatever
            # this function is called, `handler()` below calls it by name, so the name is load
            # bearing and does not move.
            #
            # **And it returns its retime stats, which is the whole response** (CF). Route C
            # is a test until the samples are seen, so it gets no `configuration`, no
            # rationale and no manifest equivalence — the production response shape is
            # deferred rather than dropped, and nobody can rule it before the samples exist.
            # **One route, so no branch.** This was `if not upscale: _retime() else: _run()`;
            # `_run` left with the upscale path and `validation` now refuses anything that
            # resolves to an upscale, so the else arm was both unreachable and a call to a name
            # that no longer exists — a latent NameError preserved in the shape of a step.
            response = _retime(request, machine, warnings, workdir, progress, started, trace,
                               clock)
            outcome["status"] = "refused" if response.get("cf_error") else "ok"
            outcome["error"] = response.get("cf_error")
            return response
        except WorkerError as exc:
            _write_diagnostics(request, machine, attempts, exc, captured, failed=True,
                               trace=trace, warnings=warnings, job=job, started=started)
            payload = exc.to_dict()
            payload["cf_error"]["log_tail"] = captured.tail()
            outcome["status"] = "refused"
            # The code and the reason, never the log tail: this is a record, not a bundle, and a
            # tail is the bundle's job.
            outcome["error"] = {k: v for k, v in payload["cf_error"].items() if k != "log_tail"}
            return _decorate(payload, machine, attempts, warnings, progress, started)
        except Exception as exc:  # noqa: BLE001 — a job must return an envelope, never raise
            traceback.print_exc()
            _write_diagnostics(request, machine, attempts, exc, captured, failed=True,
                               trace=trace, warnings=warnings, job=job, started=started)
            payload = {"cf_error": {
                "code": errors.INTERNAL,
                "message": "{}: {}".format(type(exc).__name__, exc),
                "log_tail": captured.tail(),
            }}
            outcome["status"] = "internal"
            outcome["error"] = {"code": errors.INTERNAL,
                                "message": "{}: {}".format(type(exc).__name__, exc)}
            return _decorate(payload, machine, attempts, warnings, progress, started)
        finally:
            _write_run_record(outcome, request, machine, attempts, warnings, progress,
                              trace, job, started, load_strip)
            shutil.rmtree(workdir, ignore_errors=True)


def _retime(request, machine, warnings, workdir, progress, started, trace=None, clock=None):
    """Route C end to end: fetch, decode, interpolate, encode, upload. **No model of ours.**

    **The only route.** It was written deliberately not as a second `_run` and not as a copy of
    its master-to-upload half — it calls the same `storage` helpers and nothing else — and that
    restraint is why it survived the excision intact while `_run` did not. Every guarantee `_run`
    carried was about model memory (contract §6), which is not a question this path asks.
    """
    import interpolate as interpolate_module  # noqa: PLC0415 — GPU-box imports, like the rest
    import rife  # noqa: PLC0415
    import routec  # noqa: PLC0415

    download = os.path.join(workdir, "source")
    # **The download's own clock and its own byte count** (`docs/archive/instrumentation-archive.md` §8a). Both
    # were inside `wall_s` with nothing separating them from decode, RIFE and encode: 580 MB in
    # and 742 MB up sharing one figure with the work, on the run whose 44% gap could not be
    # attributed to anything.
    #
    # **The seconds are banked in a `finally`, the bytes only on success.** A fetch that raised
    # still spent the time and the record must account for it or `wall_s` stops adding up; how
    # many bytes arrived before it failed is not something `fetch_source` can say, and 0 would be
    # a number rather than an absence.
    # **THE FETCH ANNOUNCES ITSELF BEFORE IT STARTS, AND THAT IS THE WHOLE OF THE SILENCE.**
    # *The first UNCONDITIONAL emission on this path was `progress.phase("load", …)` further
    # down; the one above it is gated on `request.get("tie_check")` and is not on a production
    # run at all.* **CF measured 176 seconds of silence on a 780 MB source** — two `[quiet]`
    # notices, and a healthy job indistinguishable from a dead one. *Every source this project
    # had tested fetched in about two seconds, so the defect was a function of the input.*
    #
    # **Wrapped, because an emit must never cost a delivery** — the same rule every other emit on
    # this path already follows.
    # **NO `pct`, AND THE FIRST DRAFT PASSED `pct=1`.** *That 1 was a marker for the phase and
    # not a measure of work done — and wave 7's cadence keys on work done, so it answered
    # `MAX_POLL_S`: the announcement written to end 176 seconds of silence told the client to
    # come back in 90.* **A transfer's progress is bytes, and it is carried as bytes** — the
    # cadence for these payloads comes from `expected_s`, or from the floor where no length was
    # declared, which is the honest answer for a stretch nobody can size. *Found in review.*
    try:
        progress.phase("fetching", force=True)
    except Exception as exc:  # noqa: BLE001 — never at the cost of a delivery
        print("[progress] fetching phase not emitted ({}: {})".format(
            type(exc).__name__, exc), flush=True)
    fetch_started = time.time()
    try:
        fetch_bytes = storage.fetch_source(
            request["source_url"], download,
            on_bytes=_byte_reporter(progress, "fetching",
                                    bytes_per_s=ladder.FETCH_BYTES_PER_S))
    finally:
        _note(trace, "timings", "fetch_s", round(time.time() - fetch_started, 3))
    _note(trace, "transfer", "fetch_bytes", fetch_bytes)
    extension = probe.detect_extension(download)
    source_path = probe.named_with_extension(download, extension)
    source = probe.probe_source(source_path)
    # **Into `trace`, so the `finally` can file it.** `trace` is the shared dict `handle` creates
    # for exactly this — a crashed run's most diagnostic numbers are the ones it learned before
    # it died — and `_retime` never wrote to it, so EVERY route-C run record filed `source` and
    # `output` as null, on success as much as on failure. The numbers were in hand both times.
    if trace is not None:
        trace["source"] = {"width": source["width"], "height": source["height"],
                           "fps": source["fps"], "duration_s": source["duration_s"]}

    # **The retime's own fields sit flat in the normalised config now.** *They arrived inside an
    # `interpolate` sub-object that existed to distinguish a retime from an upscale; with one
    # capability it discriminated nothing and it is deleted from the wire and from here.*
    config = request["release_3"]
    # **`scale` is RIFE's flow-pyramid resolution and it reaches two places.** The interpolator
    # pads to a multiple derived from it, and the model call runs motion estimation at it — so a
    # scale set in one and not the other would pad for a geometry the model never sees. One
    # value, both consumers.
    #
    # **Read before the fit predicate rather than after it**, because the padding rule is a
    # function of it and the predicate is a function of the padded area.
    scale = request.get("force_scale") or rife.DEFAULT_SCALE

    # **Contract §9c: a job the predicate rejects is refused with a code and a record**, not by
    # loading a model that will OOM at 90% and not by timing out — *an estimator that refuses
    # silently, or that refuses by timing out, is worse than no estimator*, because the caller
    # cannot tell it from a broken worker.
    #
    # Here rather than before the fetch, and the reason is arithmetic rather than preference: the
    # predicate is a function of the source's dimensions and nothing knows those until the source
    # has been probed. The CLIENT-side use of the same function answers before a job is submitted
    # at all, which is §9a's point; this is the worker's own last check.
    #
    # `LARGER_GPU` with a shortfall, never `NONE`: RIFE holds one frame pair whatever the clip
    # length, so there is no rung to step down to and a bigger card is the entire remedy.
    ok, fit = estimator.fits(source["width"], source["height"], machine, scale=scale)
    # **Banked whether it fits or not, and that is what makes the fit falsifiable.** §9a claims
    # residuals under half a percent; the residual is `peak_vram_gb` measured against
    # `needed_gb` predicted, and `peak_vram_gb` only exists on a run that RAN. Recording the
    # prediction only when the job is refused would keep it off every record that carries the
    # measurement to compare it against — an instrument read on the one path where nothing can
    # check it.
    _note(trace, "estimate", "fit", fit)
    if ok is False:
        raise WorkerError(
            errors.CAPACITY_EXCEEDED,
            "{}x{} at scale {} needs about {} GiB at the peak and this card offers {} GiB "
            "usable ({} total less a {} GiB reserve) — short by {}. RIFE holds one frame pair "
            "whatever the clip length, so there is no smaller configuration to step down to: a "
            "larger card is the only thing that changes this answer. {}".format(
                source["width"], source["height"], scale, fit["needed_gb"], fit["usable_gb"],
                fit["vram_total_gb"], fit["reserve_gb"], fit["shortfall_gb"], fit["basis"]),
            remedy=Remedy.LARGER_GPU, shortfall=fit)
    if ok is None:
        # **Priced nothing, refused nothing.** A snapshot with no `vram_total_gb` is a machine
        # this worker cannot read, and refusing on that would turn an unreadable card into an
        # unservable one. Said out loud so the silence is not mistaken for a pass.
        warnings.append("the fit predicate could not price this job: the hardware snapshot "
                        "reports no vram_total_gb, so nothing was checked")

    # **The tie check, and the environment is read BEFORE the module is imported** — §2g-1's
    # third constraint is that an unset run pays nothing, and an import at the top of this file
    # would be a module read and a parse on every cold start of every job nobody is asking about.
    # `requested()` is a `str.strip()` on an env var; the sweep behind it is a billion
    # comparisons. **`F-2026-08-26-3` is the standing lesson**: an instrument present in the runs
    # it is not being asked about changes what it measures.
    #
    # **Here rather than after the retime, and the reason is the run-record.** `trace` is what a
    # crashed run's numbers survive in, so a sweep banked before the interpolate loop is a sweep
    # that still reaches the record if the retime then dies — and the sweep is the deliverable on
    # this job, while the retime is what keeps the job from being spent purely on an errand
    # (§2g-1). It runs after the fit predicate has refused or passed, so it never allocates on a
    # card the job was about to be turned away from.
    # **The request field is read BEFORE the module is imported**, which is §2g-1's third
    # constraint and the only part of it that survived: a job nobody asked pays not one module
    # read. `request.get` is a dict lookup; the sweep behind it is a billion comparisons.
    if request.get("tie_check"):
        import tiecheck  # noqa: PLC0415 — see above; the import IS the cost being avoided

        progress.phase("load", pct=2, force=True, note="tie check")
        # **Wrapped in `keeping_the_promise`, because the sweep is a longer callback-free stretch
        # than the drain that class was written for.** One payload is minted on the line above and
        # then nothing calls back until a billion comparisons have finished — from outside, a
        # frozen `at` on a `/status` that keeps answering is indistinguishable from a dead worker,
        # which is the 11-minute silence `progress.keeping_the_promise` documents. It would be
        # that silence on the one job the sweep was asked for.
        with progress.keeping_the_promise():
            swept, why_not = tiecheck.run()
        # **Guarded, because `trace` is documented as optional on this path** and every other
        # writer in this function respects that. Unguarded, a direct caller with the variable set
        # would take a `TypeError` AFTER the sweep had finished and thrown its result away.
        if trace is not None:
            trace["tie_check"] = swept
        # **The reason rides the record, not the log** — see `tiecheck.run`. Four distinct
        # failures otherwise produce a record byte-identical to a run that never asked.
        if why_not:
            warnings.append(why_not)

    progress.phase("load", pct=3, force=True, note="interpolator")
    # **`load_s` is the one §9a stage that happens outside `routec`** — the checkpoint read and
    # the cast, which `progress.begin_phase` deliberately excludes from the per-frame rate and
    # which nothing has ever measured on its own. The clock is created here rather than in
    # `retime` so it spans the load as well as the loop, and it is a local handed down as an
    # argument: contract §4b, and `docs/archive/instrumentation-archive.md` §9 says why an attribute would have
    # been wrong on this particular object.
    # **Handed in by `handle`, which banked it in `trace` before anything could fail.** `trace`
    # is the dict a crashed run's numbers survive in, and a clock created down here would lose
    # every stage on exactly the runs the split was built for — the thrashing arm, the reap, the
    # OOM, the refusal. `None` is still supported for a direct caller.
    if clock is None:
        clock = stages.StageClock()
    with clock.timing("load_s"):
        interpolator = interpolate_module.Interpolator(
            rife.Rife.load(scale=scale), scale=scale).prepare()

    master = keys.master_name(False, source["width"], source["height"],
                              name=request["output"].get("name"))
    master_path = os.path.join(workdir, master)
    # **§5-0's gate, and it is created HERE so its evidence outlives a run that dies.** It was
    # built inside `routec.retime` and published only in that function's success return — so a
    # reaped ffmpeg discarded every frame already compared and filed a record indistinguishable
    # from a run that never armed it. Banked in `trace` before the call, which is the dict a
    # crashed run's numbers survive in, and read by the `finally` that writes the record on every
    # exit. **The same argument `stages.StageClock` carries**, one wave later and one object over.
    checker = routec.ConvertCheck() if request.get("convert_check") else None
    if trace is not None and checker is not None:
        trace[CONVERT_CHECK_LIVE] = checker
    # §3b-1's gate, hoisted for the same reason: a run reaped mid-encode still files what it
    # compared. `_tensors` is consumed lazily inside the stream, so this object outlives the
    # decode as well as the encode.
    input_checker = routec.InputCheck() if request.get("input_check") else None
    if trace is not None and input_checker is not None:
        trace[INPUT_CHECK_LIVE] = input_checker

    # **§6d's provenance container, HANDED IN AND BANKED BEFORE THE ENCODE.**
    #
    # **The branch itself is NOT here, and that is §6d-1's correction.** It keys on the DELIVERED
    # frame — `width * height` of what `MasterWriter` is constructed with — and this function
    # holds `probe.probe_source`'s dimensions, which are a different reader's answer. The branch
    # lives in `routec.retime`, at the two locals the writer is built from, so the branch and the
    # encode cannot disagree about how big the frame is.
    #
    # **An empty dict rather than a return value**, for `ConvertCheck`'s reason exactly: `retime`
    # fills it in place, and this object is already in `trace` by then — so a run reaped inside
    # ffmpeg still files what its encoder settings were resolved from. A provenance travelling
    # out in `stats` would be null on precisely the runs worth recording.
    encode_defaults = {}
    if trace is not None:
        trace["encode_defaults"] = encode_defaults
        # **§15c READS AN ABSENT `codec` AS h264, SO A RUN THAT DIES WITHOUT FILING ONE IS NOT A
        # GAP — IT IS A WRONG VALUE.** Every record written before §15 was produced by an
        # unconditional `libx264`, so absence is h264 by construction and the kit states that as
        # an inference rule. A reaped h265 run filing nothing would be read as an h264 row by
        # that rule, and averaged into the h264 corpus by everything downstream of it.
        #
        # **So it is banked BEFORE the encode, from the validated request** — the same argument
        # `encode_defaults` above is banked for, one degree stronger: that block's absence is
        # honest, and this one's is not. **Overwritten below with what the writer actually
        # used**, which is the value §15b requires and the only one that stays true the day
        # `codec: "source"` is implemented and the ask stops equalling the outcome.
        trace["codec"] = request["release_3"]["codec"]
        # §6f, banked for §15c's reason exactly: absence on the record means 8, so a reaped
        # 10-bit run that filed nothing would be read as an 8-bit row — a wrong value, not a gap.
        trace["bit_depth"] = request["release_3"]["bit_depth"]

    progress.phase("interpolate", pct=10, force=True)
    stats = routec.retime(
        source, source_path, master_path, interpolator,
        target_fps=config["target_fps"], identity=identity_tags(
            request, source["width"], source["height"]),
        snap_tolerance=config["snap_tolerance"],
        crf=request.get("crf"),
        # **§6a's other four. `preset` is still `validation`'s to default; the three below are
        # §6d's and arrive already resolved from the branch above.** Passed by name rather than
        # assembled here: the worker builds the encoder's parameter string from validated fields
        # and never accepts one, which §6a rules and `encoder.x264_params` implements.
        preset=request.get("preset"),
        # **PASSED THROUGH AS `None` WHERE THE CALLER SENT NOTHING**, which is the whole of §6d's
        # mandatory property: absence stays distinguishable from a value all the way to the
        # branch, and the branch is downstream in `retime` where the delivered frame exists.
        threads=request.get("threads"),
        sliced_threads=request.get("sliced_threads"),
        convert_check=checker,
        input_check=input_checker,
        rc_lookahead=request.get("rc_lookahead"),
        encode_defaults=encode_defaults,
        # **§6e. One field, and it is the one this repository already had** — `envelope` has
        # carried `CODECS` since `rife-seed` and `validation` refused everything but the default;
        # what moved is the refusal. Passed through by name so the library is chosen in exactly
        # one place, `encoder.CODEC_LIBRARIES`.
        codec=request["release_3"]["codec"],
        # **§6i's two x265 levers, read from the same release-3 block as `codec` and passed
        # through as `None` where the caller sent nothing** — §6d's mandatory property, applied
        # to two more fields. `envelope.derive` puts them on both of its return paths, so this
        # `.get` cannot be the reason one is missing.
        frame_threads=request["release_3"].get("frame_threads"),
        pools=request["release_3"].get("pools"),
        # §6f, travelling beside the codec because they are refused as a PAIR at the door.
        bit_depth=request["release_3"]["bit_depth"],
        # **contract §6g's fifth gate.** `retime` computes the disk bound from the delivered
        # frame size and the planned count and REFUSES before the first frame, so an unaffordable
        # request costs nothing rather than the whole model pass.
        reference_score=request.get("reference_score"),
        # **`docs/archive/instrumentation-archive.md` §8g-1's other half, and it is the half that makes the kit's
        # decline honest rather than convenient.** §8g-1 lets the acceptance kit DECLINE TO GRADE
        # an armed run's first ETA, because the instruments land in `wall_s` and §12 rules that
        # pollution not recoverable by subtraction. **A gate that looks away while the estimate
        # says nothing would hand an armed run's caller a silently wrong ETA**, which is a worse
        # arrangement than the failing row it replaces — so the estimate names the instruments and
        # the decline follows from something the answer said out loud.
        #
        # **Read off the REQUEST and not off the two live objects below it.** `convert_check` and
        # `input_check` reach `retime` as objects, but `tie_check` and `decode_probe` do not reach
        # it at all — they run in this function. Deriving the list from what `retime` happens to
        # hold would silently answer for two of the four, which is the shape of a check that runs
        # in the direction the code takes.
        # **`reference_score` joins as the FIFTH** (contract §6g). §12's rule is the same for
        # all five — an armed run's `compute_s` is comparable to nothing — and this is the one
        # that also makes the run's DISK and its `write_wait_s` incomparable, because retention
        # writes to the same volume the master is written to.
        armed=[field for field in ("convert_check", "input_check", "tie_check", "decode_probe",
                                   "reference_score")
               if request.get(field)],
        # **Passed at last.** `retime` has declared this parameter since it was written and
        # `_retime` never supplied it, so the retime path published two payloads for a
        # 239-second job — `progress_emitted: 2`, the worker counting its own silence.
        progress=progress,
        audio_source=source_path if request["keep_audio"] else None,
        variant=request.get("force_variant") or "direct", scale=scale, clock=clock)

    client = storage.client_for(request["output"])
    # **The upload's own clock and byte count, same rule as the fetch** (§8a). The size is read
    # before the PUT rather than after it: the object on the far side is what the byte count is
    # ABOUT, and reading it from the local file is the only reading that cannot have been
    # truncated by the very failure the number would be explaining.
    upload_bytes = os.path.getsize(master_path)
    # ── §18's THIRD PHASE, AND THE ONLY ONE WITH AN HONEST FIGURE ───────────────────────────
    #
    # **DERIVED FROM THIS JOB'S OWN FETCH, NOT FROM THE CORPUS.** `estimator` carries
    # `TIME_TRANSFER_PER_MP`, but it was fitted against fetch AND upload together and is a rate
    # per MEGAPIXEL, not per byte — using it for one direction of one transfer would be a fitted
    # aggregate spent on something it does not describe.
    #
    # **The fetch that already happened on THIS job, over THIS link, to THIS worker is the better
    # instrument**, and it is `_next_poll_s`'s own argument: *"the worker is the only party that
    # knows the answer, and it knows it by measurement."* Bytes per second observed inbound,
    # applied to the bytes about to go out.
    #
    # **`None` WHENEVER THAT DIVISION IS NOT AVAILABLE** — a reused source transfers nothing, a
    # zero-second fetch measures nothing — and then §18b's floor is the answer. *A rate computed
    # from a zero denominator is precisely the nearly-right quantity §2b refuses.*
    # **The whole derivation is wrapped, because it runs AFTER the master exists.** `float()` on
    # a trace value that is not a number raises, and `.get` on a `timings` that is not a mapping
    # raises — either would kill a job between writing the product and uploading it, for a poll
    # interval. *A support computation for an emit inherits the emit's rule: it must not be able
    # to cost a delivery.*
    # ── §11: THE UPLOAD IS PRICED AT THE UPLOAD'S OWN CORPUS RATE ──────────────────────────
    #
    # **THIS WAS DERIVED FROM THE FETCH RATE AND WAS 6.7x LOW.** *A delivered 8K run published
    # `phase_expected_s: 35.2` for an upload that took 236.4 s, so the client polled at 17 s
    # through 200 seconds of silence and collected two quiet notices.* **The two directions are
    # not one mechanism**: the corpus puts the fetch median at 54.5 MB/s and the upload's at 13.1,
    # a factor of four before any per-run variation — *and the fetch spread is 35.5x against the
    # upload's 3.3x, so the run's own fetch was the least reproducible number available to price
    # it with.* **From the corpus rate it would have said 214 s against 236, a 9 per cent error.**
    try:
        _upload_expected = (float(upload_bytes) / ladder.UPLOAD_BYTES_PER_S
                            if upload_bytes else None)
    except Exception:  # noqa: BLE001 — no figure is a state §18b already rules honest
        _upload_expected = None
    try:
        progress.phase("uploading", force=True, expected_s=_upload_expected)
    except Exception as exc:  # noqa: BLE001 — never at the cost of a delivery
        print("[progress] uploading phase not emitted ({}: {})".format(
            type(exc).__name__, exc), flush=True)
    upload_started = time.time()
    try:
        master_key = storage.upload(client, request["output"], master, master_path,
                                    keys.content_type(master),
                                    on_bytes=_byte_reporter(
                                        progress, "uploading",
                                        bytes_per_s=ladder.UPLOAD_BYTES_PER_S))
    finally:
        _note(trace, "timings", "upload_s", round(time.time() - upload_started, 3))
    _note(trace, "transfer", "upload_bytes", upload_bytes)

    # **THE KEY IS BANKED THE INSTANT THE OBJECT EXISTS, AND EVERYTHING ELSE FILLS IN LATER.**
    # `master_key` lived in exactly two places — the assignment above and `output_entry`, which
    # was built a hundred lines below out of `probe.probe_output(...)` on its first line. So a
    # raise anywhere in between returned `internal` with a ~140 MB master sitting in the bucket,
    # ~500 s of GPU paid, and nothing in the response or the run record naming where it went.
    # **Not the armed path — every delivery.** *Found in review; the gate filed it as
    # F-2026-08-29-12.*
    #
    # **Banked HERE rather than merely wrapping the three reads that were named.** Between this
    # line and where `output_entry` used to be built sit the §6g PNG uploads and §11's decode
    # probe — three full re-decodes at a half-hour timeout each. Guarding only the probe reads
    # would have left the same defect behind the slowest thing on the path. *The dict is mutated
    # in place below, so this is the same object the record files.*
    output_entry = {
        "key": master_key,
        # The same reading the transfer block banked, not a second one: two `getsize` calls on
        # one file are two chances to report two numbers for one fact.
        "bytes": upload_bytes,
        "channels": 3,
    }
    if trace is not None:
        trace["output"] = output_entry

    # ── contract §6g's worst-frame PNGs, uploaded beside the master ──────────────────────────
    #
    # **§17b: the PNG is not a formality.** *An aggregate nobody can look behind is the state
    # §6e's artefact ruling exists to leave* — the scores say a frame is the worst one and the
    # picture is what lets a person agree or disagree.
    #
    # **HERE, after the master's upload, and never before it.** The master is the deliverable and
    # these are evidence about it; a diagnostic upload that ran first would put its own failure
    # between a finished master and its only copy, which is the correction §11 already made to
    # the decode probe. **`worst_pngs` is rewritten from local paths to KEYS**, because a record
    # naming `/tmp/...` describes a file nobody else can open.
    # **READ OFF `stats` AND NOT OFF `trace`, and the difference is a silent failure.** `trace`
    # does not carry this block until forty lines below, where the retime stats are filed — so a
    # read from `trace` here is None on every run, no PNG is ever uploaded, and `worst_pngs`
    # reaches the record still holding LOCAL paths. **Nothing would have caught it**: the kit
    # grades `frames` and the three metric blocks and never opens the keys, so the record would
    # have passed while naming files nobody else can read. *It is the same dict object either
    # way, so rewriting `worst_pngs` here is what `trace` files below.*
    reference_block = stats.get("reference")
    if reference_block:
        keys_uploaded = []
        for png_path in reference_block.get("worst_pngs") or []:
            try:
                name = os.path.basename(png_path)
                keys_uploaded.append(storage.upload(client, request["output"], name, png_path,
                                                    keys.content_type(name)))
                _add(trace, "transfer", "upload_bytes", os.path.getsize(png_path))
            except Exception as exc:  # noqa: BLE001 — evidence must never cost a delivery
                print("[reference] worst-frame upload failed for {}: {}".format(
                    png_path, exc), flush=True)
        # **Only the keys that actually landed.** A key listed for an upload that failed would
        # make the record claim evidence it does not have — §17b's own point, one step earlier.
        reference_block["worst_pngs"] = keys_uploaded
        # **BANKED HERE, THE INSTANT THE KEYS ARE REAL, AND NOT FIFTY LINES DOWN.** The §11 decode
        # probe sits between this point and where the retime stats are filed, and it is three full
        # re-decodes at a half-hour timeout each — so a raise or a reap in that span would leave a
        # run that reserved the disk, wrote the reference, scored it, cut and uploaded the PNGs
        # and paid for all of it, filing a record with no `reference` key. **§17a rules that
        # absence to mean nobody asked**, so the most expensive instrument in the worker would
        # report as unarmed on exactly the runs that went wrong. *Found in review; it is the same
        # failure the missing `reference=` argument caused, narrowed to a slow one.*
        if trace is not None:
            trace["reference"] = reference_block

    # **`docs/archive/instrumentation-archive.md` §11, and it runs AFTER THE UPLOAD — which is a correction.**
    # It sat between the finished master and its upload, where three full re-decodes at a
    # half-hour timeout each put **up to 5400 seconds between a master existing on disk and
    # existing anywhere else.** The master lives in `workdir`, which `handle`'s `finally`
    # removes, so an execution timeout or a reap during the probe would have destroyed a master
    # that was already made — and no exception handling inside the probe covers that, because it
    # is not an exception. **A probe must never cost a delivered master, and in that position it
    # could.** Below the upload it costs only itself.
    #
    # **After the retime for a second reason that still holds: `frames`.** §11c grades the block
    # against `retime.n_in`, the DECODER'S count, which does not exist until the decode has
    # happened — the alternative was recounting the source here, a second answer to "how long is
    # this clip".
    #
    # **Wrapped in `keeping_the_promise`**, which the tie check eighty lines above already uses
    # for the same reason: nothing calls back into `Progress` for the whole of three decodes, and
    # a frozen `/status` on a healthy worker is the 11-minute silence that class was written for.
    if request.get("decode_probe"):
        import decodeprobe  # noqa: PLC0415 — an unasked run pays not one module read

        progress.phase("upload", pct=99, force=True, note="decode probe")
        with progress.keeping_the_promise():
            probe_block = decodeprobe.run(source_path, stats.get("n_in"))
        if trace is not None:
            trace["decode_probe"] = probe_block

    # **The same `output` shape the upscale path returns, and that is a correction rather than a
    # choice.** Route C first returned `{"master": key}` — a shape nobody reads. `run_one` takes
    # `output.width`, `output.height` and `output.bytes` off this object and banks them in the
    # ledger row, so the first route-C job on hardware delivered a correct file and reported
    # `output NonexNone 0 bytes` with every measured field null. The response-shape question CF
    # deferred is about `configuration` and the rationale; this is not that. This is the existing
    # output contract, which route C has no reason to differ from and every reason to satisfy —
    # the variant runs are the ones whose numbers matter, and a wave that banks nulls is a wave
    # measured by hand.
    # **Each read guarded SEPARATELY**, because they fail independently and a measurement that
    # survived should not be discarded by one that did not. A missing field is a gap; a delivery
    # reported as `internal` is a lie about where the master is.
    for field, read in (("content_type", lambda: keys.content_type(master)),
                        ("faststart", lambda: probe.is_faststart(master_path)),
                        # Measured on the far side of the encode, which is the only frame count
                        # worth reading from a container — and on this path it is also the one
                        # the witness compares.
                        ("frames", lambda: probe.written_frame_count(master_path))):
        try:
            output_entry[field] = read()
        except Exception as exc:  # noqa: BLE001 — a measurement must never displace a delivery
            warnings.append("{} could not be measured on the delivered master ({}: {})".format(
                field, type(exc).__name__, exc))
    try:
        delivered = probe.probe_output(master_path)
    except Exception as exc:  # noqa: BLE001 — same rule, and this one is the whole container read
        warnings.append("the delivered master could not be probed ({}: {})".format(
            type(exc).__name__, exc))
        delivered = {}
    # **`setdefault`, which preserves the precedence the previous shape had.** `dict(delivered)`
    # then `.update({...})` meant the explicit fields above won over the probe's; filling only
    # what is absent is the same rule stated the other way round, and it keeps `bytes` the
    # transfer's reading rather than a second `getsize`.
    for field, value in delivered.items():
        output_entry.setdefault(field, value)

    # **The stats and what they were measured on, and nothing shaped like a plan.** A
    # `configuration` block here would look, to anything reading the envelope, exactly like a
    # planned upscale — and there is no plan, because there is no model to plan for.
    _note(trace, "estimate", "time", stats.get("estimate"))
    if trace is not None:
        # **Deliberately left, and it is a no-op.** `output_entry` was banked into the trace the
        # instant the upload returned, and this is the same dict object mutated in place — so
        # this line rebinds what is already there. It stays because deleting it would leave the
        # one place a reader looks for "where does output reach the record" empty, and a fact
        # with no visible home is how the next reader concludes there isn't one.
        trace["output"] = output_entry
        # **The retime stats have NO honest slot in `runrecord.build`.** `plan` and `rationale`
        # are the estimator's fields and route C has no estimator, so putting them there would
        # make a record borrow a slot that means something else — and a record that borrows a slot
        # is worse than one with a gap, because the gap is visible. Filed to the gate rather than
        # forced; `load_strip` is the nearest thing and it is not that either.
        # **Without `estimate`, which has one home and it is the `estimate` block.** The time
        # answer travels out of `routec` inside the stats because that is the only channel it
        # has, and filing it here as well would put one fact in two places in one document — the
        # duplication that makes a stale copy indistinguishable from a live one.
        # **`convert_check` is excluded alongside `estimate` and lifted to the top level.** The
        # kit grades `convert_check.frames` AGAINST `retime.n_out`, and a check nested inside the
        # block it is checked against reads as part of that measurement rather than as the check
        # on it.
        # **`codec` is lifted out for §15's reason and not `convert_check`'s.** §15c's
        # inference rule reads `record.get("codec", "h264")` at the record's TOP level — a
        # `codec` nested inside `retime` is invisible to it, and a row that carries the field
        # somewhere the rule does not look is a row the rule reads as h264.
        trace["retime"] = {k: v for k, v in stats.items()
                           if k not in ("estimate", "convert_check", "input_check",
                                        "codec", "bit_depth", "reference")}
        # **What RAN, replacing what was asked for.** They are the same value today — §6e leaves
        # `codec: "source"` refused, so there is no resolution step between the ask and the
        # outcome — and this line is what makes that a fact about the writer rather than a fact
        # about the request that happens to hold.
        trace["codec"] = stats.get("codec", trace["codec"])
        trace["bit_depth"] = stats.get("bit_depth", trace["bit_depth"])
        # **§17a: banked ONLY when the instrument produced one**, and banked ABOVE, as soon as
        # the uploaded keys exist. This is the fallback for a run that produced scores and never
        # reached the upload — the same dict either way, so it cannot disagree with itself.
        if stats.get("reference") is not None and "reference" not in trace:
            trace["reference"] = stats["reference"]

    return _decorate({
        "status": "DELIVERED",
        "route": "C",
        "output": output_entry,
        # **`crf` and `preset` beside the x264 params** (instrumentation §2). §6a rules five
        # encode settings changeable with today's values as defaults; the record carried three, so
        # a corpus could not attribute a difference between two runs to the settings that differed.
        # Read from the same place the encoder reads them, so a default cannot be restated wrongly.
        # **The stats, whole, with nothing restated here.** This block used to add `target_fps`,
        # `snap_tolerance`, the request's `crf` and the module's `DEFAULT_PRESET` — to the
        # ENVELOPE only. `trace["retime"]` is built from `stats`, so all four were null in every
        # run record ever written while the envelope beside them carried values: the corpus could
        # not attribute a difference to the settings that differed, which is exactly what §2 says
        # recording three of five costs. `routec` now reads all five off the writer that ran and
        # returns them, so both artefacts carry one set of numbers that cannot disagree.
        # **`codec` STAYS HERE, AND THIS TUPLE DELIBERATELY DIFFERS FROM THE RECORD'S ABOVE.**
        # The record lifts it out because §15c's inference rule reads `record["codec"]` at the
        # top level and a nested copy would be both invisible to the rule and a second home for
        # one fact. **The envelope has no such rule and no top level to lift to**, so stripping
        # it here would leave a caller holding `x265_params` non-null, `x264_params` null and
        # nothing that names the codec — a response whose codec has to be INFERRED from which of
        # two strings came back empty. *One fact, one home, per document: the record's home is
        # its top level and the envelope's is here.*
        "retime": {k: v for k, v in stats.items()
                   if k not in ("estimate", "convert_check", "input_check", "reference")},
        # **`padded_megapixels` is the fit's independent variable and nothing computed it**
        # (instrumentation §1). Raw `width × height` and padded area differ by the padding rule —
        # `max(128, 128/scale)` per dimension — so a corpus banked on dimensions and a predicate
        # written against padded area are two axes that agree on nothing in particular, and the
        # difference is recoverable only by someone who remembers the padding rule of the day the
        # row was written. READ FROM `interp_plan`, which owns the rule, so the two cannot disagree.
        "source": {"width": source["width"], "height": source["height"],
                   "fps": source["fps"], "duration_s": source["duration_s"],
                   "padded_megapixels": interp_plan.padded_megapixels(
                       source["width"], source["height"], scale)},
        "build": build_identity(),
    }, machine, [], warnings, progress, started)


def _timings(trace, started):
    """§8f's `timings` block, with the wall split into the three activities that make it up.

    **`compute_s` is computed HERE and not by whoever reads the file** (§8a). A subtraction a
    reader performs is a subtraction a reader can perform differently, and the close condition
    checks the three parts against the whole — so the parts have to be the worker's arithmetic or
    the check is grading the reader.

    **Each part is rounded BEFORE the remainder is taken**, which is what makes the identity
    exact rather than exact-to-a-tolerance: `fetch + upload + (wall - fetch - upload)` is `wall`
    whatever the rounding, while rounding the remainder afterwards can drift by half a tick in
    each of three places.

    A run that never reached the fetch reports zeros for both transfers, and `compute_s` is then
    the whole wall — which is true of it: nothing was transferred.
    """
    measured = (trace or {}).get("timings") or {}
    wall = round(time.time() - started, 1)
    fetch_s = round(float(measured.get("fetch_s") or 0.0), 1)
    upload_s = round(float(measured.get("upload_s") or 0.0), 1)
    # **The stages, and the residual computed against the `compute_s` on the line below**
    # (§9a, §10a). Merged here rather than in `_retime` because that is where `compute_s` first
    # exists: the residual is `compute_s` less the stages, and a stage total banked before the
    # wall was stamped could not have known it. **Counted nowhere here on purpose** —
    # `stages.STAGES` is the one list, and §10a is the second time its length has changed.
    clock = (trace or {}).get("clock")
    compute_s = round(wall - fetch_s - upload_s, 1)
    # **A fresh empty clock rather than an empty dict when there is none.** `handle` always banks
    # one, so this is the path a direct caller or a future entry point takes — and it must file
    # them all as zeros with the residual carrying the whole of `compute_s`, not omit them. An
    # absent key and a measured zero reaching a ledger row identically is the confusion this
    # project has now paid for three times.
    stage_totals = (clock or stages.StageClock()).totals(compute_s=compute_s)
    return dict(stage_totals, **{
        "wall_s": wall,
        # The worker's own clock, which is the only one that is not somebody else's view
        # of this job — and the figure F-2026-08-19-35 showed a client cannot be trusted
        # to reconstruct after the fact.
        "started_utc": datetime.datetime.fromtimestamp(
            started, datetime.timezone.utc).isoformat(),
        "fetch_s": fetch_s,
        "upload_s": upload_s,
        "compute_s": compute_s,
    })


def _transfer(trace):
    """§8f's `transfer` block. **Zero where nothing moved, never absent.**

    A run that was refused before the fetch transferred nothing, and `0` says that. What it must
    not do is say `0` about a transfer that happened and was not counted — which is why the byte
    counts are banked on the success of each transfer and not in its `finally`.
    """
    measured = (trace or {}).get("transfer") or {}
    return {"fetch_bytes": int(measured.get("fetch_bytes") or 0),
            "upload_bytes": int(measured.get("upload_bytes") or 0)}


def _add(trace, block, field, value):
    """Accumulate into a `trace` measurement rather than replacing it. **Never raises.**

    `_note` is for a quantity measured once. This is for one measured in pieces — the upload
    seconds and bytes, which a failed run pays twice: once for the master it did not write, and
    once for the bundle explaining why.
    """
    if trace is None:
        return
    try:
        block_dict = trace.setdefault(block, {})
        block_dict[field] = (block_dict.get(field) or 0) + value
    except Exception:  # noqa: BLE001 — a measurement must never displace a real failure
        pass


def _live_block(trace, key):
    """A gate's block off its live object. **Never raises** — this runs inside the record's
    `finally`, on exactly the crashed runs those objects were hoisted out of `retime` to serve."""
    try:
        checker = (trace or {}).get(key)
        return checker.block() if checker is not None else None
    except Exception:  # noqa: BLE001 — a record must never cost a delivered master
        return None


def _convert_check_block(trace):
    """§5-0's block off the live checker."""
    return _live_block(trace, CONVERT_CHECK_LIVE)


def _note(trace, block, field, value):
    """Bank one measurement into `trace`, creating the block. **Never raises.**

    `trace` is the dict a crashed run's most diagnostic numbers survive in, and every writer of
    it so far has been a guarded assignment at the call site. Four more of those would be four
    more places to forget the guard — and these four run in `finally` blocks, where an
    AttributeError would replace the exception the `finally` was unwinding with one about
    bookkeeping.
    """
    if trace is None:
        return
    try:
        trace.setdefault(block, {})[field] = value
    except Exception:  # noqa: BLE001 — a measurement must never displace a real failure
        pass


def _write_run_record(outcome, request, machine, attempts, warnings, progress, trace, job,
                      started, load_strip=None):
    """Assemble and file the run-record. **Never raises** — see `runrecord`'s own posture.

    Wrapped even though `runrecord.write` cannot raise, because *assembling* the body reads a
    dozen fields off structures that a crashed run may have left half-built, and the whole point
    of writing this in a `finally` is that it runs on exactly those jobs.
    """
    try:
        source = (trace or {}).get("source") or {}
        document = runrecord.build(
            outcome.get("status"),
            build_identity(),
            machine,
            request=request,
            rationale=(trace or {}).get("rationale"),
            source=source,
            attempts=attempts,
            output=(trace or {}).get("output"),
            retime=(trace or {}).get("retime"),
            # §13. **Read out of `trace`, which is where the branch banked it before the encode
            # began**, so a run reaped in ffmpeg still files what its settings were resolved from.
            # Null on any run that died before the branch, which is the honest shape of a job
            # whose encode was never configured.
            # §13a: **present-and-null, never omitted.** The branch runs on every job, so an
            # absent block is a defect and not a state — but the container is created before the
            # branch, so an empty dict means "died before it ran" and must file as null rather
            # than as `{}`, which would read as a provenance that was recorded and said nothing.
            encode_defaults=(trace or {}).get("encode_defaults") or None,
            # §15. **Top level, where §15c's inference rule reads it**, and OMITTED rather than
            # null on a run that never got as far as banking one — `record.get("codec", "h264")`
            # returns `None` for a present-and-null field and "h264" for an absent one, so the
            # two spellings of "we do not know" are not interchangeable here. This is the
            # opposite of `encode_defaults` above, and the difference is which of absence and
            # null the reader was told to interpret.
            codec=(trace or {}).get("codec"),
            # §6f, and OMITTED rather than null for `codec`'s reason: §15a rules absence to mean
            # 8, so a present-and-null field would be a row that is neither depth.
            bit_depth=(trace or {}).get("bit_depth"),
            # **§17. THIS LINE IS THE WHOLE INSTRUMENT'S ONLY WAY ONTO THE RECORD, and it was
            # missing** — found in review. Without it §6g ran in full on every armed job: the
            # disk was reserved, the reference written, the scoring performed, the PNGs uploaded
            # and paid for — and the record filed nothing. **And the failure was invisible by
            # construction**, because §17a rules an absent block to be the UNARMED state, so the
            # kit reports `Skip` and a reader cannot tell "nobody asked" from "it ran and its
            # answer was dropped". *The most expensive instrument in the worker, reporting as
            # though it had never been switched on.*
            reference=(trace or {}).get("reference"),
            # §2g-2. Absent on every run that did not sweep, which is every run: the kit's
            # `--tie-check` REQUIRES the block rather than shrugging when it is missing, so a
            # null here would read as "swept and found nothing" to nobody and as a missing field
            # to the one caller that grades it.
            tie_check=(trace or {}).get("tie_check"),
            # §5-0. **Read off the live object rather than off the stats**, so a run that died
            # mid-encode still files what it had compared. Absent on every run that did not arm
            # the gate, for the same reason `tie_check` is: `--convert-check` REQUIRES the block.
            convert_check=_convert_check_block(trace),
            # §3b-1, read off its own live object for the same reason.
            input_check=_live_block(trace, INPUT_CHECK_LIVE),
            # §11a. Already a plain dict when it exists, so no live object to read through.
            decode_probe=(trace or {}).get("decode_probe"),
            # **A snapshot, not the live list.** The sampler stops when the job does and not
            # when the record is assembled, so handing `json.dumps` a list something may still be
            # appending to is handing it a list that can reallocate underneath the walk.
            load_strip=(load_strip.snapshot() if load_strip is not None else None) or None,
            host_banners=(trace or {}).get("host") or [],
            timings=_timings(trace, started),
            transfer=_transfer(trace),
            # **What was predicted, beside what happened.** Both halves of contract §9 in one
            # block: `fit` on every run that got as far as probing a source, `time` on every run
            # that got a plan. Null halves are the honest shape of a run that died earlier.
            estimate=(trace or {}).get("estimate"),
            # **As published, read off `progress` rather than recomputed** (§8f). Recomputing at
            # exit would report the ETA the run deserved instead of the one the caller got.
            eta=progress.first_eta() if progress is not None else None,
            progress=progress,
            job=job,
            error=outcome.get("error"),
            warnings=warnings,
        )
        # **The address came with the job**, like the bundle's always has. `request` may be None
        # if validation itself failed — in which case there is no URL to have been given, and the
        # skip path says so.
        runrecord.write(document, (request or {}).get(runrecord.REQUEST_FIELD))
    except Exception as exc:  # noqa: BLE001 — a record must never cost a delivered master
        print("[run-record] NOT assembled ({}: {}). The job is unaffected.".format(
            type(exc).__name__, str(exc)[:200]))


def _write_diagnostics(request, machine, attempts, exception, captured, failed,
                       trace=None, warnings=None, job=None, started=None):
    """Never fails the job. The one outcome worse than losing the diagnostics is losing the
    result because the diagnostics could not be stored."""
    try:
        body = diagnostics.bundle(
            request["request_id"], machine, attempts,
            exception=exception, log_text=captured.text(),
            request=request, rationale=(trace or {}).get("rationale"),
            warnings=warnings, job=job, started=started,
            build=build_identity(),
            extra={"source": (trace or {}).get("source")})
        # The job's own destination first, the kept reserve second. They are not interchangeable:
        # the per-job URL is what CF correlates with the request, and the reserve exists for the
        # case where there is no per-job URL to use — a request that never carried one, or one
        # whose `diagnostics` could not be minted at submit.
        # **Clocked and counted, because otherwise its seconds hide in `compute_s`.** This runs
        # inside `handle`'s except block, before the `finally` stamps `wall_s` — so a bundle PUT
        # on a failed run was time the record attributed to computing while `transfer.upload_bytes`
        # said zero bytes moved. The three-way identity held arithmetically and lied semantically,
        # which is the harder failure to notice.
        #
        # **Added to `upload_s`, not given a fourth term**, because §8g's close condition is that
        # three parts account for the wall and a fourth would break the check that makes the split
        # worth having. So `upload_*` means every byte this worker pushed out — the master and the
        # bundle both — and on a failed run there is no master, so it is exactly the bundle.
        # **The run record's own PUT can never be in here**: it is written after this figure is
        # taken, and a record cannot time its own write.
        bundle_started = time.time()
        try:
            storage.put_diagnostics(
                request.get("diagnostics") or diagnostics.reserve(), body)
        finally:
            _add(trace, "timings", "upload_s", round(time.time() - bundle_started, 3))
            _add(trace, "transfer", "upload_bytes", len(body.encode("utf-8", "replace"))
                 if isinstance(body, str) else len(body or b""))
    except Exception:  # noqa: BLE001 — see the docstring
        pass


def _decorate(payload, machine, attempts, warnings, progress, started):
    # **THE THREE CPU NUMBERS MOVED INTO `hardware.read`, WHERE THE RECORD CAN SEE THEM.**
    # This block added `usable_cores` and `affinity_cores` to the RETURNED PAYLOAD, and the run
    # record is built from `hardware.read()` alone — so for the same job the envelope reported
    # `usable_cores: 96` and the record filed null, on every run ever written. `docs/archive/instrumentation-archive.md`
    # §3 asks a RUN to report all three, never collapsed (`F-2026-08-19-37`), and **the run record
    # is the artefact that has to answer without a client having been watching** — a caller using
    # their own front-end got one that could not answer §3 at all.
    #
    # The old reasoning for putting them here was "so `hardware` keeps its own shape" and "so a
    # FAILING run carries them too". The first was a preference; the second is satisfied better by
    # the move, because `hardware.read()` is called once at the top of `handle` and reaches every
    # exit including the ones that never get decorated.
    #
    # **Still copied, and the copy is the only thing left of this block.** `handle` hands one
    # snapshot to both this and `_write_run_record`, so publishing the object itself would let
    # anything that later touched the envelope's `hardware` edit the record's — two artefacts, one
    # dict, and a mutation visible in the one nobody was looking at.
    payload["hardware"] = dict(machine or {})
    payload["execution_ms"] = int((time.time() - started) * 1000)
    if attempts:
        payload.setdefault("attempts", attempts)
    if progress.emitted:
        payload["progress_emitted"] = len(progress.emitted)
    return payload


def _write_to_the_reserve(exception, job=None, note=None):
    """The bundle for a failure with no job to attach it to. **Never raises.**

    This is the whole point of `diagnostics_reserve`. Everything `_write_diagnostics` covers has a
    validated request and a per-job URL that came with it; the failures CF cannot see at all are
    the ones that happen where neither exists — an escape past `handle`, or the process falling
    over outside any job. There is no `request` here by construction, so the bundle carries what
    can still be known: the exception, the hardware and the build.
    """
    try:
        destination = diagnostics.reserve()
        if not destination:
            return False
        body = diagnostics.bundle(
            "unattributed", hardware.read(), [], exception=exception,
            job=job, started=time.time(),
            build=build_identity(),
            extra={"note": note or "no request was in scope when this failed"})
        return storage.put_diagnostics(destination, body)
    except Exception:  # noqa: BLE001 — a last resort that raises is not one
        return False


def handler(job):
    """RunPod's entrypoint. **Errors ride the output envelope while the job still COMPLETES.**"""
    try:
        return handle(job.get("input") or {}, job)
    except Exception as exc:  # noqa: BLE001 — the last line of defence
        traceback.print_exc()
        # Past `handle`, so past everything that knew where this job's diagnostics go.
        _write_to_the_reserve(exc, job=job, note="escaped handle(); no validated request")
        return {"cf_error": {"code": errors.INTERNAL,
                             "message": "{}: {}".format(type(exc).__name__, exc)}}


if __name__ == "__main__":
    import runpod

    # **The serve loop itself, because a worker that dies here dies silently.** A driver mismatch
    # or a weight that will not load takes the process with it, and RunPod's answer to CF is a
    # worker that stopped — with the reason only in a log stream nothing scrapes.
    #
    # It reports only if a previous job left a reserve behind, so the very first container of a
    # broken build still cannot say anything. Accepted, and it is the stated limit of the design
    # rather than an oversight: an endpoint with zero successful jobs is not a subtle signal.
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _write_to_the_reserve(exc, note="the serve loop exited; no job was in scope")
        raise
