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
import hardware
import interp_plan
import keys
import phasewatch
import probe
import progress as progress_module
import runrecord
import storage
import validation
from errors import FIELD_NOT_SUPPORTED, WorkerError

WORKER_VERSION = os.environ.get("WORKER_VERSION", "0.1.0-dev")


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

#: Every `[host]` reading this job took, in order. **It has no writer, and that is the current
#: state rather than an oversight.** Every caller of `phasewatch.observe` — the only thing that
#: ever appended — was on the upscale path and left with it, so this list was already empty on
#: every route-C run before the excision touched it. **The key stays in the run record, empty**,
#: for the reason §7.1 keeps `registry_version` present and null: a field that disappears and a
#: field with nothing to say do not read alike to whoever finds the record later.
#:
#: **It is a local list now, not `phasewatch.BANNERS`.** That aliasing was ruled when two writers
#: banked into one corpus and two lists would have meant two orders; at zero writers the reason is
#: spent, and a module global nothing writes is the state §4a exists to refuse. Route C's own host
#: readings arrive when §1's guard and heartbeat work lands, and they land here.
_HOST_BANNERS = []


def handle(job_input, job=None):
    started = time.time()
    machine = hardware.read()

    # **How many cores this container may actually use, said out loud before anything else.**
    # The phase-4 tail runs at one core's worth while thirty sit idle, and the first question
    # that investigation has to answer is how many cores there were to be idle — a number that
    # was, until now, only obtainable by someone thinking to run `nproc` on a live worker during
    # a tail. Now every log carries it.
    if not _SAID_BOOT:
        _SAID_BOOT.append(True)
        print(phasewatch.boot_banner())

    # **Per job, not per worker.** A worker serves many jobs and this list is module-level, so a
    # record carrying the previous job's host readings would be a measurement attributed to the
    # wrong run — the exact defect class the build identity exists to prevent.
    del _HOST_BANNERS[:]

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
    # **What the diagnostics bundle needs, filled in as the run learns it.** The estimator's
    # rationale is built inside `_run` and the bundle is written out here, so without somewhere
    # shared to put it the most diagnostic number in a failed job — what the worker expected
    # before it started — could not be written at all.
    trace = {}
    workdir = tempfile.mkdtemp(prefix="cf-upscale-")
    progress = progress_module.Progress(job=job)

    # **The outcome, recorded where all three exits can reach it.** The run-record is written in
    # the `finally` below because that is the only point every run passes through — delivered,
    # refused and crashed alike — and "every run" is the whole content of F-2026-08-19-36. A
    # record written on the success path only would rebuild, in a new place, exactly the sampling
    # bias the finding exists to remove.
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
            # **The production path is untouched rather than preserved under another name.**
            # This is one `if` above it: an interpolate-only request goes to route C and
            # returns; everything else falls through to exactly what was there before.
            #
            # **And it returns its retime stats, which is the whole response** (CF). Route C
            # is a test until the samples are seen, so it gets no `configuration`, no
            # rationale and no manifest equivalence — the production response shape is
            # deferred rather than dropped, and nobody can rule it before the samples exist.
            if not request["release_3"]["upscale"]:
                response = _retime(request, machine, warnings, workdir, progress, started)
            else:
                response = _run(request, job, machine, warnings, attempts, workdir, progress,
                                captured, started, trace)
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
                              trace, job, started)
            shutil.rmtree(workdir, ignore_errors=True)


def _retime(request, machine, warnings, workdir, progress, started):
    """Route C end to end: fetch, decode, interpolate, encode, upload. **No model of ours.**

    Deliberately not a second `_run` and not a copy of its master-to-upload half. It calls the
    same `storage` helpers and nothing else — a test that starts absorbing the pipeline is what
    scope review exists to stop, and every guarantee `_run` carries is about model memory
    (contract §6), so borrowing its shape would borrow answers to questions route C does not ask.
    """
    import interpolate as interpolate_module  # noqa: PLC0415 — GPU-box imports, like the rest
    import rife  # noqa: PLC0415
    import routec  # noqa: PLC0415

    download = os.path.join(workdir, "source")
    storage.fetch_source(request["source_url"], download)
    extension = probe.detect_extension(download)
    source_path = probe.named_with_extension(download, extension)
    source = probe.probe_source(source_path)

    config = request["release_3"]["interpolate"]
    progress.phase("load", pct=3, force=True, note="interpolator")
    # **`scale` is RIFE's flow-pyramid resolution and it reaches two places.** The interpolator
    # pads to a multiple derived from it, and the model call runs motion estimation at it — so a
    # scale set in one and not the other would pad for a geometry the model never sees. One
    # value, both consumers.
    scale = request.get("force_scale") or rife.DEFAULT_SCALE
    interpolator = interpolate_module.Interpolator(
        rife.Rife.load(scale=scale), scale=scale).prepare()

    master = keys.master_name(False, source["width"], source["height"],
                              name=request["output"].get("name"))
    master_path = os.path.join(workdir, master)
    progress.phase("interpolate", pct=10, force=True)
    stats = routec.retime(
        source, source_path, master_path, interpolator,
        target_fps=config["target_fps"], identity=identity_tags(
            request, source["width"], source["height"]),
        snap_tolerance=config["snap_tolerance"],
        crf=request.get("crf"),
        audio_source=source_path if request["keep_audio"] else None,
        variant=request.get("force_variant") or "direct", scale=scale)

    client = storage.client_for(request["output"])
    master_key = storage.upload(client, request["output"], master, master_path,
                                keys.content_type(master))

    # **The same `output` shape the upscale path returns, and that is a correction rather than a
    # choice.** Route C first returned `{"master": key}` — a shape nobody reads. `run_one` takes
    # `output.width`, `output.height` and `output.bytes` off this object and banks them in the
    # ledger row, so the first route-C job on hardware delivered a correct file and reported
    # `output NonexNone 0 bytes` with every measured field null. The response-shape question CF
    # deferred is about `configuration` and the rationale; this is not that. This is the existing
    # output contract, which route C has no reason to differ from and every reason to satisfy —
    # the variant runs are the ones whose numbers matter, and a wave that banks nulls is a wave
    # measured by hand.
    delivered = probe.probe_output(master_path)
    output_entry = dict(delivered)
    output_entry.update({
        "key": master_key,
        "bytes": os.path.getsize(master_path),
        "content_type": keys.content_type(master),
        "faststart": probe.is_faststart(master_path),
        "channels": 3,
        # Measured on the far side of the encode, which is the only frame count worth reading
        # from a container — and on this path it is also the one the witness compares.
        "frames": probe.written_frame_count(master_path),
    })

    # **The stats and what they were measured on, and nothing shaped like a plan.** A
    # `configuration` block here would look, to anything reading the envelope, exactly like a
    # planned upscale — and there is no plan, because there is no model to plan for.
    return _decorate({
        "status": "DELIVERED",
        "route": "C",
        "output": output_entry,
        "retime": dict(stats, target_fps=config["target_fps"],
                       snap_tolerance=config["snap_tolerance"]),
        "source": {"width": source["width"], "height": source["height"],
                   "fps": source["fps"], "duration_s": source["duration_s"]},
        "build": build_identity(),
    }, machine, [], warnings, progress, started)


def _write_run_record(outcome, request, machine, attempts, warnings, progress, trace, job,
                      started):
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
            load_strip=(trace or {}).get("load_strip") or None,
            host_banners=list(_HOST_BANNERS),
            timings={
                "wall_s": round(time.time() - started, 1),
                # The worker's own clock, which is the only one that is not somebody else's view
                # of this job — and the figure F-2026-08-19-35 showed a client cannot be trusted
                # to reconstruct after the fact.
                "started_utc": datetime.datetime.fromtimestamp(
                    started, datetime.timezone.utc).isoformat(),
            },
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
        storage.put_diagnostics(
            request.get("diagnostics") or diagnostics.reserve(), body)
    except Exception:  # noqa: BLE001 — see the docstring
        pass


def _decorate(payload, machine, attempts, warnings, progress, started):
    payload["hardware"] = machine
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
