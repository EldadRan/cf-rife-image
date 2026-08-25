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
import phasewatch
import probe
import progress as progress_module
import runrecord
import storage
import validation
from errors import Remedy, WorkerError

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

#: Every `[host]` reading this job took, in order. **It has a writer at last**
#: (`docs/instrumentation.md` §8b): `handle` appends `phasewatch.boot_banner()`'s lines where it
#: prints them. Until this wave the only thing that had ever appended was `phasewatch.observe`,
#: every caller of which was on the upscale path and left with it — so the list was empty on
#: every route-C run ever written, and the record filed `"host": []` while the banner was
#: computed, printed and discarded three lines away.
#:
#: **It is a local list, not `phasewatch.BANNERS`.** That aliasing was ruled when two writers
#: banked into one corpus and two lists would have meant two orders. One writer now, and it is
#: here rather than in `phasewatch` because the LIFETIME is a job's and `phasewatch` does not
#: know what a job is. Route C's finer host readings — the load strip — are a different mechanism
#: with a different cadence and live in `trace`, not here.
_HOST_BANNERS = []


def handle(job_input, job=None):
    started = time.time()
    machine = hardware.read()

    # **Per job, not per worker.** A worker serves many jobs and this list is module-level, so a
    # record carrying the previous job's host readings would be a measurement attributed to the
    # wrong run — the exact defect class the build identity exists to prevent.
    #
    # **It clears BEFORE the banner is taken, and that ordering is the whole of §8b's fix.** The
    # clear used to sit below the print, so an append at the print site would have been wiped by
    # the next line and every record would still file `"host": []`.
    del _HOST_BANNERS[:]

    # **How many cores this container may actually use, said out loud before anything else.**
    # The phase-4 tail runs at one core's worth while thirty sit idle, and the first question
    # that investigation has to answer is how many cores there were to be idle — a number that
    # was, until now, only obtainable by someone thinking to run `nproc` on a live worker during
    # a tail. Now every log carries it.
    #
    # **TAKEN EVERY JOB, PRINTED ONCE** (`docs/instrumentation.md` §8b). The banner was computed
    # inside the print-once guard, so wiring the append there would have banked a banner on a
    # container's FIRST job and an empty list on every job after it — a record whose `host` field
    # depends on how many jobs the worker happened to have served, which is worse than the empty
    # list it replaces. The print stays once because it describes the machine and a line repeated
    # every job is a line people learn to skip; the READING is per job, because a run record is
    # per run. **Stored as lines**, which is what §8f's `host` names and what a reader of the log
    # sees.
    banner = phasewatch.boot_banner()
    if not _SAID_BOOT:
        _SAID_BOOT.append(True)
        print(banner)
    _HOST_BANNERS.extend(banner.splitlines())

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
    # **What the diagnostics bundle needs, filled in as the run learns it — AND NOTHING FILLS IT
    # IN NOW.** Its three writers were inside `_run`; `_retime` never wrote it and does not. So
    # every run record files `rationale`, `source`, `output` and `load_strip` as null while
    # `_retime` demonstrably has those numbers and returns them in the envelope. **Left as it is
    # deliberately**: what the worker records is contract §1's entry condition, and an excision
    # that started filling in a record shape would be answering a question nobody has ruled.
    # Filed to the gate; the empty dict is the honest state until it is.
    trace = {}
    workdir = tempfile.mkdtemp(prefix="cf-upscale-")
    # **The host-load series, sampled on the cadence `progress` already publishes at**
    # (`docs/instrumentation.md` §8c). It is handed in as `Progress`'s sampler rather than given
    # a thread: *a measurement whose cost is a new thread is a measurement that changes what it
    # measures*. `trace` holds the live list by reference, so whatever accumulated by the
    # `finally` is what the record files — including on a run that died mid-encode, which is the
    # run whose load history is worth the most.
    load_strip = phasewatch.LoadStrip(started)
    trace["load_strip"] = load_strip.samples
    progress = progress_module.Progress(job=job, sampler=load_strip.sample)

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
            response = _retime(request, machine, warnings, workdir, progress, started, trace)
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


def _retime(request, machine, warnings, workdir, progress, started, trace=None):
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
    # **The download's own clock and its own byte count** (`docs/instrumentation.md` §8a). Both
    # were inside `wall_s` with nothing separating them from decode, RIFE and encode: 580 MB in
    # and 742 MB up sharing one figure with the work, on the run whose 44% gap could not be
    # attributed to anything.
    #
    # **The seconds are banked in a `finally`, the bytes only on success.** A fetch that raised
    # still spent the time and the record must account for it or `wall_s` stops adding up; how
    # many bytes arrived before it failed is not something `fetch_source` can say, and 0 would be
    # a number rather than an absence.
    fetch_started = time.time()
    try:
        fetch_bytes = storage.fetch_source(request["source_url"], download)
    finally:
        _note(trace, "timings", "fetch_s", round(time.time() - fetch_started, 3))
    _note(trace, "transfer", "fetch_bytes", fetch_bytes)
    extension = probe.detect_extension(download)
    source_path = probe.named_with_extension(download, extension)
    source = probe.probe_source(source_path)
    # **Into `trace`, so the `finally` can file it.** `trace` is the shared dict `handle` creates
    # for exactly this — a crashed run's most diagnostic numbers are the ones it learned before
    # it died — and `_retime` never wrote to it, so EVERY route-C run record filed `source`,
    # `output` and `load_strip` as null, on success as much as on failure. The numbers were in
    # hand both times.
    if trace is not None:
        trace["source"] = {"width": source["width"], "height": source["height"],
                           "fps": source["fps"], "duration_s": source["duration_s"]}

    config = request["release_3"]["interpolate"]
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

    progress.phase("load", pct=3, force=True, note="interpolator")
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
        # **Passed at last.** `retime` has declared this parameter since it was written and
        # `_retime` never supplied it, so the retime path published two payloads for a
        # 239-second job — `progress_emitted: 2`, the worker counting its own silence.
        progress=progress,
        audio_source=source_path if request["keep_audio"] else None,
        variant=request.get("force_variant") or "direct", scale=scale)

    client = storage.client_for(request["output"])
    # **The upload's own clock and byte count, same rule as the fetch** (§8a). The size is read
    # before the PUT rather than after it: the object on the far side is what the byte count is
    # ABOUT, and reading it from the local file is the only reading that cannot have been
    # truncated by the very failure the number would be explaining.
    upload_bytes = os.path.getsize(master_path)
    upload_started = time.time()
    try:
        master_key = storage.upload(client, request["output"], master, master_path,
                                    keys.content_type(master))
    finally:
        _note(trace, "timings", "upload_s", round(time.time() - upload_started, 3))
    _note(trace, "transfer", "upload_bytes", upload_bytes)

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
        # The same reading the transfer block banked, not a second one: two `getsize` calls on
        # one file are two chances to report two numbers for one fact.
        "bytes": upload_bytes,
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
    if trace is not None:
        trace["output"] = output_entry
        # **The retime stats have NO honest slot in `runrecord.build`.** `plan` and `rationale`
        # are the estimator's fields and route C has no estimator, so putting them there would
        # make a record borrow a slot that means something else — and a record that borrows a slot
        # is worse than one with a gap, because the gap is visible. Filed to the gate rather than
        # forced; `load_strip` is the nearest thing and it is not that either.
        trace["retime"] = dict(stats)

    return _decorate({
        "status": "DELIVERED",
        "route": "C",
        "output": output_entry,
        # **`crf` and `preset` beside the x264 params** (instrumentation §2). §6a rules five
        # encode settings changeable with today's values as defaults; the record carried three, so
        # a corpus could not attribute a difference between two runs to the settings that differed.
        # Read from the same place the encoder reads them, so a default cannot be restated wrongly.
        "retime": dict(stats, target_fps=config["target_fps"],
                       snap_tolerance=config["snap_tolerance"],
                       crf=request.get("crf") if request.get("crf") is not None
                       else encoder.DEFAULT_CRF,
                       preset=encoder.DEFAULT_PRESET),
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
    return {
        "wall_s": wall,
        # The worker's own clock, which is the only one that is not somebody else's view
        # of this job — and the figure F-2026-08-19-35 showed a client cannot be trusted
        # to reconstruct after the fact.
        "started_utc": datetime.datetime.fromtimestamp(
            started, datetime.timezone.utc).isoformat(),
        "fetch_s": fetch_s,
        "upload_s": upload_s,
        "compute_s": round(wall - fetch_s - upload_s, 1),
    }


def _transfer(trace):
    """§8f's `transfer` block. **Zero where nothing moved, never absent.**

    A run that was refused before the fetch transferred nothing, and `0` says that. What it must
    not do is say `0` about a transfer that happened and was not counted — which is why the byte
    counts are banked on the success of each transfer and not in its `finally`.
    """
    measured = (trace or {}).get("transfer") or {}
    return {"fetch_bytes": int(measured.get("fetch_bytes") or 0),
            "upload_bytes": int(measured.get("upload_bytes") or 0)}


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
            retime=(trace or {}).get("retime"),
            load_strip=(trace or {}).get("load_strip") or None,
            host_banners=list(_HOST_BANNERS),
            timings=_timings(trace, started),
            transfer=_transfer(trace),
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
        storage.put_diagnostics(
            request.get("diagnostics") or diagnostics.reserve(), body)
    except Exception:  # noqa: BLE001 — see the docstring
        pass


def _decorate(payload, machine, attempts, warnings, progress, started):
    # **The three CPU numbers, into the block a reader looks for machine facts in**
    # (instrumentation §3). `hardware.read` already carries `cpu_quota`; `usable_cores` and
    # `affinity_cores` are the two that stopped reaching the envelope when Wave 2 removed
    # `cpu_configuration`. All three, never collapsed: a container throttled by `cpu.max` and one
    # pinned by an affinity mask are different machines that a single number reports identically
    # (F-2026-08-19-37), and CPU power is an estimator input CF has named.
    #
    # Here rather than in `hardware.read` so that module keeps its own shape, and here rather than
    # on the success path so a FAILING run carries it too — a run that does not fit is the reading
    # a fit predicate most needs, and it needs to know what machine it did not fit on.
    cpu = phasewatch.cpu_configuration()
    machine = dict(machine or {})
    for name in ("usable_cores", "affinity_cores"):
        machine[name] = cpu.get(name)
    if machine.get("cpu_quota") is None:
        machine["cpu_quota"] = cpu.get("cpu_quota")
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
