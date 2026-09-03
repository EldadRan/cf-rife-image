"""A record of every run, written by the worker, so the corpus stops depending on who watched.

**The failure bundle answers "what went wrong"; this answers "what happened".** They are different
kinds and they stay different kinds (F-2026-08-19-36, and the contract amendment of the same day).
A bundle's *presence* is certified shorthand for "a run struggled" — the R-series verdicts read it
that way, in evidence — so widening bundles to cover successful runs would retroactively change
what those verdicts were reading. The run-record therefore lands under its own `runs/` prefix, and
its presence means only that a run happened.

**Why the worker writes it and not the harness.** Every calibration row this project owns was
banked client-side by `run_one.py`, which works exactly as long as a person is watching through
our own tool. Three failures in one day showed what that costs: the first 8K customer delivery —
the most expensive single measurement in the ledger — banked `build: None` and a wall clock of
357 s for a job that ran 4147 s, because it was recovered through `--attach` rather than watched
(F-2026-08-19-35). And the callers this worker exists for will not run our harness at all. A
record the worker pushes is complete by construction: it does not care who was watching, whether
the client survived, or whether anyone was there.

What it feeds, concretely: §8b's per-card time rows (the 491-versus-69-minute prediction gap is a
missing-row problem, not a formula problem), the host tail-term anchors, and per-frame pricing per
card — each of which currently improves only when a human remembers to bank a row.

**Metadata only, never content.** No frames, no source bytes, no presigned URLs, no customer text.
The body is swept through `diagnostics.redact` before it leaves, which is defence in depth rather
than the primary control: nothing here is supposed to be able to carry a credential in the first
place, and the sweep is what makes that true of fields somebody adds later.

**The write can never fail a job.** Same posture as progress emission, and for a stronger reason:
this runs on the success path too, so an exception here would turn a delivered master into a
failed job over a bookkeeping object. Every entry point returns a value and raises nothing.

**The address arrives with the request, exactly as the failure bundle's always has** (CF, ruled
2026-08-19, superseding this module's first design). A presigned PUT URL in a `run_record` field
beside `diagnostics`, its key under `runs/`, minted by whoever builds the request — our harness
today, CF's front-end tomorrow.

An endpoint-provisioned standing credential (`RUNS_S3_*`) was built first and rejected, and the
reasons are worth keeping: the caller owns its telemetry destination and should not have to
discover ours, and a worker holding a long-lived write credential to somebody else's bucket is a
durable liability in exchange for one object per job. A presigned URL is scoped to one key, one
verb and one window, and it arrives and expires with the work.

**The window has to outlast the job.** Whoever mints the URL matches its expiry to the endpoint's
execution timeout: a URL that dies before a long job finishes manufactures record loss on exactly
the runs most worth recording — the 8K ones, which is where this finding came from in the first
place.

**Absent field is a supported state, not an error.** No URL means the record is skipped with one
line in the log. A worker that refused to run without somewhere to file its paperwork would be a
worse worker.
"""

import json

import diagnostics

#: The request field carrying this record's presigned PUT. Named beside `diagnostics` on the wire
#: because it is the same mechanism answering a different question, and validated as a string like
#: every other URL the request carries.
REQUEST_FIELD = "run_record"

#: The prefix the minted key sits under. A constant here because the worker does not build the
#: key — the minter does — but the worker's own tests and the harness both need one name for it,
#: and the whole point of a separate kind is that it can be enumerated without meeting bundles.
PREFIX = "runs/"

#: Long enough that a slow object store does not lose the record, short enough that it cannot
#: meaningfully extend a job that has already finished its real work.
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 20


#: The configuration fields an attempt carries. Named explicitly rather than taken as "everything
#: that is not a measurement": an attempt also holds outcomes and peaks, and a `plan` block that
#: quietly grew a measurement into it would be read as a decision the planner made.
CONFIGURATION_KEYS = (
    "batch_size", "chunk_size", "blocks_to_swap", "temporal_overlap",
    "output_short_edge_px", "vae_encode_tiled", "vae_decode_tiled",
    "dit_offload_device", "vae_offload_device", "tensor_offload_device",
    "swap_io_components", "crf", "tile_quality", "schedule", "name",
)


def _configuration_of(attempts):
    """The configuration the last attempt actually ran, or None when none ever started."""
    for attempt in reversed(attempts or []):
        if not isinstance(attempt, dict):
            continue
        # A nested `plan` if one is ever added; otherwise the flattened fields as they stand today.
        nested = attempt.get("plan")
        if isinstance(nested, dict) and nested:
            return nested
        chosen = {k: attempt[k] for k in CONFIGURATION_KEYS if k in attempt}
        if chosen:
            return chosen
    return None


#: The two states a record is written in (F-2026-08-20-43). `stub` is filed once the plan exists
#: and before any GPU phase; `final` overwrites it at exit, same key, same URL.
#:
#: **An unclosed stub is the finding.** A cgroup SIGKILL writes no bundle, raises no exception
#: and returns no envelope — F-41 died twice that way — so in the corpus today a run that was
#: killed and a run that never happened are the same absence. A stub turns the first into a
#: record that says what was planned, on what hardware, at what host readings, and then stops.
#: That is the only artefact that class of death can leave.
PHASE_STUB = "stub"
PHASE_FINAL = "final"


def build_stub(build_identity, machine, request=None, rationale=None, source=None,
               host_banners=None, job=None, started_utc=None):
    """The record filed on the way in: what this run is about to attempt.

    Deliberately thin. It carries the plan, the machine and the request echo — everything already
    known before the expensive part — and nothing that only exists afterwards. A stub that tried
    to guess at outcomes would be a record that disagrees with its own overwrite.
    """
    return build(
        "started", build_identity, machine, request=request, rationale=rationale, source=source,
        host_banners=host_banners, job=job,
        timings=None if not started_utc else {"started_utc": started_utc},
        phase=PHASE_STUB)


def build(status, build_identity, machine, request=None, rationale=None, source=None,
          attempts=None, output=None, load_strip=None, host_banners=None, timings=None,
          progress=None, job=None, error=None, warnings=None, phase=PHASE_FINAL,
          retime=None, transfer=None, eta=None, estimate=None, tie_check=None,
          convert_check=None, input_check=None, decode_probe=None, encode_defaults=None,
          codec=None, bit_depth=None, reference=None, cap=None):
    """The record body. Metadata only — every argument here is a number, a name or a shape."""
    body = {
        "kind": "run-record",
        # **Which of the two writes this is.** A reader finding `stub` in the corpus is holding a
        # run that began and never reported: the SIGKILL class, visible at last. A reader who
        # cannot tell a stub from a truncated final would draw the opposite conclusion from the
        # same bytes, which is why this is a field and not an inference from what is missing.
        "record_phase": phase,
        # **Version the shape, because this one is meant to be read years from now** by something
        # that was not written yet. A corpus whose entries cannot say which shape they are is a
        # corpus that can only be parsed by the code that wrote it.
        # **BUMPED 1 -> 2 BY THIS WAVE, AND IT IS THE FIELD'S FIRST JOB**
        # (`docs/archive/instrumentation-archive.md` §16b). It has been the literal `1` since the project began
        # and NOTHING READ IT — an eighth member of `F-2026-08-25-1`'s class, sitting in plain
        # sight on every row.
        #
        # **§16 makes `drain_s` mandatory, and 45 banked records legitimately lack it.** A kit
        # that failed them would retroactively un-certify work that met the spec in force when it
        # ran, so the kit keys the rule on this number: below 2 an absent `drain_s` is legal and
        # SKIPS, at 2 and above it is a defect. *Self-keying, which is §14a's form — a check
        # keyed on the build commit would need editing at every image, one keyed on a date would
        # be wrong for a replay, and one that skipped on absence would let a NEW image drop the
        # field silently.*
        #
        # **BUMPED ONCE FOR THE WHOLE WAVE, NOT PER FIELD.** `bit_depth` and the `reference`
        # block have their own absence rules — §15a's inference and §17a's armed-instrument skip
        # — and neither needs the version. `drain_s` is the only field this wave makes MANDATORY
        # on a path that already ran.
        "record_version": 2,
        "utc": diagnostics._now(),
        "status": status,
        "build": build_identity,
        "runpod": diagnostics._runpod_identity(job),
        # The card is the axis every constant in the registry is keyed on, so it is named at the
        # top rather than buried in the hardware block a reader has to know to open.
        "gpu": (machine or {}).get("gpu_name"),
        "hardware": machine,
        "request": diagnostics._request_summary(request),
        # What the planner decided and why — the half that makes a measurement re-derivable
        # instead of merely recorded. Lifted off the *winning* attempt rather than the first,
        # because a run that stepped down was measured at the configuration that finished, and
        # pairing a rung's name with another rung's peak is a corruption this ledger has already
        # met once.
        "plan": _configuration_of(attempts),
        # **Its own slot, not a borrowed one, and the reason is what survives WITHOUT the client.**
        # The retime stats ride the envelope, so a harness that completes already holds them — but
        # a harness that dies in the fetch does not, and that happened twice in one afternoon. When
        # it does, THIS RECORD IS THE ONLY THING LEFT, and without this field it would say the job
        # ran, on which machine and for how long, and NOTHING ABOUT WHAT WAS COMPUTED: no n_out, no
        # peak VRAM, no padded_megapixels, no encode settings.
        #
        # **A record whose whole purpose is to outlive the client cannot be the one artefact that
        # omits the work.** Beside `plan` and `rationale` rather than inside them: those are the
        # estimator's fields and route C has no estimator, and a record that borrows a slot is
        # worse than one with a gap because the gap is visible. Null on any path that produces no
        # retime, which is every upscale path and every refusal before the loop.
        "retime": retime,
        # **WHO CHOSE the three encoder-thread settings** (`docs/archive/instrumentation-archive.md` §13). The
        # settings themselves are in `retime`, read off the writer that ran, and they always
        # were — what no existing field can say is which of §6d's two rows fired, or whether a
        # caller overrode it. **A row reading `threads=16 sliced_threads=1` is produced
        # identically by the large row firing and by a caller sending those two values at 4K**,
        # and the corpus cannot separate two different experiments that print the same.
        #
        # **PRESENT-AND-NULL, not omitted, and that is the difference from `tie_check` below.**
        # Those blocks are opt-in diagnostics, so their absence is a state — the run did not arm
        # them — and omitting them is what keeps "did not run" distinguishable from "ran and
        # found nothing". **The branch runs on every job.** So an absence here is a run that died
        # before the encode was configured, or a defect, and never a choice — and §13's whole
        # argument is that provenance must not have to be inferred from silence.
        "encode_defaults": encode_defaults,
        # **What the frame cap computed and what it compared against, on EVERY run.** *The
        # refusal's message carries both numbers, so a refused job keeps them either way — but a
        # DELIVERED run is where they are worth most: they are the only evidence of how close to
        # the limit the fleet actually operates, and a corpus that holds them only for refusals
        # can never answer that.* **It also carries both readers' heights**, which is the first
        # instrument this project has ever had on whether `probe` and the decoder agree.
        "cap": cap,
        "rationale": rationale,
        "source": source,
        "output": output,
        # **The strip that used to be silent**, measured in halves because its two costs have
        # different causes: a CPU-and-page-cache-bound import, and a checkpoint read whose price
        # is the host's storage (F-2026-08-19-31).
        "load_strip": load_strip,
        "host": host_banners,
        "timings": timings,
        # **Bytes beside the clocks, in their own block** (`docs/archive/instrumentation-archive.md` §8f).
        # Neither half is a rate on its own, and putting the counts in `timings` would file a
        # quantity in seconds beside one in bytes under a name that says seconds. §8f names two
        # blocks and this is the second.
        "transfer": transfer,
        # **What the worker PREDICTED, beside what it then did** (contract §9). `fit` is the
        # certified VRAM predicate's answer and sits beside the measured `retime.peak_vram_gb` a
        # reader can subtract it from; `time` is §9b's point-and-spread with the corpus it came
        # from. **A prediction recorded only when it refuses can never be graded**, so both are
        # filed on every run that got far enough to make them.
        "estimate": estimate,
        # **The ETA AS FIRST PUBLISHED**, which is the only version of it worth grading. §8g
        # checks this against the outturn, and the failure it was written for was a first ETA of
        # 11,553 s on a 1,733 s job — a figure that existed, was sent, and was recorded nowhere.
        "eta": eta,
        "attempts": attempts or [],
        "warnings": list(warnings or []),
    }
    # **`docs/archive/conversion-wave-archive.md` §2g-2, and OMITTED rather than nulled when it did not run.**
    # Every ordinary job is a job that did not sweep, so a `null` here would be a field on every
    # record in the corpus saying nothing. The kit's `--tie-check` REQUIRES the block and grades
    # its absence, which is the behaviour that makes "the sweep did not happen" distinguishable
    # from "the sweep found nothing" — the distinction `F-2026-08-25-2` is about.
    if tie_check:
        body["tie_check"] = tie_check
    # **`docs/archive/conversion-wave-archive.md` §5-0, omitted rather than nulled**, exactly as `tie_check` is:
    # every ordinary job is a job that did not run the gate, and the kit's `--convert-check`
    # grades the absence so "did not run" stays distinguishable from "ran and found nothing".
    if convert_check:
        body["convert_check"] = convert_check
    # §3b-1, omitted rather than nulled for the reason `tie_check` and `convert_check` are.
    if input_check:
        body["input_check"] = input_check
    # §11a, omitted rather than nulled like every other opt-in block.
    if decode_probe:
        body["decode_probe"] = decode_probe
    # **§15, AND ITS OMISSION IS NOT THE SAME KIND AS THE FOUR BLOCKS ABOVE.** Those are opt-in
    # diagnostics whose absence means "did not run". **This one's absence has a VALUE**: §15c
    # rules that a row with no `codec` is an h264 row, because every record written before §15
    # was produced by an unconditional `libx264` with no branch that could reach anything else.
    #
    # **SO PRESENT-AND-NULL IS THE ONE SPELLING THAT MUST NOT SHIP.** The kit reads the rule as
    # `record.get("codec", "h264")` — which returns the default for an ABSENT key and `None` for
    # a present one holding null. *A null here would produce a row that is neither codec, on a
    # field whose whole purpose is that every row can be attributed to one.* Absent is h264,
    # present is what ran, and there is no third state to spell.
    if codec is not None:
        body["codec"] = codec
    # §6f, and omitted rather than nulled for `codec`'s reason exactly: §15a rules that a row
    # with no `bit_depth` is an 8-bit row, because every record written before §6f was `yuv420p`
    # from an unconditional literal. **A present-and-null field would be a row that is neither
    # depth**, on a field whose whole purpose is that every row can be attributed to one.
    if bit_depth is not None:
        body["bit_depth"] = bit_depth
    # **§17a, and OMITTED rather than nulled like the four gates above it.** `reference_score`
    # runs only when asked, so absence is a state — the run did not arm it — and the kit grades
    # that as `Skip` rather than as a pass. *A present-and-null block would be an instrument
    # asserting it ran and found nothing, which is exactly what it cannot distinguish itself from
    # if the field is always there.*
    if reference:
        body["reference"] = reference
    if progress is not None:
        body["seconds_per_frame"] = progress.seconds_per_frame()
    if error:
        body["error"] = error
    return diagnostics.redact(json.dumps(body, indent=2, default=str, sort_keys=True))


def write(document, url, log=print, label="run-record"):
    """PUT the record to the caller's presigned URL. **Never raises, never fails a job.**

    Returns True if it landed, False otherwise. Three outcomes, all reported and none fatal:
    written; skipped because the request carried no `run_record` field; or failed because the
    object store said no. The middle one is a supported state and says so, because a line that
    reads like an error every time a correctly-configured job runs is a line people learn to
    ignore — and this one will appear on every job until CF's front-end starts minting the field.

    A single PUT rather than a client, matching `storage.put_diagnostics` exactly: one object
    against a URL that already names the bucket, the key and the window it is good for.

    **The same URL is written twice on a healthy job** (F-2026-08-20-43): a stub on the way in
    and the full record at exit, the second overwriting the first. That is deliberate and it is
    why the key must be stable — a second object would make an unclosed stub indistinguishable
    from a run that filed twice, which is the one distinction this whole design exists to draw.
    `label` only names which write is speaking in the log; both go to the same place.
    """
    try:
        if not url:
            log("[{}] skipped: the request carried no {} URL. This is not an error — "
                "the record is optional and the job is unaffected.".format(label, REQUEST_FIELD))
            return False

        import requests  # noqa: PLC0415 — already a dependency; imported here to match storage

        response = requests.put(
            url,
            data=document.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
        response.raise_for_status()
        log("[{}] wrote {:,} bytes".format(label, len(document)))
        return True
    except Exception as exc:  # noqa: BLE001 — see the docstring; this must never fail the job
        # Named, not swallowed. A record that silently never appears is the same class of defect
        # as the empty log this project carried for two months. **An expired URL lands here**,
        # which is why the minter matches the expiry to the endpoint's execution timeout: the
        # longest jobs are both the most valuable to record and the most likely to outlive a
        # short window.
        try:
            log("[{}] NOT written ({}: {}). The job is unaffected; if this is a signature "
                "or expiry error the URL died before the job did.".format(
                    label, type(exc).__name__, str(exc)[:200]))
        except Exception:  # noqa: BLE001 — a last resort that raises is not one
            pass
        return False
