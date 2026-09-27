"""Fetching the source and writing artefacts under the prefix.

The credentials in `output` are temporary, write-only and scoped to the prefix, and they expire.
Everything here is built so an expired credential surfaces as a clean error rather than a hang:
bounded timeouts, few retries, no unbounded waits. Measured against real R2 on 2026-08-12 —
write-only, prefix-scoped and multipart-capable (`docs/decisions.md` 3.2).

**Keys are the worker's; the prefix and the authorization are CF's.** CF records the prefix and
not the keys, so recovery after the job record expires is a `ListObjectsV2` against names CF
never chose. That makes the naming part of the contract rather than an implementation detail:
deterministic, derivable from the request, and identical on a re-run. See `keys.py`.
"""

import os
import threading

import requests

from errors import OUTPUT_WRITE_FAILED, SOURCE_FETCH_FAILED, Remedy, WorkerError

CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# R2 ignores the region but boto3 insists on one.
R2_REGION = "auto"

# Above this, boto3 splits the write into a multipart upload. A single PUT tops out around 5 GiB
# on R2, and this worker's master will cross that on ordinary content — the media worker measured
# a 985 MB remux from a two-minute 4K source, and an upscale of the same source is larger again.
#
# Multipart needs four actions beyond PutObject: CreateMultipartUpload, UploadPart,
# CompleteMultipartUpload and AbortMultipartUpload. A credential scoped to PutObject alone fails
# at the exact moment a write crosses this threshold — **so the threshold and the credential's
# actions are one decision, not two.** All five are proved working against real R2
# (`docs/decisions.md` 3.2); what is still owed CF is the master's real size, which is what
# should set this number rather than the inherited default.
MULTIPART_THRESHOLD_BYTES = int(os.environ.get("MULTIPART_THRESHOLD_BYTES", 100 * 1024 * 1024))

#: **§24: PARTS GO UP IN PARALLEL.** *`max_concurrency=1` was a trade for memory headroom that is
#: not at risk: N parts in flight hold ~N x P of buffers — 8 x 32 MiB is 256 MiB against 46-116 GB
#: hosts whose peaks since §22 are 11-24 GB.* **N bounds the parts HELD IN MEMORY too**
#: (`max_in_memory_upload_chunks`, which s3transfer otherwise caps at 10 whatever N is — so an arm
#: of 16 or 32 ran 10 and filed its own number; found in review), plus the one being read. **PROVISIONAL**: Suite 16's sweep sets the ruled
#: default (the fastest arm whose added host memory stays under 1 GB). The request's debug fields
#: `upload_concurrency` / `upload_part_mb` move it for the sweep and nothing else. *A part is MiB —
#: the unit the fixed 32 MiB part it replaces was written in — and is recorded in bytes.*
UPLOAD_CONCURRENCY_DEFAULT = 8
UPLOAD_PART_MB_DEFAULT = 32
UPLOAD_CONCURRENCY_MIN, UPLOAD_CONCURRENCY_MAX = 1, 32
UPLOAD_PART_MB_MIN, UPLOAD_PART_MB_MAX = 8, 256

# Errors R2 returns for a credential that has expired or was never valid for this prefix.
CREDENTIAL_ERROR_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "AccessDenied",
    "InvalidToken",
}


def fetch_source(source_url, destination, on_bytes=None):
    """Stream the presigned GET to disk. No media ever arrives in the payload.

    **`on_bytes(done, expected)` IS CALLED PER CHUNK AND `expected` MAY BE `None`.** *The whole
    fetch ran in silence until 2026-09-02: `handler` started the download and the next
    unconditional emission was the interpolator load, so a 780 MB source produced 176 seconds in
    which a healthy job was indistinguishable from a dead one — two `[quiet]` notices and
    nothing else.* **The defect was a function of the INPUT and not of the code**, which is why
    it survived: every source this project had tested fetched in about two seconds.

    *The caller throttles; this reports every chunk, because the rate at which bytes arrive is
    not something this function should be deciding a poll cadence from.*

    **Returns the byte count, which it has always had and always discarded**
    (`docs/archive/instrumentation-archive.md` §8a). `received` below is measured to check the transfer against
    its declared length and was then thrown away, so the 580 MB of an 8K source shared one wall
    figure with decode, RIFE and encode and nothing could say which was which. The number is the
    same one; only its fate changes.

    **The destination is no longer returned and had no reader.** The caller passed the path in
    and already holds it; handing it back made the byte count look like an addition to a return
    value rather than what it is.
    """
    try:
        response = requests.get(
            source_url, stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
        )
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        # **`None` WHEN THE SERVER DID NOT SAY, AND THE PHASE MUST STILL EMIT.** *A name and an
        # elapsed time with no percentage is strictly better than silence, and it is the shape
        # `draining` already ships.* A chunked or compressed response has no length to declare.
        try:
            expected = int(declared) if declared is not None else None
        except (TypeError, ValueError):
            expected = None
        done = 0
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    handle.write(chunk)
                    done += len(chunk)
                    if on_bytes is not None:
                        # **A reporting failure must not lose the transfer.** *The bytes are on
                        # disk; a raising callback would abandon a fetch that is succeeding, for
                        # a progress payload.*
                        try:
                            on_bytes(done, expected)
                        except Exception:  # noqa: BLE001 — an emit never costs a delivery
                            on_bytes = None
    except requests.exceptions.RequestException as exc:
        raise WorkerError(SOURCE_FETCH_FAILED, "could not fetch source_url: {}".format(exc))

    received = os.path.getsize(destination)
    if received == 0:
        raise WorkerError(SOURCE_FETCH_FAILED, "source_url returned an empty body")

    # A truncated transfer may succeed next time, so it belongs in the retryable table — unlike
    # bytes that arrive whole and will not decode, which never will.
    if declared is not None and received < int(declared):
        raise WorkerError(
            SOURCE_FETCH_FAILED,
            "source_url returned {} bytes of a declared {}".format(received, declared),
        )
    return received


def client_for(output):
    """boto3 is imported here, not at module scope, and that is deliberate.

    Importing it costs about 100 ms, and no refusal ever reaches this function. So the cost is
    paid only by jobs that actually write something.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=output["endpoint"],
        aws_access_key_id=output["access_key_id"],
        aws_secret_access_key=output["secret_access_key"],
        aws_session_token=output.get("session_token"),
        config=Config(
            region_name=R2_REGION,
            signature_version="s3v4",
            connect_timeout=CONNECT_TIMEOUT_S,
            read_timeout=READ_TIMEOUT_S,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def upload_settings(request=None):
    """§24: `{"concurrency", "part_bytes"}` for this job's uploads — the request's debug fields
    where it sent them (validated), the provisional default where it did not. One resolution
    for every upload of the job, so the record's two fields describe all of them."""
    request = request or {}
    concurrency = request.get("upload_concurrency")
    part_mb = request.get("upload_part_mb")
    return {"concurrency": int(concurrency if concurrency is not None
                               else UPLOAD_CONCURRENCY_DEFAULT),
            "part_bytes": int(part_mb if part_mb is not None
                              else UPLOAD_PART_MB_DEFAULT) * 1024 * 1024}


class _Relay:
    """boto3's per-part byte deltas to an absolute, MONOTONIC `on_bytes(done, expected)`.

    **§24: the callback runs on boto3's worker threads.** The running total is taken under a lock;
    **a retried part reports a NEGATIVE delta** (s3transfer rewinds its progress), so the true
    total can fall, and what is published is the highest total seen — `bytes_done` never goes
    backwards. A raising `on_bytes` is dropped, never re-raised into the transfer.

    **`on_bytes` runs OUTSIDE the counting lock, and by one thread at a time**: it is a progress
    publish (the sampler, `progress_update`), and under the counting lock it paused every part's
    send while it ran (found in review). A thread that finds another publishing skips — the
    publisher always sends the latest total, and a later callback sends what arrived meanwhile —
    so the published sequence is still strictly increasing.
    """

    def __init__(self, on_bytes, expected):
        self._on_bytes = on_bytes
        self._expected = expected
        self._lock = threading.Lock()
        self._emitting = threading.Lock()
        self.total = 0
        self.published = 0
        self.emitted = 0

    def __call__(self, delta):
        with self._lock:
            self.total += int(delta)
            if self.total > self.published:
                self.published = self.total
        if self._on_bytes is None or not self._emitting.acquire(blocking=False):
            return
        try:
            with self._lock:
                latest = self.published
            if latest > self.emitted:
                self.emitted = latest
                try:
                    self._on_bytes(latest, self._expected)
                except Exception:  # noqa: BLE001 — an emit never costs a delivered master
                    self._on_bytes = None
        finally:
            self._emitting.release()

    def flush(self):
        """The final total, on the caller's thread once the transfer has returned — a report
        skipped because another thread was publishing may have been the last one."""
        self(0)


class _AbortWatch:
    """s3transfer's failure cleanup — `AbortMultipartUpload` — swallows its own failure (it logs at
    DEBUG and goes on). **Registered on the client's `after-call` events for one upload**, this
    sees the abort's own answer, so a failed abort reaches the job's error instead of leaving
    parts under the caller's prefix with no trace (found in review). Never raises."""

    EVENTS = ("after-call.s3.AbortMultipartUpload", "after-call-error.s3.AbortMultipartUpload")

    def __init__(self, client):
        self.failures = []
        self._events = getattr(getattr(client, "meta", None), "events", None)
        self._id = "cf-rife-abort-watch-{}".format(id(self))
        if self._events is not None:
            for event in self.EVENTS:
                self._events.register(event, self._heard, unique_id=self._id + event)

    def _heard(self, http_response=None, parsed=None, exception=None, **_kwargs):
        try:
            if exception is not None:
                self.failures.append("{}: {}".format(type(exception).__name__, exception))
            elif http_response is not None and http_response.status_code >= 300:
                error = (parsed or {}).get("Error") or {}
                # **`NoSuchUpload` is the upload already gone** — an abort that succeeded, whose
                # answer was lost, retried by botocore. Not a failure. Found in review.
                if error.get("Code") == "NoSuchUpload":
                    return
                self.failures.append("{} {}".format(error.get("Code") or http_response.status_code,
                                                    error.get("Message") or "").strip())
        except Exception:  # noqa: BLE001 — a hook must never break the upload
            pass

    def close(self):
        if self._events is not None:
            for event in self.EVENTS:
                try:
                    self._events.unregister(event, unique_id=self._id + event)
                except Exception:  # noqa: BLE001
                    pass

    def said(self):
        """The sentence a failed upload's error ends with, or empty."""
        if not self.failures:
            return ""
        return (" — and aborting the multipart upload failed too ({}), so its parts may remain "
                "under the prefix".format("; ".join(self.failures)))


def upload(client, output, name, path, content_type, on_bytes=None, settings=None):
    """Write one file under the prefix. The key is deterministic, so a re-run overwrites.

    **`on_bytes(done, expected)` reports absolute bytes**, not boto3's per-part delta — the
    accumulation happens here so every caller does not repeat it, and `expected` is the file's
    own size, which is known before the first byte moves. *Monotonic under threads* (`_Relay`).

    **§24: `settings` is `upload_settings`'s** — N parts of P bytes in flight; None is the
    provisional default. **A part that fails fails the upload** as it always has
    (`OUTPUT_WRITE_FAILED`), **and the multipart upload is aborted**: s3transfer registers
    `AbortMultipartUpload` as the failure cleanup the moment the upload is created
    (`s3transfer/tasks.py`, `CreateMultipartUploadTask`), so no parts are left under the caller's
    prefix — **unless the abort fails too** (an expired credential refuses it as it refused the
    part), and then the error SAYS so (`_AbortWatch`). Witnessed by the builder's
    `tests/test_upload_wave.py`, both ways.
    """
    import botocore.exceptions
    from boto3.s3.transfer import TransferConfig

    prefix = output["prefix"]
    key = "{}{}".format(prefix if prefix.endswith("/") else prefix + "/", name)
    #: **boto3 hands the callback a DELTA and the phase wants a total.** *Accumulated here, and
    #: the expected figure is read once before the transfer rather than per part.*
    try:
        expected = os.path.getsize(path)
    except OSError:
        expected = None
    settings = settings or upload_settings()
    relay = _Relay(on_bytes, expected)
    config = TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD_BYTES,
        # §24: N parts of P bytes in flight, on boto3's threads. *"One part at a time" stood
        # here, for a headroom the hosts this endpoint serves do not need (decisions.md §24).*
        multipart_chunksize=settings["part_bytes"],
        max_concurrency=settings["concurrency"],
        use_threads=True)
    # **A file handle's parts are read into memory, and those take a SEPARATE bound**, 10 by
    # default: without this, N above 10 ran 10.
    config.max_in_memory_upload_chunks = settings["concurrency"]

    watch = _AbortWatch(client)
    try:
        # upload_fileobj switches to multipart above the threshold and stays a single PUT below
        # it, so a poster keeps exactly the behaviour a single PUT would have given it.
        with open(path, "rb") as handle:
            client.upload_fileobj(
                handle,
                output["bucket"],
                key,
                ExtraArgs={"ContentType": content_type},
                Callback=relay if on_bytes is not None else None,
                Config=config,
            )
        relay.flush()
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in CREDENTIAL_ERROR_CODES:
            # **The remedy this refusal spent 8 255 s of H200 time not having**
            # (F-2026-08-20-39). The GPU work all succeeded; only the door was locked. Naming
            # what to do about it is worth more here than on almost any other code, because the
            # answer is cheap and the alternative is a caller concluding the job is impossible.
            raise WorkerError(
                OUTPUT_WRITE_FAILED,
                "output credentials rejected writing {} ({}); they are temporary and may have "
                "expired. The work itself succeeded — resubmit the same request with a freshly "
                "minted credential whose lifetime covers this endpoint's execution timeout. "
                "Retrying the identical request will fail identically: the credential is the "
                "part that has to change{}.".format(key, code, watch.said()),
                remedy=Remedy.RETRY_SAME,
            )
        raise WorkerError(OUTPUT_WRITE_FAILED,
                          "could not write {}: {}{}".format(key, exc, watch.said()),
                          remedy=Remedy.RETRY_SAME)
    except (botocore.exceptions.BotoCoreError, OSError) as exc:
        # A transport failure against the caller's own bucket. The same card would serve the same
        # job again — this is the textbook `retry_same`, and it was returning null.
        raise WorkerError(OUTPUT_WRITE_FAILED,
                          "could not write {}: {}{}".format(key, exc, watch.said()),
                          remedy=Remedy.RETRY_SAME)
    finally:
        watch.close()
    return key


def put_diagnostics(diagnostics_url, body, content_type="application/json"):
    """PUT the diagnostics bundle to CF's presigned URL. **Never raises.**

    A single presigned PUT rather than a second scoped credential, deliberately: it is one
    object against a different bucket, and a second credential would be another thing to scope,
    mint, expire and get wrong for an object written once or never.

    **Returns True/False rather than raising, and that is the whole point.** The one outcome
    worse than losing the diagnostics is losing the result because the diagnostics could not be
    stored. This is called on a path where the job has already failed, and a bare `except` here
    is correct rather than lazy.
    """
    if not diagnostics_url:
        return False
    try:
        response = requests.put(
            diagnostics_url,
            data=body,
            headers={"Content-Type": content_type},
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
        response.raise_for_status()
        return True
    except Exception:  # noqa: BLE001 — see the docstring; this must never fail the job
        return False
