"""`frame_repair`'s listed files — `decisions.md` §26e and §26g: the disk they need, what each one
is, and where a video segment's decode begins.

**EVERY FILE IS FETCHED ONCE AND PROBED ONCE, however many segments use it** — source 0 is the
source already on disk. This module decides whether each file is fit to use and says why not; the
fetches themselves are `handler`'s (`_fetch_into` for a video, `fetch_stills` below for the
stills), so every byte lands in the one `fetch_s` / `fetch_bytes` / `files_fetched` the record
reads.

**No numpy, no cv2, no torch**: `storage`, `probe`, `splice` and ffmpeg — so it runs on the tests
tree, where the builder's tests stub the free-disk reading and the network.

    disk_before_bodies    §26g check 1: every listed file + the source again as the master's
                          floor, +10%, before any body is fetched
    disk_after_source     §26g check 2: the files still to come + the master at the path's
                          estimate, +10%, once the source is probed
    probe_video,          a video beyond source 0, probed; then refused if VFR (19d), or if its
      check_video         rate or size differs from source 0's — `sources_mismatch`
    probe_still,          a still, probed; then refused unless PNG / WebP / JPEG at source 0's
      check_still         size with no transparent pixel
    keyframe_before       K, the last keyframe at or before a segment's start_frame, and the
                          copy path's seek that places a decode there
"""
import concurrent.futures
import os
import shutil
import subprocess
import threading
from fractions import Fraction

import probe
import repair_plan
import storage
from errors import (CAPACITY_EXCEEDED, INTERNAL, INVALID_SOURCE, SOURCES_MISMATCH,
                    WorkerError)

#: §26e: "plus 10%".
DISK_MARGIN = 0.10
#: §26e: a video's frame rate may differ from source 0's by at most this fraction.
RATE_TOLERANCE = Fraction(1, 1000)
#: §26e: stills come down "the single stream, up to 8 at a time".
STILL_FETCHES = 8
#: The still codecs §26e accepts, as ffprobe names them, and the format the record files.
STILL_FORMATS = {"png": "png", "webp": "webp", "mjpeg": "jpeg"}
#: What the alpha check reads at a time: a whole number of pixels at either depth (4 or 8 bytes).
_ALPHA_CHUNK = 8 * 1024 * 1024
FFPROBE_TIMEOUT_S = 600
#: The alpha check's decode, bounded like every other ffmpeg call here (the review's F8).
ALPHA_TIMEOUT_S = 600


def free_bytes(path):
    """The free disk under `path`. **A function so a test can stub it** (Suite 18's
    `capacity_exceeded` row is the builder's test with a stubbed reading)."""
    return shutil.disk_usage(path).free


def listed_sizes(source_url, sources, stills):
    """`{"source": n, "sources": [n, ...], "stills": [n, ...]}` — each file's size from a
    `Range: bytes=0-0` GET, or None where the server gave none (§26g). **Up to 8 at a time**, as
    the stills are fetched: one after another, a long stills list multiplied the probe's timeout
    (the §26 review)."""
    urls = [source_url] + list(sources) + list(stills)
    with concurrent.futures.ThreadPoolExecutor(max_workers=STILL_FETCHES) as pool:
        sizes = list(pool.map(storage.remote_size, urls))
    return {"source": sizes[0], "sources": sizes[1:1 + len(sources)],
            "stills": sizes[1 + len(sources):]}


def _unknown(sizes):
    names = ["source_url"] if sizes["source"] is None else []
    names += ["sources[{}]".format(i + 1) for i, n in enumerate(sizes["sources"]) if n is None]
    names += ["stills[{}]".format(i) for i, n in enumerate(sizes["stills"]) if n is None]
    return names


def _refuse_if_short(need, free, what):
    if need > free:
        raise WorkerError(
            CAPACITY_EXCEEDED,
            "this repair needs {:,} bytes of disk {} and the worker has {:,} free — refused "
            "before the download rather than failing part-way through it. Send fewer or smaller "
            "files, or split the job.".format(need, what, free))


def disk_before_bodies(sizes, workdir, warnings):
    """§26g check 1, BEFORE ANY BODY IS FETCHED: every listed file's size, plus the source's
    bytes once more as the master's floor, plus 10%, against the free disk. **A file whose probe
    gave no size counts 0, and `warnings[]` says which.** Returns the bytes needed."""
    unknown = _unknown(sizes)
    if unknown:
        warnings.append("the disk check could not size {} (the server gave no length for a "
                        "ranged probe), so {} counted as 0 bytes".format(
                            ", ".join(unknown), "it was" if len(unknown) == 1 else "they were"))
    files = sum(n or 0 for n in [sizes["source"]] + sizes["sources"] + sizes["stills"])
    need = int((files + (sizes["source"] or 0)) * (1 + DISK_MARGIN))
    _refuse_if_short(need, free_bytes(workdir),
                     "for its {} listed file(s) and the master".format(
                         1 + len(sizes["sources"]) + len(sizes["stills"])))
    return need


def disk_after_source(sizes, master_bytes, workdir):
    """§26g check 2, AFTER THE SOURCE's PROBE and before any other body: the files still to come
    plus the master at its path's estimate (`ladder.repair_master_bytes`), plus 10%, against the
    free disk NOW — *the source is on disk already and has left the free figure.* Returns the
    bytes needed."""
    files = sum(n or 0 for n in sizes["sources"] + sizes["stills"])
    need = int((files + int(master_bytes)) * (1 + DISK_MARGIN))
    _refuse_if_short(need, free_bytes(workdir),
                     "for the master and the {} file(s) still to fetch".format(
                         len(sizes["sources"]) + len(sizes["stills"])))
    return need


def file_refusal(label, exc):
    """A listed file's fetch or probe failure, re-raised NAMING the file (§26e: "all refusals name
    the segment id or the file index"). *Same code and remedy; the message's URL queries cut —
    a presigned query is a credential (§25d).*"""
    # **`storage`'s fetch messages say `source_url`, which is source 0** — here it is this file
    # (the §26 review's F2).
    message = storage.scrub_query(exc.message).replace("source_url", "the URL")
    return repair_plan.Refused(exc.code, "{}: {}".format(label, message), label,
                               remedy=exc.remedy, shortfall=exc.shortfall)


def named_failure(label, exc):
    """Any failure fetching or reading a listed file, NAMING it (§26e: "all refusals name the
    segment id or the file index"): a `WorkerError` keeps its code (`file_refusal`); anything
    else — a full disk, a broken pipe — is `internal`, as it would have been, with the file named
    and any query cut (the review's F4)."""
    if isinstance(exc, WorkerError):
        return file_refusal(label, exc)
    return repair_plan.Refused(INTERNAL, "{}: {}: {}".format(
        label, type(exc).__name__, storage.scrub_query(exc)), label)


def still_stub(index, nbytes, why):
    """A still that landed and was not described — `stills[]`'s every key (§26f), nulls where
    nothing was read, and why — so the record counts the download it made (the review's F9)."""
    return {"index": index, "format": None, "width": None, "height": None, "bytes": nbytes,
            "error": why}


def _rate_of(counted, probed):
    try:
        return Fraction(counted.get("r_frame_rate") or probed.get("r_frame_rate"))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_video(index, path):
    """A video beyond source 0, probed once: `(probed, counted)`, or refused naming it."""
    label = "sources[{}]".format(index)
    try:
        return (probe.probe_source(path),
                probe.counted_frames(path, INVALID_SOURCE, "counting {}'s packets".format(label)))
    except WorkerError as exc:
        raise file_refusal(label, exc) from exc


def check_video(index, probed, counted, source0, counted0):
    """§26e's refusals for a video beyond source 0, on its probe (`probe_video`). **Separate from
    the probe so the caller files the probe first** — a refused file was still fetched, and the
    record's count of downloads must close against what it lists (the §26 review).

    - **`invalid_source`**: not constant frame rate, by the reading the source gets (§19d).
    - **`sources_mismatch`**: a frame rate off source 0's by more than 0.1%, or a different
      displayed size (§26e) — *a segment replaces frames in place and is neither resized nor
      retimed.*
    """
    label = "sources[{}]".format(index)
    if counted["missing_pts"] or counted["tick_spread"] > 1:
        raise repair_plan.Refused(
            INVALID_SOURCE, "{} is not constant frame rate: its packet timestamps step by {} "
            "ticks. A segment addresses its frames by index, and on a variable-rate stream an "
            "index does not name one moment.".format(label, counted["tick_gaps"]), label)
    if (probed["width"], probed["height"]) != (source0["width"], source0["height"]):
        raise repair_plan.Refused(
            SOURCES_MISMATCH, "{} is {}x{} and source 0 is {}x{}; a segment replaces frames in "
            "place and is not resized".format(label, probed["width"], probed["height"],
                                              source0["width"], source0["height"]), label)
    rate, rate0 = _rate_of(counted, probed), _rate_of(counted0, source0)
    if rate is None or rate0 is None or abs(rate / rate0 - 1) > RATE_TOLERANCE:
        raise repair_plan.Refused(
            SOURCES_MISMATCH, "{} runs at {} fps and source 0 at {}; §26e allows 0.1% — a "
            "segment's frames are placed by index, so a different rate would be a retime".format(
                label, counted.get("r_frame_rate") or probed.get("r_frame_rate"),
                counted0.get("r_frame_rate") or source0.get("r_frame_rate")), label)


def _probe_still(path, label):
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,pix_fmt", "-of", "default=nw=1", path],
            capture_output=True, timeout=FFPROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise repair_plan.Refused(INVALID_SOURCE, "{}: ffprobe timed out".format(label), label)
    fields = dict(line.split("=", 1) for line in done.stdout.decode(errors="replace").splitlines()
                  if "=" in line)
    if done.returncode != 0 or not fields.get("codec_name"):
        raise repair_plan.Refused(
            INVALID_SOURCE, "{} is not a readable image ({})".format(
                label, done.stderr.decode(errors="replace").strip()[-200:] or "no video stream"),
            label)
    return fields


def _has_transparency(path, label, pix_fmt):
    """Whether any pixel of the still is less than fully opaque, read in whole-pixel chunks so an
    8K still is never held whole.

    **Read at the file's own depth**: a file deeper than 8 bits as RGBA64 — so a 16-bit alpha of
    65534, which 8-bit RGBA would round to opaque, is seen — and every other file as RGBA. *Not
    RGBA64 for all: swscale widens an 8-bit alpha of 255 to 65532, not 65535, so an opaque
    8-bit PNG read that way is called transparent (found by this module's own test).*"""
    deep = probe.bits_per_component(pix_fmt) > 8 or any(
        tag in (pix_fmt or "") for tag in ("64", "48", "16", "12", "10"))
    fmt = "rgba64le" if deep else "rgba"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
                             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    transparent = False
    read = 0
    # **Killed at the bound**, so a decode that hangs costs a refusal, not the platform's wall.
    expired = []

    def expire():
        expired.append(True)
        proc.kill()
    watchdog = threading.Timer(ALPHA_TIMEOUT_S, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        while True:
            chunk = proc.stdout.read(_ALPHA_CHUNK)
            if not chunk:
                break
            read += len(chunk)
            alpha = chunk[6::8] + chunk[7::8] if deep else chunk[3::4]
            if alpha.count(b"\xff") != len(alpha):
                transparent = True
                break
    finally:
        watchdog.cancel()
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if expired:
        raise repair_plan.Refused(INVALID_SOURCE, "{}: the alpha check's decode did not finish "
                                  "in {} s".format(label, ALPHA_TIMEOUT_S), label)
    if not transparent and read == 0:
        raise repair_plan.Refused(INVALID_SOURCE, "{} did not decode".format(label), label)
    return transparent


def probe_still(index, path):
    """A still, probed once: `(entry, pix_fmt)` — the record's `{index, format, width, height,
    bytes}` (§26f; `format` null for a codec §26e does not take) — or `invalid_source` for a file
    ffprobe cannot read, naming it."""
    label = "stills[{}]".format(index)
    fields = _probe_still(path, label)
    width, height = int(fields.get("width") or 0), int(fields.get("height") or 0)
    return ({"index": index, "format": STILL_FORMATS.get(fields.get("codec_name")),
             "codec": fields.get("codec_name"), "width": width, "height": height,
             "bytes": os.path.getsize(path)}, fields.get("pix_fmt"))


def check_still(entry, path, pix_fmt, source0):
    """§26e's refusals for a still, on its probe (`probe_still`), filed first for
    `check_video`'s reason: `invalid_source` for a file that is not a PNG, WebP or JPEG or that
    has ANY transparent pixel (an RGBA still whose every pixel is opaque is accepted);
    `sources_mismatch` for a size other than source 0's."""
    label = "stills[{}]".format(entry["index"])
    fmt = entry["format"]
    if fmt is None:
        raise repair_plan.Refused(
            INVALID_SOURCE, "{} is {}, and a still is PNG, WebP or JPEG".format(
                label, entry["codec"]), label)
    width, height = entry["width"], entry["height"]
    if (width, height) != (source0["width"], source0["height"]):
        raise repair_plan.Refused(
            SOURCES_MISMATCH, "{} is {}x{} and source 0 is {}x{}; a still replaces a frame in "
            "place and is not resized".format(label, width, height, source0["width"],
                                              source0["height"]), label)
    if fmt != "jpeg" and _has_transparency(path, label, pix_fmt):
        raise repair_plan.Refused(
            INVALID_SOURCE, "{} has a transparent pixel; a repaired frame is opaque, so a still "
            "must be too (an RGBA still whose every pixel is opaque is accepted)".format(label),
            label)


def fetch_stills(urls, directory, landed=None):
    """§26e: every still, the single stream, up to 8 at a time. Returns `(paths, bytes)`.

    **Every started fetch is waited for**; the first failure in LIST order is raised naming its
    index, so the same request fails the same way whichever finished first — **and any failure,
    not only a `WorkerError`** (a full disk, say: the review's F4). `landed`, a list, receives
    `(index, path, bytes)` for every still that DID land, success or not, so a caller files the
    downloads a failed batch made (the review's F3)."""
    os.makedirs(directory, exist_ok=True)
    paths = [os.path.join(directory, "still-{}".format(i)) for i in range(len(urls))]
    results = [None] * len(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=STILL_FETCHES) as pool:
        futures = {pool.submit(storage.fetch_single, url, path): i
                   for i, (url, path) in enumerate(zip(urls, paths))}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                results[i] = ("ok", future.result())
                if landed is not None:
                    landed.append((i, paths[i], results[i][1]))
            except Exception as exc:  # noqa: BLE001 — raised below, in list order
                results[i] = ("error", exc)
    for i, (state, value) in enumerate(results):
        if state == "error":
            raise named_failure("stills[{}]".format(i), value) from value
    return paths, sum(value for _, value in results)


def keyframe_before(path, start_frame, end_frame, label, cache=None):
    """`(K, seek, colour)` for a video segment (§26e, §26g N1): K is the last keyframe at or
    before `start_frame` in presentation order; `seek` is `(input args, select)`; `colour` the
    video's own matrix and range. **Refused `invalid_source`, naming `label`**, for a video whose
    packets cannot be mapped or that has no keyframe at or before the frame.

    **The seek is the copy path's `PacketMap.from_frame(K)` — a rough input `-ss` 3 s before K
    — and the select is BOUNDED ON BOTH SIDES by PTS**, K's and `end_frame`'s. *A select open at
    the top, capped by a frame count, could not fail: a seek that landed PAST K passed the frames
    after it, and they were labelled K .. end, every placed frame shifted and nothing refused (the
    §26 review's F1).* Bounded, an overshoot decodes SHORT of `end - K + 1`, and `Window` and
    `repair._check_segment` refuse it — so a count that agrees is the witness that the decode
    began at K.

    `cache`, a dict, holds one packet map per file: §26e's "probed once", however many segments
    use one video (the review's F7)."""
    import splice  # noqa: PLC0415 — stdlib and ffmpeg only

    pmap = (cache or {}).get(path)
    if pmap is None:
        try:
            pmap = splice.PacketMap(path)
        except splice.NotEligible as exc:
            raise repair_plan.Refused(INVALID_SOURCE, "{}: its packets cannot be mapped to "
                                      "frames ({})".format(label, exc), label)
        if cache is not None:
            cache[path] = pmap
    keys = [k for k, row in enumerate(pmap.frames[:start_frame + 1]) if row[2]]
    if not keys:
        raise repair_plan.Refused(INVALID_SOURCE, "{}: no keyframe at or before frame {}, so no "
                                  "decode can begin there".format(label, start_frame), label)
    k = keys[-1]
    inputs, _open_select = pmap.from_frame(k)
    select = "select=between(pts\\,{}\\,{})".format(pmap.frames[k][0],
                                                     pmap.frames[end_frame][0])
    return k, (inputs, select), pmap.matrix()
