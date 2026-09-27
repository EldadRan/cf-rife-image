"""§19e — the poster, the proxy and the spritesheet, each made from the DELIVERED MASTER, at
§21's speed.

**PORTED FROM THE ffmpeg SERVICE AT `ffmpeg@9d7495b`, WHICH IS A DIFFERENT TREE FROM THIS ONE.**
Every argument list and constant below cites where it came from in that tree; *a cite into it is
evidence only with the commit beside it, so every cite here carries one.* **What was taken is the
ffmpeg side — the filters, the encoders, the layout rule and the per-role fields.** What was not:

- **Admission, `contextvars` allocations and `wait4` accounting.** *That service runs several
  jobs per replica and charges each; this worker runs one job per container* (§19h).
- **Its 2-thread profile** (§21, CF 2026-09-27). *Sized for many jobs sharing a replica; this
  worker runs one job on 96-128 cores*, and at 8K two full decodes at 2 threads were most of a
  446 s `derive_s`. **The decoder is uncapped and the proxy's x264 takes the worker's own bound**
  (`encoder.resolve_defaults` on the proxy's area, as the master's encode takes it on its own).

**WHAT §21 CHANGED, AND WHAT A DERIVE PRODUCES DID NOT** (the roles, fields, keys, sizes, CRF,
preset, layout and frame choice are §19e's still):

- **One decode for the proxy and the sheet** — the service's own `split` (`operations.py:1051-1062`,
  `:1097-1115`), taken now where it was left before. *§19e's containment is kept by falling back:*
  **if the shared pass fails, each role is made alone**, so one role's failure still costs only
  itself — at the price of the decode it was meant to save, on the failing job only.
- **The poster SEEKS** to its frame by that frame's exact PTS rather than decoding from frame 0,
  and is still exactly frame `round(at_fraction × (frames − 1))`.
- **Cancellable**: `handler` makes the derives WHILE the master uploads, and a master upload that
  fails cancels them (`cancel`).

**NOTHING HERE MAY COST THE MASTER.** *By the time this runs the master is uploaded; a derive is
re-makeable from it and the master is not re-makeable from anything cheap* (§0, §19e). `make`
raises for a role that fails and the caller catches it per role, into `warnings[]`.

**Stdlib and ffmpeg only**, so the tests tree exercises it with real files and no GPU.
"""
import json
import os
import subprocess
import threading
import time

import keys
import probe
from errors import INTERNAL, WorkerError

#: `ffmpeg@9d7495b` `handler/operations.py:95`.
FFMPEG_BASE_ARGS = ["-y", "-hide_banner", "-loglevel", "error", "-nostdin"]

#: **Bounded, because a derive runs after the master exists and before the job returns.** *A
#: wedged ffmpeg here would hold a delivered job open against the platform's 3,600 s wall.* The
#: longest — the shared proxy-and-sheet decode of an 8K master — is minutes, not tens of them.
FFMPEG_TIMEOUT_S = 1200

#: `ffmpeg@9d7495b` `handler/profiles.py:15, 22, 24-26, 103, 112`.
PROXY_LONG_EDGE = 1280
PROXY_CRF = 26
PROXY_PRESET = "veryfast"
PROXY_AUDIO_BITRATE = "128k"
AUDIO_SAMPLE_RATE = "48000"
POSTER_ENCODER = ["-c:v", "libwebp", "-quality", "90"]

#: `ffmpeg@9d7495b` `handler/profiles.py:157-159, 162`.
SHEET_TILE_BUDGET = 32
SHEET_TILE_WIDTH = 160
SHEET_COLUMNS = 10
SHEET_ENCODER = ["-c:v", "libwebp", "-quality", "80"]

#: `ffmpeg@9d7495b` `handler/profiles.py:198-202`. Long edge to 1280, aspect kept, never upscaled,
#: both dimensions even — one filter for landscape, portrait and square.
PROXY_SCALE_FILTER = (
    "scale="
    "'if(gte(iw,ih),min({edge},iw),-2)'"
    ":'if(gte(iw,ih),-2,min({edge},ih))'"
).format(edge=PROXY_LONG_EDGE)


def sheet_layout(frame_count):
    """`(frame_step, count, columns, rows)` — `ffmpeg@9d7495b` `handler/profiles.py:165-178`.

    **§19e restates this rule because the kit tests it**: `frame_step = ceil(frames / 32)`,
    `count = ceil(frames / frame_step)`, `columns = min(10, count)`, `rows = ceil(count/columns)`.
    """
    frames = max(1, int(frame_count or 1))
    step = max(1, -(-frames // SHEET_TILE_BUDGET))
    count = -(-frames // step)
    columns = min(SHEET_COLUMNS, count)
    rows = -(-count // columns)
    return step, count, columns, rows


def poster_index(at_fraction, frame_count):
    """`round(at_fraction × (frames − 1))`, clamped — `ffmpeg@9d7495b` `operations.py:1206-1209`.

    *Python's `round` on a float, as that service computes it*; the kit's oracle computes on the
    exact value and says no case sits on a half where the two could differ.
    """
    frame_count = max(1, int(frame_count or 1))
    index = int(round(at_fraction * (frame_count - 1)))
    return max(0, min(index, frame_count - 1))


def _gop(fps):
    """`-g`: a keyframe every two seconds — `ffmpeg@9d7495b` `profiles.py:342-346`."""
    if not fps or fps <= 0:
        return 48
    return max(1, int(round(fps * 2)))


#: **The instant after which no derive may still be running**, set per job by `begin` — a module
#: value because the makers share it and each makes up to two ffmpeg calls. *One job per
#: container (§19h), and `begin` is called at the top of every job's derive step, so a previous
#: job's deadline cannot outlive it.*
_DEADLINE = [None]

#: **§21's cancellation.** The derives run on a thread beside the master's upload; a master upload
#: that fails sets this and kills the running ffmpeg, and no further call starts. `begin` clears it,
#: so a cancelled job cannot cancel the next. `_LIVE` is the one process running, under `_LOCK`.
_CANCELLED = threading.Event()
_LIVE = []
_LOCK = threading.Lock()


def begin(deadline):
    """Arm the job's deadline (`time.time()` seconds), or None for no deadline; clear a cancel."""
    _DEADLINE[0] = deadline
    _CANCELLED.clear()


def cancel():
    """Stop the derives: no further ffmpeg starts, and the running one is killed. **Never
    raises** — it runs on the path of a master upload that has already failed."""
    _CANCELLED.set()
    with _LOCK:
        for proc in list(_LIVE):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 — the kill is best effort; the wait is the join's
                pass


def _run(args):
    if _CANCELLED.is_set():
        raise WorkerError(INTERNAL, "the derives were cancelled: the master was not delivered")
    command = ["ffmpeg"] + FFMPEG_BASE_ARGS + args
    timeout = FFMPEG_TIMEOUT_S
    if _DEADLINE[0] is not None:
        # **Every call is bounded by what is left of the job, not only by its own ceiling** —
        # three roles at 1,200 s each could otherwise outlast the platform's wall after the
        # master was delivered, and a reaped container returns nothing and files nothing.
        timeout = min(timeout, _DEADLINE[0] - time.time())
        if timeout <= 0:
            raise WorkerError(INTERNAL, "no time left before the platform's wall to make this "
                                        "derive; the master is delivered")
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with _LOCK:
        _LIVE.append(proc)
    try:
        # A cancel between the check above and the registration would otherwise be missed.
        if _CANCELLED.is_set():
            proc.kill()
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise WorkerError(INTERNAL, "ffmpeg timed out after {:.0f} s".format(timeout))
    finally:
        with _LOCK:
            _LIVE.remove(proc)
    if _CANCELLED.is_set():
        raise WorkerError(INTERNAL, "the derives were cancelled: the master was not delivered")
    if proc.returncode != 0:
        raise WorkerError(INTERNAL, "ffmpeg failed: {}".format(
            err.decode("utf-8", "replace").strip()[-500:]))


def _describe(path, role, key):
    return {"role": role, "key": key, "bytes": os.path.getsize(path),
            "content_type": keys.content_type(key)}


def _dimensions(path):
    out = probe.probe_output(path)
    return out["width"], out["height"], out


#: How far before the poster's frame the rough input seek aims. *The seek only has to land at or
#: before the frame; `select` on the exact PTS picks it.* (`cf-rife-image` `splice.SEEK_SLACK_S`,
#: whose reason is the same: a time `-ss` alone landed a frame off on a source without an edit
#: list, one way measured from the video's first PTS and the other from the format's start.)
POSTER_SEEK_SLACK_S = 3.0


def frame_pts(master_path):
    """`(sorted video PTS in ticks, time base, format start)` — frame i's PTS is entry i.

    **Bounded by the job's deadline and refused after a cancel**, as `_run` is: a packet listing
    of an 8K master is seconds, and a failed master upload should not wait for one. Found in
    review."""
    if _CANCELLED.is_set():
        raise WorkerError(INTERNAL, "the derives were cancelled: the master was not delivered")
    timeout = probe.FFPROBE_TIMEOUT_S
    if _DEADLINE[0] is not None:
        timeout = min(timeout, _DEADLINE[0] - time.time())
        if timeout <= 0:
            raise WorkerError(INTERNAL, "no time left before the platform's wall to make this "
                                        "derive; the master is delivered")
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-of", "json", "-select_streams", "v:0", "-show_entries",
             "packet=pts:stream=time_base:format=start_time", master_path],
            capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise WorkerError(INTERNAL, "listing the master's packets timed out after {:.0f} s"
                          .format(timeout))
    if out.returncode != 0:
        raise WorkerError(INTERNAL, "ffprobe could not list the master's packets: {}".format(
            out.stderr.decode("utf-8", "replace").strip()[-300:]))
    data = json.loads(out.stdout or b"{}")
    from fractions import Fraction  # noqa: PLC0415
    ticks = sorted(int(p["pts"]) for p in data.get("packets") or () if "pts" in p)
    time_base = Fraction((data.get("streams") or [{}])[0].get("time_base") or "1/1")
    start = Fraction((data.get("format") or {}).get("start_time") or "0")
    return ticks, time_base, start


def poster(entry, master_path, master, workdir, name=None):
    """`ffmpeg@9d7495b` `operations.py:1202-1236`. **The master's own frame and resolution.**

    **§21: SEEKED, by the frame's exact PTS.** A rough input seek lands at or before it; `-copyts`
    keeps the master's own timestamps through the decoder; `select` takes the first frame at or
    after that PTS, which is the frame. *At 8K the old decode from frame 0 was the whole master to
    the poster's frame; this is one GOP plus the slack.*
    """
    key = keys.for_role("poster", name)
    path = os.path.join(workdir, key)
    index = poster_index(entry.get("at_fraction", 0.25), master["frames"])
    ticks, time_base, start = frame_pts(master_path)
    if len(ticks) != master["frames"]:
        raise WorkerError(INTERNAL, "the master lists {} packets with a PTS and was counted at "
                                    "{} frames".format(len(ticks), master["frames"]))
    pts = ticks[index]
    rough = max(0.0, float(pts * time_base - start) - POSTER_SEEK_SLACK_S)
    _run(["-ss", "{:.6f}".format(rough), "-copyts", "-i", master_path,
          "-vf", "select=gte(pts\\,{})".format(pts),
          "-fps_mode", "passthrough", "-frames:v", "1", "-an"] + POSTER_ENCODER + [path])
    width, height, _ = _dimensions(path)
    return path, dict(_describe(path, "poster", key), width=width, height=height)


def spritesheet(entry, master_path, master, workdir, name=None):
    """`ffmpeg@9d7495b` `operations.py:1187-1199` and `:1154-1184`; filter `profiles.py:181-195`.

    **Tile i is master frame i × frame_step, frame 0 first.** *The tile height is MEASURED — the
    sheet's height over its rows — because `scale=160:-2` rounds from the frame's own aspect and
    repeating that arithmetic here would be a second place for it to be wrong* (the service's
    reason, `operations.py:1161-1164`).
    """
    key = keys.for_role("spritesheet", name)
    path = os.path.join(workdir, key)
    chain, (step, count, columns, rows) = _sheet_chain(master["frames"])
    _run(["-i", master_path, "-vf", chain, "-frames:v", "1",
          "-fps_mode", "passthrough", "-an"] + SHEET_ENCODER + [path])
    return path, _sheet_entry(path, key, step, count, columns, rows)


def _sheet_chain(frames):
    step, count, columns, rows = sheet_layout(frames)
    return ("select='not(mod(n\\,{}))',scale={}:-2,tile={}x{}".format(
        step, SHEET_TILE_WIDTH, columns, rows), (step, count, columns, rows))


def _sheet_entry(path, key, step, count, columns, rows):
    width, height, _ = _dimensions(path)
    return dict(_describe(path, "spritesheet", key), width=width, height=height,
                columns=columns, rows=rows, tile_width=width // columns,
                tile_height=height // rows, frame_step=step, count=count)


def proxy(entry, master_path, master, workdir, name=None):
    """`ffmpeg@9d7495b` `operations.py:1072-1138`; arguments `profiles.py:392-401, 502-522`.

    **h264 8-bit, 1280 on the long edge, crf 26, veryfast, 128k AAC when the master has audio,
    faststart** — and a 10-bit h265 master still yields an 8-bit h264 proxy, because a proxy
    that will not play in the viewer's browser is worse than one two bits shallower. *Threads:
    the worker's own bound (§21), `_proxy_encoder`.*
    """
    key = keys.for_role("proxy", name)
    path = os.path.join(workdir, key)
    _run(["-i", master_path] + _proxy_output(entry, master, ["-vf", PROXY_SCALE_FILTER], path))
    return path, _proxy_entry(path, key)


def _proxy_encoder(master):
    """The proxy's x264, with the thread bound `encoder.resolve_defaults` gives the proxy's own
    delivered area — the rule the master's encode follows for its area (§21). *Uncapped by the
    service's 2.* An 8-bit h264 at 1280 on the long edge, whatever the master is."""
    import encoder  # noqa: PLC0415 — light (no torch); kept out of module scope like the rest

    width, height = int(master.get("width") or 0), int(master.get("height") or 0)
    long_edge = max(width, height) or PROXY_LONG_EDGE
    scale = min(1.0, float(PROXY_LONG_EDGE) / long_edge)
    area = max(1, int(width * scale) * int(height * scale))
    settings, _ = encoder.resolve_defaults(area, codec="h264")
    return ["-c:v", "libx264", "-preset", PROXY_PRESET, "-crf", str(PROXY_CRF),
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(_gop(master.get("fps"))),
            "-x264-params", encoder.x264_params(settings["threads"], settings["sliced_threads"],
                                                settings["rc_lookahead"])]


def _proxy_output(entry, master, video, path, mapped=False):
    """Everything after the input for the proxy's output file. `video` is its `-vf` or its
    `-map` of a filter-graph pad; `mapped` says the audio must be mapped too."""
    cap = entry.get("max_duration_s")
    audio = (["-c:a", "aac", "-b:a", PROXY_AUDIO_BITRATE, "-ar", AUDIO_SAMPLE_RATE]
             if master.get("has_audio") else ["-an"])
    if mapped and master.get("has_audio"):
        video = video + ["-map", "0:a:0?"]
    return (video + ([] if cap is None else ["-t", "{:.6f}".format(cap)])
            + _proxy_encoder(master) + audio + ["-movflags", "+faststart", path])


def _proxy_entry(path, key):
    # **§19e says FASTSTART and it is checked, not assumed** — the service's `ensure_faststart`
    # (`operations.py:160-172`): a mux does not always honour the flag. *§21: the flag is set in
    # the pass that writes the proxy; the remux below runs only if the mux did not honour it,
    # and says so.*
    if not probe.is_faststart(path):
        print("[derive] the proxy mux did not honour +faststart; remuxing it", flush=True)
        shuffled = path + ".faststart.mp4"
        _run(["-i", path, "-c", "copy", "-movflags", "+faststart", shuffled])
        os.replace(shuffled, path)
        # **And re-read after the rewrite**, or "checked, not assumed" is true of the first
        # attempt only. Found in review.
        if not probe.is_faststart(path):
            raise WorkerError(INTERNAL, "the proxy is not faststart even after a remux")
    width, height, info = _dimensions(path)
    counted = probe.counted_frames(path, INTERNAL, "counting the proxy's packets")
    return dict(_describe(path, "proxy", key), width=width, height=height,
                duration_s=info.get("duration_s"), fps=info.get("fps"),
                frames=counted["frames"], has_audio=info.get("has_audio"))


def proxy_and_sheet(proxy_entry, master_path, master, workdir, name=None):
    """§21: the proxy and the spritesheet from ONE decode — a `split` filter graph, as the ffmpeg
    service shares them (`ffmpeg@9d7495b` `operations.py:1051-1062`, `:1097-1115`).

    Returns `{role: (path, entry)}` for both, or raises; `make_all` then makes each alone, so a
    failure here costs no role anything but time. **The same filters, the same encoders, the same
    arguments per output** as the two passes they replace: the proxy's `-t` is an OUTPUT option
    and cuts the proxy only, while the sheet's branch reads the whole master.
    """
    proxy_key = keys.for_role("proxy", name)
    sheet_key = keys.for_role("spritesheet", name)
    proxy_path = os.path.join(workdir, proxy_key)
    sheet_path = os.path.join(workdir, sheet_key)
    chain, (step, count, columns, rows) = _sheet_chain(master["frames"])
    graph = "[0:v]split=2[p][s];[p]{}[pv];[s]{}[sv]".format(PROXY_SCALE_FILTER, chain)
    _run(["-i", master_path, "-filter_complex", graph]
         + _proxy_output(proxy_entry, master, ["-map", "[pv]"], proxy_path, mapped=True)
         + ["-map", "[sv]", "-frames:v", "1", "-fps_mode", "passthrough"] + SHEET_ENCODER
         + [sheet_path])
    return {"proxy": (proxy_path, _proxy_entry(proxy_path, proxy_key)),
            "spritesheet": (sheet_path, _sheet_entry(sheet_path, sheet_key, step, count,
                                                     columns, rows))}


MAKERS = {"poster": poster, "proxy": proxy, "spritesheet": spritesheet}


def make(entry, master_path, master, workdir, name=None):
    """One role. Returns `(local_path, record_entry)`; raises on failure — the caller contains it.

    `master` is `{"frames", "fps", "has_audio"}` read off the DELIVERED file — its counted packets
    and its probe — **never the source's**, which is §19e's first rule.
    """
    return MAKERS[entry["role"]](entry, master_path, master, workdir, name=name)


def make_all(entries, master_path, master, workdir, name=None):
    """Every role asked for, in the order asked. Returns `[(entry, path, made, error)]`: `error`
    is None or the exception that role failed with. **Never raises**: one role's failure is
    contained to that role (§19e), and the caller files it.

    The proxy and the sheet share one decode when both are asked (`proxy_and_sheet`); if that
    pass fails, each is made alone through `make` — the containment §19e promises, paid for in
    time on the failing job only. **A cancel stops everything not yet made.**
    """
    results = {}
    roles = {entry["role"]: entry for entry in entries}
    if "proxy" in roles and "spritesheet" in roles:
        try:
            for role, (path, made) in proxy_and_sheet(roles["proxy"], master_path, master,
                                                      workdir, name=name).items():
                results[role] = (path, made, None)
        except Exception as exc:  # noqa: BLE001 — each role is retried alone below
            print("[derive] the shared proxy+sheet pass failed ({}: {}); making each alone"
                  .format(type(exc).__name__, str(exc)[:200]), flush=True)
            results.clear()
    out = []
    for entry in entries:
        role = entry["role"]
        if role not in results and _CANCELLED.is_set():
            # **A cancel stops every role not yet made**, rather than each finding out in its
            # own first call. Found in review.
            results[role] = (None, None, WorkerError(
                INTERNAL, "the derives were cancelled: the master was not delivered"))
        if role not in results:
            try:
                path, made = make(entry, master_path, master, workdir, name=name)
                results[role] = (path, made, None)
            except Exception as exc:  # noqa: BLE001 — contained to this role
                results[role] = (None, None, exc)
        path, made, error = results[role]
        out.append((entry, path, made, error))
    return out
