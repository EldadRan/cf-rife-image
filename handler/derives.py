"""§19e — the poster, the proxy and the spritesheet, each made from the DELIVERED MASTER.

**PORTED FROM THE ffmpeg SERVICE AT `ffmpeg@9d7495b`, WHICH IS A DIFFERENT TREE FROM THIS ONE.**
Every argument list and constant below cites where it came from in that tree; *a cite into it is
evidence only with the commit beside it, so every cite here carries one.* **What was taken is the
ffmpeg side — the filters, the encoders, the layout rule and the per-role fields.** What was not:

- **The proxy-and-sheet shared decode** (`operations.py:1051-1062`, `:1097-1115`). *There the
  sheet rides the proxy's pass as a `split` branch when both are asked for.* **Here each role is
  its own pass**, so a failed proxy cannot take the sheet with it — §19e's rule that a derive's
  failure is contained is easier to keep true when no two share a process.
- **Admission, `contextvars` allocations and `wait4` accounting.** *That service runs several
  jobs per replica and charges each; this worker runs one job per container* (§19h). **The one
  number kept is the proxy's thread count, 2**, because it is a fact about a proxy encode and
  not about sharing.

**NOTHING HERE MAY COST THE MASTER.** *By the time this runs the master is uploaded; a derive is
re-makeable from it and the master is not re-makeable from anything cheap* (§0, §19e). `make`
raises for a role that fails and the caller catches it per role, into `warnings[]`.

**Stdlib and ffmpeg only**, so the tests tree exercises it with real files and no GPU.
"""
import os
import subprocess

import keys
import probe
from errors import INTERNAL, WorkerError

#: `ffmpeg@9d7495b` `handler/operations.py:95`.
FFMPEG_BASE_ARGS = ["-y", "-hide_banner", "-loglevel", "error", "-nostdin"]

#: **Bounded, because a derive runs after the master exists and before the job returns.** *A
#: wedged ffmpeg here would hold a delivered job open against the platform's 3,600 s wall.* The
#: longest of the three — a proxy of an 8K master at 2 threads — is minutes, not tens of them.
FFMPEG_TIMEOUT_S = 1200

#: `ffmpeg@9d7495b` `handler/profiles.py:15, 22, 24-26, 103, 112`.
PROXY_LONG_EDGE = 1280
PROXY_THREADS = 2
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
#: value because the three makers share it and each makes up to two ffmpeg calls. *One job per
#: container (§19h), and `begin` is called at the top of every job's derive step, so a previous
#: job's deadline cannot outlive it.*
_DEADLINE = [None]


def begin(deadline):
    """Arm the job's deadline (`time.time()` seconds), or None for no deadline."""
    _DEADLINE[0] = deadline


def _run(args):
    import time  # noqa: PLC0415

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
    try:
        done = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise WorkerError(INTERNAL, "ffmpeg timed out after {:.0f} s".format(timeout))
    if done.returncode != 0:
        raise WorkerError(INTERNAL, "ffmpeg failed: {}".format(
            done.stderr.decode("utf-8", "replace").strip()[-500:]))


def _describe(path, role, key):
    return {"role": role, "key": key, "bytes": os.path.getsize(path),
            "content_type": keys.content_type(key)}


def _dimensions(path):
    out = probe.probe_output(path)
    return out["width"], out["height"], out


def poster(entry, master_path, master, workdir, name=None):
    """`ffmpeg@9d7495b` `operations.py:1202-1236`. **The master's own frame and resolution.**"""
    key = keys.for_role("poster", name)
    path = os.path.join(workdir, key)
    index = poster_index(entry.get("at_fraction", 0.25), master["frames"])
    _run(["-i", master_path, "-vf", "select='eq(n,{})'".format(index),
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
    step, count, columns, rows = sheet_layout(master["frames"])
    chain = "select='not(mod(n\\,{}))',scale={}:-2,tile={}x{}".format(
        step, SHEET_TILE_WIDTH, columns, rows)
    _run(["-threads", str(PROXY_THREADS), "-i", master_path, "-vf", chain, "-frames:v", "1",
          "-fps_mode", "passthrough", "-an"] + SHEET_ENCODER + [path])
    width, height, _ = _dimensions(path)
    return path, dict(_describe(path, "spritesheet", key), width=width, height=height,
                      columns=columns, rows=rows, tile_width=width // columns,
                      tile_height=height // rows, frame_step=step, count=count)


def proxy(entry, master_path, master, workdir, name=None):
    """`ffmpeg@9d7495b` `operations.py:1072-1138`; arguments `profiles.py:392-401, 502-522`.

    **h264 8-bit, 1280 on the long edge, crf 26, veryfast, 2 threads, 128k AAC when the master
    has audio, faststart** — and a 10-bit h265 master still yields an 8-bit h264 proxy, because a
    proxy that will not play in the viewer's browser is worse than one two bits shallower.
    """
    key = keys.for_role("proxy", name)
    path = os.path.join(workdir, key)
    cap = entry.get("max_duration_s")
    audio = (["-c:a", "aac", "-b:a", PROXY_AUDIO_BITRATE, "-ar", AUDIO_SAMPLE_RATE]
             if master.get("has_audio") else ["-an"])
    _run(["-threads", str(PROXY_THREADS), "-i", master_path]
         + ([] if cap is None else ["-t", "{:.6f}".format(cap)])
         + ["-threads", str(PROXY_THREADS), "-c:v", "libx264", "-preset", PROXY_PRESET,
            "-crf", str(PROXY_CRF), "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-g", str(_gop(master.get("fps"))), "-vf", PROXY_SCALE_FILTER]
         + audio + ["-movflags", "+faststart", path])
    # **§19e says FASTSTART and it is checked, not assumed** — the service's `ensure_faststart`
    # (`operations.py:160-172`): a mux does not always honour the flag.
    if not probe.is_faststart(path):
        shuffled = path + ".faststart.mp4"
        _run(["-i", path, "-c", "copy", "-movflags", "+faststart", shuffled])
        os.replace(shuffled, path)
        # **And re-read after the rewrite**, or "checked, not assumed" is true of the first
        # attempt only. Found in review.
        if not probe.is_faststart(path):
            raise WorkerError(INTERNAL, "the proxy is not faststart even after a remux")
    width, height, info = _dimensions(path)
    counted = probe.counted_frames(path, INTERNAL, "counting the proxy's packets")
    return path, dict(_describe(path, "proxy", key), width=width, height=height,
                      duration_s=info.get("duration_s"), fps=info.get("fps"),
                      frames=counted["frames"], has_audio=info.get("has_audio"))


MAKERS = {"poster": poster, "proxy": proxy, "spritesheet": spritesheet}


def make(entry, master_path, master, workdir, name=None):
    """One role. Returns `(local_path, record_entry)`; raises on failure — the caller contains it.

    `master` is `{"frames", "fps", "has_audio"}` read off the DELIVERED file — its counted packets
    and its probe — **never the source's**, which is §19e's first rule.
    """
    return MAKERS[entry["role"]](entry, master_path, master, workdir, name=name)
