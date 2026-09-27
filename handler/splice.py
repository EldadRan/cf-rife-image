"""`frame_repair`'s copy path, the ffmpeg half: `decisions.md` §20, as the spike settled it (§20i).

**A repair re-encodes the GOPs its items touch and copies every other packet bit for bit.** This
module owns everything about that which is ffmpeg and bytes: the packet map, the seams, the
eligibility checks and the declines, the copied segments, the span encoder's arguments, the join,
the mux, the two-set `avcC`, and the check before upload. **The frames that go INTO a span
encoder are `repair.run_copy`'s**: that is where the model is. This module imports no numpy, no
cv2 and no torch, so it runs on the tests tree.

**PORTED FROM `builder_scripts/splice_spike.py` (cf-rife-project `60a5c20`, certified by the gate
2026-09-27), which ported from `ffmpeg@4b3b8c1 handler/smartcut.py`.** Each piece cites where it
came from. Where the spike departed from smartcut, the reason is the spike's, and §20i rules it.

    source ──packets──▶ PacketMap ──seams──▶ repair_plan.spans ──▶ plan()  (declines here)
       │
       ├─ segment muxer, -c copy ──▶ partNNN.ts            every GOP outside the spans
       └─ repair.run_copy ─frames─▶ span_encoder() ──▶ spanNNN.ts   one per span
                                          │
            concat (declared durations) ──▶ joined.ts ──▶ mux (+audio, faststart)
                                          ──▶ rewrite_avcc (source id 0 + span id 1) ──▶ verify()

**Two exceptions, both a fallback and never a job failure** (CF ruling C): `NotEligible` before
anything is encoded, and `CheckFailed` after. The caller runs the full path and files the reason.
"""
import json
import os
import struct
import subprocess
from fractions import Fraction

import repair_plan

#: §20b, as amended by §20i R5. *The container is ffprobe's first `format_name` token: `mov` is
#: what it prints for both MP4 and MOV.*
ELIGIBLE_CODECS = ("h264",)
ELIGIBLE_CONTAINERS = ("mov",)
ELIGIBLE_PIX_FMTS = ("yuv420p", "yuvj420p", "yuv420p10le")

#: What an MPEG-TS round trip ADDS to a packet: an AUD on every one, SPS and PPS before every
#: IDR. Stripped at the final mux, which is only byte-exact if the source carries none of its own
#: (§20i R5; the spike's F2).
ROUND_TRIP_NALS = (7, 8, 9)

#: The id the spans' SPS and PPS are written under (§20i R2). The source's are id 0; a clash is
#: refused rather than resolved.
SPAN_PARAM_ID = 1

#: How far before a wanted frame a rough input seek aims. **Frame identity is never a time
#: `-ss`**: measured from the video's first PTS it landed 2 frames early on a source without an
#: edit list, and measured from the format start 1 frame late (the spike's F5). The seek is only
#: rough, `-copyts` keeps the source's own PTS, and `select` picks the exact tick.
SEEK_SLACK_S = 3.0

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
FFMPEG_TIMEOUT_S = 3600
FFPROBE_TIMEOUT_S = 600

#: x264 profile names ffprobe prints, as `-profile:v` spells them.
PROFILES = {"baseline": "baseline", "constrainedbaseline": "baseline", "main": "main",
            "high": "high", "high10": "high10"}


class NotEligible(Exception):
    """This source or this request cannot be copied. `path_reason` is `not_eligible: <this>`."""


class CheckFailed(Exception):
    """The spliced master failed §20e's check. `path_reason` is `check_failed: <this>`."""


def _run(argv, what, failure=CheckFailed, timeout=FFMPEG_TIMEOUT_S):
    try:
        done = subprocess.run(argv, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise failure("{} timed out after {} s".format(what, timeout))
    if done.returncode != 0:
        raise failure("{} failed (exit {}): {}".format(
            what, done.returncode, done.stderr.decode(errors="replace").strip()[-300:]))
    return done.stdout


def _probe_json(args, what, failure):
    return json.loads(_run([FFPROBE, "-v", "error", "-of", "json"] + args, what, failure,
                           FFPROBE_TIMEOUT_S) or b"{}")


# ── eligibility, before anything is read beyond the probe ────────────────────────────────────

#: What the MP4 muxer takes by copy: `encoder.MP4_NATIVE_AUDIO`'s list, restated because this
#: module imports no encoder. A source with other audio has it transcoded on the full path.
COPYABLE_AUDIO = ("aac", "mp3", "alac", "ac3", "eac3")


def eligibility(source, keep_audio=False):
    """§20b's cheap half, on `probe.probe_source`'s dict. None when eligible, else the reason.

    **Audio is checked too, when it is kept**: §20d copies the source's audio whole, and a codec
    the MP4 muxer will not take would fail the mux after every span was encoded. The full path
    transcodes it, so the decline is cheap and the fallback would not be. Found in review."""
    if source.get("codec") not in ELIGIBLE_CODECS:
        return "the source codec is {}, and this wave copies h264 only (§20h)".format(
            source.get("codec"))
    if source.get("container") not in ELIGIBLE_CONTAINERS:
        return "the source container is {}, not MP4 or MOV".format(source.get("container"))
    if source.get("pix_fmt") not in ELIGIBLE_PIX_FMTS:
        return "the source pixel format is {}, not one of {}".format(
            source.get("pix_fmt"), ", ".join(ELIGIBLE_PIX_FMTS))
    if keep_audio and source.get("has_audio") and source.get("audio_codec") not in COPYABLE_AUDIO:
        return "the source's audio is {}, which an MP4 cannot carry by copy".format(
            source.get("audio_codec"))
    return None


# ── the packet map ───────────────────────────────────────────────────────────────────────────

class PacketMap:
    """The source's video packets and everything the splice plans from.

    `decode` is decode order as ffprobe lists it; `frames` is presentation order, so entry i is
    frame i (smartcut `frame_times` :162-204). **Read once**; `seams` reads the samples once more
    to find the IDRs, streamed a packet at a time.
    """

    def __init__(self, path):
        self.path = path
        data = _probe_json(
            ["-select_streams", "v:0", "-show_entries",
             "stream=codec_name,pix_fmt,profile,level,width,height,r_frame_rate,time_base,"
             "color_range,color_space,color_primaries,color_transfer:"
             "packet=pts,dts,flags,size:format=start_time", path],
            "reading the source's packets", NotEligible)
        self.stream = (data.get("streams") or [{}])[0]
        self.width = int(self.stream["width"])
        self.height = int(self.stream["height"])
        self.pix_fmt = self.stream.get("pix_fmt")
        self.rate = Fraction(self.stream["r_frame_rate"])
        self.time_base = Fraction(self.stream["time_base"])
        self.format_start = Fraction((data.get("format") or {}).get("start_time") or "0")
        rows = []
        for packet in data.get("packets") or ():
            try:
                pts = int(packet["pts"])
                dts = int(packet.get("dts", pts))
            except (KeyError, TypeError, ValueError):
                raise NotEligible("a video packet carries no timestamp")
            rows.append((pts, dts, "K" in (packet.get("flags") or ""), int(packet["size"])))
        if not rows:
            raise NotEligible("the source has no video packets")
        self.decode = rows
        self.frames = sorted(rows, key=lambda row: row[0])
        if len({row[0] for row in rows}) != len(rows):
            raise NotEligible("two video packets share a presentation timestamp")
        self.count = len(rows)
        self.pts0 = self.frames[0][0]
        self._seams = None

    @property
    def depth(self):
        return 10 if "10" in (self.pix_fmt or "") else 8

    @property
    def frame_bytes(self):
        """One raw frame in the source's own pixel format. 4:2:0 only (`ELIGIBLE_PIX_FMTS`)."""
        return self.width * self.height * 3 // 2 * (2 if self.depth == 10 else 1)

    def nal_types(self):
        """NAL unit types inside each sample, decode order, read from the samples as stored.

        **Streamed a packet at a time**: an 8K source is gigabytes, and this needs one packet's
        bytes at once. The sizes come from ffprobe and the bytes from ffmpeg; if the two disagree
        by one byte, every later packet would be cut at the wrong offset, so the total is
        checked and a mismatch declines (§20i R5).
        """
        head, _, _, _ = read_avcc(self.path)
        length = (head[4] & 0x03) + 1
        proc = subprocess.Popen([FFMPEG, "-v", "error", "-nostdin", "-i", self.path,
                                 "-map", "0:v:0", "-c", "copy", "-f", "data", "-"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        types = []
        try:
            for _, _, _, size in self.decode:
                packet = _read_exactly(proc.stdout, size)
                if len(packet) != size:
                    raise NotEligible("reading the samples returned {} of {} packets".format(
                        len(types), self.count))
                here, i = [], 0
                while i + length < len(packet):
                    n = int.from_bytes(packet[i:i + length], "big")
                    here.append(packet[i + length] & 0x1F)
                    i += length + n
                types.append(here)
            if proc.stdout.read(1):
                raise NotEligible("reading the samples returned more bytes than the packets "
                                  "ffprobe lists")
        finally:
            proc.stdout.close()
            proc.wait()
        return types

    @property
    def seams(self):
        """§20c as amended by §20i R1: presentation indices of the CLEAN IDRs.

        Clean: every packet before it in decode order presents earlier, and every packet after
        it no earlier, which also makes its decode index equal to its presentation index — what
        lets the segment muxer (counting packets) and the plan (counting frames) agree. IDR: NAL
        type 5, because the span carries its own SPS and H.264 switches SPS only at an IDR.
        """
        if self._seams is None:
            n = self.count
            prefix_max, running = [None] * n, None
            for d, row in enumerate(self.decode):
                prefix_max[d] = running
                running = row[0] if running is None else max(running, row[0])
            suffix_min, running = [None] * n, None
            for d in range(n - 1, -1, -1):
                suffix_min[d] = running
                running = self.decode[d][0] if running is None else min(running,
                                                                        self.decode[d][0])
            rank = {row[0]: i for i, row in enumerate(self.frames)}
            types = self.nal_types()
            self.inband = sorted({t for here in types for t in here} & set(ROUND_TRIP_NALS))
            clean, unclean = [], {}
            for d, (pts, _, key, _) in enumerate(self.decode):
                if not key:
                    continue
                if not ((prefix_max[d] is None or prefix_max[d] < pts)
                        and (suffix_min[d] is None or suffix_min[d] >= pts)):
                    unclean[rank[pts]] = "open GOP"
                elif 5 not in types[d]:
                    unclean[rank[pts]] = "not an IDR"
                else:
                    clean.append(rank[pts])
            self._seams = (sorted(clean), unclean)
        return self._seams[0]

    @property
    def unclean(self):
        self.seams  # noqa: B018 — computes both
        return self._seams[1]

    def reorder_delay(self):
        """PTS minus DTS at the seams, in whole frames (§20i R3). The span has no B-frames, so
        its DTS equals its PTS, while the source's runs this many frames behind; spliced as-is,
        the muxer bends the DTS at the span's end into 1-tick steps (the spike's F3)."""
        by_pts = {row[0]: row[1] for row in self.decode}
        delays = {(self.frames[k][0] - by_pts[self.frames[k][0]]) * self.time_base * self.rate
                  for k in self.seams}
        if len(delays) != 1 or next(iter(delays)).denominator != 1:
            raise NotEligible("the source's reorder delay is not one whole number of frames ({})"
                              .format(", ".join(sorted(str(d) for d in delays))))
        return int(next(iter(delays)))

    def from_frame(self, index):
        """Input arguments and a filter that start the video exactly at frame `index`, by PTS."""
        pts = self.frames[index][0]
        rough = max(0.0, float(pts * self.time_base - self.format_start) - SEEK_SLACK_S)
        return (["-ss", "{:.6f}".format(rough), "-copyts", "-i", self.path],
                "select=gte(pts\\,{})".format(pts))

    def matrix(self):
        """The source's colour matrix and range, for swscale, explicitly (§20c). An untagged
        source is BT.601, which is what swscale would assume anyway; saying it is the point."""
        space = self.stream.get("color_space") or ""
        matrix = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601",
                  "bt2020nc": "bt2020"}.get(space, "bt601")
        full = self.stream.get("color_range") == "pc" or self.pix_fmt == "yuvj420p"
        return matrix, ("pc" if full else "tv")


def _read_exactly(stream, size):
    chunks, got = [], 0
    while got < size:
        chunk = stream.read(size - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


# ── the plan ─────────────────────────────────────────────────────────────────────────────────

def plan(pmap, items):
    """§20c's spans for this source, or `NotEligible`. `items` carry `id`, `a`, `b`.

    The declines are §20i R5's, checked in the order the cheapest first: the seams (which
    reads the samples), what the samples carry, frame 0, the reorder delay, and a span that is
    the whole file — *nothing to copy, so the full path is the same work without the splice's
    risk*.
    """
    seams = pmap.seams
    if pmap.inband:
        raise NotEligible("the source carries NAL types {} (SPS/PPS/AUD) inside its samples, "
                          "and stripping what the join adds would strip those too".format(
                              pmap.inband))
    if not seams or seams[0] != 0:
        raise NotEligible("frame 0 is not a clean IDR")
    delay = pmap.reorder_delay()
    spans = repair_plan.spans(pmap.count, seams, items)
    if spans and spans[0][0] == 0 and spans[0][1] == pmap.count:
        described = ", ".join("{} ({})".format(k, why) for k, why in sorted(
            pmap.unclean.items())[:6])
        raise NotEligible(
            "the span is the whole file: {} clean IDR(s) at {}{}".format(
                len(seams), seams[:6], "; keyframes not usable as seams: " + described
                if described else ""))
    return spans, delay


def layout(frame_count, spans):
    """The whole file as consecutive `(kind, start, end)`, kind "copy" or "encode"."""
    out, cursor = [], 0
    for start, end, _ in spans:
        if start > cursor:
            out.append(("copy", cursor, start))
        out.append(("encode", start, end))
        cursor = end
    if cursor < frame_count:
        out.append(("copy", cursor, frame_count))
    return out


# ── the copied GOPs ──────────────────────────────────────────────────────────────────────────

def copy_parts(pmap, segments, workdir):
    """Every run of the layout, cut at its boundaries in ONE `-c copy` pass to MPEG-TS.

    Returns `{start frame: path}` for every run; the encode runs' parts are discarded by the
    caller. **The segment muxer counts packets in decode order**, which at a clean seam equals
    the presentation index (`PacketMap.seams`); the check before upload proves the cut landed.
    """
    cuts = [start for _, start, _ in segments if start > 0]
    pattern = os.path.join(workdir, "part%05d.ts")
    argv = [FFMPEG, "-v", "error", "-y", "-nostdin", "-i", pmap.path, "-map", "0:v:0",
            "-c", "copy", "-f", "segment", "-segment_format", "mpegts",
            "-reset_timestamps", "0"]
    if cuts:
        argv += ["-segment_frames", ",".join(str(c) for c in cuts)]
    _run(argv + [pattern], "copying the untouched GOPs")
    return {start: pattern % n for n, (_, start, _) in enumerate(segments)}, argv


# ── the span encoder ─────────────────────────────────────────────────────────────────────────

def span_encoder(pmap, frames, out_ts, delay, crf, preset, threading=None):
    """The argv of one span's encoder: raw frames in the source's own pixel format on stdin,
    MPEG-TS out.

    `ffmpeg@4b3b8c1 smartcut._edge_encode_args` :296-350: libx264, the source's profile, level,
    pixel format and colour tags, `-refs 1 -bf 0` (§20i R4). **What the spike added and §20i
    rules:** `sps-id=1` (R2), a GOP of the span plus one so the span holds exactly one IDR at its
    start, scene-cut off, and the DTS moved back by the source's reorder delay (R3) — in the
    encoder's time base, which is one frame, so it is exact before any 90 kHz rounding.

    **`threading` is the full path's own x264 bound** — `encoder.x264_params`' string off
    `encoder.resolve_defaults` for the delivered area, the caller's fields winning. *Unbounded,
    x264 on the 96-core host runs 128 frame threads, and that configuration filled 46 GiB and got
    the first 8K run reaped* (`encoder.py`). Found in review.
    """
    args = [FFMPEG, "-v", "error", "-y", "-nostdin", "-f", "rawvideo",
            "-pix_fmt", pmap.pix_fmt, "-s", "{}x{}".format(pmap.width, pmap.height),
            "-framerate", "{}/{}".format(pmap.rate.numerator, pmap.rate.denominator),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf),
            "-pix_fmt", pmap.pix_fmt, "-refs", "1", "-bf", "0",
            "-x264-params", x264_params(threading), "-g", str(frames + 1)]
    profile = PROFILES.get((pmap.stream.get("profile") or "").lower().replace(" ", ""))
    if profile:
        args += ["-profile:v", profile]
    level = pmap.stream.get("level")
    if isinstance(level, int) and level > 0:
        args += ["-level", str(level)]
    for key, flag in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                      ("color_primaries", "-color_primaries"),
                      ("color_transfer", "-color_trc")):
        value = pmap.stream.get(key)
        if value and value != "unknown":
            args += [flag, str(value)]
    if delay:
        args += ["-bsf:v", "setts=dts=DTS-{}".format(delay)]
    return args + ["-f", "mpegts", out_ts]


def x264_params(threading=None):
    """The span encoder's `-x264-params`: the thread bound, then §20i's two."""
    return "{}scenecut=0:sps-id={}".format(threading + ":" if threading else "", SPAN_PARAM_ID)


def span_decoder(pmap, start, frames):
    """The argv that decodes `frames` source frames from `start`, exactly, in the source's own
    pixel format — the untouched frames of a span, which never go through RGB (§20c)."""
    inputs, select = pmap.from_frame(start)
    return ([FFMPEG, "-v", "error", "-nostdin"] + inputs +
            ["-map", "0:v:0", "-vf", select, "-fps_mode", "passthrough",
             "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", pmap.pix_fmt, "-"])


def rgb_decoder(path, from_args, select, frames, matrix, rng):
    """The argv that decodes `frames` frames as cv2-layout BGR (what `routec._tensors` takes),
    converting with an EXPLICIT matrix and range. `from_args`/`select` place the first frame."""
    vf = "scale=in_color_matrix={}:in_range={},format=bgr24".format(matrix, rng)
    if select:
        vf = select + "," + vf
    return ([FFMPEG, "-v", "error", "-nostdin"] + from_args +
            ["-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(frames),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])


def rgb_to_source(pmap, frames):
    """The argv that turns `frames` rgb24 frames on stdin into the source's own pixel format,
    with the source's matrix and range stated explicitly (§20c) — replaced frames only."""
    matrix, rng = pmap.matrix()
    return [FFMPEG, "-v", "error", "-y", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(pmap.width, pmap.height), "-i", "-", "-frames:v", str(frames),
            "-vf", "scale=out_color_matrix={}:out_range={},format={}".format(
                matrix, rng, pmap.pix_fmt),
            "-f", "rawvideo", "-pix_fmt", pmap.pix_fmt]


# ── §22: the full path as one span ───────────────────────────────────────────────────────────
#
# **A full re-encode is one span from frame 0 to the end** (`decisions.md` §22a): the whole file
# decoded once in the source's own YUV, untouched frames straight to the encoder, replaced frames
# through RGB with the source's matrix and range both ways. *No seam, so none of the copy's
# constraints: the encoder keeps the preset's B-frames and references (§22a as amended), writes
# MP4 directly, and only the PTS are held to the source's.* What cannot be planned for runs the
# frame loop (§22b-1), and `native_decline` says why.

#: What a writer's 8- or 10-bit output is when the source already has that layout: `yuvj420p`
#: stays `yuvj420p` (the same bytes, full range), so no conversion runs at all.
_SAME_LAYOUT = {("yuvj420p", "yuv420p"): "yuvj420p"}


def native_decline(source, pmap):
    """Why the native full path cannot take this source, or None (§22b-1).

    **Every reason is a timeline the encode could not reproduce, or input it does not read**:
    h264 in MP4/MOV at 4:2:0 (the copy's own list, `ELIGIBLE_*`); a time base whose inverse is a
    whole timescale; a frame duration that is a whole number of ticks; and every PTS exactly on
    that grid from the first. *The encoder is fed at the rate and placed with `-itsoffset`, so
    its PTS are the grid: a source off the grid would fail `verify_pts` after the whole encode.*
    """
    if source.get("codec") not in ELIGIBLE_CODECS:
        return "the source codec is {}, and the native path reads h264".format(
            source.get("codec"))
    if source.get("container") not in ELIGIBLE_CONTAINERS:
        return "the source container is {}, not MP4 or MOV".format(source.get("container"))
    if pmap.pix_fmt not in ELIGIBLE_PIX_FMTS:
        return "the source pixel format is {}, not one of {}".format(
            pmap.pix_fmt, ", ".join(ELIGIBLE_PIX_FMTS))
    timescale = 1 / pmap.time_base
    if timescale.denominator != 1:
        return "the source's time base {} is not a whole timescale".format(pmap.time_base)
    tick = 1 / (pmap.rate * pmap.time_base)
    if tick.denominator != 1:
        return "a frame at {} fps is {} ticks of {}, not a whole number".format(
            pmap.rate, tick, pmap.time_base)
    if pmap.pts0 < 0:
        return "the source's first PTS is negative ({})".format(pmap.pts0)
    for k, row in enumerate(pmap.frames):
        if row[0] != pmap.pts0 + k * tick.numerator:
            return ("the source's PTS leave the {}-tick grid at frame {} ({} where {} was "
                    "expected)".format(tick.numerator, k, row[0], pmap.pts0 + k * tick.numerator))
    return None


def native_input(pmap, pix_fmt_out):
    """`encoder.MasterWriter`'s `native` block for this source (§22a).

    `pix_fmt_out` is `encoder.pixel_format` of the requested depth. **The source's own layout is
    kept where it is the same** (`_SAME_LAYOUT`); otherwise the change is a `format` in the
    encoder's filter graph, in YUV, with the range stated on both sides so a full-range source is
    not squeezed. The output carries the source's colour tags — a full-range source's `pc` said
    explicitly, since a converted `yuv420p10le` carries no range of its own."""
    out = _SAME_LAYOUT.get((pmap.pix_fmt, pix_fmt_out), pix_fmt_out)
    _, rng = pmap.matrix()
    vf = None
    if out != pmap.pix_fmt:
        vf = "scale=in_range={0}:out_range={0},format={1}".format(rng, out)
    output_args = []
    for key, flag in (("color_space", "-colorspace"), ("color_primaries", "-color_primaries"),
                      ("color_transfer", "-color_trc")):
        value = pmap.stream.get(key)
        if value and value != "unknown":
            output_args += [flag, str(value)]
    tagged = pmap.stream.get("color_range")
    if rng == "pc" or (tagged and tagged != "unknown"):
        output_args += ["-color_range", rng]
    output_args += ["-video_track_timescale", str((1 / pmap.time_base).numerator)]
    lead = pmap.pts0 * pmap.time_base
    return {"pix_fmt": pmap.pix_fmt, "frame_bytes": pmap.frame_bytes, "pix_fmt_out": out,
            "vf": vf, "input_args": ["-itsoffset", "{:.9f}".format(float(lead))] if lead else [],
            "output_args": output_args}


def native_decoder(pmap):
    """The argv that decodes the whole source, every frame once, in its own pixel format.
    **Uncapped**, so a decoder that runs long is counted rather than cut (§19d)."""
    return [FFMPEG, "-v", "error", "-nostdin", "-i", pmap.path, "-map", "0:v:0",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pmap.pix_fmt, "-"]


def yuv_to_rgb(pmap, frames):
    """The argv that turns `frames` raw frames of the source's pixel format on stdin into
    cv2-layout BGR — `rgb_decoder`'s conversion, with the same explicit matrix and range, fed
    from frames the native decode already holds rather than from a second decode."""
    matrix, rng = pmap.matrix()
    return [FFMPEG, "-v", "error", "-y", "-nostdin", "-f", "rawvideo", "-pix_fmt", pmap.pix_fmt,
            "-s", "{}x{}".format(pmap.width, pmap.height), "-i", "-", "-frames:v", str(frames),
            "-vf", "scale=in_color_matrix={}:in_range={},format=bgr24".format(matrix, rng),
            "-f", "rawvideo", "-pix_fmt", "bgr24"]


def verify_pts(master_path, pmap):
    """§22a's added line on the full path: the master's time base and every PTS equal to the
    source's, tick for tick. **Not the DTS** (§22a as amended: B-frames stay). Raises
    `CheckFailed`."""
    data = _probe_json(["-select_streams", "v:0", "-show_entries",
                        "stream=time_base:packet=pts", master_path],
                       "reading the master's timestamps", CheckFailed)
    stream = (data.get("streams") or [{}])[0]
    if Fraction(stream.get("time_base") or "0/1") != pmap.time_base:
        raise CheckFailed("time base {} against the source's {}".format(
            stream.get("time_base"), pmap.time_base))
    try:
        got = sorted(int(p["pts"]) for p in data.get("packets") or ())
    except (KeyError, TypeError, ValueError):
        raise CheckFailed("a master packet carries no PTS")
    want = [row[0] for row in pmap.frames]
    if got != want:
        first = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w),
                     min(len(got), len(want)))
        raise CheckFailed("{} PTS against the source's {}; the first difference is at frame {}"
                          .format(len(got), len(want), first))


# ── the join and the mux ─────────────────────────────────────────────────────────────────────

def join(pmap, files, workdir):
    """smartcut `_join` :523-555: the concat demuxer with each segment's duration DECLARED, to
    one TS. *Without the declarations a segment ending on a B-frame reports one frame short.*
    The TS-then-MP4 order is smartcut's :54-70 (MP4 segments break DTS order; TS straight to MP4
    breaks `stts`)."""
    listing = os.path.join(workdir, "concat.txt")
    with open(listing, "w") as handle:
        for path, frames in files:
            handle.write("file '{}'\n".format(path.replace("'", r"'\''")))
            handle.write("duration {:.9f}\n".format(float(frames / pmap.rate)))
    joined = os.path.join(workdir, "joined.ts")
    argv = [FFMPEG, "-v", "error", "-y", "-nostdin", "-f", "concat", "-safe", "0",
            "-i", listing, "-map", "0:v:0", "-c", "copy", "-f", "mpegts", joined]
    _run(argv, "joining the parts")
    return joined, argv


def mux(pmap, joined, master_path, audio_source):
    """The joined TS to MP4: `avc1`, `+faststart`, the source's video timescale, the round
    trip's AUD/SPS/PPS stripped, and the source's audio copied whole (§20d).

    **Two things put the source's timeline back to the tick** (the spike's F5): the concat
    rebases the video to 0, so `-itsoffset` restores the source's video lead over its format
    start (83 ms of A/V offset on a source without an edit list); and an edit list is written
    only if the source had one, so the first PTS and every DTS are the source's.
    """
    argv = [FFMPEG, "-v", "error", "-y", "-nostdin"]
    lead = pmap.pts0 * pmap.time_base - pmap.format_start
    if lead:
        argv += ["-itsoffset", "{:.9f}".format(float(lead))]
    argv += ["-i", joined]
    if audio_source:
        argv += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0?"]
    else:
        argv += ["-map", "0:v:0"]
    argv += ["-c", "copy", "-bsf:v", "filter_units=remove_types=" + "|".join(
        str(t) for t in ROUND_TRIP_NALS)]
    if pmap.decode[0][1] >= 0:
        argv += ["-use_editlist", "0"]
    timescale = 1 / pmap.time_base
    if timescale.denominator == 1:
        argv += ["-video_track_timescale", str(timescale.numerator)]
    argv += ["-tag:v", "avc1", "-movflags", "+faststart", master_path]
    _run(argv, "muxing the master")
    return argv


# ── avcC ─────────────────────────────────────────────────────────────────────────────────────

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _boxes(data, start, end):
    at = start
    while at + 8 <= end:
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[at + 8:at + 16])[0]
            header = 16
        elif size == 0:
            size = end - at
        if size < header:
            return
        yield at, size, header, kind
        at += size


def _find_avcc(data):
    """`(chain from moov down to avc1 as (offset, header, kind), avcC offset, avcC size)`."""
    def walk(start, end, chain):
        for at, size, header, kind in _boxes(data, start, end):
            here = chain + [(at, header, kind)]
            if kind in _CONTAINERS:
                found = walk(at + header, at + size, here)
                if found:
                    return found
            elif kind == b"stsd":
                for eat, esize, eheader, ekind in _boxes(data, at + header + 8, at + size):
                    if ekind in (b"avc1", b"avc3"):
                        # A VisualSampleEntry's fields run 78 bytes before its child boxes.
                        for cat, csize, _, ckind in _boxes(data, eat + eheader + 78,
                                                           eat + esize):
                            if ckind == b"avcC":
                                return here + [(eat, eheader, ekind)], cat, csize
        return None
    return walk(0, len(data), [])


def _moov_bytes(path):
    """The file's top-level boxes read by header, and the moov read whole. **The mdat is never
    read**: at 8K it is gigabytes, and nothing here needs it."""
    boxes = []
    with open(path, "rb") as handle:
        at, size_total = 0, os.path.getsize(path)
        while at + 8 <= size_total:
            handle.seek(at)
            head = handle.read(16)
            size, kind = struct.unpack(">I4s", head[:8])
            header = 8
            if size == 1:
                size, header = struct.unpack(">Q", head[8:16])[0], 16
            elif size == 0:
                size = size_total - at
            if size < header:
                break
            boxes.append((at, size, header, kind))
            at += size
        moov = [b for b in boxes if b[3] == b"moov"]
        if not moov:
            raise CheckFailed("{} has no moov".format(os.path.basename(path)))
        handle.seek(moov[0][0])
        return boxes, moov[0], handle.read(moov[0][1])


def parse_avcc(payload):
    """`(head, [sps], [pps], tail)` of an avcC payload."""
    head = payload[:5]
    at, sps, pps = 6, [], []
    for _ in range(payload[5] & 0x1F):
        n = struct.unpack(">H", payload[at:at + 2])[0]
        sps.append(payload[at + 2:at + 2 + n])
        at += 2 + n
    count = payload[at]
    at += 1
    for _ in range(count):
        n = struct.unpack(">H", payload[at:at + 2])[0]
        pps.append(payload[at + 2:at + 2 + n])
        at += 2 + n
    return head, sps, pps, payload[at:]


def build_avcc(head, sps, pps, tail):
    body = bytes(head) + bytes([0xE0 | len(sps)])
    for unit in sps:
        body += struct.pack(">H", len(unit)) + unit
    body += bytes([len(pps)])
    for unit in pps:
        body += struct.pack(">H", len(unit)) + unit
    return body + bytes(tail)


def read_avcc(path):
    _, _, moov = _moov_bytes(path)
    found = _find_avcc(moov)
    if not found:
        raise NotEligible("the source has no avcC")
    _, at, size = found
    return parse_avcc(moov[at + 8:at + size])


def param_id(unit):
    """The id an SPS or PPS NAL declares: the SPS's follows profile, flags and level (24 bits),
    the PPS's comes first. Both ue(v), and too early for emulation prevention to reach."""
    bits = "".join("{:08b}".format(b) for b in unit[1:12])
    at = 24 if unit[0] & 0x1F == 7 else 0
    zeros = 0
    while bits[at + zeros] == "0":
        zeros += 1
    return int(bits[at + zeros:at + 2 * zeros + 1], 2) - 1


def annexb_units(data):
    """NAL units of an Annex B byte string."""
    starts, at = [], 0
    while True:
        i = data.find(b"\x00\x00\x01", at)
        if i < 0:
            break
        starts.append(i + 3)
        at = i + 3
    units = []
    for k, s in enumerate(starts):
        unit = data[s:starts[k + 1] - 3 if k + 1 < len(starts) else len(data)]
        units.append(unit.rstrip(b"\x00"))
    return units


def span_param_sets(span_ts):
    """The SPS and PPS a span's encoder wrote, off its first packet."""
    raw = _run([FFMPEG, "-v", "error", "-nostdin", "-i", span_ts, "-map", "0:v:0",
                "-frames:v", "1", "-c", "copy", "-f", "h264", "-"], "reading a span's SPS")
    units = annexb_units(raw)
    return ([u for u in units if u[0] & 0x1F == 7], [u for u in units if u[0] & 0x1F == 8])


def avcc_sets(source_sps, source_pps, span_sets):
    """The source's sets, then each distinct span set. **Two different sets under one id would
    make the file ambiguous, so that declines rather than picking one** (§20i R5)."""
    sps, pps = list(source_sps), list(source_pps)
    for new_sps, new_pps in span_sets:
        for have, units in ((sps, new_sps), (pps, new_pps)):
            for unit in units:
                if unit in have:
                    continue
                if any(param_id(u) == param_id(unit) for u in have):
                    raise CheckFailed("two different parameter sets under id {}".format(
                        param_id(unit)))
                have.append(unit)
    return sps, pps


def rewrite_avcc(path, sps, pps, head):
    """Write the two-set avcC into the master in place (§20i R2).

    Every enclosing box grows by the same number of bytes and, the moov being before the mdat
    (faststart), every chunk offset moves with it — audio included. `head` (profile,
    compatibility, level) is the SOURCE's: the muxer built the existing avcC from whichever part
    came first, which is a span when a repair starts at frame 0. **The NAL length size stays the
    OUTPUT's**, because it describes the samples the muxer wrote. The mdat is streamed across,
    never held.
    """
    boxes, (moov_at, moov_size, moov_header, _), moov = _moov_bytes(path)
    moov = bytearray(moov)
    found = _find_avcc(moov)
    if not found:
        raise CheckFailed("the master has no avcC")
    chain, at, size = found
    _, _, _, tail = parse_avcc(bytes(moov[at + 8:at + size]))
    written = bytes(moov[at + 8:at + 13])
    head = bytes(head[:4]) + bytes([(head[4] & 0xFC) | (written[4] & 0x03)])
    payload = build_avcc(head, sps, pps, tail)
    delta = 8 + len(payload) - size
    moov[at:at + size] = struct.pack(">I4s", 8 + len(payload), b"avcC") + payload
    for box_at, header, _ in chain:
        if header != 8:
            raise CheckFailed("a 64-bit box size in the moov path, which this does not rewrite")
        old = struct.unpack(">I", moov[box_at:box_at + 4])[0]
        moov[box_at:box_at + 4] = struct.pack(">I", old + delta)
    mdat = [b for b in boxes if b[3] == b"mdat"]
    if not mdat or moov_at > mdat[0][0]:
        raise CheckFailed("the moov is not before the mdat; faststart did not happen")
    if delta:
        _shift_chunk_offsets(moov, delta)
    temporary = path + ".avcc"
    try:
        with open(path, "rb") as source, open(temporary, "wb") as out:
            out.write(source.read(moov_at))
            out.write(moov)
            source.seek(moov_at + moov_size)
            while True:
                block = source.read(1 << 24)
                if not block:
                    break
                out.write(block)
        os.replace(temporary, path)
    except OSError as exc:
        # **A partial copy of an 8K mdat must not survive the fallback**, which writes its own
        # master beside it. Found in review.
        if os.path.exists(temporary):
            os.remove(temporary)
        raise CheckFailed("rewriting the avcC: {}".format(exc))
    return delta


def _shift_chunk_offsets(moov, delta):
    def walk(start, end):
        for at, size, header, kind in _boxes(moov, start, end):
            if kind in _CONTAINERS:
                walk(at + header, at + size)
            elif kind in (b"stco", b"co64"):
                count = struct.unpack(">I", moov[at + 12:at + 16])[0]
                width, fmt = (4, ">I") if kind == b"stco" else (8, ">Q")
                for k in range(count):
                    p = at + 16 + k * width
                    value = struct.unpack(fmt, moov[p:p + width])[0]
                    moov[p:p + width] = struct.pack(fmt, value + delta)
    header = 8 if struct.unpack(">I", moov[:4])[0] != 1 else 16
    walk(header, len(moov))


# ── the check before upload ──────────────────────────────────────────────────────────────────

def verify(master_path, pmap, spans):
    """§20e and §20i's added line, on the written master. Raises `CheckFailed`; returns None.

      frames equal, r_frame_rate equal, the same time base
      every PTS and every DTS equal to the source's, tick for tick
      the sample entry avc1, faststart
      every seam decodes clean: each span from its start through the whole GOP after its end,
          which is where an open GOP or a wrong parameter set breaks
    """
    data = _probe_json(["-select_streams", "v:0", "-show_entries",
                        "stream=codec_tag_string,r_frame_rate,time_base:packet=pts,dts",
                        master_path], "reading the master's packets", CheckFailed)
    stream = (data.get("streams") or [{}])[0]
    packets = data.get("packets") or []
    failures = []
    if len(packets) != pmap.count:
        failures.append("the master holds {} frames and the source {}".format(
            len(packets), pmap.count))
    if Fraction(stream.get("r_frame_rate") or "0/1") != pmap.rate:
        failures.append("r_frame_rate {} against the source's {}".format(
            stream.get("r_frame_rate"), pmap.rate))
    if Fraction(stream.get("time_base") or "0/1") != pmap.time_base:
        failures.append("time base {} against the source's {}".format(
            stream.get("time_base"), pmap.time_base))
    if stream.get("codec_tag_string") != "avc1":
        failures.append("sample entry {}".format(stream.get("codec_tag_string")))
    try:
        got_pts = sorted(int(p["pts"]) for p in packets)
        got_dts = [int(p["dts"]) for p in packets]
    except (KeyError, TypeError, ValueError):
        got_pts, got_dts = None, None
        failures.append("a master packet carries no timestamp")
    if got_pts is not None:
        want_pts = [row[0] for row in pmap.frames]
        want_dts = [row[1] for row in pmap.decode]
        if got_pts != want_pts:
            first = next((i for i, (g, w) in enumerate(zip(got_pts, want_pts)) if g != w), None)
            failures.append("PTS differ from the source's, first at frame {}".format(first))
        if got_dts != want_dts:
            first = next((i for i, (g, w) in enumerate(zip(got_dts, want_dts)) if g != w), None)
            failures.append("DTS differ from the source's, first at packet {}".format(first))
    boxes, _, _ = _moov_bytes(master_path)
    kinds = [b[3] for b in boxes]
    if b"mdat" not in kinds or kinds.index(b"moov") > kinds.index(b"mdat"):
        failures.append("the moov is not before the mdat")
    if failures:
        raise CheckFailed("; ".join(failures))
    seams = pmap.seams
    for start, end, _ in spans:
        after = [k for k in seams if k > end]
        stop = after[0] if after else pmap.count
        inputs, select = PacketMap.from_frame(pmap, start)
        inputs[-1] = master_path
        try:
            done = subprocess.run(
                [FFMPEG, "-v", "error", "-nostdin"] + inputs +
                ["-map", "0:v:0", "-vf", select, "-fps_mode", "passthrough",
                 "-frames:v", str(stop - start), "-f", "framemd5", "-"],
                capture_output=True, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise CheckFailed("the seam decode at [{}, {}) timed out".format(start, end))
        said = done.stderr.decode(errors="replace").strip()
        # **Counted, not only listened to**: a decode that stops early says nothing on stderr.
        decoded = sum(1 for line in done.stdout.decode(errors="replace").splitlines()
                      if line and not line.startswith("#"))
        if done.returncode != 0 or said or decoded != stop - start:
            raise CheckFailed("the seam at [{}, {}) does not decode clean through frame {}: "
                              "{} of {} frame(s){}".format(
                                  start, end, stop, decoded, stop - start,
                                  ", " + said[:200] if said else ""))


def colour_of(path):
    """A file's colour matrix and range for swscale, explicitly — `PacketMap.matrix`'s rule, for
    a file that is not the source (a segment)."""
    data = _probe_json(["-select_streams", "v:0", "-show_entries",
                        "stream=pix_fmt,color_space,color_range", path],
                       "reading a segment's colour tags", CheckFailed)
    stream = (data.get("streams") or [{}])[0]
    matrix = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601",
              "bt2020nc": "bt2020"}.get(stream.get("color_space") or "", "bt601")
    full = stream.get("color_range") == "pc" or stream.get("pix_fmt") == "yuvj420p"
    return matrix, ("pc" if full else "tv")
