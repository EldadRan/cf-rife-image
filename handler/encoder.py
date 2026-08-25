"""Writing the master, once, with everything CF requires in the same mux.

**This module exists because the vendored writer cannot satisfy the contract** (see
`docs/decisions.md` 0.3). SeedVR2's `save_frames_to_video` offers `mp4v` through OpenCV or
`libx264 -crf 12` through an ffmpeg pipe, and neither writes faststart, `-metadata` tags or
audio. CF requires all three **in the mux already being done** and explicitly forbids a second
pass — a remux that exists only to add tags is a rewrite, and CF has measured a stream-copy
faststart pass take a 342-frame trim to 343.

So the worker owns the encode. Frames arrive as they are produced and go straight down a pipe;
nothing is staged on disk and no finished file is ever reopened to fix what the first pass should
have set.

**On `+faststart` being one pass and not two.** ffmpeg relocates the moov atom after writing the
last frame, inside the same invocation. That is not a remux of a finished file: no packet is
re-encoded, no container is reinterpreted, and no edit list exists to be moved — this worker's
output has none, because it writes a fresh timeline rather than bounding an existing one. The
distinction matters because the failure CF measured comes from *reinterpreting* a container that
carried an edit list, which cannot arise here.
"""

import os
import subprocess

import probe

from errors import INTERNAL, WorkerError

#: Audio codecs that mux into MP4 as-is. Anything else is re-encoded to AAC — the media worker's
#: rule, and the reason a copied track stays bit-exact where it can.
MP4_NATIVE_AUDIO = ("aac", "mp3", "alac", "ac3", "eac3")

#: The master's encoder settings. Not measured yet — the encode figures CF was owed were never
#: would justify them, and until those exist these are a starting point rather than a decision.
#: CRF 12 is the vendored writer's own choice, kept so the first measurements compare against
#: something rather than against a number invented here.
DEFAULT_CRF = 12
DEFAULT_PRESET = "medium"


def _identity_tags(identity):
    """`-metadata` arguments. **Identity only** — this file is delivered.

    Timings, hardware, tiling configuration, worker ids and anything resembling a credential stay
    in the manifest and the diagnostics bundle. What goes in the container is what the file needs
    to say what it is when it is found in R2 with no job and no manifest beside it.

    It is a recovery aid and never a source of truth: CF's standing rule is to read the worker's
    reported fields rather than re-probe the file, and these tags are what someone falls back to
    when the response and the manifest are both gone.
    """
    args = []
    for key, value in identity.items():
        if value is None:
            continue
        args += ["-metadata", "{}={}".format(key, value)]
    return args


def still_master_extension(width, height):
    """Always `.png`.

    **PNG rather than lossless WebP, chosen for the people who have to look at the output.** Both
    are lossless and WebP is materially smaller at these dimensions, which is what this returned
    before. But a master is the thing a person opens to check the work, hands to a customer, or
    drags into an editor, and PNG opens everywhere without a thought while WebP still meets tools
    that will not preview it. Paying storage for that is the right trade: the file is written once
    and looked at many times, and an artefact nobody can conveniently open is an artefact nobody
    checks.

    It also removes a ceiling. WebP is limited to 16383 pixels per side by the format itself,
    which sits inside the range this worker is aimed at — 12K fits, 16K does not — so the old
    two-format rule had a real edge in it. PNG has no practical limit, so there is one format, one
    path, and no dimension at which the master silently changes type.

    Lossless WebP is still used for the `crop` derives, where CF asked for it by name and the
    files are small evidence images rather than the deliverable.

    The arguments are kept so the signature does not change if a size-dependent rule ever comes
    back.
    """
    del width, height
    return ".png"


def _peak_rss_gb(pid):
    """The largest resident set this process reached, in GiB, or None where it cannot be read.

    **`/proc/<pid>/status`'s `VmHWM`, sampled while the process is alive.** `getrusage`'s
    `RUSAGE_CHILDREN` would be the easy answer and is the wrong one: it reports the maximum across
    every child this worker has ever reaped, so an ffprobe from a previous phase and the encode
    would be indistinguishable — a plausible number about a different process, which is the class
    this project keeps finding. `VmHWM` is that process's own high-water mark and it disappears
    when the process does, which is why it is sampled rather than read at the end.

    None on anything that is not Linux, which is honest: a figure that is absent says nothing and
    a figure that is zero says the encode used no memory.

    **It under-reads on a clip shorter than the encoder's buffering window, and there it under-
    reads totally.** The last sample is taken when the last `write()` returns; nothing samples
    while x264 drains its lookahead and flushes. On a long encode that phase is memory-
    non-increasing — frames are released and none admitted — so the shortfall is near zero. On a
    fixture of a few dozen frames the whole encode happens after the final write, and the number
    describes the buffering footprint rather than the encode. **A reassuringly low peak from a
    small fixture means nothing**, which matters because a small fixture is what somebody
    reaches for when checking that the measurement works.
    """
    try:
        with open("/proc/{}/status".format(pid), encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 2)
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return None
    return None


class MasterWriter:
    """A one-pass ffmpeg encode fed frame by frame.

    Used as a context manager so the pipe is closed and the process reaped on any path,
    including an OOM raised mid-generation by the model upstream of it.
    """

    def __init__(self, path, width, height, fps, identity,
                 audio_source=None, audio_codec=None, audio_limit_s=None,
                 crf=DEFAULT_CRF, preset=DEFAULT_PRESET, x264_params=None):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        #: **ffmpeg's own high-water mark, not ours.** The 8K run died at ~46 GiB in x264's
        #: working set while this side held one frame and a cached pair — and nothing reported
        #: it, so the ceiling had to be inferred from a kernel kill. A test path built to find a
        #: memory ceiling that does not report memory has to be run twice to learn anything, and
        #: each run is fifty minutes of A40. None where it cannot be measured.
        self.encoder_peak_rss_gb = None
        self._proc = None
        self._identity = dict(identity or {})
        self._audio_source = audio_source
        self._audio_codec = audio_codec
        #: Seconds of audio to read, or None to read it all. Bounds the carried track to the
        #: picture without the muxer being allowed to bound the picture to the track.
        self._audio_limit_s = audio_limit_s
        #: Frames the container reports once ffmpeg has exited, or None where it does not say.
        #: The only frame count this class holds that was measured after the encode.
        self.verified_frames = None
        self._crf = crf
        self._preset = preset
        #: **An override, and the production path never passes it** (contract §8c). Route C is a
        #: test path and says so at the call site rather than in a threshold: a resolution gate —
        #: "frugal above 12 megapixels" — would leave the upscale path's bytes depending on a
        #: number somebody has to keep right, and `codec_default_unmoved` would be protecting a
        #: boundary rather than a behaviour. An argument nobody passes cannot move anything.
        self._x264_params = x264_params

    def set_frame_size(self, width, height):
        """Adopt the size the model actually produced, before ffmpeg is started.

        **This is why ffmpeg is started lazily.** `-s` on a rawvideo input is a promise about
        bytes that carry no shape of their own: declare 8210×4320 and feed 8208×4320, and ffmpeg
        does not complain — it reads across frame boundaries and writes a master that shears
        progressively, exiting 0 with a plausible file size. The still path caught the same
        disagreement as a byte-count refusal; this path would not have caught it at all.
        """
        if self._proc is not None:
            raise WorkerError(INTERNAL, "the master's frame size was changed after ffmpeg started")
        self.width, self.height = int(width), int(height)
        self._identity["cf_output"] = "{}x{}".format(self.width, self.height)

    def _build_command(self):
        width, height, fps = self.width, self.height, self.fps
        identity = self._identity
        audio_source, audio_codec = self._audio_source, self._audio_codec
        crf, preset = self._crf, self._preset
        path = self.path

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # Input 0: raw frames on stdin, exactly as the model produces them.
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(width, height), "-r", str(fps), "-i", "-",
        ]

        carry_audio = audio_source is not None
        if carry_audio:
            # **The trim goes on the audio input, never on the output.** `-t` before `-i` bounds
            # how much of *that* input is read and can only ever shorten the audio.
            if self._audio_limit_s:
                command += ["-t", "{:.6f}".format(float(self._audio_limit_s))]
            command += ["-i", audio_source]

        command += ["-map", "0:v:0", "-c:v", "libx264",
                    "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if self._x264_params:
            # `-x264-params` rather than more `-preset` flags: the preset stays the caller's and
            # these are the specific knobs §8c names, so a reader can see which of the two moved.
            command += ["-x264-params", self._x264_params]

        if carry_audio:
            # `?` makes the mapping optional, so a source whose audio stream vanished between the
            # probe and the mux does not fail the encode of an expensive master.
            command += ["-map", "1:a:0?"]
            command += ["-c:a", "copy"] if audio_codec in MP4_NATIVE_AUDIO else \
                       ["-c:a", "aac", "-b:a", "192k"]
            # **No `-shortest` here, and the reason is a delivered defect.** The flag was added
            # to stop a longer source track leaving an audio-only tail past the last frame, under
            # the comment "the video stream is authoritative". It is symmetric and does the
            # opposite: it ends the output when *any* input ends, so an audio track shorter than
            # the video truncates the video. On 2026-08-15 a 1.984 s AAC track against 2.000 s of
            # picture cost two frames of a delivered master; reproduced locally at 45 of 48 frames
            # with the flag and 48 of 48 without, from the same source and the same mux.
            #
            # AAC frames are 1024 samples, so a track almost never lands exactly on the video's
            # duration and is usually a fraction short. Every audio job was exposed. It had never
            # shown because every fixture that had been run at size was silent.
            #
            # The tail it was defending against is handled by `_audio_limit_s` above, which cannot
            # touch the picture. Where no limit is known the tail is accepted: audio playing past
            # the last frame is a cosmetic fault, and a master missing frames is not.

        command += _identity_tags(identity)
        # Two flags, and the second is not optional despite looking like a detail.
        #
        # `+faststart` puts the moov atom at the front, in this pass. Never a later one.
        #
        # `+use_metadata_tags` is what makes the identity tags above actually exist. **The MP4
        # muxer silently discards any metadata key it does not recognise** — `comment` and
        # `title` survive, `cf_request_id` does not — with a zero exit code and no warning.
        # Measured 2026-08-12 (`docs/decisions.md` 3.3). Without it the whole "a file found in
        # R2 with no job and no manifest still says what it is" mechanism is absent from every
        # file while every check around it passes.
        command += ["-movflags", "+faststart+use_metadata_tags", path]

        return command

    def __enter__(self):
        return self

    def _start(self):
        self.command = self._build_command()
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise WorkerError(INTERNAL, "could not start ffmpeg: {}".format(exc))

    def write(self, frame_bytes):
        """One frame, already `rgb24` and `width × height × 3` bytes."""
        # **The check the still path had and this one did not.** rawvideo carries no shape, so a
        # frame of the wrong length is not an error to ffmpeg — it is the first bytes of the next
        # frame, and the master shears from that point on while the process exits 0. Cheap to
        # check, and it turns the worst failure mode in this file into a refusal.
        expected = self.width * self.height * 3
        if len(frame_bytes) != expected:
            raise WorkerError(INTERNAL, "frame {} is {} bytes, expected {} for {}x{}".format(
                self.frames_written, len(frame_bytes), expected, self.width, self.height))
        if self._proc is None and not self.frames_written:
            self._start()
        if self._proc is None or self._proc.poll() is not None:
            raise WorkerError(INTERNAL, self._died("ffmpeg exited before the frames did"))
        try:
            self._proc.stdin.write(frame_bytes)
        except BrokenPipeError:
            raise WorkerError(INTERNAL, self._died("ffmpeg closed the pipe"))
        self.frames_written += 1
        # **Sampled here because this is where the loop already blocks.** `write` returns when
        # the pipe accepts the frame, which is exactly when the encoder is working — so the
        # samples land across the whole encode without a thread, and the maximum survives the
        # process that produced it. One `/proc` read per frame is noise against a 50 MiB write.
        peak = _peak_rss_gb(self._proc.pid)
        if peak is not None and peak > (self.encoder_peak_rss_gb or 0.0):
            self.encoder_peak_rss_gb = peak

    def _died(self, why):
        stderr = b""
        if self._proc is not None:
            try:
                stderr = self._proc.stderr.read() or b""
            except Exception:  # noqa: BLE001 — we are already reporting a failure
                pass
        detail = stderr.decode(errors="replace")[-400:].strip()
        return "{}{}".format(why, ": " + detail if detail else "")

    def __exit__(self, exc_type, exc, traceback):
        if self._proc is None:
            return False
        try:
            self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        self._proc.wait()
        # An exception on the way in owns the failure; do not replace it with one about ffmpeg,
        # which most likely died *because* of it. The original diagnosis is the useful one —
        # especially for an OOM, where the exception carries the phase and the allocation that
        # failed and no log gives better.
        if exc_type is not None:
            return False
        if self._proc.returncode != 0:
            raise WorkerError(INTERNAL, self._died(
                "ffmpeg exited {}".format(self._proc.returncode)))
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            raise WorkerError(INTERNAL, "ffmpeg wrote no output to {}".format(self.path))

        # **The first count this worker takes on the far side of the encode**, and the reason it
        # exists is that every other one is on the near side. `decoded_in` and `written_out` are
        # both counters in this process, so `frames_match` compares the write loop to itself and
        # passes on a master the muxer silently truncated. It did exactly that on 2026-08-15.
        #
        # Refusing is the right end for it. A short master is not a degraded success — it is a
        # video that plays correctly and is missing frames, which is the one failure a caller
        # cannot detect downstream either. `internal` is honest: the request was fine and this
        # worker wrote the wrong file.
        self.verified_frames = probe.written_frame_count(self.path)
        if self.verified_frames is not None and self.verified_frames != self.frames_written:
            raise WorkerError(INTERNAL, (
                "the master was written with {} frames but the file holds {} — the encode lost "
                "{} frame(s) after the write loop. The file plays; it is short.").format(
                    self.frames_written, self.verified_frames,
                    self.frames_written - self.verified_frames))
        return False


def _run(command, what):
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise WorkerError(INTERNAL, "could not start ffmpeg while {}: {}".format(what, exc))
    if completed.returncode != 0:
        raise WorkerError(INTERNAL, "ffmpeg failed while {}: {}".format(
            what, completed.stderr.decode(errors="replace")[-400:].strip()))
    return completed
