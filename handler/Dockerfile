# CF frame-interpolation worker.
#
# **Ordered by how often each step changes**, which is what makes the build cache hit: base image,
# system packages, pinned third-party source, Python dependencies, the dependency import check,
# **weights**, then this repo's code last. A handler edit rebuilds one small layer.
#
# **Any step inserted below the bake invalidates it and costs the whole ~17 GB download.** A
# `pip install` added in the wrong place is the mistake to watch for.
#
# Weights are baked rather than mounted because **RunPod network volumes exist in only a few
# datacentres, which often do not overlap with GPU availability** — baking is what lets the
# endpoint run wherever there is a free card. That is the reason; build speed is not.

# Python 3.11, stated in the tag rather than deduced from an Ubuntu release (docs/decisions.md
# 2.1). torch 2.8.0 rather than the image worker's 2.4.0.
#
# **This used to claim CUDA 12.8.1 "widens the card pool this worker's headroom comes from". It
# does the opposite**, and it was never measured — the base image's `cuda>=12.8` requirement is a
# *floor on the host driver*, so a newer CUDA admits fewer machines, not more. Measured when most
# of the fleet refused to start the container while the sibling worker's `cuda12.4.1` build ran on
# the same cards. The `NVIDIA_REQUIRE_CUDA` override below is what recovers the pool.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# A pinned static ffmpeg, not the distribution's. Ubuntu 22.04 ships 4.4, this repo is developed
# against 9.x, and the gap is not cosmetic: `-vsync` was removed in 9.x while `-fps_mode` arrived
# in 5.0, so a command written for one is invalid on the other (docs/decisions.md 3.3). Pinning
# one build for the container makes the encode path a known quantity instead of a distribution
# detail. It also guarantees libwebp, which every poster and every crop needs.
#
# **The release tag is a MONTH-END snapshot and must stay one.** BtbN keeps fourteen rolling
# dailies and retains month-end builds for years; the previous pin was a daily and started
# 404ing the day it aged out, which fails this RUN two minutes into a twenty-one minute build.
# A newer-looking daily tag is a fourteen-day fuse and, since the dailies have moved past this
# ffmpeg commit, also a toolchain change. Moving the tag while holding the asset name keeps the
# ffmpeg commit, the libavcodec version and the x265 build; it does not keep the bytes, which
# differ between releases. What holds the build steady is the capability assertion below.
ARG FFMPEG_RELEASE=autobuild-2026-07-31-14-10
ARG FFMPEG_ASSET=ffmpeg-n8.1.2-34-g9b6c8969e0-linux64-gpl-8.1

# **The base image's `cuda>=12.8` label is what kept this off most of RunPod's fleet.**
#
# `nvidia-container-cli` checks `NVIDIA_REQUIRE_CUDA` against the *host driver* before `runc`
# creates the process, so a host on 12.4 refuses the container with
# `unsatisfied condition: cuda>=12.8` and nothing of ours ever runs. Measured against the sibling
# worker on the same account and the same A40s: its `cuda12.4.1` base asks for `cuda>=12.4` and
# runs; this one asked for 12.8 and was refused. **The comment below claiming CUDA 12.8.1 widens
# the card pool is exactly backwards, and is corrected there.**
#
# 12.0 rather than removing the check: CUDA guarantees *minor version compatibility*, so an
# application built against 12.8 runs on any 12.x driver — that is the guarantee PyTorch itself
# relies on. Lowering the bound admits precisely 12.0–12.7, which is the range that guarantee
# covers. **Not `NVIDIA_DISABLE_REQUIRE=1`**, the usual suggestion: that removes the check
# altogether and would let an 11.x host start the container, turning a clean refusal into a
# confusing failure after the worker has been billed.
#
# The residual risk is that a host which still cannot run this now fails at model load rather than
# at container start. `hardware.read()` records the driver version on every job so that failure is
# one lookup rather than a hunt.
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0"

# **Measured: 26.40 GiB of a 96 GB card held by the allocator and unusable.** A 2K-to-4K clip at
# the `resident` rung OOMed with 62.34 GiB actually allocated, 26.40 GiB reserved-but-unallocated,
# and 5.58 GiB free — failing on a single 19.78 GiB request inside the VAE decode. More than a
# quarter of the card was fragmentation, and PyTorch names this setting in the error text itself.
#
# `expandable_segments` lets the allocator grow a segment rather than pick from fixed-size blocks,
# which is exactly the shape of this workload: a few enormous, short-lived tensors of varying size
# per chunk. It is the largest single memory lever found so far, and it costs nothing to set.
#
# **Set here rather than in the handler** because the CUDA caching allocator reads it once, when
# it initialises — which happens on the first CUDA call, well before any of this repo's code could
# reliably set it. An environment variable in the image cannot be too late.
#
# Untested as of 2026-08-13: measured only that the fragmentation exists, not that this fixes it.
# The run that answers it is `fhd_source.mp4 --target 2160 --rung resident`, which OOMed here.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ENV RIFE_MODEL_DIR=/app/models/RIFE \
    PYTHONUNBUFFERED=1

# ── system packages ─────────────────────────────────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 12 --retry-delay 15 --retry-all-errors --retry-max-time 600 -o /tmp/ffmpeg.tar.xz \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_RELEASE}/${FFMPEG_ASSET}.tar.xz" \
 && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
 && install -m755 "/tmp/${FFMPEG_ASSET}/bin/ffmpeg" /usr/local/bin/ffmpeg \
 && install -m755 "/tmp/${FFMPEG_ASSET}/bin/ffprobe" /usr/local/bin/ffprobe \
 && rm -rf /tmp/ffmpeg.tar.xz "/tmp/${FFMPEG_ASSET}"

# **Assert the toolchain rather than commenting it.** Each of these is a fact this repo's code
# depends on, and each has a silent failure mode: a Python that is not 3.11 means the venv these
# dependencies were resolved against is not the one they run on; a missing libwebp means every
# poster and crop fails on the first job rather than at build time; and `use_metadata_tags` is
# what makes the identity tags exist at all — without it ffmpeg succeeds, the file plays, and the
# tags are silently absent (docs/decisions.md 3.3).
#
# **`libvmaf` IS ASSERTED HERE RATHER THAN INSTALLED ANYWHERE, AND THAT IS THE POINT**
# (contract §6g). The pinned build already carries it — its embedded configuration line reads
# `--enable-libvmaf` beside `--enable-libx264` and `--enable-libx265` — so this file needs no new
# build-time dependency for CAMBI. **What it needs is the assertion**, because the same is true of
# libx264 and nothing in this file names that either: a capability nobody checks is a capability
# that disappears when the pin moves.
#
# **AND IT RUNS THE INVOCATION RATHER THAN GREPPING FOR A NAME, WHICH IS THE WHOLE DIFFERENCE**
# (`docs/gate-findings.md` F-2026-08-28-15). A name grep catches a pin that moved to a build
# without the capability. **It cannot catch a build where the capability is present and the
# invocation is wrong** — and that is `F-2026-08-28-7` delivered: ffmpeg accepted `-x264-params`
# under `libx265`, exited 0, and wrote a master with no bound.
#
# **TWO UNPROVEN FACTS DIE HERE FOR THE PRICE OF ONE.** Neither the gate nor the builder could
# establish, by any method either had, that this build's `libvmaf` accepts `feature=name=...` —
# `strings` finds `log_path` and `log_fmt` in the binary and does NOT find `feature`, which is
# ambiguous rather than damning, and string-probing a 138 MB binary is the method that produced
# `F-2026-08-28-7` in the first place. **The second fact is the metric KEY NAMES**: `psnr_y`,
# `float_ssim` and `cambi` are the same class of assumption, and `reference.parse_log` REFUSES
# rather than defaulting when one is missing — because a metric scored as 0.0 reads as excellent
# CAMBI and as a broken PSNR from one silence. *Both are settled in two seconds, at build time.*
#
# **AND IT ASSERTS EVERY PRECONDITION `reference.parse_log` REFUSES ON, not just the key names.**
# A build can satisfy a key check and still be refused by the parser on every delivered job —
# `parse_log` also requires the frame COUNT to match, `frameNum` to be zero-based and contiguous,
# and each value to convert to `float`. *A `{"cambi": null}` satisfies `k in m` and raises a
# `TypeError` — not even the coded refusal — inside a real run.* **Each of those is asserted here,
# on both frames**, so the build fails in two minutes rather than the instrument failing silently
# on every job for ever. Found in review of this wave.
#
# **CHECKED WHERE THE PARSER READS IT, NOT ANYWHERE IN THE FILE.** A `grep` for `"cambi"` proves
# the string appears SOMEWHERE in the log; `reference.parse_log` requires the key inside
# `frames[i]["metrics"]`. A build whose libvmaf reported these only under `pooled_metrics` would
# pass a grep and refuse every delivered job — **the assertion exercised from the direction the
# code takes rather than the direction the failure comes from**, which is the class this whole
# block is being hardened against. *Found in review of this wave.*
#
# **The two sources DIFFER on purpose.** Identical inputs give infinite PSNR, and `inf` is not
# valid JSON — an assertion that wrote a log nothing could parse would fail for the wrong reason.
#
# **AND THE SIZE IS A REAL FRAME SIZE, WHICH IS THE CORRECTION.** This first read `128x128` on the
# reasoning that CAMBI's window is 63 pixels wide, and **the build failed on it** — libvmaf exits
# `-22` inside `vmaf_read_pictures` with *"problem reading pictures"*. Measured against a local
# libvmaf 3.2.0 rather than guessed at:
#
#     208x208  FAIL      212x212  FAIL      216x216  OK      1920x1080  OK
#     640x120  OK        120x640  OK        216x208  OK      208x216    OK
#
# **So it is not a per-side minimum and not an area threshold** — 216x208 passes at 44,928 pixels
# while 212x212 fails at 44,944 — it is the LONGER side needing roughly 216. *A rule that shape is
# one nobody should be deriving from a comment, so the assertion uses a frame size the worker
# actually delivers and stays nowhere near it.* Two frames of 1080p cost 0.13 s.
#
# **THE FAILURE WAS THIS SIZE AND NOT THE SPELLING.** `feature=name=cambi|name=psnr|name=float_ssim`
# PARSED — no "Option not found" — so the option surface neither session could establish by probing
# a binary is now established by a build that ran it. *The assertion caught a defect in itself,
# which is the only kind of check worth having.*
#
# **`libx265` IS ASSERTED BESIDE IT ON A RULED EXCEPTION, AND THE EXCEPTION IS RECORDED RATHER
# THAN ABSORBED.** The gap is real — §6e made x265 a shipping path and nothing anywhere asserted
# the encoder exists, so a pin moving to a build without it would fail every h265 job at RUN time
# instead of here. **It is not part of §6f, §6g, §16 or §17**, and it was reverted once as scope
# growth before the gate ordered it explicitly: what made it payable is that this wave already
# opens this block for `libvmaf`, so deferring it would mean touching the block twice and paying a
# second build for one line. *A ruled exception, not a precedent.*
#
# **Written to files rather than piped into `grep -q`.** `grep -q` exits the instant it matches,
# which closes the pipe under ffmpeg and kills it with SIGPIPE — exit 141, which `-o pipefail`
# then propagates and `set -e` turns into a failed build. It is a race against ffmpeg finishing
# its write, so it passes on a fast machine and fails on a slow one; it passed in CI and failed
# here on the same commit. Grepping a file cannot race.
RUN set -eux; \
    python3 -c "import sys; assert sys.version_info[:2] == (3, 11), 'expected Python 3.11, got %s' % sys.version"; \
    ffmpeg -hide_banner -encoders > /tmp/encoders.txt; \
    ffmpeg -hide_banner -h muxer=mp4 > /tmp/mp4muxer.txt; \
    grep -q libwebp /tmp/encoders.txt; \
    grep -q libx264 /tmp/encoders.txt; \
    grep -q libx265 /tmp/encoders.txt; \
    grep -q use_metadata_tags /tmp/mp4muxer.txt; \
    ffmpeg -hide_banner -loglevel error \
      -f lavfi -i "testsrc2=size=1920x1080:rate=1:duration=2,format=yuv420p" \
      -f lavfi -i "testsrc=size=1920x1080:rate=1:duration=2,format=yuv420p" \
      -lavfi "[0:v][1:v]libvmaf=feature=name=cambi|name=psnr|name=float_ssim:log_path=/tmp/vmaf.json:log_fmt=json" \
      -f null -; \
    python3 -c "import json,sys; K=('cambi','psnr_y','float_ssim'); f=json.load(open('/tmp/vmaf.json'))['frames']; assert len(f)==2, 'scored %d frames, expected 2' % len(f); [ (lambda m: [ (k in m) or sys.exit('frames[%d].metrics lacks %s; it reports %s' % (i,k,sorted(m))) for k in K ] + [ float(m[k]) for k in K ])(e['metrics']) for i,e in enumerate(f) ]; [ e.get('frameNum') in (None,i) or sys.exit('frameNum %r at position %d; the parser needs zero-based contiguous' % (e.get('frameNum'),i)) for i,e in enumerate(f) ]; print('libvmaf ok: %d frames, %s' % (len(f), sorted(f[0]['metrics'])))"; \
    ffmpeg -version > /tmp/ffver.txt; \
    echo "toolchain: $(python3 -V), $(head -1 /tmp/ffver.txt)"; \
    rm -f /tmp/encoders.txt /tmp/mp4muxer.txt /tmp/ffver.txt /tmp/vmaf.json

# ── Python dependencies ─────────────────────────────────────────────────────────────────────
# **Nothing here pins `torch` or `torchvision`, and that is load-bearing rather than an omission.**
# The base image ships the CUDA build; a version constraint added to `requirements.txt` would have
# pip replace it with a PyPI default. The vendored requirements used to be installed alongside
# ours and listed both unpinned for the same reason — that file is gone, and this one must not
# acquire the constraint it was careful not to have.
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── weights ─────────────────────────────────────────────────────────────────────────────────
# Their own layer, so editing the handler below does not re-download them. Nothing may be
# inserted between here and the code copy.
# `df` on both sides of the bake, in the same layer so it does not disturb the ordering above.
# This is the only place the real cost of the weights is visible: every earlier disk figure in
# docs/decisions.md 3.5 is about the base, and no build had reached this line when the size
# projections there were written.
COPY bake_weights.py RIFE-LICENSES.txt ./
RUN set -eux; \
    echo "--- disk before the bake ---"; df -h /; \
    mkdir -p "$RIFE_MODEL_DIR"; \
    python3 bake_weights.py; \
    install -m644 RIFE-LICENSES.txt "$RIFE_MODEL_DIR/LICENSE.txt"; \
    echo "--- disk after the bake ---"; df -h /; \
    du -sh "$RIFE_MODEL_DIR"

# ── build identity ──────────────────────────────────────────────────────────────────────────
# **An image that cannot say what it is makes every measurement taken on it an assertion.** The
# 2026-08-17 calibration campaign banked ten runs' worth of VRAM peaks and recorded
# `"image": null` on all ten, because the handler read `IMAGE_TAG` and nothing ever set it; the
# claim that those ten share one build rests on nobody having re-created the endpoint. The
# registry key that prices this worker is keyed on exactly that.
#
# **Below the bake, deliberately.** These change on every single build, and a layer above the
# weights would re-download 17 GB to record a commit hash. Here they cost one small layer.
#
# The digest is not among them and cannot be: it is computed when the image is pushed, after the
# last layer is sealed. `IMAGE_REF` is the next best thing and is enough — CI pushes an immutable
# `:sha-<commit>` tag alongside `:latest`, so the reference resolves to one digest forever.
ARG IMAGE_REF=""
ARG BUILD_COMMIT=""
ARG BUILD_UTC=""
# **`WORKER_VERSION` was never set, so every build block said `0.1.0-dev`**
# (F-2026-08-18-10) — the same class of gap as the missing `IMAGE_TAG` that release item 5
# closed, and found the same way: by reading a bundle back and noticing a constant where a
# reading belonged. It is a hand-set version rather than a derived one because that is what it
# is for: `BUILD_COMMIT` already answers "which code", truthfully and without anyone having to
# remember to bump anything, while this answers "which release of it" and only a person can say.
ARG WORKER_VERSION="0.2.0"
ENV IMAGE_TAG=${IMAGE_REF} \
    BUILD_COMMIT=${BUILD_COMMIT} \
    BUILD_UTC=${BUILD_UTC} \
    WORKER_VERSION=${WORKER_VERSION}

# **`decode_s` IS 106.8 s AT 8K — 18.70% of `compute_s`, larger than the entire conversion
# pipeline two waves were spent deleting** (`docs/archive/instrumentation-archive.md` §11). And the pointer that
# says this is worth trying: **the worker decodes at 0.556 s/frame where a laptop's
# SINGLE-THREADED ffmpeg does the same file at 0.285** — so the capture backend is very likely
# not threading at all.
#
# **This is the whole of item 2 and it is deliberately zero code.** OpenCV's FFmpeg backend reads
# this variable and passes the options straight to libavcodec; `threads;8` asks for frame
# threading. **Expect roughly 3x, not linear** — h264 frame threading is bounded by the
# dependency structure of the stream, not by core count.
#
# **A DOCKERFILE `ENV` AND NOT A REQUEST FIELD, and that is consistent with CF's rule rather than
# an exception to it.** The rule is that a per-JOB control belongs in the request; this is a
# property of how the image is built and would be identical on every job. It also cannot be a
# request field even in principle: OpenCV reads it once, when the capture backend initialises.
#
# **NOTHING HERE PROVES IT WORKED.** `params.decode_probe` (§11) is what says whether the
# backend honoured it and what is left — and `decode_s` being unchanged after this lands is
# itself the answer, not a null result.
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;8"

# ── this repo's code, last ──────────────────────────────────────────────────────────────────
# **`*.py` is currently everything the handler needs, and that has not always been true.**
# `calibration.json` was named here too — measured VRAM data the estimator planned from — and it
# leaves with the estimator. The finding that put it on this line is worth keeping even though the
# file is gone: when it was added to the repo but NOT to this COPY, the image shipped without it
# in silence. Nothing failed, because an absent table read as "take the floor" — a correct state,
# indistinguishable from having no measurements. The only symptom was a rung nobody expected.
#
# **So a non-`.py` file that the handler reads must be named here explicitly**, and the next one
# added is the next chance to repeat that. `COPY *` is not the answer: a wildcard sweeps in
# whatever lands in this directory later.
COPY *.py ./

# The handler must import with the heavy chain still lazy — if this ever pulls in torch, the
# rung-1 suite stops running in CI and nobody finds out until a GPU-less runner fails.
#
# **`decodeprobe` is named here because nothing else imports it at module scope.** It is pulled
# in lazily, inside the one function that runs it, so a syntax error or a bad module-scope import
# in that file reaches a GPU host and not this line. Naming it costs nothing and it does not
# weaken the assertion beside it: `decodeprobe` imports `subprocess` and `time` and nothing else
# (`handler/decodeprobe.py:45-46`; its only other import is `os`, inside a function at :121), so
# the lazy-torch property this line exists to hold is unchanged by adding it.
#
# **AND AN IMPORT CHECK COVERS IMPORTS.** It catches a module that will not parse and a
# module-scope import that will not resolve. It does not enter `run`, `_pass` or any of the three
# ffmpeg argument lists, and it is not evidence about any of them.
RUN python3 -c "import handler, decodeprobe; import sys; assert 'torch' not in sys.modules, 'handler imported torch eagerly'; print('handler + decodeprobe import OK, torch still lazy')"

# **THE LINE ABOVE'S OWN SCOPE LIMIT, CLOSED BY ONE STDLIB PASS** (`docs/gate-findings.md`
# `F-2026-08-29-4`). An import proves a module parses and that its module-scope imports resolve;
# **it does not enter a single function.** `routec.py` shipped with five uses of `os` and no
# `import os` — the module imported cleanly, the line above passed, CI was green on three jobs,
# and the wave's most expensive instrument had never executed once. **It cost a 4K GPU dispatch to
# find, and this finds it in under a second.**
#
# **IT IS `F-2026-08-28-15` APPLIED TO THE HALF IT WAS NEVER APPLIED TO.** That finding changed the
# libvmaf assertion in this file from grepping a NAME to running the INVOCATION. The Python
# assertion still imported a MODULE and never ran a PATH; this does not run a path either, but it
# answers the one question a `NameError` is the answer to. *`name_check.py` states plainly what it
# does not do — no attribute resolution, no execution — so nobody reads it as more than it is.*
RUN python3 name_check.py

CMD ["python3", "-u", "handler.py"]
