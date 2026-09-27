# cf-rife-image

A RunPod serverless GPU worker that **does what RIFE can do, one operation per job**: `retime`
changes a clip's frame rate by synthesising the frames between the ones it was given, and
`frame_repair` replaces damaged frames in place — by RIFE between the good frames either side, or
by frames from a clip the caller supplies. No upscaler. Bytes move
through S3-compatible object storage in both directions: a job names a source URL and an output
destination, and no image data travels in the job envelope.

Interpolation is RIFE (Practical-RIFE v4.26), pinned by commit and by the sha256 of the archive
it ships in, baked into the image at build time.

## What is here

```
handler/                    the worker, and the Docker build context
handler/Dockerfile          the whole build
.github/workflows/          the build-and-publish workflow
```

**This repository was seeded from `cf-upscale-image@rife-seed` and had the SeedVR2 upscaler
excised from it.** Twelve modules, the vendored third-party tree, the rung ladder, the estimator
and the derive-and-manifest half are gone; `handler/` is 20 modules and carries no reference to
`inference_cli`, `SEEDVR2_DIR` or `src.core.*` anywhere — `.py`, `Dockerfile` or otherwise. The
plan that governed the removal and the evidence for each disposition are in `cf-rife-project`,
which is private.

**Every request names its operation in `op`, and there is no default.** Each operation owns its
own `params`; a name belonging to the other one, a name belonging to the departed upscale path,
and any name the contract does not define are refused by name with `field_not_supported`. A field
that validates and then does nothing reads as supported to every client, which is why they are
refused rather than ignored.

**A repair re-encodes only the GOPs it touches and copies every other packet bit for bit.** On an
h264 MP4 or MOV source in 4:2:0 at 8 or 10 bits, each item's span runs from the clean IDR at or
before its first replaced frame to the next one after its last; those spans are re-encoded and
everything outside them is the source's own packets, joined through MPEG-TS into an `avc1` MP4
whose `avcC` carries the source's parameter sets and the spans'. The spliced file is checked
before upload — frame count, rate, every PTS and every DTS equal to the source's, and a clean
decode across every seam — and anything that is not eligible, or fails that check, is re-encoded
in full instead and says why. `params.reencode: "full"` forces the full re-encode, and a repair
with no `params.output` keeps the source's format. The splice is `handler/splice.py`.

**Either operation may ask for `derive`** — a WebP poster, a 1280-px h264 proxy and a spritesheet,
each made from the delivered master and uploaded beside it. A derive that fails is reported in
the run record's `warnings[]` and does not cost the master. The derives are made while the master
uploads — the proxy and the spritesheet from one decode, the poster by seeking to its exact frame,
at the worker's own thread bound — and upload after it; `timings.derive_exposed_s` is the part of
their time the master's upload did not hide.

## Building

The workflow has **no `push` trigger, deliberately** — not every commit is meant to become an
image. Builds are dispatched on purpose, from the Actions tab or with
`gh workflow run docker-publish.yml -R <owner>/cf-rife-image --ref main`, and a pull request never
publishes. **It takes no inputs**: there is one image, so there is nothing to choose.

`publish` is gated on `toolchain-gate` (`needs: toolchain-gate`), which installs the pinned ffmpeg
and asserts that `libwebp`, `libx264` and `use_metadata_tags` are present before a build starts.
A bad pin costs seconds that way rather than a full build — and the assertion is only meaningful
because the gate installs the exact binary the image will carry, not a distribution's.

A dispatched build publishes two tags to GHCR:

```
ghcr.io/<owner>/cf-rife-image:latest
ghcr.io/<owner>/cf-rife-image:sha-<commit>
```

**Endpoints pin the `sha-` tag, never `latest`.** `latest` moves, and a worker that pulled it
cannot say afterwards which build it ran. The image stamps its own `BUILD_COMMIT`, `IMAGE_REF`
and `BUILD_UTC` at build time and reports them in every envelope and every run record, so a
measurement can always be traced to the bytes that produced it.

The weights are ~23 MB rather than the upscaler's ~16 GiB, so a build is minutes rather than
tens of minutes and the image is a fraction of the size it was.

To build the same thing locally:

```
docker build handler/
```

## What every run reports

**The worker's job is to say what it did, not only to do it.** A run that does not fit is the
reading a predictor most needs, and a run is not repeatable — an 8K job costs twenty minutes of
an A40 whether or not anyone remembered to bank its padded area. So every envelope carries:

- `op`, and `retime` or `frame_repair` beside it — for a retime `n_out`, `n_synth`, `n_copy`,
  `n_hold`, `real_share`, `variant`, `scale`, `snap_tolerance`; for a repair each item's `id`,
  `type`, `a`, `b` and `n`, which `path` it took (`copy` or `full`) and why (`path_reason`), the
  re-encoded `spans[]` on a copy, and `frames_copied` and `frames_encoded`; for both
  `peak_vram_gb`, `encoder_peak_rss_gb` and all five encode settings (`crf`, `preset`,
  `x264_params`)
- `source` and `output` — each file's ffprobe, the rate as its exact rational and the frames
  counted from its packets, so a repair's claim to have changed nothing but frames is checkable
- `derived[]` — one entry per derive delivered, when any was asked for
- `source.padded_megapixels` — the padded area, **computed by `interp_plan`, which owns the
  padding rule**, rather than restated. Raw dimensions and padded area differ by
  `max(128, 128/scale)` per dimension, and a corpus banked on one against a predicate written
  for the other agree on nothing in particular
- `hardware` — the card, the VRAM, the host RAM **limit** rather than the machine's, and three
  separate CPU numbers: `usable_cores`, `affinity_cores`, `cpu_quota`. A container throttled by
  `cpu.max` and one pinned by an affinity mask are different machines that a single number
  reports identically
- `build` — nine identity keys, including the RIFE revision and the archive's sha256

Progress is **frame-level**: frames written against the planned count. Decode, interpolation and
encode are one streaming loop — the writer pulls each frame through the whole chain — so
*"decode complete"* is never true and the only quantity true per frame is the frame count. A
repair that copies counts a copied run as done the moment it is cut, and each re-encoded frame as
it reaches its span's encoder.

## Tests

**The contract suite is not in this repository, and that is deliberate rather than missing.** It
lives in `cf-rife-project` with the harness it exercises, and it is run before a build is
dispatched rather than by this workflow. The oracle there states the frame plan's arithmetic
independently and imports nothing from this worker; agreement between them is evidence rather
than a tautology.

So what CI enforces is narrower than a green suite, and worth stating plainly: **`toolchain-gate`
checks the toolchain, not the worker.** A dispatched build proves the image assembles, that its
ffmpeg has the capabilities the encode path needs, and that `import handler` succeeds with torch
still lazy. Whether the worker behaves is established before the dispatch, not by it.
