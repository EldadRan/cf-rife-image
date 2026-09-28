# cf-rife-image

A RunPod serverless GPU worker that **does what RIFE can do, one operation per job**: `retime`
changes a clip's frame rate by synthesising the frames between the ones it was given, and
`frame_repair` replaces damaged frames in place — by RIFE between the good frames either side, or
by frames the caller supplies: a frame range from the source itself or from up to three other
videos, or a list of stills. No upscaler. Bytes move
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

**The full re-encode is an encode, not a frame loop.** The source is decoded once in its own YUV
and every untouched frame goes straight to the encoder without passing through RGB; only the
replaced frames go through RGB, with the source's colour matrix and range stated both ways, and a
codec or depth change is a conversion in YUV in the encoder's filter graph. Decode, RIFE and the
encode run at once. The master keeps the encoder's own B-frames and is checked before upload for
frame count, rate, duration and every PTS equal to the source's. A source this cannot plan for,
an armed `convert_check`, `input_check` or `reference_score`, or a native encode that fails its
check runs the per-frame loop instead, filed as `frame_repair.pipeline` and `pipeline_reason`
with a warning. This is `repair.run_native`.

**A segment names its frames by index.** `params.sources` lists up to three videos beyond the
source (source 0 is `source_url`) and `params.stills` any number of PNG, WebP or JPEG images; a
segment takes `source` with an inclusive `start_frame`/`end_frame`, or `stills` — indices in
output order, repeats allowed — and is fitted to the frames it replaces as before (`exact` or
`resample`). Every listed file is fetched once and probed once however many segments use it, and
a video segment is decoded from the keyframe at or before its `start_frame`, never from frame 0.
A still is converted with the source's matrix and range, as RIFE's output is; a still with any
transparent pixel is refused. The disk is checked twice, each with 10% to spare:
before any file's body is fetched, for every listed file's size (from a ranged probe) plus the
source's again as the master's floor; and after the source's probe, for the files still to come
plus the master at its path's estimate — the larger of the source's size and 0.05 MB per
megapixel-frame on the full path, and on the copy path the larger of that and twice the source's
size, since a copy writes its parts before joining them and can fall back to the full path. The
same size estimate prices the master's upload inside the ETA's derive term. A video whose rate or size differs from the source's, or a still of
another size, is refused `sources_mismatch`, and a range past a video's end
`segment_exceeds_source`; neither is retryable. This is `handler/repair_files.py`.

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
  counted from its packets, so a repair's claim to have changed nothing but frames is checkable;
  `source_probes[]`, one per video (source 0 first), and `stills[]`, one per still. A segment item
  names its `source` or `stills`, its `start_frame` and `end_frame`, the keyframe its decode began
  at (`decode_start_frame`) and the frames its decoder produced (`frames_decoded`)
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

Every upload — the master, the derives, the reference PNGs — goes up in parallel parts, 16 in
flight, each file's part taken from its own size: a thirty-second of the file, rounded up to a
whole MiB and kept within 8-64 MiB, and multipart from 16 MiB (`handler/envelope.py`). The caller
has no say. `transfer.upload_part_bytes` is the master's. The upload phase is priced at 200 MB/s — the
parallel rate — so the poll cadence asks again when the upload should be done. A part that fails fails the upload and
the multipart upload is aborted; an abort that fails too is said in the error.

The source, and every other video, comes down in parallel ranges: a `Range: bytes=0-0` probe must
answer 206 with a total size and a strong ETag, and then 8 streams of 32 MiB write each slice in
place into a pre-sized file. Stills come down as single streams, up to 8 at a time. Every range carries `If-Match`
with the probe's ETag, so a source replaced mid-fetch fails rather than being stitched from two
versions; a slice is tried three times before the job fails `source_fetch_failed`, and the
finished size must equal the probe's total. A server that refuses the probe, or a source under
64 MB, is fetched as one stream, and the record says why (`transfer.fetch_mode_reason`);
`transfer.files_fetched` counts the downloads. The debug field `fetch_sha256` hashes the fetched
source into `transfer.fetch_sha256`.

The ETA exists from the moment the frame plan does, before the model loads, and it includes the
part of any requested derives that the master's upload does not hide. `next_poll_s` never asks for
longer than that ETA, and never less than 5 seconds. A repair's h264 encode takes the most x264
threads of 64, 32 and 16 whose predicted host memory — 0.0222 GB per megapixel per thread — stays
under 27.94 GB, not sliced: 64 at 4K and below, 32 at 8K. Retime and h265 are unchanged, and the
request's debug fields still override. The rules live in `handler/ladder.py` (`derive_expected`)
and `handler/encoder.py` (`repair_threads`).

A repair's first ETA is its own, not retime's table: frames times a per-frame time taken from the
frame's pixels — linear between 1080p, 4K and 8K and flat beyond — plus the derives' exposed part,
never below 5 s. On the copy path the frames are the ones re-encoded, plus a term per source MB. It
is published once the path is known, with `eta_basis` `predicted_repair_v2` and, once the job is
priced, retime's flat band (`eta_low_s` / `eta_high_s`). The rates live in `handler/ladder.py`
(`repair_t`, `repair_work`). Retime keeps its table.

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
