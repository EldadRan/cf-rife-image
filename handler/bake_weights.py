"""Build-time: bake the RIFE weights into the image.

**Run as a script by the `Dockerfile`, imported as a module by `build_identity`.** The second is
why the fetches sit behind `if __name__ == "__main__"` at the bottom: the identity reads
`RIFE_REVISION`, `RIFE_ARCHIVE_SHA256` and `WEIGHTS_FILE` from here rather than restating them
(excision plan §7.1), and an unguarded call would have made `import handler` download 23 MB.

**The pins and the assertions that check them live in one file, deliberately.** `verify` compares
size and sha256 against the constants above it on every fetch, so a pin and its verification
cannot drift apart by being edited in different places. `WEIGHTS_FILE` keys straight into
`RIFE_MEMBERS` for the same reason — the member table asserts the hash of the file the identity
names.

**`flownet.pkl` is a pickle and `torch.load` on a pickle executes code.** The hash below makes it
the SAME pickle on every build; it does not make it an inert one. Recorded rather than mitigated
(CF, 2026-08-23), so nobody reads "hash-asserted" as "safe to load from anywhere".

**The SeedVR2 half of this file is gone** — its checkpoint, its pins, its file table and the
`BAKE_WEIGHTS` flag that gated it. `sha256_of`, `verify` and `_fetch` are NOT its: they were
always shared, and `_fetch` in particular sits between the two halves, so a block deletion of
"the SeedVR2 part" would have taken RIFE's own downloader with it.
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.request
import zipfile


RIFE_REPO = "hzwer/RIFE"
RIFE_REVISION = "01fdc7e97404120c243c3ea7b427046e5dc7643e"
RIFE_ARCHIVE = "RIFEv4.26_0921.zip"
RIFE_ARCHIVE_BYTES = 22869906
RIFE_ARCHIVE_SHA256 = "1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e"

#: The four files the pipeline needs, by their name inside the archive's single directory, with
#: the size and sha256 of each. Everything else in the zip — a `.DS_Store`, a `__pycache__` of
#: another Python's bytecode, and Finder's `__MACOSX` shadows — is dropped rather than shipped.
#: **The archive is NOT self-sufficient, which is the thing to know about it.** `RIFE_HDv3.py`
#: opens with `from model.warplayer import warp` and `from model.loss import *` — a package that
#: lives in the Practical-RIFE *repository* and not in the weights zip. The reference script hides
#: this by taking a whole checkout as `--rife-dir`; an image that baked only the archive would
#: import-error on the first interpolation, having verified four hashes on the way.
#:
#: Two files, both pure Python, both tiny, pinned by commit rather than by tag. Raw file bytes at
#: a commit sha are stable in a way a repository tarball is not, so the same assertion the weights
#: get applies here. `loss.py` needs torchvision, which the base image already carries — its
#: `VGGPerceptualLoss` would fetch VGG19 weights, but nothing constructs it: `RIFE_HDv3.Model`
#: has that line commented out upstream, so no second download hides in this import.
RIFE_SOURCE_COMMIT = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_SOURCE_URL = ("https://raw.githubusercontent.com/hzwer/Practical-RIFE/{}/model/{}")
RIFE_SOURCE_FILES = {
    "warplayer.py": (
        1058, "eed94da2f2e8056fa0ceabed88b87fedf25ec849494991a956b9f2cbad33632c"),
    "loss.py": (
        4641, "9e4679cd685a37add8d8bb4a963b9822df0e1d344b82d01f975fc3426c8fc77a"),
}

#: **The weights file itself, named once.** `build_identity` reports it as `model` and reads it
#: from here (§7.1: read, not restated) — it used to be `MODEL_BUILD`, an env read with a SeedVR2
#: filename as its literal default, so dropping that ENV would have pinned every envelope and
#: every muxed tag to a checkpoint this image does not contain rather than nulling them.
WEIGHTS_FILE = "flownet.pkl"

RIFE_MEMBERS = {
    WEIGHTS_FILE: (
        24636301, "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"),
    "RIFE_HDv3.py": (
        3101, "81bbd0648e499de79e44768d284005d9d57d0f6eb7c30adae407f22675055730"),
    "IFNet_HDv3.py": (
        6433, "655b4c772b037967b86c2dd31c8fa3b5323b79dd9a0e0088708d89149bbc8a32"),
    "refine.py": (
        3510, "0c5698b4a05b9f6ab551740575c1c35e248e5b1829bab6445186081ebe15f032"),
}


def sha256_of(path):
    """Streamed, because the DiT is 15.3 GiB and the runner has no room to hold it twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, label, want_size, want_sha):
    """Size then sha256, exiting on either. Size first because it is free and a truncated
    download is the common failure; the hash then says the bytes are the ones measured."""
    size = os.path.getsize(path)
    if size != want_size:
        sys.exit("{}: expected {} bytes, got {}. The pin resolved to different content than the "
                 "calibration measured.".format(label, want_size, size))
    got_sha = sha256_of(path)
    if got_sha != want_sha:
        sys.exit("{}: sha256 mismatch.\n  expected {}\n  got      {}\nThe pin resolved to "
                 "different content than the calibration measured.".format(
                     label, want_sha, got_sha))
    print("verified {} {} bytes sha256 {}".format(label, size, got_sha), flush=True)


def _fetch(url, destination, attempts=5, pause=5):
    """One small file, retried. **`hf_hub_download` retries internally and this did not** — a
    single blip on the plain `urlopen` failed the RUN, which is why the ffmpeg install a hundred
    lines up in the Dockerfile spends `--retry 12` on the same class of fetch. `timeout` is
    per-socket-operation rather than total, so it bounds a stall but not a trickle; the retry is
    what bounds the blip."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, \
                    open(destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except Exception as exc:  # noqa: BLE001 — any transport failure is worth one more try
            if attempt == attempts:
                sys.exit("could not fetch {} after {} attempts: {}".format(url, attempts, exc))
            print("fetch {} failed ({}); retrying in {}s".format(url, exc, pause), flush=True)
            time.sleep(pause)


def bake_rife():
    """Practical-RIFE's `train_log` — the model code and its weights, from one pinned archive.

    **Unconditional, and it always was.** `BAKE_WEIGHTS` gated the SeedVR2 checkpoint and never
    gated this one — the image without an upscaler is exactly the image that has to interpolate.
    With SeedVR2 gone there is nothing left for the flag to decide and it went with it.

    The archive is verified whole before anything is extracted, and each extracted file is
    verified again after. Two checks rather than one because they answer different questions: the
    first says the pin resolved to the bytes the calibration measured, the second says the
    extraction produced the files those bytes contain — a truncated write and a wrong download
    are different failures and only the first is visible upstream.
    """
    rife_dir = os.environ["RIFE_MODEL_DIR"]
    train_log = os.path.join(rife_dir, "train_log")
    os.makedirs(train_log, exist_ok=True)

    from huggingface_hub import hf_hub_download  # noqa: PLC0415 — build-time only

    archive = hf_hub_download(repo_id=RIFE_REPO, filename=RIFE_ARCHIVE, local_dir=rife_dir,
                              revision=RIFE_REVISION)
    verify(archive, RIFE_ARCHIVE, RIFE_ARCHIVE_BYTES, RIFE_ARCHIVE_SHA256)

    with zipfile.ZipFile(archive) as bundle:
        # **Named members, not `extractall`.** The zip was built on a Mac and carries a
        # `.DS_Store`, a `__pycache__` of another Python's bytecode, and a `__MACOSX` shadow of
        # every entry. `extractall` would ship all of it, and stale `.pyc` files beside their
        # sources are a way to run code nobody can see. It is also the answer to zip-slip: a
        # member is looked up by the name we asked for, so a crafted path cannot escape.
        # Keyed on the basename, which would collide last-wins if the archive ever held two
        # files sharing one. It cannot bite: the archive is pinned by hash, so its shape cannot
        # change, and **the `verify` after each copy is what makes that safe rather than this
        # lookup** — a wrong file under a right name fails its sha256 and exits. Said here
        # because an edit that moved the verification would remove the protection without
        # touching the line that looks load-bearing.
        members = {os.path.basename(name): name
                   for name in bundle.namelist() if not name.startswith("__MACOSX/")}
        for wanted, (want_size, want_sha) in sorted(RIFE_MEMBERS.items()):
            inside = members.get(wanted)
            if inside is None:
                sys.exit("{} is not in {} — the archive's shape changed under a pinned hash, "
                         "which should be impossible".format(wanted, RIFE_ARCHIVE))
            destination = os.path.join(train_log, wanted)
            with bundle.open(inside) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
            verify(destination, wanted, want_size, want_sha)
            print("baked {} -> {} ({} bytes)".format(
                wanted, destination, os.path.getsize(destination)), flush=True)

    os.remove(archive)

    # **Prove only the four arrived, rather than trusting that only four were asked for.** The
    # archive was built on a Mac and carries a `__MACOSX/` sidecar tree, a 6,148-byte `.DS_Store`
    # and two `.pyc` files compiled by another Python. `extractall` would have shipped all of it,
    # and **bytecode beside the source it claims to be is unreviewable** — reading the `.py` says
    # nothing about the `.pyc`, which is the same class of trust as the pickle. CPython would
    # very probably ignore them, since extraction rewrites the mtime a `.pyc` is validated
    # against; probably is not a reason to ship them. Members are taken by name above, and this
    # asserts the result.
    landed = set(os.listdir(train_log))
    if landed != set(RIFE_MEMBERS):
        sys.exit("train_log holds {} but should hold exactly {} — the extraction shipped "
                 "something nobody named".format(sorted(landed), sorted(RIFE_MEMBERS)))

    # **The `model` package the archive does not carry.** Fetched from the pinned commit and
    # verified the same way, into a sibling of `train_log` so the two names the vendored code
    # imports — `train_log.*` and `model.*` — are both reachable from one directory on the path.
    model_pkg = os.path.join(rife_dir, "model")
    os.makedirs(model_pkg, exist_ok=True)
    for name, (want_size, want_sha) in sorted(RIFE_SOURCE_FILES.items()):
        destination = os.path.join(model_pkg, name)
        url = RIFE_SOURCE_URL.format(RIFE_SOURCE_COMMIT, name)
        _fetch(url, destination)
        verify(destination, "model/" + name, want_size, want_sha)
        print("baked model/{} -> {}".format(name, destination), flush=True)

    print("baked RIFE into {} -> train_log {} · model {}".format(
        rife_dir, sorted(os.listdir(train_log)), sorted(os.listdir(model_pkg))), flush=True)


# **One bake, unconditional.** `BAKE_WEIGHTS` gated the SeedVR2 checkpoint and never gated this
# one — RIFE was baked on every variant of the image including the weightless one — so with
# SeedVR2 gone the flag has nothing left to decide and leaves with it. `build_identity` reports
# `weights_baked` as a constant `true` for the same reason (excision plan §7.1).
#
# **The guard is not the flag.** `if __name__ == "__main__"` is here so `build_identity` can read
# the pins above without this script fetching 23 MB at import time; the Dockerfile still runs it
# as a script, so `bake_rife()` is as unconditional as it has always been.
if __name__ == "__main__":
    bake_rife()
