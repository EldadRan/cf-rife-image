"""What the host looks like, said out loud once per container.

**All that is left of this module is the host reading.** It was built to name which of the
vendored coder's four phases was running and what each cost in VRAM, by wrapping the
vendored CLI's own debug logger and parsing its phase banners — and that whole layer left with the
vendored tree it was reading. `PhaseWatch`, the phase table, the batch and phase regexes and the
torch peak helpers went with it.

**What survives is what `boot_banner` needs**: cores usable, host RAM, resident RSS, the cgroup
slice and the build identity. Route C calls it once per container, on the first job it handles.

**`observe` and `BANNERS` went too, and that is worth knowing rather than rediscovering.** They
were the one door every `[host]` reading passed through, and every caller of that door was on the
upscale path — so route C never banked a banner and the corpus was already empty on every
route-C run. `handler` keeps the run-record key, empty, rather than dropping it. When contract §1's
host guard and heartbeat are written, route C gets host readings of its own and they land there.
"""



def host_rss_gb():
    """The container's own resident set, in GiB, or `None` where /proc does not say.

    **`VmRSS` from `/proc/self/status`, read rather than modelled.** Host RAM is the wall that
    kills without an exception: a breach is a cgroup SIGKILL, which writes no bundle, raises
    nothing and offers no walk — so unlike VRAM there is no failure path that reports the number
    afterwards. It has to be sampled while the process is alive or it is never known at all.

    Linux-only by construction. On a laptop this returns `None` and the banners stay silent,
    which is the right behaviour for a figure that means nothing off the container.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 — an unreadable /proc is a missing number, not a failure
        return None
    return None


def host_total_gb():
    """The RAM this container may actually use — **the ceiling, not the machine's total**.

    This used to read `SC_PHYS_PAGES`, which is the physical host and was the third independent
    place doing so (F-2026-08-19-37). It feeds the `of X GiB (Y% peak)` figure on every `[host]`
    banner, so a sliced worker was reporting its tail as a comfortable fraction of a number it
    could never reach: 107.20 GiB of "3019.4" read as 4%, where against a real 377 GiB slice it
    is 28%. Same measurement, entirely different meaning.
    """
    try:
        import hardware  # noqa: PLC0415 — stdlib-only; the one choke point for this number
        return hardware.effective_ram_gb()
    except Exception:  # noqa: BLE001
        return None


def cpu_count():
    """Cores this container may actually use.

    **`sched_getaffinity`, not `cpu_count`.** The second reports the machine's cores and the
    first reports the ones this process is allowed on, and a container pinned to a fraction of a
    large host is exactly the case worth knowing about: the phase-4 tail runs at one core's worth
    while thirty sit idle, and the first question that investigation has to answer is how many
    cores there were to be idle. Printed at boot so every log carries the answer without anyone
    having to have thought to ask.
    """
    counts = []
    try:
        import os as _os
        counts.append(len(_os.sched_getaffinity(0)))
    except Exception:  # noqa: BLE001
        try:
            import os as _os
            counts.append(_os.cpu_count())
        except Exception:  # noqa: BLE001
            pass
    # **And the quota, which affinity cannot see** (F-2026-08-19-37). A container pinned by mask
    # is visible to `sched_getaffinity`; one throttled by `cpu.max` sees every CPU in its mask and
    # is simply stopped when its slice is spent. Independent mechanisms, so the usable figure is
    # the smaller — and this worker is allocated 24 vCPUs of a 128-core host.
    try:
        import hardware  # noqa: PLC0415 — stdlib-only
        quota = hardware.cpu_quota()
        if quota:
            counts.append(int(max(1, quota)))
    except Exception:  # noqa: BLE001
        pass
    counts = [c for c in counts if c]
    return min(counts) if counts else None


def affinity_cores():
    """Cores in this process's mask, BEFORE any quota is applied.

    **Reported beside the quota because they answer different questions**, and this project has
    already been bitten by conflating them (F-2026-08-19-37): a container pinned by an affinity
    mask is visible here; one throttled by `cpu.max` sees every CPU in its mask and is simply
    stopped when its slice is spent. A single number reports two different machines identically.
    """
    try:
        import os as _os  # noqa: PLC0415

        return len(_os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        try:
            import os as _os  # noqa: PLC0415

            return _os.cpu_count()
        except Exception:  # noqa: BLE001
            return None


def cpu_configuration():
    """The three CPU numbers a corpus needs, never collapsed into one.

    **Deleted in Wave 2 under §4e's boundary and restored here by CF's ruling.** Every caller was
    inside `_run`, so it went with the upscale path — correct on the evidence then, and wrong for
    what this project now needs: CPU power is an estimator input, and §6a's open question is
    whether `threads=4` caps something x264 could use, which no corpus can answer without
    recording how many cores the container actually had.

    `usable_cores` is the minimum of the two mechanisms and is what the worker actually gets.
    """
    quota = None
    try:
        import hardware  # noqa: PLC0415 — stdlib-only

        quota = hardware.cpu_quota()
    except Exception:  # noqa: BLE001
        pass
    return {"usable_cores": cpu_count(),
            "affinity_cores": affinity_cores(),
            "cpu_quota": int(quota) if quota else None}


#: The environment variable the image bakes its commit into. Named here rather than spelled into
#: the format string so this banner and `handler.build_identity` cannot come to read different
#: names — the rung-1 witness asserts the two agree, and this constant is what makes that cheap.
BUILD_COMMIT_ENV = "BUILD_COMMIT"

#: What CI pushed this image as, set from `IMAGE_REF` in the Dockerfile. **The only honest source
#: for the reference**: one commit can produce more than one image, so the tag cannot be derived
#: from the commit. `handler.build_identity()` reads the same variable for `build.image`.
IMAGE_TAG_ENV = "IMAGE_TAG"


def build_banner():
    """Which build is running, in the worker's own log.

    **The image has always known this and the log has never said it.** `BUILD_COMMIT` is baked
    into every image's config by CI (`Dockerfile:223`, `docker-publish.yml:282`) and the gate
    reads it back off the registry blob on every verification — but that answers *what did we
    publish*, from outside. A worker log answers *what am I*, from inside, and until now it could
    not: a log pulled off a running endpoint named the host, the slice and the data centre, and
    left unstated the two facts that decide whether any of the others are worth reading. Two,
    not one: the commit says which source built it, and the reference says which of the images
    that commit produced is running — a distinction that did not exist until one commit started
    producing both a full and a weightless build. Ten calibration runs were once banked against
    an image reporting `"image": null`, and the only evidence they shared a build was that
    nobody remembered changing one.

    Beside the host lines deliberately: one glance at any worker log now names the build, the
    host slice and the DC together, which is the set a measurement has to be sorted by.

    **Absent is a supported state and says which name was tried** — the same convention the data
    centre line above already uses. A handler run locally has no build identity and must not
    crash for lacking one, and "the variable was never set" must not read identically to "we read
    the wrong name" for whoever finds the log later.
    """
    import os as _os  # noqa: PLC0415 — local, matching every other stdlib touch here

    commit = _os.environ.get(BUILD_COMMIT_ENV)
    if not commit:
        return ("[host] boot: build unknown — {} is not set. A CI image always carries it, so "
                "this is a local or hand-built one.".format(BUILD_COMMIT_ENV))
    # **The reference is READ, never assembled from the commit.** This line used to build
    # `sha-<commit>` itself, which was right for every image that had ever existed — CI tagged
    # them all exactly that — and became wrong the first time one commit produced two images. The
    # route-C build is tagged `sha-<commit>-routec`, and the constructed form 404s against the
    # registry while the worker prints it as fact. A person reads this line at the moment they
    # are establishing what they are looking at; it was used to identify an image the night it
    # first lied, and nearly certified from.
    #
    # **Absent says absent.** No constructed fallback, because "no reference recorded" reads
    # correctly and a 404ing tag reads as fact — the guess is worse than the admission. The
    # commit still leads, since it is what a person types and what the ledger sorts by.
    #
    # The full sha as well as the short form: read beside a registry digest and a
    # `docs/deployment.md` lineage row, an abbreviation is one collision away from another commit.
    reference = _os.environ.get(IMAGE_TAG_ENV)
    return "[host] boot: build {} (sha-{}) image {}".format(
        commit[:7], commit,
        reference if reference else "not recorded — {} is not set".format(IMAGE_TAG_ENV))


def boot_banner():
    """The one-line CPU, host-RAM and data-centre statement every run should open with.

    **The data centre is on this line because the `[load]` strip is meaningless without it.** Two
    H200 data centres behaved differently on the same image on the same day — one could not pull
    from GHCR, another streams layers lazily and pays 6 to 12 minutes faulting the checkpoint in
    on a fresh worker. Every log now carries the axis those figures have to be sorted by, and an
    absent one says *which* names were tried, so "the platform stopped exposing it" and "we read
    the wrong key" stop looking identical.
    """
    import hardware  # noqa: PLC0415 — stdlib-only module; imported here to keep the cycle absent

    cores, total = cpu_count(), host_total_gb()
    # **Read ONCE and bound.** This was `host_rss_gb() or 0.0` guarded by a second, separate
    # `host_rss_gb() is not None` — and the guard is evaluated before the value, so a read that
    # succeeded for the condition and failed for the value printed `0.00 GiB resident`. **A
    # fabricated number carrying a unit is the one thing this project refuses by name**: contract
    # §1 on the guard that returns silently rather than inventing, and `interp_plan`'s own
    # coefficients, absent because a placeholder is indistinguishable from a measurement. One
    # call, one binding, and `unknown` when there is nothing to report.
    rss = host_rss_gb()
    dc, source = hardware.datacenter()
    physical = hardware.physical_ram_gb()
    limit = hardware.memory_limit_gb()
    # **Both numbers, always, so a sliced host is visible in every log** (F-2026-08-19-37). The
    # ceiling alone would be indistinguishable from a small machine, and the physical alone is
    # what this worker planned against for its whole life. Printed as a pair, the slice is a fact
    # anybody reading a log can see without knowing to go looking for it.
    return "[host] boot: {} core(s) usable, {} host RAM, {} resident, dc {}".format(
        cores if cores is not None else "?",
        "{:.1f} GiB".format(total) if total else "unknown",
        "{:.2f} GiB".format(rss) if rss is not None else "unknown",
        "{} (from {})".format(dc, source) if dc
        else "not exposed — tried {}".format(", ".join(hardware.DATACENTER_ENV))) + (
        "\n[host] boot: cgroup slice {:.1f} GiB of {:.1f} GiB physical — the slice is what "
        "kills".format(limit, physical) if limit and physical
        else "\n[host] boot: no cgroup memory limit; the machine's {} is the ceiling".format(
            "{:.1f} GiB".format(physical) if physical else "unknown RAM")) + (
        "\n" + build_banner())
