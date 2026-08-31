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


#: **The smallest series that satisfies §8g, banked regardless of the floor below.** The floor
#: governs the RATE and must not govern the MINIMUM: a run whose frame work finishes inside one
#: interval would otherwise bank the constructor's sample alone and fail the close condition for
#: a reason that is not a defect. Two readings a few hundred milliseconds apart are a thin series
#: and an honest one; a red witness on a healthy short run is neither.
MIN_SAMPLES = 2

#: **How often the strip is allowed to record, in seconds.** Not a clock and not a schedule: it
#: is a floor on the same rate limiter `progress._emit` already applies to itself, and it fires
#: only when `progress` was going to emit anyway. Route C calls `progress.frames()` once per
#: written frame with `boundary=True`, which is FORCED past that limiter — so without a floor
#: here a long clip would bank one sample per frame and the strip would grow with the job.
LOAD_SAMPLE_INTERVAL_S = 15.0


def cuda_memory_gb():
    """`(allocated, reserved)` in GiB, or `(None, None)` where there is no CUDA to ask.

    Both, because they answer different questions and the gap between them IS the caching
    allocator's pool — the same pool that makes `vram_free_gb` a lie on a warm worker
    (`F-2026-08-25-6`). Allocated alone would make a warm container look empty; reserved alone
    would make it look full.
    """
    try:
        import torch  # noqa: PLC0415 — a GPU-box import, like every other torch touch here

        if not torch.cuda.is_available():
            return None, None
        gib = 1024.0 ** 3
        return (round(torch.cuda.memory_allocated() / gib, 3),
                round(torch.cuda.memory_reserved() / gib, 3))
    except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
        return None, None


class LoadStrip:
    """The host-load time series, sampled on the timer `progress` already owns.

    **`docs/archive/instrumentation-archive.md` §8c, and the constraint is as binding as the measurement.** A
    host-load series is what separates a starved encode from a slow one, and it is the only
    quantity on §8f's list that no existing line already computes — but *"a measurement whose
    cost is a new thread is a measurement that changes what it measures"*. So this class owns no
    thread and no timer. It exposes `sample()`, `progress._emit` calls it when it publishes, and
    the cadence is the cadence of the news the worker was already sending.

    **Nothing here can raise.** It runs inside the emit path on every job including a refusal,
    and a sampler that fails a delivered master over a diagnostic would be the worst trade in
    this repository.

    **`cpu_pct` is a DELTA and needs two readings**, so the first sample reports it as `None`
    rather than as zero — a fabricated number carrying a unit is what this project refuses by
    name, and `0.0` on the first row of a series is indistinguishable from an idle container.
    Successive samples divide CPU-seconds burned by wall seconds elapsed and multiply by 100, so
    the figure is a percentage of ONE core and reads above 100 on a busy container. That is the
    convention `top` uses and the one the 96-core question is asked in.
    """

    def __init__(self, started, interval_s=LOAD_SAMPLE_INTERVAL_S):
        import threading as _threading  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        self._time = _time
        self._started = started
        self._interval_s = interval_s
        self._last_at = None
        self._last_cpu_s = None
        #: **One sampler, two possible threads.** `progress._emit` is reached from the main loop
        #: and, once `Progress.keeping_the_promise` has a caller on this route, from its heartbeat
        #: daemon as well. Two threads past the floor together would both read the stale
        #: `_last_at`, and the second would divide a CPU-second delta by a sub-millisecond wall
        #: gap — banking a `cpu_pct` in the thousands, indistinguishable from a real spike, in the
        #: one series whose whole purpose is to say whether the encode was starved. **A
        #: fabricated number carrying a unit is what this project refuses by name.**
        #:
        #: `keeping_the_promise` has no caller on route C today, so the race is latent rather than
        #: live. It is closed now because the lock costs nothing and the defect it prevents is
        #: invisible in the data it corrupts.
        self._lock = _threading.Lock()
        #: The list the run record files. Handed to `handler`'s `trace` by reference, so whatever
        #: has accumulated by the `finally` is what lands — including on a run that died.
        self.samples = []
        self.sample(force=True)

    def sample(self, force=False):
        """Bank one reading, or decline because the floor has not passed. **Never raises.**"""
        try:
            with self._lock:
                return self._sample_locked(force)
        except Exception:  # noqa: BLE001 — see the class docstring
            return False

    def _sample_locked(self, force):
        """The body of `sample`, with `_last_at`/`_last_cpu_s` read and written under the lock.

        **Read and write are one critical section, not two.** The floor check, the delta and the
        two writes all key off `_last_at`; splitting them would leave exactly the window the lock
        exists to close.
        """
        now = self._time.time()
        if (not force and len(self.samples) >= MIN_SAMPLES
                and self._last_at is not None
                and (now - self._last_at) < self._interval_s):
            return False
        cpu_s = _cpu_usage_s()
        cpu_pct = None
        if (cpu_s is not None and self._last_cpu_s is not None
                and self._last_at is not None and now > self._last_at):
            cpu_pct = round(100.0 * (cpu_s - self._last_cpu_s) / (now - self._last_at), 1)
        allocated, reserved = cuda_memory_gb()
        memory_gb, memory_source = _container_memory_gb()
        self.samples.append({
            "elapsed_s": round(now - self._started, 1),
            "cpu_pct": cpu_pct,
            # **`host_mem_gb`, NOT `host_rss_gb`, and the rename is the ruling** (claim C-2,
            # §8f). §8c says the strip exists to show whether the encoder is starved; ffmpeg is
            # a CHILD, so `VmRSS` from `/proc/self` cannot see it and a field named after VmRSS
            # and filled with VmRSS would report a near-idle container across exactly the
            # stretch the strip was built to measure. The reading was right and the name was
            # wrong.
            #
            # **`host_mem_source` says which reading answered**, because one that varies with the
            # host must — the same thing `eta_basis` does for the ETA. Without it a corpus holds
            # two rows meaning different things with nothing on the row saying so.
            "host_mem_gb": memory_gb,
            "host_mem_source": memory_source,
            "cuda_allocated_gb": allocated,
            "cuda_reserved_gb": reserved,
        })
        self._last_at = now
        self._last_cpu_s = cpu_s
        return True

    def snapshot(self):
        """A copy of the series, taken under the lock. **What the record files.**

        The run record is `json.dumps`ed while this strip is still armed — the sampler stops when
        the job does, not when the record is assembled — and a list being appended to while an
        encoder walks it is a list that can reallocate underneath the walk. The record gets a
        copy; nothing else may.
        """
        try:
            with self._lock:
                return list(self.samples)
        except Exception:  # noqa: BLE001 — a measurement must never cost a delivered master
            return list(self.samples)


def _cpu_usage_s():
    try:
        import hardware  # noqa: PLC0415 — stdlib-only, same choke point as everything else here

        return hardware.cpu_usage_s()
    except Exception:  # noqa: BLE001
        return None


#: What `host_mem_source` may say, and each names the FILE the number came out of rather than a
#: category. "cgroup" and "process" would be a taxonomy somebody has to hold in their head; a path
#: is checkable by whoever finds the row.
MEMORY_SOURCE_CGROUP = "cgroup.memory.current"
MEMORY_SOURCE_VMRSS = "proc.self.VmRSS"
MEMORY_SOURCE_NONE = "unavailable"


def _container_memory_gb():
    """`(gb, source)` — the container's memory charge and which reading answered.

    **The cgroup first, because it is the only one that can see the encoder.** `memory.current` is
    what the OOM killer counts and it charges every task in the container, ffmpeg included. VmRSS
    is this process alone and is the fallback for a host with no cgroup — a laptop, mostly, where
    §8c's question does not arise.

    **Never a bare number.** `(None, "unavailable")` still SAYS something: the source is stated
    even when the reading is not, so a sample with no memory figure is a sampler that failed
    rather than a container that used none. §8f permits null only in `cpu_pct` on sample 0, so
    that state is a red witness — correctly.
    """
    try:
        import hardware  # noqa: PLC0415

        current = hardware.memory_current_gb()
        if current is not None:
            return round(current, 3), MEMORY_SOURCE_CGROUP
    except Exception:  # noqa: BLE001
        pass
    rss = host_rss_gb()
    if rss is None:
        return None, MEMORY_SOURCE_NONE
    return round(rss, 3), MEMORY_SOURCE_VMRSS


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
