"""The CPU/CUDA rounding sweep — `docs/conversion-wave.md` §2e prerequisite 2 and §2g.

**`docs/conversion-wave.md` §5 claims the outbound wave carries ZERO quality risk, and that claim
rests entirely on one unverified premise: that the rounding on the device breaks ties the way the
rounding on the host does.** Both document half-to-even. Nobody has ever checked it on a card, and
neither session has torch or numpy on the machine where the documents are written. **This module is
what turns the premise into a measurement.**

**IT RUNS ON THE ENDPOINT AND NOWHERE ELSE**, because this project has no interactive GPU — the
worker is a serverless container behind a job API. So the sweep rides in as image code, is turned
on for one job by an environment variable, and reports into the run-record like everything else.
§2g's reason for the record rather than the log is worth restating: *a claim that survives only in
a console is a claim that stops existing when the console scrolls.*

**THREE ARMS, NOT TWO.** §2g-2 asked for two that *"differ ONLY in device"* — unsatisfiable, and
amended after a builder claim on 2026-08-27, because the deployed chain is not one library:
`routec._to_rgb24` clamps in TORCH and then multiplies, rounds and casts in NUMPY. "The deployed
chain on CUDA" is not an object that exists. **The pair §5 actually rests on is `ndarray.round`
against `torch.round`**, so a torch-CPU-against-torch-CUDA sweep would answer a neighbouring
question with a billion samples of confidence. Filed as a builder claim, 2026-08-27.

    A   `routec._to_rgb24` itself                     numpy, host      the DEPLOYED chain
    B   `routec._to_rgb24_device`                     torch, host      the SHIPPED chain, CPU
    C   the same                                      torch, device    the SHIPPED chain, CUDA

**A vs C is the graded number** and is §5's question, which is whether the DELIVERED MASTER
changes — a question that does not care which axis a difference came from. **B exists to name that
axis when something disagrees**: without it a non-zero count says there is a problem and not
whether the library or the device owns it, and §2g-3 sends exactly that result to CF for a gate
redesign — a decision that wants the cause and not just the count. Ruled by the gate, 2026-08-27,
and §2g-2 amended with it.

**ALL THREE ARMS IMPORT, AND NONE RESTATES** (§2g-2). The chunk is shaped
`1x3x1xW` and handed to `routec._to_rgb24` as a frame, so the transpose, the contiguity and the
cast under test are the ones that ship; a later edit to that function changes this result instead
of silently invalidating it. **Arms B and C were written fresh when this was built** and said so —
§3a's chain was a snippet in a document then. **The conversion wave landed it and §2g-2's
obligation came due: they import `routec._to_rgb24_device` now.** The proof is about the code
that ships.

**UNASKED, THIS MODULE IS NEVER IMPORTED.** `handler` reads the request field first and imports
second, so a run nobody is asking about pays nothing — not a module read, not an allocation, not a
line of cold start. `F-2026-08-26-3` is this project's standing lesson about an instrument that
changed what it was measuring.
"""

import os
import struct
import time

#: §2e — every fp32 bit pattern from `0x00000000` to `0x3F800000` INCLUSIVE: `+0.0` through
#: `1.0`, which is the closed interval `clamp` can hand the rest of the chain. Negative zero and
#: everything above 1.0 are clamped before they reach the arithmetic and are covered by the
#: sentinels instead.
DOMAIN_HI = 0x3F800000
DOMAIN = DOMAIN_HI + 1

#: **The chunk-size override, and it is ALL that is left in the environment** (§3b-0 item 4).
#: `CF_RIFE_TIECHECK` used to ARM the sweep as well, and CF's rule retired that: **the flag was
#: left set, taxed two jobs ~12.6 s each, and was invisible on the endpoint page** — found only
#: because a record carried a block it should not have, and never traced to where it was set.
#: **Arming is `params.tie_check` now**, refused by name when misspelled and echoed by the record
#: that reports the result.
#:
#: This one survives as an environment variable because it is not arming: it is a
#: deployment-scoped escape hatch for a card that ran out of memory, and a stale one costs a
#: differently-chunked sweep rather than a sweep nobody asked for.
ENV = "CF_RIFE_TIECHECK_CHUNK"

#: Values per chunk. **A multiple of 3 because a chunk is shaped as an RGB frame** — arm A goes
#: through `_to_rgb24`, which expects `1xCxHxW` — and sized so the host side stays small against
#: the 46.57 GiB slice.
#:
#: **PEAK HOST COST IS ROUGHLY `23x` THE CHUNK'S VALUE COUNT IN BYTES, and the first draft of
#: this comment said `15x`.** Counted at the moment `routec.py:98`'s `.round()` allocates: the
#: uint32 `arange` (4x, and it lives the whole iteration), the `.copy()` behind `values` (4x),
#: `_to_rgb24`'s own `clamp` copy (4x), the `x255` product (4x) and the rounded result (4x) — 20x
#: before the uint8 stages and before the previous iteration's three result arrays, which are
#: still bound while the new chunk allocates. **At the default that is ~1.15 GB, not ~750 MB.**
#: The two the earlier figure missed were the `arange` and the clamp copy inside the deployed
#: function. Device peak is ~650 MB. *This is the number an operator reasons from when picking an
#: override after an OOM, which is the only reason it is worth counting to this precision.*
#:
#: **Overridable through `CF_RIFE_TIECHECK_CHUNK`** — because the one thing
#: nobody can test from here is how this behaves on the card, and a sweep that dies on memory
#: with no way to retry smaller would cost a whole job to learn one number.
CHUNK_VALUES = 3 * 16_777_216

#: **The floor under an overriding chunk size**, kept from when one variable both armed the sweep
#: and sized it: `CF_RIFE_TIECHECK=1` was what a person wrote to mean *on*, and read as a chunk
#: size it was three values per pass and 355 million passes — a sweep that never returns and a
#: job spent learning nothing. **Arming moved to the request and the ambiguity went with it**;
#: the floor stays, because the override exists for a card that ran out of memory and no such
#: retry asks for a chunk this small.
MIN_CHUNK_VALUES = 3 * 65_536

#: §2g-2. Named here so the record, this module and the kit read one list.
SENTINELS = ("nan", "pos_inf", "neg_inf", "negative", "above_one", "subnormal")

#: The sentinel values themselves. **`subnormal` is the smallest positive denormal** and
#: `above_one` sits just past the domain's top so `clamp` has something to actually clamp.
_SENTINEL_BITS = {
    "nan": 0x7FC00000,
    "pos_inf": 0x7F800000,
    "neg_inf": 0xFF800000,
    "negative": 0xBF800000,      # -1.0
    "above_one": 0x3F800001,     # nextafter(1.0, inf)
    "subnormal": 0x00000001,
}


def _chunk_size():
    """Values per pass. **A number below `MIN_CHUNK_VALUES` is ignored, not honoured** — see
    `MIN_CHUNK_VALUES`."""
    raw = (os.environ.get(ENV) or "").strip()
    try:
        asked = int(raw)
    except (TypeError, ValueError):
        return CHUNK_VALUES
    if asked < MIN_CHUNK_VALUES:
        return CHUNK_VALUES
    # A whole number of pixels, because the chunk is handed to `_to_rgb24` shaped as a frame.
    return (asked // 3) * 3


def _deployed(values, torch):
    """ARM A — the shipped chain, called as itself.

    `values` is a 1-D float32 tensor whose length is a multiple of 3. Shaped `1x3x1xW` and handed
    to `routec._to_rgb24`, which returns `rgb24` bytes; the transpose it applies is undone here so
    the result lines up with the input order. **Undone rather than avoided**: avoiding it would
    mean not calling the deployed function.
    """
    import numpy as np  # noqa: PLC0415 — a GPU-box import, like every other numpy touch
    import routec  # noqa: PLC0415 — imported HERE so an unset run never reaches it

    width = values.numel() // 3
    frame = values.reshape(1, 3, 1, width)
    out = np.frombuffer(routec._to_rgb24(frame), dtype=np.uint8)
    # `_to_rgb24` emits `HxWx3` after `transpose(1, 2, 0)`; this is `1xWx3` -> `3x1xW` -> flat.
    return out.reshape(1, width, 3).transpose(2, 0, 1).reshape(-1)


def _proposed(values, torch):
    """ARMS B and C — **`routec._to_rgb24_device` itself, imported, not restated.**

    **§2g-2's OBLIGATION, DISCHARGED.** When this module was written the shipped chain did not
    exist: §3a held it as a snippet in a document, so arms B and C were necessarily written fresh
    and this file said so rather than letting the sweep look more self-validating than it was.
    **The conversion wave landed it. So the copy goes**, and the sweep now proves something about
    the code that ships instead of about a faithful transcription of it — which is `retime_oracle`'s
    argument and the same one arm A has rested on from the start.

    Shaped `1x3x1xW` and un-permuted exactly as `_deployed` does, because the two arms must be
    compared at aligned indices and `_to_rgb24_device` emits `HxWx3` bytes like its host twin.
    """
    import numpy as np  # noqa: PLC0415
    import routec  # noqa: PLC0415 — imported HERE so an unset run never reaches it

    width = values.numel() // 3
    frame = values.reshape(1, 3, 1, width)
    out = np.frombuffer(routec._to_rgb24_device(frame), dtype=np.uint8)
    return out.reshape(1, width, 3).transpose(2, 0, 1).reshape(-1)


def _alignment_ok(torch, device, log):
    """**THE POSITIVE CONTROL, and without it this instrument's worst failure is unreadable.**

    The sweep maps a billion inputs onto 256 output levels, so **an index misalignment of `k`
    positions produces on the order of `k x 255` mismatches — a few hundred for a small `k`.**
    That is numerically the same signature a genuine tie difference produces, because there are
    only ~255 tie points in the whole domain. A permutation bug would therefore be reported as
    *"CUDA breaks ties differently at 217 points"* and would send §2g-3's escalation to CF for a
    gate redesign on the strength of a reshape.

    **The scenario is live rather than hypothetical**: this module calls `routec._to_rgb24` on
    purpose so that an edit there cannot silently invalidate the result (§2g-2), and `_deployed`
    un-permutes what that function emits. **An edit to the transpose is exactly the event the
    import is meant to catch, and it is the event that breaks the un-permute.**

    So: a short vector whose expected `uint8` is known INDEPENDENTLY OF THE ROUNDING RULE. Each
    value is `(k + 0.25) / 255`, which lands a quarter-step off every tie, so it rounds to `k`
    under half-to-even, half-away-from-zero, or anything else. **The control tests the
    PERMUTATION and nothing else** — comparing the arms against each other could not, since
    agreement is the thing being measured.

    A failure aborts the sweep rather than reporting a number, because a misaligned sweep's
    number is worse than no number: it looks like an answer.
    """
    width = 256
    # **The pattern is rotated per CHANNEL so its period is the whole vector and not 256.** A
    # plain `i % 256` repeats once per channel, and a misalignment of exactly one channel — 256
    # positions, which is precisely what a wrong `transpose` produces — would have slid one
    # repeat onto the next and matched perfectly. **A control blind to the bug it exists for is
    # worse than no control**, and this one was, in its first draft. Each channel is still a
    # rotation of the full 0-255 range, so every output level is exercised in every channel.
    expected = [(i % 256 + 85 * (i // 256)) % 256 for i in range(3 * width)]
    values = torch.tensor([(k + 0.25) / 255.0 for k in expected], dtype=torch.float32)
    for name, got in (("deployed", list(_deployed(values, torch))),
                      ("torch-cpu", [int(v) for v in _proposed(values, torch)]),
                      ("torch-cuda", [int(v) for v in _proposed(values.to(device), torch)])):
        if len(got) != len(expected):
            log("[tiecheck] ALIGNMENT CONTROL FAILED on the {} arm: it returned {} values where "
                "{} were expected. The sweep is NOT run.".format(name, len(got), len(expected)))
            return False
        if got != expected:
            # A length-guarded scan, so this cannot itself raise on the path whose whole job is
            # to report clearly.
            wrong = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
            log("[tiecheck] ALIGNMENT CONTROL FAILED on the {} arm: {} of {} values wrong, first "
                "at index {} ({} where {} was expected). The sweep is NOT run — a misaligned "
                "comparison reports a mismatch count that reads exactly like a tie "
                "difference.".format(name, len(wrong), len(expected), wrong[0],
                                     got[wrong[0]], expected[wrong[0]]))
            return False
    return True


def _sentinels(torch, device):
    """Each sentinel through arm A and arm C. **Reported, never fatal** (§2g-3)."""
    import numpy as np  # noqa: PLC0415

    out = {}
    for name in SENTINELS:
        entry = {"cpu": None, "cuda": None, "torch_cpu": None, "agree": False}
        try:
            bits = _SENTINEL_BITS[name]
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            # Three of them, because `_to_rgb24` needs a whole pixel to exist.
            host = torch.tensor([value] * 3, dtype=torch.float32)
            entry["cpu"] = int(_deployed(host, torch)[0])
            entry["cuda"] = int(_proposed(host.to(device), torch)[0])
            # **Arm B, and the sweep would be inconsistent without it.** `mismatches_library` and
            # `mismatches_device` exist so a disagreement names its axis; the sentinels are the
            # results MOST likely to disagree — `clamp` propagating NaN, and the NaN-to-`uint8`
            # cast — and they are most likely to differ by LIBRARY rather than by device. Naming
            # the axis everywhere except where it is most needed is not a saving.
            entry["torch_cpu"] = int(_proposed(host, torch)[0])
            entry["agree"] = entry["cpu"] == entry["cuda"]
        except Exception as exc:  # noqa: BLE001 — a sentinel that explodes is a reportable result
            entry["error"] = "{}: {}".format(type(exc).__name__, str(exc)[:120])
        out[name] = entry
    return out


def run(log=print):
    """Sweep the domain and return `(block, reason)`. **Never raises.**

    **`block` is `None` on four distinct paths and the REASON is returned beside it**, because
    without it they collapse into one indistinguishable absence: no CUDA, torch or numpy missing,
    the alignment control failing, or any exception at all. A run-record with no `tie_check` is
    then identical to a run that never set the variable — and the operator who spent a whole GPU
    job on the errand cannot tell whether the card had no CUDA, the image shipped without this
    module, or the sweep ran out of memory. **Three answers, three different next actions.**

    The reason is a warning the caller files into the record, not a log line. *This module's own
    opening argument is that a claim surviving only in a console stops existing when the console
    scrolls, and the first draft of this function made that mistake about its own failures.*
    """
    started = time.perf_counter()
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            reason = "the tie check was requested but this host has no CUDA; nothing was swept"
            log("[tiecheck] " + reason)
            return None, reason
        device = torch.device("cuda", torch.cuda.current_device())
        if not _alignment_ok(torch, device, log):
            reason = ("the tie check's alignment control failed, so the sweep was not run — see "
                      "the log for the arm and index; a misaligned sweep reports a mismatch "
                      "count indistinguishable from a real tie difference")
            return None, reason
        step = _chunk_size()
        log("[tiecheck] sweeping {:,} fp32 bit patterns on {} in chunks of {:,}"
            .format(DOMAIN, device, step))

        swept = 0
        mismatches = 0
        first = []
        by_library = 0   # A vs B — numpy against torch, both on the host
        by_device = 0    # B vs C — torch against torch, host against card
        base = 0
        while base < DOMAIN:
            count = min(step, DOMAIN - base)
            # **Padded up to a whole pixel and the pad is discarded below**, rather than relying
            # on the domain size dividing by three. It does — but a chunk boundary that depends
            # on an arithmetic coincidence is a boundary that breaks when somebody changes the
            # chunk size, which is a thing this module invites through the env var.
            padded = count + (-count % 3)
            bits = np.arange(base, base + padded, dtype=np.uint32)
            if padded > count:
                bits[count:] = base  # benign filler, never compared
            values = torch.from_numpy(bits.view(np.float32).copy())

            got_a = _deployed(values, torch)[:count]
            got_b = _proposed(values, torch)[:count]
            got_c = _proposed(values.to(device), torch)[:count]
            # **Lengths asserted rather than assumed, and the reason is what silence costs
            # here.** On numpy before 1.25 a `!=` between two 1-D arrays of different lengths
            # returns the SCALAR `False`; `flatnonzero(False)` is empty and `count_nonzero(False)`
            # is zero, so a shape bug would file `swept: 1,065,353,217, mismatches: 0` having
            # compared nothing at all. **That is the one outcome this whole errand exists to be
            # unable to produce falsely.**
            if not (len(got_a) == len(got_b) == len(got_c) == count):
                raise ValueError(
                    "arm lengths {}/{}/{} against a chunk of {} — the comparison would have "
                    "broadcast rather than compared".format(
                        len(got_a), len(got_b), len(got_c), count))

            differ = np.flatnonzero(got_a != got_c)
            mismatches += int(differ.size)
            by_library += int(np.count_nonzero(got_a != got_b))
            by_device += int(np.count_nonzero(got_b != got_c))
            # **Bounded at 16** (§2g-2): an unbounded list of a billion disagreements is a record
            # nobody can fetch, and the count is what decides the verdict.
            for index in differ[:max(0, 16 - len(first))]:
                first.append({"bits": int(base + int(index)),
                              "cpu": int(got_a[index]), "cuda": int(got_c[index])})
            swept += count
            base += count

        result = {
            "swept": swept,
            "mismatches": mismatches,
            "first_mismatches": first,
            "sentinels": _sentinels(torch, device),
            "device": str(device),
            "elapsed_s": round(time.perf_counter() - started, 3),
            # **Recorded, ungraded, and named by the gate rather than invented here.** The kit
            # grades `mismatches` alone; these two say which axis owns a non-zero one — the
            # library (numpy against torch) or the device (host against card).
            "mismatches_library": by_library,
            "mismatches_device": by_device,
            "arms": {"cpu": "routec._to_rgb24 (numpy, deployed)",
                     "cuda": "clamp/mul/round/to(uint8) (torch, §3a)"},
        }
        log("[tiecheck] swept {:,} of {:,}; {:,} mismatches in {:.1f}s "
            "(library {:,}, device {:,})".format(
                swept, DOMAIN, mismatches, result["elapsed_s"], by_library, by_device))
        disagreed = [k for k, v in result["sentinels"].items() if not v.get("agree")]
        if disagreed:
            log("[tiecheck] sentinels disagree on {} — reported, not fatal (§2g-3); a NaN "
                "reaching _to_rgb24 is a broken frame under the current code too"
                .format(", ".join(disagreed)))
        return result, None
    except Exception as exc:  # noqa: BLE001 — an errand must never cost a delivered master
        reason = "the tie check did not complete ({}: {})".format(
            type(exc).__name__, str(exc)[:160])
        log("[tiecheck] " + reason + ". The job is unaffected.")
        return None, reason
