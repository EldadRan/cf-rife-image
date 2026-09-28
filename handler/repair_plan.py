"""`frame_repair`'s request, its plan, and the order its frames are emitted in — `decisions.md` §19c/§19d.

**PURE PYTHON, AND THAT IS THE MODULE's WHOLE REASON FOR BEING SEPARATE.** *Everything here is
arithmetic on integers and `Fraction`s and a walk over opaque frame objects*, so it imports on a
tree with no numpy, no cv2 and no torch — which is where `validation` runs, where the builder's
conformance test runs it against `gate_scripts/repair_cases.CASES`, and where the harness's dry
run consults it. *The GPU wiring is `repair.py`, and it owns nothing this file decides.*

**THREE THINGS, AND WHICH MOMENT OWNS EACH IS §19d's RULE:**

  `validate_items(params, url)`  at the door — everything the request alone can settle, §26e's
                                 file lists included
  `plan(frames, ranges, segs,    after the probes — the bound on `b`, each segment's `M` against
        sources, stills)`        its video's frames, and the whole output mapping, with the
                                 near-item warnings
  `emit(mapping, ...)`           the one forward pass that turns the mapping into frames

**Every refusal names the item `id`, and carries it as `.item`**, because the conformance test
compares `(code, item)` against the kit and a message is not a field.

**`t` and `p` are `Fraction`s all the way to the model call.** *`roles.md`: identity comparisons
never use floats* — the copy-or-blend decision is `t == 0`, and it is exact here.
"""
from fractions import Fraction

from errors import (FIELD_NOT_SUPPORTED, INVALID_FIELD_VALUE, INVALID_SOURCE,
                    MISSING_REQUIRED_FIELD, SEGMENT_EXCEEDS_SOURCE, WorkerError)

#: §19d — a range's ceiling. *A safety bound, not a policy: the caller applies its own.* **Ranges
#: only**; a segment carries its own frames and has no synthesis to bound.
MAX_RANGE_N = 24

#: §19d — at or under this many GOOD frames between two items, the record carries `near_ranges`.
NEAR_GAP = 3

FITS = ("exact", "resample")
DEFAULT_FIT = "exact"

RANGE_FIELDS = ("id", "a", "b")
#: §26e: a video by index with an inclusive frame range, or a list of stills. *`source_url`,
#: `trim_head` and `trim_tail` — the pre-§26 form — are unlisted, so refused as any unlisted name.*
SEGMENT_FIELDS = ("id", "a", "b", "fit", "source", "start_frame", "end_frame", "stills")

#: §26e — videos beyond source 0. *Stills are not counted (CF: frames do not cost).*
MAX_SOURCES = 3

#: The entry kinds `plan` puts in a mapping, spelled once. **The tuple shapes are the kit's**
#: (`repair_oracle.plan`'s docstring) so the conformance test compares like with like — the VALUES
#: in them are computed here and nowhere else.
SRC, RANGE, SEG = "src", "range", "seg"


class Refused(WorkerError):
    """A `WorkerError` that knows which item it is about. `item` is the id, or None."""

    def __init__(self, code, message, item=None, remedy=None, shortfall=None):
        super().__init__(code, message, remedy=remedy, shortfall=shortfall)
        self.item = item


def _label(item):
    return "{} '{}'".format(item["type"], item["id"])


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _shape(value):
    """What a refused value IS, never what it says — `a str`, `a list of 2` — so no URL a caller
    put where an index belongs is echoed back (§25d: a presigned query is a credential)."""
    if isinstance(value, list):
        return "a list of {}".format(len(value))
    return "a {}".format(type(value).__name__)


# ── the door ─────────────────────────────────────────────────────────────────────────────────

def _int_field(entry, field, where, item_id, default=None, minimum=None):
    """A required (default None) or optional integer. **`bool` is refused**: `True` is an `int`
    in Python and would otherwise be frame 1."""
    value = entry.get(field)
    if value is None:
        if default is not None:
            return default
        raise Refused(MISSING_REQUIRED_FIELD,
                      "field '{}' is required on {}".format(field, where), item_id)
    if not _is_int(value):
        raise Refused(INVALID_FIELD_VALUE,
                      "field '{}' on {} must be an integer frame index, got {!r}".format(
                          field, where, value), item_id)
    if minimum is not None and value < minimum:
        raise Refused(INVALID_FIELD_VALUE,
                      "field '{}' on {} must be {} or more, got {}".format(
                          field, where, minimum, value), item_id)
    return value


def _segment_source(entry, named, item_id, n_videos, n_stills):
    """§26e's segment fields, normalised: `source`, `start_frame`, `end_frame` (None = the video's
    last frame, resolved by `plan`) and `stills` (None on a video segment). **An absent field and
    a null one are the same absence**, so a normalised item passes through here unchanged.

    `n_videos` counts source 0, so a valid `source` is `0 .. n_videos - 1`."""
    source, picks = entry.get("source"), entry.get("stills")
    if (source is None) == (picks is None):
        raise Refused(INVALID_FIELD_VALUE,
                      "{} names {}: a segment takes EXACTLY ONE of 'source' (a video by index, "
                      "with start_frame/end_frame) and 'stills' (indices into "
                      "'params.stills')".format(named, "both" if source is not None else
                                                "neither"), item_id)
    if picks is not None:
        # **No value is echoed that is not an index**: the pre-§26 caller sends URLs here, and a
        # presigned query is a credential (the §26 review's F6).
        if not isinstance(picks, list) or not picks:
            raise Refused(INVALID_FIELD_VALUE,
                          "field 'stills' on {} must be a non-empty list of indices into "
                          "'params.stills', got {}".format(named, _shape(picks)), item_id)
        for pick in picks:
            if not _is_int(pick) or not 0 <= pick < n_stills:
                raise Refused(INVALID_FIELD_VALUE,
                              "field 'stills' on {} holds {}, and 'params.stills' lists {} "
                              "still(s) — an index must be 0 .. {}".format(
                                  named, pick if _is_int(pick) else _shape(pick), n_stills,
                                  n_stills - 1), item_id)
        for field in ("start_frame", "end_frame"):
            if entry.get(field) is not None:
                raise Refused(INVALID_FIELD_VALUE,
                              "field '{}' on {} is a video's frame range, and this segment "
                              "is stills: its frames are the stills it lists, in order".format(
                                  field, named), item_id)
        return {"source": None, "start_frame": None, "end_frame": None, "stills": list(picks)}
    if not _is_int(source) or not 0 <= source < n_videos:
        raise Refused(INVALID_FIELD_VALUE,
                      "field 'source' on {} is {}, and the request lists {} video(s) — source 0 "
                      "is source_url and 1 .. 3 are 'params.sources', so an index must be "
                      "0 .. {}".format(named, source if _is_int(source) else _shape(source),
                                       n_videos, n_videos - 1), item_id)
    start = _int_field(entry, "start_frame", named, item_id, default=0, minimum=0)
    end = entry.get("end_frame")
    if end is not None:
        end = _int_field(entry, "end_frame", named, item_id)
        if end < start:
            raise Refused(INVALID_FIELD_VALUE,
                          "{}: end_frame {} is before start_frame {} — the range is inclusive, so "
                          "end_frame must be start_frame or later".format(named, end, start),
                          item_id)
    return {"source": source, "start_frame": start, "end_frame": end, "stills": None}


def _entry(entry, kind, position, n_videos=1, n_stills=0, normalised=False):
    """One item off the wire, normalised. Structure only; the rules between items are `_rules`.

    **Idempotent on its own output** (a null field is an absent one), so `plan` can take either
    the door's items or the wire's — the kit's cases are the latter. **`type` is the normalised
    item's own key and is accepted only there** (`normalised`, `plan`'s call on an item that
    carries its own kind): on the wire it is an unlisted field like any other (the §26 review's
    F5)."""
    listing = "ranges" if kind == "range" else "segments"
    where = "entry {} of 'params.{}'".format(position, listing)
    if not isinstance(entry, dict):
        raise Refused(INVALID_FIELD_VALUE, "{} must be an object, got {!r}".format(where, entry))
    allowed = RANGE_FIELDS if kind == "range" else SEGMENT_FIELDS
    # **Strict, like every other level of the request.** *`fit` on a range is the likely mistake
    # — it is a segment's field — and accepting it would read as honoured.*
    own = {"type"} if normalised and entry.get("type") == kind else set()
    for name in sorted(set(entry) - set(allowed) - own):
        raise Refused(FIELD_NOT_SUPPORTED,
                      "field '{}' on {} is not accepted on a {}; a {} takes {}".format(
                          name, where, kind, kind, ", ".join(allowed)),
                      entry.get("id") if isinstance(entry.get("id"), str) else None)
    item_id = entry.get("id")
    if item_id is None:
        raise Refused(MISSING_REQUIRED_FIELD, "field 'id' is required on {}".format(where))
    if not isinstance(item_id, str) or not item_id:
        raise Refused(INVALID_FIELD_VALUE,
                      "field 'id' on {} must be a non-empty string, got {!r}".format(
                          where, item_id), item_id if isinstance(item_id, str) else None)
    named = "{} '{}'".format(kind, item_id)
    item = {"type": kind, "id": item_id,
            "a": _int_field(entry, "a", named, item_id),
            "b": _int_field(entry, "b", named, item_id)}
    if kind == "segment":
        fit = DEFAULT_FIT if entry.get("fit") is None else entry["fit"]
        if fit not in FITS:
            raise Refused(INVALID_FIELD_VALUE,
                          "field 'fit' on {} must be one of {}, got {!r}. 'resample' is the "
                          "name of what the handoff called 'retime', which is now an "
                          "operation.".format(named, ", ".join(FITS), fit), item_id)
        item["fit"] = fit
        item.update(_segment_source(entry, named, item_id, n_videos, n_stills))
    return item


def _rules(items, last_frame=None):
    """The rules BETWEEN items and against the source, in `repair_oracle.plan`'s order.

    `last_frame` is None at the door — the bound on `b` needs the file (§19d) — and the source's
    last index after the probe. Returns the items sorted by `a`.
    """
    if not items:
        raise Refused(INVALID_FIELD_VALUE,
                      "a frame_repair needs at least one item across 'params.ranges' and "
                      "'params.segments'; this request carries none")
    seen = set()
    for item in items:
        if item["id"] in seen:
            raise Refused(INVALID_FIELD_VALUE,
                          "id '{}' is used twice; ids are unique across ranges and segments "
                          "together, because the record reports each item by it".format(
                              item["id"]), item["id"])
        seen.add(item["id"])
    for item in items:
        a, b = item["a"], item["b"]
        if a < 0:
            raise Refused(INVALID_FIELD_VALUE,
                          "{}: a = {} — a is the last GOOD frame before the run, so a bad first "
                          "frame has no left anchor and cannot be repaired".format(
                              _label(item), a), item["id"])
        if b - a < 2:
            raise Refused(INVALID_FIELD_VALUE,
                          "{}: b - a = {} — the frames replaced are a+1 .. b-1, so b must be at "
                          "least a + 2".format(_label(item), b - a), item["id"])
        if last_frame is not None and b > last_frame:
            raise Refused(INVALID_FIELD_VALUE,
                          "{}: b = {} and the source's last frame is {} — b is the first GOOD "
                          "frame after the run, so a bad last frame has no right anchor".format(
                              _label(item), b, last_frame), item["id"])
        if item["type"] == "range" and b - a - 1 > MAX_RANGE_N:
            raise Refused(INVALID_FIELD_VALUE,
                          "{} replaces {} frames and a range is capped at {}. The ceiling is a "
                          "safety bound, not a policy — split the run, or send the frames as a "
                          "segment".format(_label(item), b - a - 1, MAX_RANGE_N), item["id"])
    ordered = sorted(items, key=lambda it: it["a"])
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt["a"] < prev["b"]:
            raise Refused(INVALID_FIELD_VALUE,
                          "{} (a={}) overlaps {} (b={}); items may share an anchor "
                          "(next.a == prev.b) and may not overlap".format(
                              _label(nxt), nxt["a"], _label(prev), prev["b"]), nxt["id"])
    return ordered


def _unused(items, n_videos, n_stills):
    """§26e: **a listed file no segment uses is refused**, naming it — it would be fetched for
    nothing. *Source 0 is the source itself and is always used.*"""
    videos = {it["source"] for it in items if it.get("source") is not None}
    stills = {pick for it in items for pick in (it.get("stills") or ())}
    for index in range(1, n_videos):
        if index not in videos:
            raise Refused(INVALID_FIELD_VALUE,
                          "sources[{}] is listed and no segment uses it; every listed file is "
                          "fetched, so an unused one is refused rather than downloaded for "
                          "nothing".format(index), "sources[{}]".format(index))
    for index in range(n_stills):
        if index not in stills:
            raise Refused(INVALID_FIELD_VALUE,
                          "stills[{}] is listed and no segment uses it; every listed file is "
                          "fetched, so an unused one is refused rather than downloaded for "
                          "nothing".format(index), "stills[{}]".format(index))


def _url_list(params, listing, cap=None):
    """`params.sources` / `params.stills`: absent, or a list of non-empty strings."""
    raw = params.get(listing)
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(u, str) or not u.strip() for u in raw):
        # **The value is NOT echoed**: it is a list of presigned URLs, and a query is a
        # credential (the §26 review). Its shape is enough to act on.
        if not isinstance(raw, list):
            bad = "got a {}".format(type(raw).__name__)
        else:
            i, entry = next((i, u) for i, u in enumerate(raw)
                            if not isinstance(u, str) or not u.strip())
            bad = "entry {} is {}".format(i, "an empty string" if isinstance(entry, str)
                                          else "a {}".format(type(entry).__name__))
        raise Refused(INVALID_FIELD_VALUE,
                      "field 'params.{}' must be a list of URL strings; {}".format(listing, bad),
                      listing)
    if cap is not None and len(raw) > cap:
        raise Refused(INVALID_FIELD_VALUE,
                      "field 'params.sources' lists {} videos and a repair takes at most {} "
                      "beyond source 0 (source_url) — videos cost, stills do not (CF)".format(
                          len(raw), cap), listing)
    return raw


def _without_query(url):
    return url.split("?", 1)[0]


def validate_items(params, source_url=None):
    """The door: `params.ranges`, `params.segments`, `params.sources` and `params.stills`,
    normalised, or `Refused`. Returns the items in submission order — ranges first, then segments
    — each carrying `type`.

    **§26e: one object listed twice is refused, `source_url` included, compared without its query
    string** — two presigned URLs for one object differ only there, and fetching it twice is what
    "every file is fetched once" rules out.
    """
    sources = _url_list(params, "sources", cap=MAX_SOURCES)
    stills = _url_list(params, "stills")
    seen = {}
    if isinstance(source_url, str):
        seen[_without_query(source_url)] = "source_url"
    for name, urls, first in (("sources", sources, 1), ("stills", stills, 0)):
        for position, url in enumerate(urls, start=first):
            label = "{}[{}]".format(name, position)
            key = _without_query(url)
            if key in seen:
                raise Refused(INVALID_FIELD_VALUE,
                              "{} names the same object as {} (compared without the query "
                              "string); each file is listed once and fetched once, however "
                              "many segments use it".format(label, seen[key]), label)
            seen[key] = label
    items = []
    for kind, listing in (("range", "ranges"), ("segment", "segments")):
        raw = params.get(listing)
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise Refused(INVALID_FIELD_VALUE,
                          "field 'params.{}' must be an array, got {!r}".format(listing, raw))
        items.extend(_entry(entry, kind, position, 1 + len(sources), len(stills))
                     for position, entry in enumerate(raw))
    if items:
        _unused(items, 1 + len(sources), len(stills))
    _rules(items)
    return items


def check_bounds(frame_count, items):
    """§19d's first file-dependent rule, on its own: `b` against the source's last frame.

    **Called straight after the source's probe, before any other file is fetched**, so a request
    whose anchors are off the end does not pay for its segments' downloads before it is told.
    """
    _rules(items, last_frame=frame_count - 1)


# ── the plan ─────────────────────────────────────────────────────────────────────────────────

def _segment_positions(item, counts):
    """`([(lo, hi, t), ...], m)` for output i = 0 .. N-1 — `lo`/`hi` are frame indices in the
    segment's VIDEO (start_frame included), or indices into `params.stills`.

    §19c's fit on §26e's M: `exact` needs M == N; `resample` puts output i at
    `p = i·(M−1)/(N−1)` (`(M−1)/2` when N = 1) and blends positions ⌊p⌋ and ⌈p⌉ at `t = p − ⌊p⌋`
    — **a copy when t = 0**. M == N is a straight copy under either fit.
    """
    n = item["b"] - item["a"] - 1
    fit = item.get("fit", DEFAULT_FIT)
    if item.get("stills") is not None:
        picks = item["stills"]
        m = len(picks)
        at = picks.__getitem__
    else:
        last = counts[item["source"]] - 1
        start = item["start_frame"]
        end = last if item.get("end_frame") is None else item["end_frame"]
        if end > last:
            raise Refused(SEGMENT_EXCEEDS_SOURCE,
                          "{}: end_frame {} and source {}'s last frame is {} — a segment's frames "
                          "must exist in the video it names".format(
                              _label(item), end, item["source"], last), item["id"])
        m = end - start + 1
        if m < 1:
            raise Refused(INVALID_FIELD_VALUE,
                          "{}: start_frame {} is past source {}'s last frame {}, so the segment "
                          "holds no frames".format(_label(item), start, item["source"], last),
                          item["id"])

        def at(offset):
            return start + offset
    if fit == "exact" and m != n:
        raise Refused(INVALID_FIELD_VALUE,
                      "{}: fit 'exact' needs the segment to hold exactly the {} frames it "
                      "replaces, and it holds {}. Send fit 'resample' to have it fitted".format(
                          _label(item), n, m), item["id"])
    if fit == "resample" and m != n and m < 2:
        raise Refused(INVALID_FIELD_VALUE,
                      "{}: fit 'resample' needs at least 2 frames to resample {} into {}".format(
                          _label(item), m, n), item["id"])
    positions = []
    for i in range(n):
        if m == n:
            p = Fraction(i)
        elif n == 1:
            p = Fraction(m - 1, 2)
        else:
            p = Fraction(i * (m - 1), n - 1)
        lo = p.numerator // p.denominator
        t = p - lo
        positions.append((at(lo), at(lo if t == 0 else lo + 1), t))
    return positions, m


def plan(frame_count, ranges=(), segments=(), sources=(), stills=0):
    """`(mapping, warnings, summary)` or `Refused`.

    `mapping[n]` for output index n is one of — the kit's own shapes (`repair_oracle.plan`):

        ("src", n)                           source frame n, untouched
        ("range", id, k, t)                  RIFE(a, b) at t = k/(N+1)
        ("seg", id, i, lo, hi, t)            §26e: a video segment's frames lo/hi in ITS video,
                                             or a stills segment's still indices; t == 0 is a
                                             copy of lo

    `ranges` and `segments` are item dicts — off `validate_items`, or the wire's (the kit's cases).
    **`sources` are the FRAME COUNTS of `params.sources`** (videos 1 .. 3; source 0's is
    `frame_count`) and `stills` how many `params.stills` are listed — only the files know the
    counts. `summary` is the record's `items[]`.

    **Anchors are read from the source and nowhere else** (§19c), so an item's entries depend on
    its own fields alone: one item's patch is the same whether it is sent alone or with others.
    """
    counts = [frame_count] + list(sources or ())
    if len(counts) - 1 > MAX_SOURCES:
        raise Refused(INVALID_FIELD_VALUE,
                      "{} videos beyond source 0, and a repair takes at most {}".format(
                          len(counts) - 1, MAX_SOURCES), "sources")
    items = [_entry(r, "range", k, normalised=True) for k, r in enumerate(ranges or ())] + \
            [_entry(s, "segment", k, len(counts), stills, normalised=True)
             for k, s in enumerate(segments or ())]
    # **The oracle's order**: no items, then an unused file, then the anchors.
    if items:
        _unused(items, len(counts), stills)
    ordered = _rules(items, last_frame=frame_count - 1)

    warnings = []
    for prev, nxt in zip(ordered, ordered[1:]):
        gap = nxt["a"] - prev["b"] + 1
        if gap <= NEAR_GAP:
            warnings.append({"code": "near_ranges", "ids": [prev["id"], nxt["id"]],
                             "good_gap": gap})

    mapping = [(SRC, n) for n in range(frame_count)]
    summary = {}
    for item in ordered:
        a, b = item["a"], item["b"]
        n = b - a - 1
        entry = {"id": item["id"], "type": item["type"], "a": a, "b": b, "n": n}
        if item["type"] == "range":
            for k in range(1, n + 1):
                mapping[a + k] = (RANGE, item["id"], k, Fraction(k, n + 1))
        else:
            positions, m = _segment_positions(item, counts)
            for i, (lo, hi, t) in enumerate(positions):
                mapping[a + 1 + i] = (SEG, item["id"], i, lo, hi, t)
            # **`fit_applied` is what RAN, not what was asked**: a `resample` whose M already
            # equals N is a straight copy, and the record says so. §26f: which file, and — for a
            # video — the inclusive range as resolved (an absent end_frame is the last frame).
            if item.get("stills") is not None:
                entry["stills"] = list(item["stills"])
            else:
                entry.update(source=item["source"], start_frame=item["start_frame"],
                             end_frame=item["start_frame"] + m - 1)
            entry.update(segment_frames_used=m, fit_applied="exact" if m == n else "resample")
        summary[item["id"]] = entry
    # The record lists items in the order they were SENT, which is the order a caller reads back.
    return mapping, warnings, [summary[item["id"]] for item in items]


def spans(frame_count, seams, items):
    """§20c: the re-encoded spans, `[(start, end_exclusive, [ids])]`, sorted and merged where
    they overlap or touch. `seams` are presentation-order frame indices a copy may start at —
    the clean IDRs (§20i R1), which `splice.PacketMap.seams` reads; `items` carry `id`, `a`, `b`.

    For each item the greatest seam at or before its first replaced frame `a+1`, and the least
    seam after its last `b-1`, or the end of the file. **Pure**, like the rest of this module:
    the builder's conformance test runs it against `gate_scripts/repair_cases.SPAN_CASES`.
    """
    keys = sorted(set(seams))
    if not keys or keys[0] != 0:
        raise ValueError("frame 0 must be a seam; got {}".format(keys[:3]))
    raw = []
    for item in items:
        first, last = item["a"] + 1, item["b"] - 1
        start = max(k for k in keys if k <= first)
        end = next((k for k in keys if k > last), frame_count)
        raw.append((start, end, item["id"]))
    raw.sort(key=lambda row: (row[0], row[1]))
    merged = []
    for start, end, item_id in raw:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2] + [item_id])
        else:
            merged.append((start, end, [item_id]))
    return merged


def synthesised(mapping):
    """How many output frames need the model: every range frame and every segment blend."""
    return sum(1 for m in mapping if m[0] == RANGE or (m[0] == SEG and m[5] != 0))


# ── the one forward pass ─────────────────────────────────────────────────────────────────────

class Window:
    """Forward-only access to a decoded stream, holding only the frames still needed.

    **The plan's source indices never go backwards**, so a frame behind the lowest index anything
    still needs can be dropped — two frames in hand at most, whatever the clip length.

    **`first` is the index of the stream's first frame** (§26e): a segment's video is decoded
    from the keyframe at or before its `start_frame`, not from frame 0, so its first decoded
    frame is frame `first` and the plan's indices are that video's own.
    """

    def __init__(self, frames, what, item=None, first=0):
        self._frames = iter(frames)
        self._held = {}
        self.first = int(first)
        self.highest = self.first - 1
        self._what = what
        self._item = item

    def at(self, index, keep=()):
        if index < self.first:
            raise Refused(INVALID_SOURCE,
                          "{} was decoded from frame {} and the plan needs frame {} — before "
                          "where its decode began".format(self._what, self.first, index),
                          self._item)
        while self.highest < index:
            try:
                frame = next(self._frames)
            except StopIteration:
                raise Refused(INVALID_SOURCE,
                              "{} ended after {} decoded frame(s) and the plan needs frame {}. "
                              "The count the plan was built from was read from the container's "
                              "packets, and the decoder disagrees with it — frame indices are "
                              "exact or they are wrong, so this is refused rather than "
                              "delivered shifted".format(self._what, self.highest + 1 - self.first,
                                                         index), self._item)
            self.highest += 1
            self._held[self.highest] = frame
            for stale in [k for k in self._held if k < self.highest and k not in keep]:
                del self._held[stale]
        for stale in [k for k in self._held if k < index and k not in keep]:
            del self._held[stale]
        return self._held[index]


def emit(mapping, anchors, source, segments, synth):
    """Yield the output frames in order. **Frames are opaque here**: this decides WHICH, never HOW.

    `anchors` is `{range_id: (a, b)}`; `source` an iterable of the source's decoded frames in
    order; `segments` `{segment_id: frames}`, where `frames` is an iterable decoded from that
    segment's frame 0, **or anything with `at(index, keep=())`** — a `Window` over a video decoded
    from its keyframe (§26e), or a caller's stills reader, whose indices need not rise. `synth(key,
    frame_a, frame_b, t)` returns the synthesis at `t` (a `Fraction`, never 0 here) between two
    frames, `key` naming the PAIR so a caller can cache it. **How many frames each stream handed
    over is the CALLER's count** (`routec.DecodeCount`, at the decoder), not this function's.

    **A range's anchors are SOURCE frames a and b.** *The frames between them are decoded and
    dropped unread — they are the damage* — and b is held until its own output index copies it.
    """
    src = Window(source, "the source")
    windows = {sid: (frames if hasattr(frames, "at")
                     else Window(frames, "segment '{}'".format(sid), sid))
               for sid, frames in segments.items()}
    for n, entry in enumerate(mapping):
        kind = entry[0]
        if kind == SRC:
            yield src.at(n)
        elif kind == RANGE:
            _, rid, _k, t = entry
            a, b = anchors[rid]
            frame_a = src.at(a, keep=(a, b))
            frame_b = src.at(b, keep=(a, b))
            yield synth((RANGE, rid), frame_a, frame_b, t)
        else:
            _, sid, _i, lo, hi, t = entry
            window = windows[sid]
            if t == 0:
                yield window.at(lo)
            else:
                frame_lo = window.at(lo, keep=(lo, hi))
                frame_hi = window.at(hi, keep=(lo, hi))
                # **`hi` is in the key** (§26e): a stills segment may pair one still with two
                # different neighbours, and a key naming only `lo` would reuse the wrong pair.
                yield synth((SEG, sid, lo, hi), frame_lo, frame_hi, t)
