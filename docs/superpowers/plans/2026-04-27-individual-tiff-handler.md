# Single-TIFF Handler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop `SingleChannelTiffHandler`, rename `MultiChannelTiffHandler` → `SingleTiffHandler`, and teach the renamed handler squid's current per-image TIFF layout (`<acq>/<t>/<region>_<fov>_<z>_<channel>.tiff` — `<region>` is any non-underscore string), while keeping the legacy `<acq>/0/current_<fov>_<z>_<channel>.tiff` layout working. Format name (`single_tiff`) and filename regex match `cephla-lab/ndviewer_light` for ecosystem consistency.

**Architecture:** Two regexes (legacy then squid — squid's `[^_]+` region group also matches `current`, so legacy must be tried first) committed at detect time and recorded on `acq.extra["layout"]`. `Acquisition` gains `timepoints` and `selected_timepoint`; `FormatHandler` protocol grows a `timepoint: str = "0"` kwarg threaded through to `iter_z_slices` / `load_full_stack` / `iter_full_channel_stacks` / `cache_key`. Composite `<region>_<fov>` strings reuse the existing `acq.fovs` axis. Gallery row UI gains a Time combo (hidden when `len(timepoints) == 1`) wired identically to the FOV combo. Cache keys gain a `t<t>` segment; `CACHE_VERSION` bumps to 4 to evict pre-existing entries.

**Tech Stack:** Python 3.11+, PyQt6 / qtpy, tifffile, numpy, pytest. Spec: `docs/superpowers/specs/2026-04-27-individual-tiff-handler-design.md`.

---

## File Structure

| Path                                              | Action  | Responsibility                                            |
|---------------------------------------------------|---------|-----------------------------------------------------------|
| `src/gallery_view/sources/single_channel_tiff.py` | delete  | gone; AION format dropped                                 |
| `src/gallery_view/sources/multi_channel_tiff.py`  | rename → `single_tiff.py` | renamed handler; rewrites parse/discovery for squid+legacy |
| `src/gallery_view/sources/__init__.py`            | modify  | drop SingleChannel import, rename Multi→Individual         |
| `src/gallery_view/sources/base.py`                | modify  | add `timepoint="0"` kwarg to protocol methods             |
| `src/gallery_view/sources/ome_tiff.py`            | modify  | accept-and-ignore `timepoint` kwarg                       |
| `src/gallery_view/types.py`                       | modify  | `Acquisition.timepoints`, `selected_timepoint`            |
| `src/gallery_view/loader.py`                      | modify  | `Job.timepoint`; pass to handler                          |
| `src/gallery_view/cache.py`                       | modify  | `CACHE_VERSION = 4`                                       |
| `src/gallery_view/ui/gallery_window.py`           | modify  | Time combo, RowKey & mip_data shape                       |
| `src/gallery_view/ui/lut_dialog.py`               | modify  | pass timepoint to `cache_key`                             |
| `tests/conftest.py`                               | modify  | drop single-channel fixture; rename multi→individual; add `make_squid_single_tiff_acq` |
| `tests/sources/test_single_channel_tiff.py`       | delete  | gone                                                       |
| `tests/sources/test_multi_channel_tiff.py`        | rename → `test_single_tiff.py` | rename; add squid-layout cases                |
| `tests/sources/test_registry.py`                  | modify  | drop single-channel test, rename multi→individual         |
| `tests/sources/test_ome_tiff.py`                  | modify  | accept-and-ignore timepoint test                          |
| `tests/test_scan.py`                              | modify  | drop AION case, add squid case                            |
| `tests/test_handler_cache_integration.py`         | modify  | drop single-channel, assert per-timepoint key shape       |
| `tests/test_cache.py`                             | modify  | `CACHE_VERSION` assertion                                 |
| `README.md`                                       | modify  | format-table row updated, single-channel removed          |

---

## Task 1: Drop SingleChannelTiffHandler

**Files:**
- Delete: `src/gallery_view/sources/single_channel_tiff.py`
- Delete: `tests/sources/test_single_channel_tiff.py`
- Modify: `src/gallery_view/sources/__init__.py`
- Modify: `src/gallery_view/scan.py` (remove `_merge_single_channel_siblings`, remove single-channel special cases in `ingest()`)
- Modify: `tests/conftest.py` (remove `make_single_channel_tiff_acq`)
- Modify: `tests/sources/test_registry.py` (remove single-channel test, drop fixture references)
- Modify: `tests/sources/test_ome_tiff.py` (remove single-channel cross-references)
- Modify: `tests/test_scan.py` (remove AION case)
- Modify: `tests/test_handler_cache_integration.py` (remove single-channel cases)
- Modify: `README.md` (remove single-channel row)

- [ ] **Step 1: Inventory references**

```bash
grep -rln "single_channel\|SingleChannelTiff\|make_single_channel" src/ tests/ README.md
```

Expected: matches in the files listed above.

- [ ] **Step 2: Delete the handler module and its test file**

```bash
git rm src/gallery_view/sources/single_channel_tiff.py tests/sources/test_single_channel_tiff.py
```

- [ ] **Step 3: Drop the handler from the registry**

In `src/gallery_view/sources/__init__.py`:

```python
"""Handler registry. ``detect()`` walks ``HANDLERS`` in priority order and
returns the first one whose ``detect()`` returns True (or None).

Order matters: ``ome_tiff`` is checked first because it's identified by a
specific file path; ``multi_channel_tiff`` matches per-image-TIFF folders by
filename pattern.
"""

from .base import FormatHandler
from .multi_channel_tiff import MultiChannelTiffHandler
from .ome_tiff import OmeTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    MultiChannelTiffHandler(),
]
```

(Keep the file otherwise unchanged. Task 2 renames `MultiChannelTiffHandler`.)

- [ ] **Step 4: Drop the conftest fixture**

In `tests/conftest.py`, delete the `make_single_channel_tiff_acq` fixture and any helpers used only by it. Search for it first:

```bash
grep -n "make_single_channel\|single_channel" tests/conftest.py
```

Remove the fixture function and any imports unique to it.

- [ ] **Step 5: Remove single-channel from `tests/sources/test_registry.py`**

```python
"""Handler registry: detect() returns the right handler per format."""

from gallery_view import sources


def test_detect_returns_ome_tiff_handler(make_ome_tiff_acq):
    h = sources.detect(str(make_ome_tiff_acq()))
    assert h is not None
    assert h.name == "ome_tiff"


def test_detect_returns_multi_channel_handler(make_multi_channel_tiff_acq):
    h = sources.detect(str(make_multi_channel_tiff_acq()))
    assert h is not None
    assert h.name == "multi_channel_tiff"


def test_detect_returns_none_for_unrelated_folder(tmp_path):
    (tmp_path / "random").mkdir()
    assert sources.detect(str(tmp_path / "random")) is None
```

(`test_detect_returns_single_channel_handler` is gone.)

- [ ] **Step 6: Simplify `src/gallery_view/scan.py`**

The current `ingest()` has two single-channel special cases that need removing:

1. `if handler is not None and handler.name != "single_channel_tiff":` → drop the `and …` clause; the handler can never be single-channel any more.
2. `if sub_handler.name == "single_channel_tiff":` block + the `single_channel_buckets` accumulator + the trailing `out.extend(_merge_single_channel_siblings(…))` call.
3. Delete `_merge_single_channel_siblings` and any imports it pulled in (`from collections import defaultdict` if no longer used; check for other uses first via `grep -n "defaultdict" src/gallery_view/scan.py`).

After cleanup, `ingest()` should fold all detected handlers identically:

```python
def ingest(
    path: str,
    *,
    _seen: set[str] | None = None,
    _depth: int = 0,
) -> list[Acquisition]:
    if _seen is None:
        _seen = set()
    if not os.path.isdir(path) or os.path.islink(path) or _is_hidden(path):
        return []
    real = os.path.realpath(path)
    if real in _seen:
        return []
    _seen.add(real)

    handler = sources.detect(path)
    if handler is not None:
        params = common.parse_acquisition_params(path) or {}
        acq = handler.build(path, params)
        return [acq] if acq is not None else []

    if _depth >= MAX_DEPTH:
        return []

    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return []

    out: list[Acquisition] = []
    for entry in entries:
        sub = os.path.join(path, entry)
        if _is_hidden(sub) or not os.path.isdir(sub) or os.path.islink(sub):
            continue
        sub_handler = sources.detect(sub)
        if sub_handler is None:
            out.extend(ingest(sub, _seen=_seen, _depth=_depth + 1))
            continue
        params = common.parse_acquisition_params(sub) or {}
        acq = sub_handler.build(sub, params)
        if acq is not None:
            out.append(acq)
    return out
```

Update the docstring to drop the "Sibling single-channel folders…" bullet.

- [ ] **Step 7: Remove single-channel from other test files**

```bash
grep -ln "single_channel\|make_single_channel\|SingleChannelTiff" tests/ | xargs -I{} grep -n "single_channel\|make_single_channel\|SingleChannelTiff" {}
```

For each match, delete the test function or fixture argument that depends on the dropped fixture/handler. Common locations: `tests/test_scan.py`, `tests/test_handler_cache_integration.py`, `tests/sources/test_ome_tiff.py`. The deletions are mechanical; do not fold dropped tests into others.

- [ ] **Step 8: Remove single-channel row from README**

Edit the *Supported formats* table in `README.md` to leave only `ome_tiff` and `multi_channel_tiff` rows. Don't change wording yet (Task 11 rewrites it).

- [ ] **Step 9: Run the test suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all remaining tests pass; total count reduced (single-channel tests gone).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Drop SingleChannelTiffHandler

AION's single-channel-per-folder format is no longer in scope. Remove
the handler, its test file, the conftest fixture, registry registration,
and the references in cross-handler tests and the README.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Rename MultiChannelTiffHandler → SingleTiffHandler

**Files:**
- Rename: `src/gallery_view/sources/multi_channel_tiff.py` → `single_tiff.py`
- Rename: `tests/sources/test_multi_channel_tiff.py` → `test_single_tiff.py`
- Modify: `src/gallery_view/sources/__init__.py`
- Modify: `tests/conftest.py` (rename fixture)
- Modify: `tests/sources/test_registry.py`
- Modify: any other test using `make_multi_channel_tiff_acq` or `MultiChannelTiffHandler`
- Modify: `README.md` (table row label only)

- [ ] **Step 1: `git mv` the source file**

```bash
git mv src/gallery_view/sources/multi_channel_tiff.py src/gallery_view/sources/single_tiff.py
git mv tests/sources/test_multi_channel_tiff.py tests/sources/test_single_tiff.py
```

- [ ] **Step 2: Rename the class and `name` attribute**

In `src/gallery_view/sources/single_tiff.py`:

```python
class SingleTiffHandler:
    name = "single_tiff"
```

(Update the module docstring too — replace "multi-channel TIFF" with "single TIFF" wording.)

- [ ] **Step 3: Update the registry import**

In `src/gallery_view/sources/__init__.py`:

```python
from .base import FormatHandler
from .single_tiff import SingleTiffHandler
from .ome_tiff import OmeTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    SingleTiffHandler(),
]
```

- [ ] **Step 4: Rename the conftest fixture**

In `tests/conftest.py`:

```bash
sed -i.bak 's/make_multi_channel_tiff_acq/make_single_tiff_acq/g' tests/conftest.py && rm tests/conftest.py.bak
```

(Verify the substitution: `grep "make_multi_channel\|make_individual" tests/conftest.py` should show only `make_single_tiff_acq`.)

- [ ] **Step 5: Update test files using the fixture**

```bash
grep -rln "make_multi_channel_tiff_acq\|MultiChannelTiffHandler\|multi_channel_tiff" tests/ src/
```

For each match, replace:
- `make_multi_channel_tiff_acq` → `make_single_tiff_acq`
- `MultiChannelTiffHandler` → `SingleTiffHandler`
- `"multi_channel_tiff"` (handler name string) → `"single_tiff"`

`test_single_tiff.py` will have the most replacements. Update `tests/sources/test_registry.py`'s test function name to `test_detect_returns_individual_handler` and assert `h.name == "single_tiff"`.

- [ ] **Step 6: Update README format-table label**

Change `multi_channel_tiff` → `single_tiff` in the *Supported formats* table. Don't change descriptions yet.

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass (mechanical rename, no behavior change).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Rename MultiChannelTiffHandler → SingleTiffHandler

The 'multi-channel' label was misleading: the format is one TIFF per
(z, channel), not multiple channels per file. Mechanical rename only —
class, module, handler.name string, conftest fixture, and references
in tests and README. No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `timepoint` kwarg through the data model and protocol

**Files:**
- Modify: `src/gallery_view/types.py`
- Modify: `src/gallery_view/sources/base.py`
- Modify: `src/gallery_view/sources/ome_tiff.py`
- Modify: `src/gallery_view/sources/single_tiff.py`
- Modify: `src/gallery_view/loader.py`
- Modify: `src/gallery_view/ui/lut_dialog.py`
- Modify: `tests/sources/test_ome_tiff.py`

This task is plumbing only: defaults are backward-compatible, no behavior changes yet. No new failing test for behavior; we add a regression test that the OME-TIFF handler accepts and ignores the new kwarg, and verify all existing tests still pass.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/sources/test_ome_tiff.py`:

```python
def test_iter_z_slices_accepts_timepoint_kwarg(handler, make_ome_tiff_acq):
    """OME-TIFF has no <t>/ subdir but the protocol method takes the
    kwarg. Pass an arbitrary value and confirm slices come back unchanged."""
    folder = make_ome_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0], timepoint="7"))
    assert len(slices) == 3
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/sources/test_ome_tiff.py::test_iter_z_slices_accepts_timepoint_kwarg -v
```

Expected: FAIL — `iter_z_slices() got an unexpected keyword argument 'timepoint'`.

- [ ] **Step 3: Add the fields to `Acquisition`**

In `src/gallery_view/types.py`:

```python
@dataclass
class Acquisition:
    handler: "FormatHandler"
    path: str
    folder_name: str
    display_name: str
    params: dict
    channels: list[Channel]
    fovs: list[str]
    shape_zyx: ShapeZYX | None = None
    selected_fov: str = "0"
    timepoints: list[str] = field(default_factory=lambda: ["0"])
    selected_timepoint: str = "0"
    extra: dict = field(default_factory=dict)
```

- [ ] **Step 4: Add `timepoint` to the protocol**

In `src/gallery_view/sources/base.py`, update each method signature:

```python
def cache_key(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> tuple[str, str]:
    ...

def iter_z_slices(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> Iterator[np.ndarray]:
    ...

def load_full_stack(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> np.ndarray:
    ...

def iter_full_channel_stacks(
    self, acq: Acquisition, fov: str,
    timepoint: str = "0",
) -> Iterator[tuple[Channel, np.ndarray]]:
    ...
```

Update each docstring to mention timepoint where relevant ("Used by handlers whose data is partitioned by timepoint; OME-TIFF ignores this argument.").

- [ ] **Step 5: Update OmeTiffHandler to accept (and ignore)**

In `src/gallery_view/sources/ome_tiff.py`, add `timepoint: str = "0"` to the same four methods. The body doesn't need to use `timepoint`. Example:

```python
def iter_z_slices(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> Iterator[np.ndarray]:
    ome_path = acq.extra["ome_path"]
    ch_idx = self._channel_index(acq, channel)
    # OME-TIFF stores all timepoints in one file; timepoint kwarg ignored.
    with TiffFile(ome_path) as tif:
        ...
```

(Apply the same kwarg addition to `load_full_stack`, `iter_full_channel_stacks`, and `cache_key`.)

- [ ] **Step 6: Update SingleTiffHandler to accept the kwarg**

In `src/gallery_view/sources/single_tiff.py`, add `timepoint: str = "0"` to the same four methods. Body unchanged for now — the timepoint will be wired into the glob in Task 9.

- [ ] **Step 7: Add `timepoint` to `Job` and pass it through**

In `src/gallery_view/loader.py`:

```python
@dataclass
class Job:
    acq_id: int
    acq: Acquisition
    fov: str
    channel: Channel
    ch_idx: int
    timepoint: str = "0"
```

Find every reference to `cache_key` and `iter_z_slices` in this file and pass `timepoint=job.timepoint`:

```python
src, ch_id = job.acq.handler.cache_key(
    job.acq, job.fov, job.channel, timepoint=job.timepoint
)
...
for slice_yx in job.acq.handler.iter_z_slices(
    job.acq, job.fov, job.channel, timepoint=job.timepoint
):
```

- [ ] **Step 8: Update `lut_dialog.py`**

In `src/gallery_view/ui/lut_dialog.py:220`, find the `cache_key` call and pass the current acquisition's `selected_timepoint`:

```python
src, ch_id = acq.handler.cache_key(
    acq, fov, channel, timepoint=acq.selected_timepoint
)
```

- [ ] **Step 9: Run the test from Step 1**

```bash
.venv/bin/python -m pytest tests/sources/test_ome_tiff.py::test_iter_z_slices_accepts_timepoint_kwarg -v
```

Expected: PASS.

- [ ] **Step 10: Run the full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "Add timepoint plumbing to data model, protocol, and Job

Adds Acquisition.timepoints (default ['0']) and selected_timepoint,
threads a timepoint='0' kwarg through FormatHandler.iter_z_slices /
load_full_stack / iter_full_channel_stacks / cache_key, and propagates
it via Job and the LUT dialog. OME-TIFF and the renamed single TIFF
handler accept the kwarg; OME-TIFF ignores it (no <t>/ subdir).
Backward-compatible defaults — no behavior change yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Bump CACHE_VERSION and add `t<t>` to cache key paths

**Files:**
- Modify: `src/gallery_view/cache.py`
- Modify: `src/gallery_view/sources/ome_tiff.py` (cache_key shape)
- Modify: `src/gallery_view/sources/single_tiff.py` (cache_key shape)
- Modify: `tests/test_cache.py`
- Modify: `tests/test_handler_cache_integration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/sources/test_ome_tiff.py`:

```python
def test_cache_key_includes_timepoint(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    _, ch_id = handler.cache_key(acq, "0", acq.channels[0], timepoint="3")
    assert "/t3/" in ch_id
```

Append to `tests/sources/test_single_tiff.py`:

```python
def test_cache_key_includes_timepoint(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    _, ch_id = handler.cache_key(acq, "0", acq.channels[0], timepoint="2")
    assert "/t2/" in ch_id
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/sources/test_ome_tiff.py::test_cache_key_includes_timepoint tests/sources/test_single_tiff.py::test_cache_key_includes_timepoint -v
```

Expected: FAIL on the assert (current keys lack `/t<t>/`).

- [ ] **Step 3: Update `cache_key` in OmeTiffHandler**

```python
def cache_key(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> tuple[str, str]:
    return acq.extra["ome_path"], f"fov{fov}/t{timepoint}/wl_{channel.wavelength}"
```

- [ ] **Step 4: Update `cache_key` in SingleTiffHandler**

```python
def cache_key(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> tuple[str, str]:
    return acq.path, f"fov{fov}/t{timepoint}/{channel.name}"
```

- [ ] **Step 5: Bump `CACHE_VERSION`**

In `src/gallery_view/cache.py:18`:

```python
CACHE_VERSION = 4  # v4: cache key includes /t<t>/ for multi-timepoint single_tiff
```

- [ ] **Step 6: Update `tests/test_cache.py` if it asserts a specific version**

```bash
grep -n "CACHE_VERSION" tests/test_cache.py
```

If a test pins the value (e.g. `assert cache.CACHE_VERSION == 2`), update to `== 4`. The version-mismatch eviction test at line 52 (`payload["version"] = np.int32(cache.CACHE_VERSION + 99)`) is already version-relative and needs no change.

- [ ] **Step 7: Update `tests/test_handler_cache_integration.py`**

Locate the test that round-trips `cache_key` → `cache.save` → `cache.load` (around line 33). The key shape changed; if any assertion inspects the literal `ch_id`, update to match `fov{fov}/t{t}/...`.

If `test_cache_keys_distinct_across_acquisitions` is the only test, no change is likely required since it only compares two keys for inequality. Confirm with:

```bash
.venv/bin/python -m pytest tests/test_handler_cache_integration.py -v
```

If anything fails, fix the literal comparison and re-run.

- [ ] **Step 8: Run the new tests + full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Add /t<t>/ segment to cache keys; bump CACHE_VERSION → 4

Cache keys now include the timepoint so multi-timepoint single_tiff
acquisitions don't collide. CACHE_VERSION bumps to 4 so any pre-existing
v3 entries are evicted on first read and recomputed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add squid filename regex (parsing only, no detection wiring yet)

**Files:**
- Modify: `src/gallery_view/sources/single_tiff.py`
- Modify: `tests/sources/test_single_tiff.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/sources/test_single_tiff.py`:

```python
from gallery_view.sources.single_tiff import (
    parse_squid_filename,
    parse_legacy_filename,
)


def test_parse_squid_filename_numeric_region():
    p = parse_squid_filename("0_1_2_Fluorescence_488_nm_Ex.tiff")
    assert p == {"region": "0", "fov": "1", "z": "2",
                 "channel": "Fluorescence_488_nm_Ex"}


def test_parse_squid_filename_well_as_region():
    """Squid puts the well in the region slot when wells are used."""
    p = parse_squid_filename("A1_0_2_Fluorescence_488_nm_Ex.tiff")
    assert p == {"region": "A1", "fov": "0", "z": "2",
                 "channel": "Fluorescence_488_nm_Ex"}


def test_parse_squid_filename_rejects_garbage():
    assert parse_squid_filename("not_a_match.tiff") is None
    assert parse_squid_filename("only_two_components.tiff") is None


def test_parse_legacy_filename_basic():
    p = parse_legacy_filename("current_3_5_Fluorescence_488_nm_Ex.tiff")
    assert p == {"fov": "3", "z": "5", "channel": "Fluorescence_488_nm_Ex"}


def test_parse_legacy_filename_rejects_squid():
    assert parse_legacy_filename("0_1_2_Fluorescence_488_nm_Ex.tiff") is None


def test_squid_regex_also_matches_legacy_filenames():
    """Important: squid regex's [^_]+ region accepts 'current' too. Detection
    code must try legacy first, then squid — never the other way around."""
    p = parse_squid_filename("current_0_0_Fluorescence_488_nm_Ex.tiff")
    assert p is not None
    assert p["region"] == "current"  # documents the overlap
```

- [ ] **Step 2: Run them to verify failure**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py -v -k "parse_"
```

Expected: ImportError (the parsers don't exist yet).

- [ ] **Step 3: Implement the parsers**

In `src/gallery_view/sources/single_tiff.py`, add at module top (after imports, before the class). The regex matches `cephla-lab/ndviewer_light`'s `FPATTERN` exactly:

```python
import re

# Matches cephla-lab/ndviewer_light's FPATTERN. <region> is any non-underscore
# string — squid stuffs the well there for plate acquisitions and a numeric
# region id for non-plate runs, so a single capture handles both cases.
_SQUID_RE = re.compile(
    r"^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)\.tiff?$",
    re.IGNORECASE,
)
_LEGACY_RE = re.compile(
    r"^current_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)\.tiff?$"
)


def parse_squid_filename(name: str) -> dict | None:
    """Parse a squid per-image TIFF filename: ``<region>_<fov>_<z>_<channel>.tiff``.

    ``region`` is any non-underscore string (numeric id, well like ``A1``,
    etc.). NOTE: this also matches legacy ``current_<fov>_<z>_<ch>.tiff``
    filenames (where ``current`` lands in the region slot). Detection code
    must try ``parse_legacy_filename`` first.
    """
    m = _SQUID_RE.match(name)
    if not m:
        return None
    return m.groupdict()


def parse_legacy_filename(name: str) -> dict | None:
    """Parse the older ``current_<fov>_<z>_<channel>.tiff`` filename."""
    m = _LEGACY_RE.match(name)
    if not m:
        return None
    return m.groupdict()
```

- [ ] **Step 4: Verify tests pass**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py -v -k "parse_"
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Add squid + legacy filename parsers to single_tiff

Two regexes — module-level so they're shared across detect/build/glob
and unit-tested independently. Squid format: optional <well>_ prefix
followed by <region>_<fov>_<z>_<channel>.tiff. Legacy: current_<fov>_
<z>_<channel>.tiff. The two regexes are disjoint (squid requires a
leading digit; legacy requires the literal 'current_').

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Detect squid layout and discover composite FOVs in `build()`

**Files:**
- Modify: `src/gallery_view/sources/single_tiff.py`
- Modify: `tests/conftest.py` (add `make_squid_single_tiff_acq`)
- Modify: `tests/sources/test_single_tiff.py`

- [ ] **Step 1: Add the squid fixture to conftest**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def make_squid_single_tiff_acq(tmp_path):
    """Build an acquisition in squid's per-image TIFF layout.

    ``<acq>/<t>/[<well>_]<region>_<fov>_<z>_<channel>.tiff``
    """

    def _build(
        wavelengths=("488", "561"),
        nz=3,
        ny=8,
        nx=10,
        nt=1,
        regions=1,
        fovs_per_region=1,
        with_well_prefix=False,
        well="A1",
        folder_name="25x_A1_2026-04-26_12-00-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
        mag=25,
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        _write_params(folder, {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
        })
        _write_channels_yaml(folder, [
            {
                "name": f"Fluorescence_{wl}_nm_Ex",
                "enabled": True,
                "camera_settings": {"exposure_time_ms": 100.0 * (i + 1)},
                "illumination_settings": {"intensity": 25.0 * (i + 1)},
            }
            for i, wl in enumerate(wavelengths)
        ])
        prefix = f"{well}_" if with_well_prefix else ""
        for t in range(nt):
            t_dir = folder / str(t)
            t_dir.mkdir()
            for r in range(regions):
                for f in range(fovs_per_region):
                    for z in range(nz):
                        for c, wl in enumerate(wavelengths):
                            stack = _gradient_3d(nz, ny, nx, c)
                            tifffile.imwrite(
                                t_dir / f"{prefix}{r}_{f}_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                                stack[z],
                            )
        return folder

    return _build
```

(Reuses the `_write_params`, `_write_channels_yaml`, and `_gradient_3d` helpers already in `conftest.py`.)

- [ ] **Step 2: Write failing tests**

Append to `tests/sources/test_single_tiff.py`:

```python
def test_detect_squid_layout(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_squid_layout_with_well_prefix(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(with_well_prefix=True)
    assert handler.detect(str(folder)) is True


def test_legacy_layout_still_detects(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_build_discovers_composite_fovs(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=2)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.fovs == ["0_0", "0_1", "1_0", "1_1"]
    assert acq.selected_fov == "0_0"


def test_legacy_build_uses_zero_region(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.fovs == ["0_0"]
```

- [ ] **Step 3: Run them to verify failure**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py -v -k "detect_squid or composite_fovs or legacy_build_uses_zero"
```

Expected: FAILs on the squid-specific tests.

- [ ] **Step 4: Implement detect + build (and update `list_fovs`)**

In `src/gallery_view/sources/single_tiff.py`, also replace the existing `list_fovs` (currently hard-coded to `["0"]`) to return the acquisition's discovered list:

```python
def list_fovs(self, acq: Acquisition) -> list[str]:
    return acq.fovs
```

Then replace `detect` and `build`:

```python
def detect(self, folder: str) -> bool:
    layout = self._detect_layout(folder)
    return layout is not None

def build(self, folder: str, params: dict) -> Acquisition | None:
    layout = self._detect_layout(folder)
    if layout is None:
        return None
    timepoints = self._timepoints_for(folder, layout)
    if not timepoints:
        return None
    fovs = self._fovs_for(folder, layout, timepoints[0])
    if not fovs:
        return None
    channels = common.parse_acquisition_channels_yaml(folder)
    if not channels:
        channels = self._channels_from_filenames(folder, layout, timepoints[0])
    if not channels:
        return None
    folder_name = os.path.basename(folder)
    return Acquisition(
        handler=self,
        path=folder,
        folder_name=folder_name,
        display_name=common.display_name_for(folder_name),
        params=params,
        channels=channels,
        fovs=fovs,
        selected_fov=fovs[0],
        timepoints=timepoints,
        selected_timepoint=timepoints[0],
        extra={"layout": layout},
    )

@staticmethod
def _detect_layout(folder: str) -> str | None:
    """Return ``"squid"`` or ``"legacy"`` if folder matches; else None."""
    legacy_dir = os.path.join(folder, "0")
    if os.path.isdir(legacy_dir):
        for f in glob.glob(os.path.join(legacy_dir, "*.tiff")):
            base = os.path.basename(f)
            if parse_legacy_filename(base) is not None:
                return "legacy"
            if parse_squid_filename(base) is not None:
                return "squid"
    # Squid layout may have <t>/ dirs other than 0 too
    for entry in sorted(os.listdir(folder) if os.path.isdir(folder) else []):
        if not entry.isdigit():
            continue
        t_dir = os.path.join(folder, entry)
        if not os.path.isdir(t_dir):
            continue
        for f in glob.glob(os.path.join(t_dir, "*.tiff")):
            if parse_squid_filename(os.path.basename(f)) is not None:
                return "squid"
    return None

@staticmethod
def _timepoints_for(folder: str, layout: str) -> list[str]:
    if layout == "legacy":
        return ["0"]
    return sorted(
        (e for e in os.listdir(folder) if e.isdigit()
         and os.path.isdir(os.path.join(folder, e))),
        key=int,
    )

@staticmethod
def _fovs_for(folder: str, layout: str, timepoint: str) -> list[str]:
    t_dir = os.path.join(folder, timepoint)
    parser = parse_squid_filename if layout == "squid" else parse_legacy_filename
    seen: set[tuple[str, str]] = set()
    for f in glob.glob(os.path.join(t_dir, "*.tiff")):
        p = parser(os.path.basename(f))
        if p is None:
            continue
        region = p.get("region", "0")
        seen.add((region, p["fov"]))
    return [f"{r}_{f}" for r, f in sorted(seen, key=lambda x: (int(x[0]), int(x[1])))]

@staticmethod
def _channels_from_filenames(folder: str, layout: str, timepoint: str) -> list[Channel]:
    t_dir = os.path.join(folder, timepoint)
    parser = parse_squid_filename if layout == "squid" else parse_legacy_filename
    names: set[str] = set()
    for f in glob.glob(os.path.join(t_dir, "*.tiff")):
        p = parser(os.path.basename(f))
        if p is None:
            continue
        names.add(p["channel"])
    out: list[Channel] = []
    for name in sorted(names):
        wl_m = re.search(r"(\d+)_nm", name)
        wl = wl_m.group(1) if wl_m else "unknown"
        out.append(Channel(name=name, wavelength=wl))
    out.sort(
        key=lambda c: int(c.wavelength) if c.wavelength.isdigit() else 999
    )
    return out
```

(Delete the old `_channels_from_filenames` if it had a different signature; this replaces it.)

- [ ] **Step 5: Run the new tests**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py -v -k "detect_squid or composite_fovs or legacy_build_uses_zero or legacy_layout_still_detects"
```

Expected: all pass.

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass. Note that some pre-existing tests may rely on `acq.fovs == ["0"]` for the legacy fixture — they should now expect `["0_0"]`. Update those assertions in `test_single_tiff.py` (only) if any fail.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Detect squid layout, discover composite <region>_<fov> FOVs

SingleTiffHandler.detect now recognises both <acq>/0/current_…
(legacy) and <acq>/<t>/<region>_<fov>_<z>_<channel>.tiff (squid, with
optional <well>_ prefix). build() records the chosen layout in
acq.extra['layout'] and populates acq.fovs with composite '<region>_
<fov>' strings sorted by (region, fov). Legacy folders synthesise
region='0' for shape consistency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Discover multi-timepoint in `build()`

**Files:**
- Modify: `tests/sources/test_single_tiff.py`

(`_timepoints_for` was implemented in Task 6; this task adds the test that exercises it on a real `nt=3` fixture.)

- [ ] **Step 1: Write the failing test**

Append to `tests/sources/test_single_tiff.py`:

```python
def test_build_discovers_multiple_timepoints(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(nt=3)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.timepoints == ["0", "1", "2"]
    assert acq.selected_timepoint == "0"


def test_legacy_acq_has_single_timepoint(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert acq.timepoints == ["0"]
    assert acq.selected_timepoint == "0"
```

- [ ] **Step 2: Run to verify it passes (Task 6 already implemented this)**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py::test_build_discovers_multiple_timepoints tests/sources/test_single_tiff.py::test_legacy_acq_has_single_timepoint -v
```

Expected: PASS for both. If the legacy test fails because `acq.timepoints` is missing, double-check Task 3 + Task 6 wiring before continuing.

- [ ] **Step 3: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Test multi-timepoint discovery in SingleTiffHandler.build

Adds explicit assertions that nt>1 squid acquisitions report all
<t>/ subdirs in acq.timepoints (and that legacy folders stay at
['0']). Implementation already landed in the previous commit; this
locks the contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Use `timepoint` in `iter_z_slices`, `load_full_stack`, `iter_full_channel_stacks`

**Files:**
- Modify: `src/gallery_view/sources/single_tiff.py`
- Modify: `tests/sources/test_single_tiff.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/sources/test_single_tiff.py`:

```python
def test_iter_z_slices_uses_timepoint(handler, make_squid_single_tiff_acq):
    """Slices for timepoint='1' must come from <acq>/1/, not <acq>/0/."""
    folder = make_squid_single_tiff_acq(nt=2, nz=3)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices_t0 = list(handler.iter_z_slices(acq, "0_0", acq.channels[0], timepoint="0"))
    slices_t1 = list(handler.iter_z_slices(acq, "0_0", acq.channels[0], timepoint="1"))
    assert len(slices_t0) == 3
    assert len(slices_t1) == 3
    # Both timepoints write the same gradient fixture, so identical content
    # would mask a wiring bug. Distinguish by file path: load_full_stack
    # via a tampered timepoint should raise once a directory is missing.


def test_iter_z_slices_unknown_timepoint_returns_empty(
    handler, make_squid_single_tiff_acq,
):
    folder = make_squid_single_tiff_acq(nt=1)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0_0", acq.channels[0], timepoint="42"))
    assert slices == []


def test_load_full_stack_uses_timepoint(handler, make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(nt=2, nz=3)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0_0", acq.channels[0], timepoint="1")
    assert stack.shape == (3, 8, 10)


def test_legacy_iter_z_slices_still_works(handler, make_single_tiff_acq):
    folder = make_single_tiff_acq(nz=4)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0_0", acq.channels[0]))
    assert len(slices) == 4
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py -v -k "uses_timepoint or unknown_timepoint or legacy_iter_z_slices"
```

Expected: FAIL — current `_tiffs_for` ignores `timepoint`.

- [ ] **Step 3: Update `_tiffs_for` to consume `timepoint` and parse the composite FOV**

In `src/gallery_view/sources/single_tiff.py`:

```python
@staticmethod
def _tiffs_for(
    acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0",
) -> list[str]:
    layout = acq.extra.get("layout", "legacy")
    region, fov_idx = fov.split("_", 1) if "_" in fov else ("0", fov)
    pattern = channel.name.replace(" ", "_")
    t_dir = os.path.join(acq.path, timepoint)
    if layout == "squid":
        # match optional well prefix via *
        candidates = glob.glob(
            os.path.join(t_dir, f"*{region}_{fov_idx}_*_{pattern}.tiff")
        )
        # filter to exact (region, fov) — the * in the well slot can also
        # match nothing or an actual well like 'A1_'.
        files: list[str] = []
        for f in candidates:
            p = parse_squid_filename(os.path.basename(f))
            if p is None:
                continue
            if p["region"] == region and p["fov"] == fov_idx:
                files.append(f)
    else:
        files = glob.glob(
            os.path.join(t_dir, f"current_{fov_idx}_*_{pattern}.tiff")
        )
    files.sort(
        key=lambda f: int(re.search(r"_(\d+)_", os.path.basename(f).split(f"_{pattern}")[0]).group(1))
    )
    return files
```

(The sort key extracts the `<z>` digit from each filename — for squid, it's the last `_<z>_` before the channel; for legacy, the `_<z>_` after `current_<fov>`. The shared regex `_(\d+)_` finds the right one because we anchor on `_<channel>` to bound the search.)

- [ ] **Step 4: Update `iter_z_slices` and `load_full_stack` to forward `timepoint`**

In the same file, ensure these call `_tiffs_for(acq, fov, channel, timepoint=timepoint)`:

```python
def iter_z_slices(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> Iterator[np.ndarray]:
    for f in self._tiffs_for(acq, fov, channel, timepoint=timepoint):
        yield imread(f).astype(np.float32)

def load_full_stack(
    self, acq: Acquisition, fov: str, channel: Channel,
    timepoint: str = "0",
) -> np.ndarray:
    tiffs = self._tiffs_for(acq, fov, channel, timepoint=timepoint)
    if not tiffs:
        raise FileNotFoundError(
            f"No TIFFs for {channel.name!r} at fov={fov!r} t={timepoint!r}"
        )
    return np.stack([imread(f) for f in tiffs])

def iter_full_channel_stacks(
    self, acq: Acquisition, fov: str,
    timepoint: str = "0",
) -> Iterator[tuple[Channel, np.ndarray]]:
    for channel in acq.channels:
        yield channel, self.load_full_stack(acq, fov, channel, timepoint=timepoint)
```

- [ ] **Step 5: Update existing legacy iter test**

Open `tests/sources/test_single_tiff.py` and find any pre-existing assertions that pass `fov="0"` to `iter_z_slices`. Change them to `fov="0_0"` to match the new composite-FOV format. (The legacy fixture now reports `acq.fovs == ["0_0"]`.)

- [ ] **Step 6: Run all new tests**

```bash
.venv/bin/python -m pytest tests/sources/test_single_tiff.py -v
```

Expected: all pass.

- [ ] **Step 7: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Use timepoint when globbing TIFFs in SingleTiffHandler

iter_z_slices / load_full_stack / iter_full_channel_stacks now look in
<acq>/<t>/ instead of always <acq>/0/, and split the composite FOV
'<region>_<fov>' into its components when building the squid glob. The
legacy code path stays at <acq>/0/current_<fov>_… as before.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Update `tests/test_scan.py` and `tests/test_handler_cache_integration.py`

**Files:**
- Modify: `tests/test_scan.py`
- Modify: `tests/test_handler_cache_integration.py`

These tests previously covered the dropped single-channel handler and the now-renamed multi-channel one. Tasks 1-2 already removed direct references; this task adds explicit squid-layout coverage at the scanner and cache-integration layers.

- [ ] **Step 1: Inventory what these files currently test**

```bash
.venv/bin/python -m pytest tests/test_scan.py tests/test_handler_cache_integration.py -v --collect-only 2>&1 | head -30
```

- [ ] **Step 2: Add a squid-scan test in `tests/test_scan.py`**

The existing tests use `scan.ingest(path)` — match that style:

```python
def test_ingest_detects_squid_single_tiff(make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(nt=2, regions=2, fovs_per_region=2)
    acqs = scan.ingest(str(folder))
    assert len(acqs) == 1
    acq = acqs[0]
    assert acq.handler.name == "single_tiff"
    assert acq.timepoints == ["0", "1"]
    assert acq.fovs == ["0_0", "0_1", "1_0", "1_1"]
```

(`scan` is already imported at the top of `tests/test_scan.py`.)

- [ ] **Step 3: Add a multi-timepoint cache-key test in `tests/test_handler_cache_integration.py`**

```python
def test_cache_keys_distinct_across_timepoints(make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(nt=2)
    from gallery_view.sources.single_tiff import SingleTiffHandler
    handler = SingleTiffHandler()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    k_t0 = handler.cache_key(acq, acq.fovs[0], acq.channels[0], timepoint="0")
    k_t1 = handler.cache_key(acq, acq.fovs[0], acq.channels[0], timepoint="1")
    assert k_t0 != k_t1
    assert "/t0/" in k_t0[1]
    assert "/t1/" in k_t1[1]
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_scan.py tests/test_handler_cache_integration.py -v
```

Expected: PASS (handler logic from prior tasks already supports this).

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Test scanner + cache-integration on squid layout

Adds a multi-timepoint, multi-FOV scan test and a per-timepoint
cache-key uniqueness test using the new squid fixture.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Wire timepoint into the gallery row UI

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py`

This is the largest UI change. No automated test — manual verification covers it.

- [ ] **Step 1: Update `RowKey`**

Search for `RowKey` definition and update both fields it carries:

```python
@dataclass(frozen=True)
class RowKey:
    """A row in the gallery: (acq_id, timepoint, fov)."""
    acq_id: int
    timepoint: str
    fov: str
```

- [ ] **Step 2: Update `mip_data` keys throughout `gallery_window.py`**

The dict's key tuple grows from `(acq_id, fov, ch_idx, axis)` to `(acq_id, timepoint, fov, ch_idx, axis)`. Use the editor's project-wide search-and-replace, then visually inspect every match. Locations to check (from earlier grep — lines may shift):

- The `mip_data` field declaration comment near line 84.
- All `for (acq_id, fov, ch_idx, axis), … in self.mip_data` loops.
- All `self.mip_data[(acq_id, fov, ch_idx, axis)] = …` assignments.
- All `self.mip_data.get((acq_id, fov, ch_idx, axis))` lookups.
- The `_on_mip_ready` signal handler and the `MipLoader.mip_ready` signal signature in `loader.py` (the signal carries `(acq_id, fov, ch_idx, …)` — append a `str` for timepoint).

After updating `mip_data` keys, every site that reads or writes them needs the new positional argument. Walk through them in order from top of file.

- [ ] **Step 3: Update `MipLoader.mip_ready` signal**

In `src/gallery_view/loader.py`:

```python
class MipLoader(QThread):
    # acq_id, timepoint, fov, ch_idx, wavelength, channel_mips, shape_zyx
    mip_ready = Signal(int, str, str, int, str, object, object)
```

And in `run()`, when emitting:

```python
self.mip_ready.emit(
    job.acq_id, job.timepoint, job.fov, job.ch_idx,
    job.channel.wavelength, channel_mips, shape_zyx,
)
```

The corresponding gallery handler `_on_mip_ready` in `gallery_window.py` gets a new positional arg:

```python
def _on_mip_ready(
    self, acq_id, timepoint, fov, ch_idx, wl, channel_mips, shape_zyx
) -> None:
    ...
```

Inside, every `(acq_id, fov, …)` key creation gains the `timepoint` slot.

- [ ] **Step 4: Add the Time combo per row**

Find where the FOV combo is built per-acquisition row (search `fov_combo` in `gallery_window.py`). Add a sibling Time combo right next to it, populated from `acq.timepoints`:

```python
time_combo: "QComboBox | None" = None
if len(acq.timepoints) > 1:
    time_combo = QComboBox()
    time_combo.addItems(acq.timepoints)
    time_combo.setCurrentText(acq.selected_timepoint)
    time_combo.currentTextChanged.connect(
        lambda t, aid=acq_id: self._on_timepoint_changed(aid, t)
    )
```

Insert `time_combo` into the row layout at the same position the `fov_combo` lives. Hide it (or skip creation) when `len(acq.timepoints) == 1`.

- [ ] **Step 5: Add `_on_timepoint_changed`**

Mirror `_on_fov_changed`:

```python
def _on_timepoint_changed(self, acq_id: int, timepoint: str) -> None:
    acq = self.flat_acqs[acq_id]
    acq.selected_timepoint = timepoint
    # Re-enqueue MIP jobs for the new (t, fov) and re-render the row
    self._enqueue_jobs_for_acq(acq_id, acq, timepoint, acq.selected_fov)
    self._rebuild_rows()
```

- [ ] **Step 6: Update `_enqueue_jobs_for_acq`**

```python
def _enqueue_jobs_for_acq(
    self, acq_id: int, acq: Acquisition, timepoint: str, fov: str,
) -> None:
    for ch_idx, channel in enumerate(acq.channels):
        self.loader.enqueue(
            Job(
                acq_id=acq_id,
                acq=acq,
                fov=fov,
                channel=channel,
                ch_idx=ch_idx,
                timepoint=timepoint,
            )
        )
```

Update every caller in this file to pass `acq.selected_timepoint` (and the FOV).

- [ ] **Step 7: Manually run and exercise**

```bash
.venv/bin/python -m gallery_view --source /path/to/squid/multi-timepoint/data
```

(Substitute a real path. If you don't have a multi-timepoint dataset, build one with `make_squid_single_tiff_acq(nt=3)` in a Python REPL and point gallery-view at the resulting tmp folder.)

Verify:
- Single-timepoint acquisitions show no Time combo.
- Multi-timepoint acquisitions show a Time combo with each `<t>` entry; switching it re-renders thumbnails for the selected timepoint.
- FOV combo continues to work (regression check).
- 3D viewer still opens and renders the selected timepoint's stack.

- [ ] **Step 8: Run tests**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: all pass (no UI tests, but other tests must continue to pass).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Add Time combo to gallery row UI for multi-timepoint acquisitions

RowKey and mip_data keys grow a timepoint slot. MipLoader.mip_ready
signal carries timepoint. Per-row Time combo appears alongside FOV
when len(acq.timepoints) > 1; switching it re-enqueues MIPs for the
selected (t, fov). Single-timepoint rows are unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit the *Supported formats* table**

Replace the existing table with:

```markdown
| Format          | Layout                                                                |
|-----------------|-----------------------------------------------------------------------|
| `ome_tiff`      | `<acq>/ome_tiff/current_0.ome.tiff` — one OME-TIFF holding every (z, channel) |
| `single_tiff` (squid) | `<acq>/<t>/[<well>_]<region>_<fov>_<z>_<channel>.tiff` — one TIFF per (t, region, fov, z, channel); multi-timepoint supported via per-row Time combo |
| `single_tiff` (legacy) | `<acq>/0/current_<fov>_<z>_<channel>.tiff` — older squid output, still supported |
```

Also update the "Features" section to mention the Time combo for multi-timepoint datasets.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "README: describe single_tiff (squid + legacy) layouts

Drops the single_channel_tiff row, splits single_tiff into squid
and legacy sub-rows so users can see at a glance which layout matches
their data, and adds a one-line note about the Time combo for
multi-timepoint acquisitions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Verification

- [ ] **Final smoke test**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m gallery_view --source /path/to/some/data
```

Open the gallery, exercise the Time combo on a multi-timepoint dataset, open the 3D viewer, close it. Watch RSS in Activity Monitor — the memory-release fix from earlier still applies.

- [ ] **Push to origin** (only after the user reviews and approves the local commits):

```bash
git push origin main
```
