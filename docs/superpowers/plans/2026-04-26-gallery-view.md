# gallery-view Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `gallery-view`, a standalone PyQt6 desktop app for browsing squid-output z-stack acquisitions with drag-and-drop ingestion, three-axis MIP projection, LUT/PNG export, and napari 3D viewer.

**Architecture:** Small Python package, format-handler dispatch (Approach 2 from the spec). Pure-function core (`mips.py`, `cache.py`, `types.py`) → format handlers (`sources/*.py`) → scanner (`scan.py`) → loader thread (`loader.py`) → UI (`ui/*.py`). Tests cover everything below the UI layer; UI is exercised via manual smoke test against real squid data.

**Tech Stack:** Python 3.11+, qtpy + PyQt6, numpy, tifffile, pyyaml, napari, matplotlib, platformdirs, pytest. Reference design: `docs/superpowers/specs/2026-04-26-gallery-view-design.md`.

**Cache key clarification (resolves a small ambiguity in the spec):** the spec's MD5 contains `<source_path>|<fov>|<channel_id>`, but the public `cache.load(src, channel_id)` signature has no `fov` arg. The plan resolves this by having each handler **fold the FOV into its returned `channel_id`** (e.g., `f"fov{fov}/wl_488"`). The `cache` module stays FOV-agnostic; on-disk uniqueness is preserved.

---

## File map

**Source files (created):**

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package config + deps + entry point + pytest config |
| `README.md` | Project overview + install + run |
| `.github/workflows/ci.yml` | pytest on ubuntu/macos × py3.11/3.12 |
| `src/gallery_view/__init__.py` | `__version__` only |
| `src/gallery_view/__main__.py` | `python -m gallery_view` → `main()` |
| `src/gallery_view/cli.py` | argparse: `--source PATH` (repeatable) |
| `src/gallery_view/types.py` | `Channel`, `AxisMip`, `Acquisition` dataclasses + type aliases |
| `src/gallery_view/mips.py` | MIP math: state, accumulate, finalize, percentiles, RGBA |
| `src/gallery_view/cache.py` | `load` / `save` / `save_lut_only` / `clear_all` |
| `src/gallery_view/scan.py` | `ingest(path)` → list[Acquisition] with bounded walk |
| `src/gallery_view/loader.py` | `MipLoader(QThread)`, queue-driven, cancel-by-acq |
| `src/gallery_view/sources/__init__.py` | `HANDLERS` registry, `detect()` |
| `src/gallery_view/sources/base.py` | `FormatHandler` Protocol |
| `src/gallery_view/sources/_squid_common.py` | shared parsers (folder name, params.json, channels.yaml) |
| `src/gallery_view/sources/ome_tiff.py` | `OmeTiffHandler` |
| `src/gallery_view/sources/multi_channel_tiff.py` | `MultiChannelTiffHandler` |
| `src/gallery_view/sources/single_channel_tiff.py` | `SingleChannelTiffHandler` |
| `src/gallery_view/ui/__init__.py` | empty |
| `src/gallery_view/ui/colors.py` | wavelength → RGB / napari colormap maps |
| `src/gallery_view/ui/zoomable_view.py` | `ZoomableImageView` (lifted) |
| `src/gallery_view/ui/sources_panel.py` | sources chip strip widget |
| `src/gallery_view/ui/gallery_window.py` | main window, drop, filters, rows |
| `src/gallery_view/ui/lut_dialog.py` | LUT sliders + PNG export |
| `src/gallery_view/ui/viewer3d.py` | `open_napari()` |

**Test files (created):**

| Path | Purpose |
|---|---|
| `tests/conftest.py` | synthetic acq fixtures (3 format generators) |
| `tests/test_mips.py` | MIP math correctness |
| `tests/test_cache.py` | npz round-trip + LUT sidecar |
| `tests/test_lut_override.py` | sidecar JSON edge cases |
| `tests/test_scan.py` | walk depth, dedupe, hidden, symlinks |
| `tests/sources/__init__.py` | empty |
| `tests/sources/test_ome_tiff.py` | handler unit tests |
| `tests/sources/test_multi_channel_tiff.py` | handler unit tests |
| `tests/sources/test_single_channel_tiff.py` | handler unit tests |

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/gallery_view/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/sources/__init__.py`

- [ ] **Step 1: Create the package skeleton**

```bash
mkdir -p src/gallery_view/sources src/gallery_view/ui tests/sources
touch src/gallery_view/sources/__init__.py src/gallery_view/ui/__init__.py
touch tests/__init__.py tests/sources/__init__.py
```

- [ ] **Step 2: Write `src/gallery_view/__init__.py`**

```python
"""gallery-view: a standalone z-stack gallery viewer for squid output."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gallery-view"
version = "0.1.0"
description = "Standalone gallery viewer for z-stack microscopy acquisitions written by squid"
requires-python = ">=3.11"
dependencies = [
    "qtpy>=2.4",
    "PyQt6>=6.5",
    "numpy>=1.24",
    "tifffile>=2023.7",
    "pyyaml>=6.0",
    "napari>=0.4.18,<0.6",
    "matplotlib>=3.7",
    "platformdirs>=3.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[project.scripts]
gallery-view = "gallery_view.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["src/gallery_view"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 4: Write a minimal `README.md`**

```markdown
# gallery-view

Standalone gallery viewer for z-stack microscopy acquisitions written by [squid](https://github.com/cephla-lab/squid).

## Install

```bash
pip install -e .[dev]
```

## Run

```bash
python -m gallery_view
# or with one or more sources preloaded:
python -m gallery_view --source /path/to/acquisitions
```

Drop folders (acquisition folders or parent folders containing many) onto the
window. The first time an acquisition is seen, full-resolution Z/Y/X MIPs are
computed and cached under `~/Library/Caches/gallery-view/` (macOS) or
`~/.cache/gallery-view/` (Linux). Subsequent loads are instant.

## Tests

```bash
pytest -q
```
```

- [ ] **Step 5: Verify install works**

Run: `pip install -e ".[dev]"`
Expected: completes without error, `pytest --version` works.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md src/gallery_view/__init__.py src/gallery_view/sources/__init__.py src/gallery_view/ui/__init__.py tests/__init__.py tests/sources/__init__.py
git commit -m "Add project scaffolding (pyproject, package skeleton, README)"
```

---

## Task 2: Type definitions (`types.py`)

**Files:**
- Create: `src/gallery_view/types.py`

No tests — these are pure data containers; they're tested transitively by every later test.

- [ ] **Step 1: Write `src/gallery_view/types.py`**

```python
"""Shared dataclasses and type aliases."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .sources.base import FormatHandler

Axis = Literal["z", "y", "x"]
ShapeZYX = tuple[int, int, int]


@dataclass(frozen=True)
class Channel:
    """A single fluorescence channel.

    ``name`` is the squid channel name (e.g. ``"Fluorescence_488_nm_Ex"``).
    ``wavelength`` is the digit string parsed from it, or ``"unknown"``.
    """

    name: str
    wavelength: str


@dataclass(frozen=True)
class AxisMip:
    """One axis's MIP plus its auto-contrast percentiles."""

    mip: np.ndarray  # 2-D float32, full resolution
    p1: float  # 0.5 percentile of mip values
    p999: float  # 99.5 percentile of mip values


# {axis: AxisMip} — one channel's MIPs along Z, Y, X
ChannelMips = dict[Axis, AxisMip]


@dataclass
class Acquisition:
    """One z-stack acquisition discovered by a FormatHandler.

    Fields are populated by ``handler.build()``; ``selected_fov`` is mutable
    and tracks the FOV picker state in the UI. ``extra`` is a per-handler
    escape hatch for format-private fields (no one outside the owning
    handler reads it).
    """

    handler: "FormatHandler"
    path: str
    folder_name: str
    display_name: str
    params: dict
    channels: list[Channel]
    fovs: list[str]
    shape_zyx: ShapeZYX | None = None
    selected_fov: str = "0"
    extra: dict = field(default_factory=dict)
```

- [ ] **Step 2: Smoke check imports work**

Run: `python -c "from gallery_view.types import Acquisition, Channel, AxisMip; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/types.py
git commit -m "Add Acquisition / Channel / AxisMip dataclasses"
```

---

## Task 3: MIP math (`mips.py`) — TDD

**Files:**
- Create: `src/gallery_view/mips.py`
- Test: `tests/test_mips.py`

These are the pure functions lifted from `explorer_ovelle.py:184-204` and `:358-367`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mips.py
"""MIP math: accumulators, finalize, percentile, RGBA."""

import numpy as np
import pytest

from gallery_view import mips


def _gradient_stack(nz=4, ny=5, nx=6):
    """Predictable 3-D test stack: value = z*100 + y*10 + x."""
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    return (z * 100 + y * 10 + x).astype(np.float32)


def test_new_axis_state_is_empty():
    state = mips.new_axis_state()
    assert state["z"] is None
    assert state["y_strips"] == []
    assert state["x_strips"] == []


def test_accumulate_one_slice_initialises_z():
    state = mips.new_axis_state()
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mips.accumulate_axes(img, state)
    assert np.array_equal(state["z"], img)
    assert len(state["y_strips"]) == 1
    assert len(state["x_strips"]) == 1


def test_accumulate_z_mip_matches_numpy_max():
    stack = _gradient_stack()
    state = mips.new_axis_state()
    for z in range(stack.shape[0]):
        mips.accumulate_axes(stack[z], state)
    finalized = mips.finalize(state)
    np.testing.assert_array_equal(finalized["z"], stack.max(axis=0))


def test_finalize_y_and_x_match_numpy_max():
    stack = _gradient_stack()
    state = mips.new_axis_state()
    for z in range(stack.shape[0]):
        mips.accumulate_axes(stack[z], state)
    finalized = mips.finalize(state)
    np.testing.assert_array_equal(finalized["y"], stack.max(axis=1))
    np.testing.assert_array_equal(finalized["x"], stack.max(axis=2).T)


def test_finalize_returns_none_when_no_slices():
    assert mips.finalize(mips.new_axis_state()) is None


def test_axis_data_with_percentiles_returns_axismip_per_axis():
    stack = _gradient_stack()
    state = mips.new_axis_state()
    for z in range(stack.shape[0]):
        mips.accumulate_axes(stack[z], state)
    finalized = mips.finalize(state)
    out = mips.axis_data_with_percentiles(finalized)
    assert set(out.keys()) == {"z", "y", "x"}
    for ax_mip in out.values():
        assert ax_mip.mip.dtype == np.float32
        assert 0 <= ax_mip.p1 <= ax_mip.p999


def test_mip_to_rgba_clips_below_p1_and_above_p999():
    img = np.array([[0.0, 50.0, 200.0]], dtype=np.float32)
    rgba = mips.mip_to_rgba(img, p1=50.0, p999=200.0, color_rgb=(255, 0, 0))
    assert rgba.shape == (1, 3, 4)
    assert rgba.dtype == np.uint8
    # value 0 -> below p1 -> normalized to 0 -> red channel 0
    assert rgba[0, 0, 0] == 0
    # value 200 -> at p999 -> normalized to 1 -> red channel 255
    assert rgba[0, 2, 0] == 255
    # alpha always 255
    assert (rgba[..., 3] == 255).all()


def test_mip_to_rgba_returns_zeros_when_p1_equals_p999():
    img = np.array([[5.0, 5.0]], dtype=np.float32)
    rgba = mips.mip_to_rgba(img, p1=5.0, p999=5.0, color_rgb=(255, 0, 0))
    assert (rgba[..., :3] == 0).all()
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_mips.py -v`
Expected: collection fails with `ModuleNotFoundError: gallery_view.mips`.

- [ ] **Step 3: Write `src/gallery_view/mips.py`**

```python
"""Pure-function MIP math (no Qt, no I/O).

Lifted from aion-explorer/explorer_ovelle.py with no behavioral changes.
"""

from typing import Iterable

import numpy as np

from .types import AxisMip, ChannelMips


def new_axis_state() -> dict:
    """Fresh accumulator for streaming MIPs along Z/Y/X.

    Holds a running ``z`` MIP (2-D) plus per-Z 1-D ``y_strips`` and
    ``x_strips``; ``finalize`` stacks the strips into 2-D Y and X MIPs.
    """
    return {"z": None, "y_strips": [], "x_strips": []}


def accumulate_axes(img: np.ndarray, state: dict) -> None:
    """Update ``state`` with one Z-slice (YX, float32)."""
    if state["z"] is None:
        state["z"] = img.copy()
    else:
        np.maximum(state["z"], img, out=state["z"])
    state["y_strips"].append(img.max(axis=0))  # 1-D along X
    state["x_strips"].append(img.max(axis=1))  # 1-D along Y


def finalize(state: dict) -> dict | None:
    """Turn accumulator state into ``{axis: 2D mip}`` or ``None`` if empty."""
    if state["z"] is None:
        return None
    return {
        "z": state["z"],
        "y": np.stack(state["y_strips"]) if state["y_strips"] else state["z"][:1],
        "x": np.stack(state["x_strips"]).T
        if state["x_strips"]
        else state["z"][:, :1].T,
    }


def axis_data_with_percentiles(axis_mips: dict) -> ChannelMips:
    """Wrap each MIP into an ``AxisMip`` with auto-contrast percentiles."""
    out: ChannelMips = {}
    for ax, mip in axis_mips.items():
        p1 = float(np.percentile(mip, 0.5))
        p999 = float(np.percentile(mip, 99.5))
        out[ax] = AxisMip(mip=mip.astype(np.float32), p1=p1, p999=p999)
    return out


def mip_to_rgba(
    mip: np.ndarray,
    p1: float,
    p999: float,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Normalize ``mip`` to [0, 1] using ``[p1, p999]`` and false-color it.

    Returns an HxWx4 uint8 array with alpha=255.
    """
    if p999 > p1:
        norm = np.clip((mip - p1) / (p999 - p1), 0.0, 1.0)
    else:
        norm = np.zeros_like(mip)
    h, w = norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (norm * color_rgb[0]).astype(np.uint8)
    rgba[:, :, 1] = (norm * color_rgb[1]).astype(np.uint8)
    rgba[:, :, 2] = (norm * color_rgb[2]).astype(np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def stream_mips(slices: Iterable[np.ndarray]) -> tuple[dict | None, int]:
    """Convenience: drain an iterable of YX float32 slices, return finalized
    MIPs and the number of slices seen. Returns ``(None, 0)`` if empty."""
    state = new_axis_state()
    n = 0
    for img in slices:
        accumulate_axes(img, state)
        n += 1
    return finalize(state), n
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_mips.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gallery_view/mips.py tests/test_mips.py
git commit -m "Add MIP math module with accumulator + percentile + RGBA helpers"
```

---

## Task 4: Cache layer (`cache.py`) — TDD

**Files:**
- Create: `src/gallery_view/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache.py
"""Cache I/O: .npz save/load round-trip + .lut.json sidecar."""

from pathlib import Path

import numpy as np
import pytest

from gallery_view import cache
from gallery_view.types import AxisMip


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect the module's cache dir to a tmpdir for every test."""
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def _make_axis_data():
    return {
        "z": AxisMip(mip=np.arange(20, dtype=np.float32).reshape(4, 5),
                    p1=1.0, p999=18.0),
        "y": AxisMip(mip=np.arange(15, dtype=np.float32).reshape(3, 5),
                    p1=0.5, p999=14.0),
        "x": AxisMip(mip=np.arange(12, dtype=np.float32).reshape(3, 4),
                    p1=0.5, p999=11.0),
    }


def test_load_returns_none_when_missing():
    mips, shape = cache.load("/nonexistent/path", "wl_488")
    assert mips is None and shape is None


def test_save_then_load_roundtrip():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    loaded, shape = cache.load("/some/src", "wl_488")
    assert shape == (10, 4, 5)
    for ax in ("z", "y", "x"):
        np.testing.assert_array_equal(loaded[ax].mip, data[ax].mip)
        assert loaded[ax].p1 == pytest.approx(data[ax].p1)
        assert loaded[ax].p999 == pytest.approx(data[ax].p999)


def test_load_returns_none_when_version_mismatch(isolated_cache_dir):
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    # Corrupt: rewrite with a wrong version field
    npz_path = cache._cache_path("/some/src", "wl_488")
    payload = dict(np.load(npz_path))
    payload["version"] = np.int32(cache.CACHE_VERSION + 99)
    np.savez_compressed(npz_path, **payload)
    loaded, shape = cache.load("/some/src", "wl_488")
    assert loaded is None and shape is None


def test_save_lut_only_writes_sidecar_without_touching_npz():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    npz_path = cache._cache_path("/some/src", "wl_488")
    npz_mtime_before = npz_path.stat().st_mtime
    cache.save_lut_only(
        "/some/src",
        "wl_488",
        {"z": (data["z"].mip, 5.0, 50.0), "y": (data["y"].mip, 1.0, 10.0)},
    )
    assert npz_path.stat().st_mtime == npz_mtime_before
    sidecar = cache._lut_override_path("/some/src", "wl_488")
    assert sidecar.exists()


def test_lut_override_applies_on_top_of_cached_defaults():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    cache.save_lut_only(
        "/some/src",
        "wl_488",
        {"z": (data["z"].mip, 100.0, 200.0)},
    )
    loaded, _ = cache.load("/some/src", "wl_488")
    assert loaded["z"].p1 == 100.0
    assert loaded["z"].p999 == 200.0
    # Other axes keep their cached percentiles
    assert loaded["y"].p1 == pytest.approx(data["y"].p1)


def test_save_clears_pre_existing_lut_override():
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    cache.save_lut_only("/some/src", "wl_488",
                        {"z": (data["z"].mip, 100.0, 200.0)})
    sidecar = cache._lut_override_path("/some/src", "wl_488")
    assert sidecar.exists()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))  # fresh compute
    assert not sidecar.exists()


def test_clear_all_removes_cache_dir(isolated_cache_dir):
    data = _make_axis_data()
    cache.save("/some/src", "wl_488", data, (10, 4, 5))
    assert isolated_cache_dir.exists()
    cache.clear_all()
    assert not isolated_cache_dir.exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_cache.py -v`
Expected: collection fails with `ModuleNotFoundError: gallery_view.cache`.

- [ ] **Step 3: Write `src/gallery_view/cache.py`**

```python
"""On-disk MIP cache and per-channel LUT-override sidecar.

Keys are MD5 of ``"<src_path>|<channel_id>"`` truncated to 16 hex chars; the
FOV is encoded into ``channel_id`` by each handler when relevant. The cache
module itself is FOV-agnostic.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import platformdirs

from .types import AxisMip, ChannelMips, ShapeZYX

CACHE_VERSION = 1
AXES = ("z", "y", "x")
CACHE_DIR = str(Path(platformdirs.user_cache_dir("gallery-view")) / "mips")


def _cache_path(src_path: str, channel_id: str) -> Path:
    key = f"{src_path}|{channel_id}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return Path(CACHE_DIR) / f"{h}.npz"


def _lut_override_path(src_path: str, channel_id: str) -> Path:
    return _cache_path(src_path, channel_id).with_suffix(".lut.json")


def _load_lut_override(src_path: str, channel_id: str) -> dict | None:
    p = _lut_override_path(src_path, channel_id)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
        return {ax: (float(v["p1"]), float(v["p999"])) for ax, v in data.items()}
    except (OSError, ValueError, KeyError):
        return None


def load(
    src_path: str, channel_id: str
) -> tuple[ChannelMips | None, ShapeZYX | None]:
    """Read the cached MIPs (with any LUT override applied) for one channel.

    Returns ``(None, None)`` if the file is missing or the version is stale.
    """
    path = _cache_path(src_path, channel_id)
    if not path.exists():
        return None, None
    try:
        data = np.load(path)
        if "version" not in data.files or int(data["version"]) != CACHE_VERSION:
            return None, None
        out: ChannelMips = {}
        for ax in AXES:
            if f"mip_{ax}" not in data.files:
                return None, None
            out[ax] = AxisMip(
                mip=np.asarray(data[f"mip_{ax}"]),
                p1=float(data[f"p1_{ax}"]),
                p999=float(data[f"p999_{ax}"]),
            )
        shape: ShapeZYX | None = None
        if "nz_orig" in data.files:
            shape = (
                int(data["nz_orig"]),
                int(data["ny_orig"]),
                int(data["nx_orig"]),
            )
    except (OSError, ValueError, KeyError):
        return None, None

    overrides = _load_lut_override(src_path, channel_id)
    if overrides:
        for ax, (p1, p999) in overrides.items():
            if ax in out:
                out[ax] = AxisMip(mip=out[ax].mip, p1=p1, p999=p999)
    return out, shape


def save(
    src_path: str,
    channel_id: str,
    axis_data: ChannelMips,
    shape_zyx: ShapeZYX | None = None,
) -> None:
    """Full save: arrays + auto-percentiles + shape. Removes any LUT override."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(src_path, channel_id)
    payload: dict = {"version": np.int32(CACHE_VERSION)}
    for ax, ax_mip in axis_data.items():
        payload[f"mip_{ax}"] = ax_mip.mip.astype(np.float32)
        payload[f"p1_{ax}"] = np.float32(ax_mip.p1)
        payload[f"p999_{ax}"] = np.float32(ax_mip.p999)
    if shape_zyx is not None:
        payload["nz_orig"] = np.int32(shape_zyx[0])
        payload["ny_orig"] = np.int32(shape_zyx[1])
        payload["nx_orig"] = np.int32(shape_zyx[2])
    np.savez_compressed(path, **payload)
    sidecar = _lut_override_path(src_path, channel_id)
    if sidecar.exists():
        try:
            sidecar.unlink()
        except OSError:
            pass


def save_lut_only(
    src_path: str,
    channel_id: str,
    axis_data: dict,
) -> None:
    """Quick LUT-only save. Writes a tiny JSON next to the .npz cache.

    ``axis_data`` is ``{axis: (mip_unused, p1, p999)}`` — same shape the LUT
    dialog hands us; only the floats are written.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {
        ax: {"p1": float(p1), "p999": float(p999)}
        for ax, (_mip, p1, p999) in axis_data.items()
    }
    with _lut_override_path(src_path, channel_id).open("w") as f:
        json.dump(payload, f)


def clear_all() -> None:
    """Delete the cache directory if it exists. Used by 'Clear MIP cache…'."""
    import shutil

    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gallery_view/cache.py tests/test_cache.py
git commit -m "Add MIP cache layer with .npz round-trip and LUT-override sidecar"
```

---

## Task 5: LUT-override JSON edge cases — TDD

**Files:**
- Test: `tests/test_lut_override.py`

(No new source — just defensive coverage of `cache._load_lut_override`.)

- [ ] **Step 1: Write the test**

```python
# tests/test_lut_override.py
"""LUT-override JSON edge cases."""

import pytest

from gallery_view import cache


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def test_load_lut_override_returns_none_when_missing():
    assert cache._load_lut_override("/nope", "wl_488") is None


def test_load_lut_override_returns_none_for_malformed_json(isolated_cache_dir):
    isolated_cache_dir.mkdir(parents=True)
    p = cache._lut_override_path("/src", "wl_488")
    p.write_text("{not json")
    assert cache._load_lut_override("/src", "wl_488") is None


def test_load_lut_override_returns_none_when_keys_missing(isolated_cache_dir):
    isolated_cache_dir.mkdir(parents=True)
    p = cache._lut_override_path("/src", "wl_488")
    p.write_text('{"z": {"p1": 1.0}}')  # no p999
    assert cache._load_lut_override("/src", "wl_488") is None


def test_load_lut_override_partial_axes_ok(isolated_cache_dir):
    cache.save_lut_only(
        "/src",
        "wl_488",
        {"z": (None, 5.0, 50.0)},  # only Z
    )
    out = cache._load_lut_override("/src", "wl_488")
    assert out == {"z": (5.0, 50.0)}
```

- [ ] **Step 2: Run tests and verify they pass (cache.py already supports this)**

Run: `pytest tests/test_lut_override.py -v`
Expected: all tests pass without any source change.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lut_override.py
git commit -m "Test LUT-override JSON edge cases"
```

---

## Task 6: `FormatHandler` Protocol (`sources/base.py`)

**Files:**
- Create: `src/gallery_view/sources/base.py`

No tests yet — Protocol checks happen via the handler tests in Tasks 9-11.

- [ ] **Step 1: Write `src/gallery_view/sources/base.py`**

```python
"""FormatHandler protocol.

Each squid on-disk format gets one handler. A handler is stateless and is
constructed once at import time; the global registry in ``__init__.py``
holds the singletons in priority order.

The cache module is FOV-agnostic; handlers fold the FOV into the
``channel_id`` they return from ``cache_key``.
"""

from typing import Iterator, Protocol

import numpy as np

from ..types import Acquisition, Channel, ShapeZYX


class FormatHandler(Protocol):
    name: str

    def detect(self, folder: str) -> bool:
        """Return True iff ``folder`` is in this handler's format."""
        ...

    def build(self, folder: str, params: dict) -> Acquisition | None:
        """Construct an Acquisition from the folder. Returns None if the
        folder is unparseable for any reason (channels missing, etc.)."""
        ...

    def list_fovs(self, acq: Acquisition) -> list[str]:
        """List of FOV ids available in the acquisition. v1: always ['0']."""
        ...

    def read_shape(
        self, acq: Acquisition, fov: str
    ) -> ShapeZYX | None:
        """Return ``(nz, ny, nx)`` for one FOV without reading the full stack."""
        ...

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> tuple[str, str]:
        """Return ``(src_path_for_hash, channel_id)`` for ``cache.load/save``.

        ``channel_id`` MUST be unique per (acq, fov, channel). Encode the FOV
        into the string when the handler can have more than one FOV.
        """
        ...

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]:
        """Yield each Z-slice as a YX float32 array, in Z-order."""
        ...

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray:
        """Return the full ZYX stack (float32 or native dtype) for napari."""
        ...

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        """Per-channel extras for the PNG export footer.

        Returns ``{}`` when the metadata isn't available. Recognized keys:
          - ``exposure_ms`` (float)
          - ``intensity`` (float, 0-100% laser power)
        """
        ...
```

- [ ] **Step 2: Smoke check imports**

Run: `python -c "from gallery_view.sources.base import FormatHandler; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/sources/base.py
git commit -m "Add FormatHandler protocol"
```

---

## Task 7: Squid metadata helpers (`sources/_squid_common.py`)

**Files:**
- Create: `src/gallery_view/sources/_squid_common.py`

No dedicated tests — exercised via handler tests. These are lifted from `aion-explorer/explorer_ovelle.py:131-163` and trimmed (no Cephla-specific noise filtering, no AION single-channel folder-name parsing — that's in the single-channel handler).

- [ ] **Step 1: Write `src/gallery_view/sources/_squid_common.py`**

```python
"""Shared parsers for squid metadata files. SQUID_TESTED_AGAINST: master @ 2026-04-26."""

import json
import os
import re

import yaml

from ..types import Channel

KNOWN_WAVELENGTHS = frozenset({"405", "488", "561", "638", "730"})


def parse_acquisition_params(folder: str) -> dict | None:
    """Read ``<folder>/acquisition parameters.json`` or return None."""
    p = os.path.join(folder, "acquisition parameters.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def parse_acquisition_channels_yaml(folder: str) -> list[Channel]:
    """Read ``<folder>/acquisition_channels.yaml`` -> list of Channel.

    Returns ``[]`` if the file is absent or unparseable. Channels marked
    ``enabled: false`` are excluded. Output is sorted by wavelength.
    """
    yaml_path = os.path.join(folder, "acquisition_channels.yaml")
    if not os.path.exists(yaml_path):
        return []
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return []
    channels: list[Channel] = []
    for ch in config.get("channels", []):
        if not ch.get("enabled", True):
            continue
        name = ch.get("name", "")
        if not name:
            continue
        wl_match = re.search(r"(\d+)\s*nm", name)
        wl = wl_match.group(1) if wl_match else "unknown"
        channels.append(Channel(name=name, wavelength=wl))
    channels.sort(
        key=lambda c: int(c.wavelength) if c.wavelength.isdigit() else 999
    )
    return channels


def channel_extras_from_yaml(
    folder: str, channel: Channel
) -> dict:
    """Look up exposure_ms / intensity for one channel in
    ``acquisition_channels.yaml``. Returns ``{}`` if not found."""
    yaml_path = os.path.join(folder, "acquisition_channels.yaml")
    if not os.path.exists(yaml_path):
        return {}
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    for ch in config.get("channels", []):
        if ch.get("name") == channel.name or ch.get("name") == channel.name.replace("_", " "):
            return {
                "exposure_ms": ch.get("camera_settings", {}).get("exposure_time_ms"),
                "intensity": ch.get("illumination_settings", {}).get("intensity"),
            }
    return {}


def display_name_for(folder_name: str) -> str:
    """A short human label for a squid acquisition folder.

    Default: the folder name with the trailing timestamp stripped.
    """
    m = re.match(
        r"(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)$", folder_name
    )
    return m.group(1) if m else folder_name


def parse_timestamp(folder_name: str) -> tuple[str, str] | None:
    """Return ``("MM-DD", "HH:MM")`` parsed from the folder-name suffix or None."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", folder_name)
    if not m:
        return None
    return f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}:{m.group(5)}"


def parse_mag(folder_name: str) -> int | None:
    """Pull the leading magnification (e.g. ``25`` from ``25x_…``) or None."""
    m = re.match(r"(\d+)x_", folder_name)
    return int(m.group(1)) if m else None
```

- [ ] **Step 2: Smoke check imports**

Run: `python -c "from gallery_view.sources._squid_common import parse_acquisition_channels_yaml; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/sources/_squid_common.py
git commit -m "Add shared squid metadata parsers"
```

---

## Task 8: Synthetic test fixtures (`tests/conftest.py`)

**Files:**
- Create: `tests/conftest.py`

These are the deterministic on-disk fixtures every handler test consumes.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Synthetic squid-format acquisition fixtures.

Each generator writes a real on-disk acquisition into ``tmp_path`` so handler
tests exercise the actual file-layout detection and reading code paths.
The synthetic data is a deterministic ``z*100 + y*10 + x`` gradient so MIP
results are predictable.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml


def _gradient_3d(nz: int, ny: int, nx: int, channel_offset: int = 0) -> np.ndarray:
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    return (z * 100 + y * 10 + x + channel_offset * 1000).astype(np.uint16)


def _write_params(folder: Path, params: dict) -> None:
    (folder / "acquisition parameters.json").write_text(json.dumps(params))


def _write_channels_yaml(folder: Path, channels: list[dict]) -> None:
    (folder / "acquisition_channels.yaml").write_text(
        yaml.safe_dump({"channels": channels})
    )


@pytest.fixture
def make_ome_tiff_acq(tmp_path):
    """Build an OME-TIFF acquisition with axes ZCYX."""

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
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
        ome_dir = folder / "ome_tiff"
        ome_dir.mkdir()
        nc = len(wavelengths)
        # Build a (nz, nc, ny, nx) ZCYX volume from per-channel 3-D gradients.
        zcyx = np.stack([_gradient_3d(nz, ny, nx, c) for c in range(nc)], axis=1)
        assert zcyx.shape == (nz, nc, ny, nx), zcyx.shape
        tifffile.imwrite(
            ome_dir / "current_0.ome.tiff",
            zcyx,
            metadata={"axes": "ZCYX"},
        )
        return folder

    return _build


@pytest.fixture
def make_multi_channel_tiff_acq(tmp_path):
    """Build a per-Z-per-channel TIFF acquisition (``./0/current_0_<z>_<chname>.tiff``)."""

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
        folder_name="25x_B2_2026-04-26_12-30-00.000000",
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
        fov_dir = folder / "0"
        fov_dir.mkdir()
        for c, wl in enumerate(wavelengths):
            stack = _gradient_3d(nz, ny, nx, c)
            for z in range(nz):
                tifffile.imwrite(
                    fov_dir / f"current_0_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                    stack[z],
                )
        return folder

    return _build


@pytest.fixture
def make_single_channel_tiff_acq(tmp_path):
    """Build a single-channel-per-folder acquisition (one folder per wavelength).

    Returns ``(group_root, [folder_per_wavelength, …])``. The scanner detects
    each wavelength folder and merges them into one acquisition keyed by
    ``(mag, well)``.
    """

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
        well="C3",
        mag=25,
        timestamp="2026-04-26_13-00-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
    ) -> tuple[Path, list[Path]]:
        group_root = tmp_path / f"group_{well}"
        group_root.mkdir()
        folders: list[Path] = []
        for c, wl in enumerate(wavelengths):
            folder = group_root / f"{mag}x_{well}_{wl}_LR1000_{timestamp}"
            folder.mkdir()
            _write_params(folder, {
                "sensor_pixel_size_um": sensor_pixel_size_um,
                "dz(um)": dz_um,
            })
            fov_dir = folder / "0"
            fov_dir.mkdir()
            stack = _gradient_3d(nz, ny, nx, c)
            for z in range(nz):
                tifffile.imwrite(
                    fov_dir / f"current_0_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                    stack[z],
                )
            folders.append(folder)
        return group_root, folders

    return _build
```

- [ ] **Step 2: Smoke check the fixtures import (no tests yet)**

Run: `pytest tests/conftest.py --collect-only -q`
Expected: collection succeeds, 0 tests selected.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "Add synthetic squid acquisition fixtures for handler tests"
```

---

## Task 9: OME-TIFF handler — TDD

**Files:**
- Create: `src/gallery_view/sources/ome_tiff.py`
- Test: `tests/sources/test_ome_tiff.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sources/test_ome_tiff.py
"""OmeTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.ome_tiff import OmeTiffHandler


@pytest.fixture
def handler():
    return OmeTiffHandler()


def test_detect_returns_true_for_ome_tiff_acq(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_returns_false_for_other_formats(
    handler, make_multi_channel_tiff_acq, make_single_channel_tiff_acq
):
    folder = make_multi_channel_tiff_acq()
    assert handler.detect(str(folder)) is False
    _, folders = make_single_channel_tiff_acq()
    assert handler.detect(str(folders[0])) is False


def test_build_populates_acquisition(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(wavelengths=("488", "638"), nz=5)
    params = {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    acq = handler.build(str(folder), params)
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["488", "638"]
    assert acq.handler is handler
    assert acq.fovs == ["0"]


def test_read_shape_matches_written(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(nz=6, ny=12, nx=14)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert handler.read_shape(acq, "0") == (6, 12, 14)


def test_iter_z_slices_count_and_dtype(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 3
    assert slices[0].dtype == np.float32
    assert slices[0].shape == (4, 5)


def test_cache_key_is_stable_and_distinct_per_channel(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(wavelengths=("488", "561"))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    k1 = handler.cache_key(acq, "0", acq.channels[0])
    k2 = handler.cache_key(acq, "0", acq.channels[0])
    k3 = handler.cache_key(acq, "0", acq.channels[1])
    assert k1 == k2
    assert k1 != k3


def test_load_full_stack_shape(handler, make_ome_tiff_acq):
    folder = make_ome_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)


def test_channel_yaml_extras_returns_exposure_and_intensity(
    handler, make_ome_tiff_acq
):
    folder = make_ome_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    extras = handler.channel_yaml_extras(acq, acq.channels[0])
    assert extras["exposure_ms"] == 100.0
    assert extras["intensity"] == 25.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/sources/test_ome_tiff.py -v`
Expected: collection fails with `ModuleNotFoundError: gallery_view.sources.ome_tiff`.

- [ ] **Step 3: Write `src/gallery_view/sources/ome_tiff.py`**

```python
"""OME-TIFF handler for squid output (axes ZCYX, TCYX, or CYX)."""

from typing import Iterator

import numpy as np
from tifffile import TiffFile

from ..types import Acquisition, Channel, ShapeZYX
from . import _squid_common as common

OME_PATH = ("ome_tiff", "current_0.ome.tiff")


class OmeTiffHandler:
    name = "ome_tiff"

    def detect(self, folder: str) -> bool:
        import os

        return os.path.exists(os.path.join(folder, *OME_PATH))

    def build(self, folder: str, params: dict) -> Acquisition | None:
        import os

        ome_path = os.path.join(folder, *OME_PATH)
        channels = common.parse_acquisition_channels_yaml(folder)
        if not channels:
            # Fall back to reading channel count from the OME header
            channels = self._channels_from_ome_header(ome_path)
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
            fovs=["0"],
            extra={"ome_path": ome_path},
        )

    def list_fovs(self, acq: Acquisition) -> list[str]:
        return ["0"]

    def read_shape(self, acq: Acquisition, fov: str) -> ShapeZYX | None:
        try:
            with TiffFile(acq.extra["ome_path"]) as tif:
                s = tif.series[0]
                axes, shape = s.axes, s.shape
                if axes == "CYX":
                    return (1, shape[1], shape[2])
                if axes in ("ZCYX", "TCYX"):
                    return (shape[0], shape[2], shape[3])
        except (OSError, ValueError, IndexError):
            return None
        return None

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> tuple[str, str]:
        return acq.extra["ome_path"], f"fov{fov}/wl_{channel.wavelength}"

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]:
        ome_path = acq.extra["ome_path"]
        ch_idx = self._channel_index(acq, channel)
        with TiffFile(ome_path) as tif:
            s = tif.series[0]
            axes, shape = s.axes, s.shape
            if axes == "CYX":
                yield tif.pages[ch_idx].asarray().astype(np.float32)
                return
            if axes in ("ZCYX", "TCYX"):
                nz, nc = shape[0], shape[1]
                for z in range(nz):
                    yield tif.pages[z * nc + ch_idx].asarray().astype(np.float32)
                return
            raise ValueError(f"Unsupported OME-TIFF axes: {axes}")

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray:
        ch_idx = self._channel_index(acq, channel)
        with TiffFile(acq.extra["ome_path"]) as tif:
            data = tif.series[0].asarray()
            axes = tif.series[0].axes
        if axes == "CYX":
            return data[ch_idx][np.newaxis, :, :]
        if axes in ("ZCYX", "TCYX"):
            return data[:, ch_idx, :, :]
        raise ValueError(f"Unsupported OME-TIFF axes: {axes}")

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        return common.channel_extras_from_yaml(acq.path, channel)

    # ── helpers ──

    @staticmethod
    def _channel_index(acq: Acquisition, channel: Channel) -> int:
        for i, c in enumerate(acq.channels):
            if c.name == channel.name:
                return i
        raise ValueError(f"Channel {channel.name!r} not found in {acq.path}")

    @staticmethod
    def _channels_from_ome_header(ome_path: str) -> list[Channel]:
        try:
            with TiffFile(ome_path) as tif:
                s = tif.series[0]
                axes, shape = s.axes, s.shape
                nc = (
                    shape[1]
                    if axes in ("ZCYX", "TCYX")
                    else (shape[0] if axes == "CYX" else 0)
                )
        except (OSError, ValueError, IndexError):
            return []
        return [Channel(name=f"channel_{i}", wavelength="unknown") for i in range(nc)]
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/sources/test_ome_tiff.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gallery_view/sources/ome_tiff.py tests/sources/test_ome_tiff.py
git commit -m "Add OME-TIFF squid handler with TDD coverage"
```

---

## Task 10: Multi-channel TIFF handler — TDD

**Files:**
- Create: `src/gallery_view/sources/multi_channel_tiff.py`
- Test: `tests/sources/test_multi_channel_tiff.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sources/test_multi_channel_tiff.py
"""MultiChannelTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.multi_channel_tiff import MultiChannelTiffHandler


@pytest.fixture
def handler():
    return MultiChannelTiffHandler()


def test_detect_returns_true_for_multi_channel(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq()
    assert handler.detect(str(folder)) is True


def test_detect_returns_false_for_single_channel_folder(
    handler, make_single_channel_tiff_acq, make_ome_tiff_acq
):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    assert handler.detect(str(folders[0])) is False
    assert handler.detect(str(make_ome_tiff_acq())) is False


def test_build_populates_acquisition(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(wavelengths=("488", "638"))
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert [c.wavelength for c in acq.channels] == ["488", "638"]
    assert acq.fovs == ["0"]


def test_read_shape_matches_written(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(nz=5, ny=8, nx=10)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert handler.read_shape(acq, "0") == (5, 8, 10)


def test_iter_z_slices_count_and_order(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(nz=4, ny=4, nx=4)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 4
    # The synthetic stack has value = z*100 + y*10 + x; z=0 has min, z=3 max
    assert slices[0].max() < slices[3].max()


def test_cache_key_is_stable(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq()
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert (
        handler.cache_key(acq, "0", acq.channels[0])
        == handler.cache_key(acq, "0", acq.channels[0])
    )


def test_load_full_stack_shape(handler, make_multi_channel_tiff_acq):
    folder = make_multi_channel_tiff_acq(nz=3, ny=4, nx=5)
    acq = handler.build(str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/sources/test_multi_channel_tiff.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `src/gallery_view/sources/multi_channel_tiff.py`**

```python
"""Per-Z-per-channel TIFF handler for squid output.

Layout: ``<acq>/0/current_0_<z>_<channel_name>.tiff`` — one TIFF per Z slice
per channel, all in a single FOV directory.
"""

import glob
import os
import re
from typing import Iterator

import numpy as np
from tifffile import TiffFile, imread

from ..types import Acquisition, Channel, ShapeZYX
from . import _squid_common as common


class MultiChannelTiffHandler:
    name = "multi_channel_tiff"

    def detect(self, folder: str) -> bool:
        fov0 = os.path.join(folder, "0")
        if not os.path.isdir(fov0):
            return False
        z0_tiffs = glob.glob(os.path.join(fov0, "current_0_0_*.tiff"))
        # Multi-channel iff > 1 distinct channel name at z=0
        names = set()
        for f in z0_tiffs:
            m = re.search(r"current_0_0_(.+)\.tiff$", os.path.basename(f))
            if m:
                names.add(m.group(1))
        return len(names) > 1

    def build(self, folder: str, params: dict) -> Acquisition | None:
        channels = common.parse_acquisition_channels_yaml(folder)
        if not channels:
            channels = self._channels_from_filenames(folder)
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
            fovs=["0"],
        )

    def list_fovs(self, acq: Acquisition) -> list[str]:
        return ["0"]

    def read_shape(self, acq: Acquisition, fov: str) -> ShapeZYX | None:
        if not acq.channels:
            return None
        tiffs = self._tiffs_for(acq, fov, acq.channels[0])
        if not tiffs:
            return None
        try:
            with TiffFile(tiffs[0]) as tif:
                ny, nx = tif.pages[0].shape
        except (OSError, ValueError):
            return None
        return (len(tiffs), ny, nx)

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> tuple[str, str]:
        return acq.path, f"fov{fov}/{channel.name}"

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]:
        for f in self._tiffs_for(acq, fov, channel):
            yield imread(f).astype(np.float32)

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray:
        tiffs = self._tiffs_for(acq, fov, channel)
        if not tiffs:
            raise FileNotFoundError(
                f"No TIFFs for {channel.name!r} in {acq.path}"
            )
        return np.stack([imread(f) for f in tiffs])

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        return common.channel_extras_from_yaml(acq.path, channel)

    # ── helpers ──

    @staticmethod
    def _tiffs_for(
        acq: Acquisition, fov: str, channel: Channel
    ) -> list[str]:
        pattern = channel.name.replace(" ", "_")
        files = glob.glob(
            os.path.join(acq.path, fov, f"current_{fov}_*_{pattern}.tiff")
        )
        files.sort(
            key=lambda f: int(
                re.search(r"current_\d+_(\d+)_", os.path.basename(f)).group(1)
            )
        )
        return files

    @staticmethod
    def _channels_from_filenames(folder: str) -> list[Channel]:
        z0_tiffs = glob.glob(os.path.join(folder, "0", "current_0_0_*.tiff"))
        out: list[Channel] = []
        for f in sorted(z0_tiffs):
            m = re.search(r"current_0_0_(.+)\.tiff$", os.path.basename(f))
            if not m:
                continue
            name = m.group(1)
            wl_m = re.search(r"(\d+)_nm", name)
            wl = wl_m.group(1) if wl_m else "unknown"
            out.append(Channel(name=name, wavelength=wl))
        out.sort(
            key=lambda c: int(c.wavelength) if c.wavelength.isdigit() else 999
        )
        return out
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/sources/test_multi_channel_tiff.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gallery_view/sources/multi_channel_tiff.py tests/sources/test_multi_channel_tiff.py
git commit -m "Add multi-channel TIFF squid handler with TDD coverage"
```

---

## Task 11: Single-channel TIFF handler — TDD

**Files:**
- Create: `src/gallery_view/sources/single_channel_tiff.py`
- Test: `tests/sources/test_single_channel_tiff.py`

In squid output, single-channel acquisitions are *one folder per channel*. The handler `detect`s individual folders; the **scanner** is responsible for grouping sibling single-channel folders into one logical acquisition (Task 12). For the handler tests, build the handler against a single folder and verify it presents a one-channel `Acquisition`. The grouping is exercised in `tests/test_scan.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sources/test_single_channel_tiff.py
"""SingleChannelTiffHandler unit tests."""

import numpy as np
import pytest

from gallery_view.sources.single_channel_tiff import SingleChannelTiffHandler


@pytest.fixture
def handler():
    return SingleChannelTiffHandler()


def test_detect_returns_true_for_single_channel_folder(
    handler, make_single_channel_tiff_acq
):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    assert handler.detect(str(folders[0])) is True


def test_detect_returns_false_for_other_formats(
    handler, make_ome_tiff_acq, make_multi_channel_tiff_acq
):
    assert handler.detect(str(make_ome_tiff_acq())) is False
    assert handler.detect(str(make_multi_channel_tiff_acq())) is False


def test_build_populates_one_channel(handler, make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    assert acq is not None
    assert len(acq.channels) == 1
    assert acq.channels[0].wavelength == "488"
    assert acq.fovs == ["0"]


def test_iter_z_slices_count(handler, make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("561",), nz=3)
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    slices = list(handler.iter_z_slices(acq, "0", acq.channels[0]))
    assert len(slices) == 3
    assert slices[0].dtype == np.float32


def test_cache_key_is_stable_per_channel_path(
    handler, make_single_channel_tiff_acq
):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    k1 = handler.cache_key(acq, "0", acq.channels[0])
    k2 = handler.cache_key(acq, "0", acq.channels[0])
    assert k1 == k2


def test_load_full_stack_shape(handler, make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",), nz=3, ny=4, nx=5)
    acq = handler.build(str(folders[0]),
                        {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5})
    stack = handler.load_full_stack(acq, "0", acq.channels[0])
    assert stack.shape == (3, 4, 5)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/sources/test_single_channel_tiff.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `src/gallery_view/sources/single_channel_tiff.py`**

```python
"""Single-channel-per-folder TIFF handler for squid output.

Layout: each folder contains exactly one channel's z-stack as
``<folder>/0/current_0_<z>_Fluorescence_<wl>_nm_Ex.tiff``. Sibling folders
of the same ``(mag, well)`` are merged by the scanner into one logical
acquisition; this handler operates on one such folder at a time.
"""

import glob
import os
import re
from typing import Iterator

import numpy as np
from tifffile import TiffFile, imread

from ..types import Acquisition, Channel, ShapeZYX
from . import _squid_common as common


class SingleChannelTiffHandler:
    name = "single_channel_tiff"

    def detect(self, folder: str) -> bool:
        fov0 = os.path.join(folder, "0")
        if not os.path.isdir(fov0):
            return False
        z0_tiffs = glob.glob(os.path.join(fov0, "current_0_0_*.tiff"))
        # Single-channel iff exactly one distinct channel name at z=0
        names = set()
        for f in z0_tiffs:
            m = re.search(r"current_0_0_(.+)\.tiff$", os.path.basename(f))
            if m:
                names.add(m.group(1))
        return len(names) == 1

    def build(self, folder: str, params: dict) -> Acquisition | None:
        channel = self._channel_for(folder)
        if channel is None:
            return None
        folder_name = os.path.basename(folder)
        return Acquisition(
            handler=self,
            path=folder,
            folder_name=folder_name,
            display_name=common.display_name_for(folder_name),
            params=params,
            channels=[channel],
            fovs=["0"],
            extra={"channel_paths": {channel.wavelength: folder}},
        )

    def list_fovs(self, acq: Acquisition) -> list[str]:
        return ["0"]

    def read_shape(self, acq: Acquisition, fov: str) -> ShapeZYX | None:
        if not acq.channels:
            return None
        tiffs = self._tiffs_for(acq, fov, acq.channels[0])
        if not tiffs:
            return None
        try:
            with TiffFile(tiffs[0]) as tif:
                ny, nx = tif.pages[0].shape
        except (OSError, ValueError):
            return None
        return (len(tiffs), ny, nx)

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> tuple[str, str]:
        ch_path = acq.extra["channel_paths"][channel.wavelength]
        return ch_path, f"fov{fov}/Fluorescence_{channel.wavelength}_nm_Ex"

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]:
        for f in self._tiffs_for(acq, fov, channel):
            yield imread(f).astype(np.float32)

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray:
        tiffs = self._tiffs_for(acq, fov, channel)
        if not tiffs:
            raise FileNotFoundError(
                f"No TIFFs for {channel.wavelength}nm in {acq.path}"
            )
        return np.stack([imread(f) for f in tiffs])

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        # Single-channel acquisitions don't ship acquisition_channels.yaml in
        # squid; if a sibling does, the scanner-merged acquisition will route
        # through here per-channel-folder. We try the channel's own folder
        # first, then fall back to the merged acq.path.
        ch_path = acq.extra["channel_paths"].get(channel.wavelength, acq.path)
        extras = common.channel_extras_from_yaml(ch_path, channel)
        if extras:
            return extras
        return common.channel_extras_from_yaml(acq.path, channel)

    # ── helpers ──

    @staticmethod
    def _tiffs_for(
        acq: Acquisition, fov: str, channel: Channel
    ) -> list[str]:
        ch_path = acq.extra["channel_paths"][channel.wavelength]
        pattern = f"Fluorescence_{channel.wavelength}_nm_Ex"
        files = glob.glob(
            os.path.join(ch_path, fov, f"current_{fov}_*_{pattern}.tiff")
        )
        files.sort(
            key=lambda f: int(
                re.search(r"current_\d+_(\d+)_", os.path.basename(f)).group(1)
            )
        )
        return files

    @staticmethod
    def _channel_for(folder: str) -> Channel | None:
        z0_tiffs = glob.glob(os.path.join(folder, "0", "current_0_0_*.tiff"))
        if len(z0_tiffs) != 1:
            return None
        m = re.search(r"current_0_0_Fluorescence_(\d+)_nm_Ex\.tiff$",
                      os.path.basename(z0_tiffs[0]))
        if not m:
            return None
        wl = m.group(1)
        return Channel(name=f"Fluorescence_{wl}_nm_Ex", wavelength=wl)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/sources/test_single_channel_tiff.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gallery_view/sources/single_channel_tiff.py tests/sources/test_single_channel_tiff.py
git commit -m "Add single-channel TIFF squid handler with TDD coverage"
```

---

## Task 12: Handler registry (`sources/__init__.py`)

**Files:**
- Modify: `src/gallery_view/sources/__init__.py`

- [ ] **Step 1: Write a small registry test**

Append to `tests/sources/__init__.py` is fine; but a clean place is a new file:

```python
# tests/sources/test_registry.py
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


def test_detect_returns_single_channel_handler(make_single_channel_tiff_acq):
    _, folders = make_single_channel_tiff_acq(wavelengths=("488",))
    h = sources.detect(str(folders[0]))
    assert h is not None
    assert h.name == "single_channel_tiff"


def test_detect_returns_none_for_unrelated_folder(tmp_path):
    (tmp_path / "random").mkdir()
    assert sources.detect(str(tmp_path / "random")) is None
```

- [ ] **Step 2: Run test and verify it fails**

Run: `pytest tests/sources/test_registry.py -v`
Expected: AttributeError because `sources.detect` doesn't exist yet (the package has only the empty `__init__.py`).

- [ ] **Step 3: Write `src/gallery_view/sources/__init__.py`**

```python
"""Handler registry. ``detect()`` walks ``HANDLERS`` in priority order and
returns the first one whose ``detect()`` returns True (or None).

Order matters: ``ome_tiff`` is checked first because it's identified by a
specific file path; ``multi_channel_tiff`` and ``single_channel_tiff`` both
look at ``./0/current_0_0_*.tiff`` filename patterns and are distinguished
by the *count* of distinct channels at z=0.
"""

from .base import FormatHandler
from .multi_channel_tiff import MultiChannelTiffHandler
from .ome_tiff import OmeTiffHandler
from .single_channel_tiff import SingleChannelTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    MultiChannelTiffHandler(),
    SingleChannelTiffHandler(),
]


def detect(folder: str) -> FormatHandler | None:
    """First-match-wins format detection. Returns None if no handler claims
    the folder."""
    for h in HANDLERS:
        if h.detect(folder):
            return h
    return None
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/sources/ -v`
Expected: all handler + registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gallery_view/sources/__init__.py tests/sources/test_registry.py
git commit -m "Add handler registry with detect() dispatch"
```

---

## Task 13: Folder scanner (`scan.py`) — TDD

**Files:**
- Create: `src/gallery_view/scan.py`
- Test: `tests/test_scan.py`

The scanner has two responsibilities:
1. Walk dropped folders, dispatch to handlers, return `Acquisition`s.
2. **Merge sibling single-channel folders** of the same `(mag, well, timestamp)` into one logical acquisition. (This is what `explorer_ovelle.py:519-544` does.)

The merge is the only piece that needs more than dispatch. Folder-name well parsing is squid-specific so it lives in `_squid_common`.

- [ ] **Step 1: Add a helper to `_squid_common.py` for well parsing**

Modify `src/gallery_view/sources/_squid_common.py` — add at the end:

```python
def parse_mag_well_wl(folder_name: str) -> tuple[int, str, str] | None:
    """Pull (mag, well, wavelength) from a single-channel squid folder name.

    Single-channel folders look like ``25x_C3_488_LR1000_2026-04-26_…``.
    Returns None when any component is missing.
    """
    m = re.match(
        r"(\d+)x_([A-H]\d{1,2})_(\d+)_.*?_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}",
        folder_name,
    )
    if not m:
        return None
    mag, well, wl = int(m.group(1)), m.group(2), m.group(3)
    if wl not in KNOWN_WAVELENGTHS:
        return None
    return mag, well, wl
```

- [ ] **Step 2: Write the failing scanner tests**

```python
# tests/test_scan.py
"""Folder ingestion: walks, dedupe, hidden-skip, single-channel merge."""

import os
from pathlib import Path

import pytest

from gallery_view import scan


def test_ingest_single_acquisition(make_ome_tiff_acq):
    folder = make_ome_tiff_acq()
    acqs = scan.ingest(str(folder))
    assert len(acqs) == 1
    assert acqs[0].handler.name == "ome_tiff"
    assert acqs[0].path == str(folder)


def test_ingest_walks_a_parent_dir(tmp_path, make_ome_tiff_acq):
    # The fixture writes into its own ``tmp_path`` (shared with this test)
    folder_a = make_ome_tiff_acq(folder_name="25x_A1_2026-04-26_12-00-00.000000")
    folder_b = make_ome_tiff_acq(folder_name="25x_A2_2026-04-26_12-30-00.000000")
    acqs = scan.ingest(str(tmp_path))
    assert len(acqs) == 2
    assert {os.path.basename(a.path) for a in acqs} == {folder_a.name, folder_b.name}


def test_ingest_skips_hidden_dirs(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    assert scan.ingest(str(tmp_path)) == []


def test_ingest_dedupes_via_realpath(tmp_path, make_ome_tiff_acq):
    folder = make_ome_tiff_acq()
    seen: set[str] = set()
    a1 = scan.ingest(str(folder), _seen=seen)
    a2 = scan.ingest(str(folder), _seen=seen)
    assert len(a1) == 1
    assert len(a2) == 0


def test_ingest_respects_max_depth(tmp_path, make_ome_tiff_acq):
    # Build the acquisition normally inside tmp_path …
    acq_folder = make_ome_tiff_acq(folder_name="25x_A1_2026-04-26_12-00-00.000000")
    # … then move it to a path 5 levels below tmp_path (exceeds MAX_DEPTH=3).
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    moved = deep / acq_folder.name
    acq_folder.rename(moved)
    assert scan.ingest(str(tmp_path)) == []


def test_ingest_merges_single_channel_siblings(make_single_channel_tiff_acq):
    group_root, folders = make_single_channel_tiff_acq(
        wavelengths=("488", "638"), well="C3"
    )
    acqs = scan.ingest(str(group_root))
    assert len(acqs) == 1
    merged = acqs[0]
    assert {c.wavelength for c in merged.channels} == {"488", "638"}
    assert "488" in merged.extra["channel_paths"]
    assert "638" in merged.extra["channel_paths"]
```

- [ ] **Step 3: Run tests and verify they fail**

Run: `pytest tests/test_scan.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 4: Write `src/gallery_view/scan.py`**

```python
"""Folder ingestion: walk dropped paths, dispatch to handlers, merge
single-channel siblings into logical multi-channel acquisitions."""

import os
from collections import defaultdict
from typing import Iterable

from . import sources
from .sources import _squid_common as common
from .types import Acquisition, Channel

MAX_DEPTH = 3


def ingest(
    path: str,
    *,
    _seen: set[str] | None = None,
    _depth: int = 0,
) -> list[Acquisition]:
    """Walk ``path`` (depth-bounded) and return every recognized acquisition.

    - Depth 0 is the dropped folder itself; the walker descends until it finds
      an acquisition or hits ``MAX_DEPTH``.
    - Hidden folders (``.*``) and symlinks are skipped.
    - Duplicate ingestion of the same realpath is suppressed via ``_seen``.
    - Sibling single-channel folders of matching ``(mag, well)`` are merged
      into one ``Acquisition`` with multiple channels.
    """
    if _seen is None:
        _seen = set()
    real = os.path.realpath(path)
    if real in _seen:
        return []
    _seen.add(real)

    if not os.path.isdir(path) or os.path.islink(path) or _is_hidden(path):
        return []

    handler = sources.detect(path)
    if handler is not None and handler.name != "single_channel_tiff":
        params = common.parse_acquisition_params(path) or {}
        acq = handler.build(path, params)
        return [acq] if acq is not None else []

    if _depth >= MAX_DEPTH:
        return []

    out: list[Acquisition] = []
    single_channel_buckets: dict[tuple[int, str], list[str]] = defaultdict(list)

    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if _is_hidden(sub) or not os.path.isdir(sub) or os.path.islink(sub):
            continue
        sub_handler = sources.detect(sub)
        if sub_handler is None:
            out.extend(ingest(sub, _seen=_seen, _depth=_depth + 1))
            continue
        if sub_handler.name == "single_channel_tiff":
            mag_well = common.parse_mag_well_wl(os.path.basename(sub))
            if mag_well is None:
                # Treat as a normal one-channel acquisition
                params = common.parse_acquisition_params(sub) or {}
                acq = sub_handler.build(sub, params)
                if acq is not None:
                    out.append(acq)
                continue
            mag, well, _wl = mag_well
            single_channel_buckets[(mag, well)].append(sub)
        else:
            params = common.parse_acquisition_params(sub) or {}
            acq = sub_handler.build(sub, params)
            if acq is not None:
                out.append(acq)

    out.extend(_merge_single_channel_siblings(single_channel_buckets))
    return out


def _merge_single_channel_siblings(
    buckets: dict[tuple[int, str], list[str]],
) -> Iterable[Acquisition]:
    handler = next(
        h for h in sources.HANDLERS if h.name == "single_channel_tiff"
    )
    for (mag, well), folders in buckets.items():
        # One acquisition per (mag, well); pick the latest folder per
        # wavelength when multiple acquisitions exist for the same channel.
        by_wl: dict[str, str] = {}
        for f in folders:
            mag_well = common.parse_mag_well_wl(os.path.basename(f))
            if mag_well is None:
                continue
            _, _, wl = mag_well
            # "Latest" by lexicographic folder name (squid timestamps sort).
            if wl not in by_wl or os.path.basename(f) > os.path.basename(by_wl[wl]):
                by_wl[wl] = f
        if not by_wl:
            continue
        first_folder = sorted(by_wl.values())[0]
        params = common.parse_acquisition_params(first_folder) or {}
        channels = [
            Channel(name=f"Fluorescence_{wl}_nm_Ex", wavelength=wl)
            for wl in sorted(by_wl, key=int)
        ]
        channel_paths = {wl: by_wl[wl] for wl in by_wl}
        folder_name = os.path.basename(first_folder)
        merged = Acquisition(
            handler=handler,
            path=first_folder,
            folder_name=folder_name,
            display_name=f"{mag}x {well}",
            params=params,
            channels=channels,
            fovs=["0"],
            extra={"channel_paths": channel_paths, "merged_mag": mag, "merged_well": well},
        )
        yield merged


def _is_hidden(path: str) -> bool:
    return os.path.basename(path).startswith(".")
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `pytest tests/test_scan.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/gallery_view/scan.py src/gallery_view/sources/_squid_common.py tests/test_scan.py
git commit -m "Add folder scanner with single-channel merge"
```

---

## Task 14: UI color maps (`ui/colors.py`)

**Files:**
- Create: `src/gallery_view/ui/colors.py`

No tests — pure constants.

- [ ] **Step 1: Write `src/gallery_view/ui/colors.py`**

```python
"""Wavelength → display color maps (lifted from explorer_ovelle.py)."""

CHANNEL_ORDER: list[str] = ["405", "488", "561", "638", "730"]

CHANNEL_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "405": (80, 120, 255),
    "488": (0, 255, 80),
    "561": (255, 255, 0),
    "638": (255, 50, 50),
    "730": (255, 0, 255),
}

NAPARI_COLORMAPS: dict[str, str] = {
    "405": "blue",
    "488": "green",
    "561": "yellow",
    "638": "red",
    "730": "magenta",
}

DEFAULT_RGB: tuple[int, int, int] = (200, 200, 200)
DEFAULT_NAPARI_CMAP: str = "gray"


def rgb_for(wavelength: str) -> tuple[int, int, int]:
    return CHANNEL_COLORS_RGB.get(wavelength, DEFAULT_RGB)


def napari_cmap_for(wavelength: str) -> str:
    return NAPARI_COLORMAPS.get(wavelength, DEFAULT_NAPARI_CMAP)
```

- [ ] **Step 2: Commit**

```bash
git add src/gallery_view/ui/colors.py
git commit -m "Add wavelength → display color maps"
```

---

## Task 15: Loader thread (`loader.py`)

**Files:**
- Create: `src/gallery_view/loader.py`

No unit tests — Qt thread tests are deferred (spec §12.3). Smoke-tested via the manual run in Task 25.

- [ ] **Step 1: Write `src/gallery_view/loader.py`**

```python
"""Long-lived MIP loader thread.

Job queue holds ``(acq_id, acq, fov, channel)`` tuples. ``cancel(acq_id)``
prunes pending jobs for that acq from the queue; in-flight jobs run to
completion. Emits ``mip_ready`` per finished channel and ``progress`` after
each step.
"""

import queue
from dataclasses import dataclass

from qtpy.QtCore import QThread, Signal

from . import cache, mips
from .types import Acquisition, AxisMip, Channel, ChannelMips, ShapeZYX


@dataclass
class Job:
    acq_id: int
    acq: Acquisition
    fov: str
    channel: Channel
    ch_idx: int


class MipLoader(QThread):
    # acq_id, fov, ch_idx, wavelength, channel_mips, shape_zyx
    mip_ready = Signal(int, str, int, str, object, object)
    # done_total, queued_total, message
    progress = Signal(int, int, str)
    idle = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._done = 0
        self._enqueued = 0
        self._cancelled_acqs: set[int] = set()
        self._stop = False

    # ── public API (call from GUI thread) ──

    def enqueue(self, job: Job) -> None:
        self._enqueued += 1
        self._queue.put(job)

    def cancel(self, acq_id: int) -> None:
        """Drop pending jobs for this acq. In-flight job runs to completion."""
        self._cancelled_acqs.add(acq_id)

    def stop(self) -> None:
        self._stop = True
        self._queue.put(None)  # wake the worker

    # ── thread loop ──

    def run(self) -> None:
        while not self._stop:
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._enqueued and self._done >= self._enqueued:
                    self.idle.emit()
                continue
            if job is None:
                break
            if job.acq_id in self._cancelled_acqs:
                self._done += 1
                self._emit_progress(f"skipped (cancelled)")
                continue
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001
                self._done += 1
                self._emit_progress(
                    f"failed {job.channel.wavelength}nm — {job.acq.display_name}: {exc}"
                )

    def _process(self, job: Job) -> None:
        src, ch_id = job.acq.handler.cache_key(job.acq, job.fov, job.channel)
        cached, shape = cache.load(src, ch_id)
        if cached is not None:
            self._emit_ready(job, cached, shape)
            self._done += 1
            self._emit_progress(
                f"{job.channel.wavelength}nm cached — {job.acq.display_name}"
            )
            return
        self._emit_progress(
            f"computing {job.channel.wavelength}nm MIP — {job.acq.display_name}"
        )
        state = mips.new_axis_state()
        n = 0
        ny = nx = 0
        for slice_yx in job.acq.handler.iter_z_slices(job.acq, job.fov, job.channel):
            mips.accumulate_axes(slice_yx, state)
            n += 1
            ny, nx = slice_yx.shape
        finalized = mips.finalize(state)
        if finalized is None:
            self._done += 1
            self._emit_progress(
                f"empty {job.channel.wavelength}nm — {job.acq.display_name}"
            )
            return
        channel_mips = mips.axis_data_with_percentiles(finalized)
        shape_zyx: ShapeZYX = (n, ny, nx)
        cache.save(src, ch_id, channel_mips, shape_zyx)
        self._emit_ready(job, channel_mips, shape_zyx)
        self._done += 1
        self._emit_progress(
            f"{job.channel.wavelength}nm computed — {job.acq.display_name}"
        )

    def _emit_ready(
        self, job: Job, channel_mips: ChannelMips, shape: ShapeZYX | None
    ) -> None:
        self.mip_ready.emit(
            job.acq_id, job.fov, job.ch_idx,
            job.channel.wavelength, channel_mips, shape,
        )

    def _emit_progress(self, message: str) -> None:
        self.progress.emit(self._done, self._enqueued, message)
```

- [ ] **Step 2: Smoke check imports**

Run: `python -c "from gallery_view.loader import MipLoader, Job; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/loader.py
git commit -m "Add long-lived MIP loader thread"
```

---

## Task 16: Zoomable image view (`ui/zoomable_view.py`)

**Files:**
- Create: `src/gallery_view/ui/zoomable_view.py`

Lifted from `aion-explorer/explorer_ovelle.py:683-765` essentially intact.

- [ ] **Step 1: Write `src/gallery_view/ui/zoomable_view.py`**

```python
"""Zoomable / pannable image view used by the LUT dialog."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QPixmap, QTransform
from qtpy.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QPushButton,
)


class ZoomableImageView(QGraphicsView):
    """Fixed-size view with mouse-wheel zoom (anchored under cursor),
    click-drag pan, and an overlay 'home' button that fits to view."""

    ZOOM_STEP = 1.25

    def __init__(self, size: int, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFixedSize(size, size)
        self.setStyleSheet(
            "QGraphicsView { background-color: #111; border: 1px solid #333; border-radius: 4px; }"
        )

        self._home_btn = QPushButton("⌂", self)
        self._home_btn.setFixedSize(26, 26)
        self._home_btn.setCursor(Qt.PointingHandCursor)
        self._home_btn.setToolTip("Fit to view")
        self._home_btn.setStyleSheet(
            "QPushButton { background-color: rgba(0, 0, 0, 160); color: white;"
            " border: 1px solid rgba(255,255,255,80); border-radius: 13px;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: rgba(60, 60, 60, 200); }"
        )
        self._home_btn.clicked.connect(self.fit)
        self._home_btn.hide()
        self._home_btn.move(size - 32, 6)

    def set_pixmap(self, pixmap: QPixmap, y_scale: float = 1.0) -> None:
        was_empty = self._pixmap_item.pixmap().isNull()
        self._pixmap_item.setPixmap(pixmap)
        transform = QTransform()
        transform.scale(1.0, max(y_scale, 1e-6))
        self._pixmap_item.setTransform(transform)
        self._scene.setSceneRect(self._pixmap_item.sceneBoundingRect())
        if was_empty:
            self.fit()
        self._update_home_visibility()

    def fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._update_home_visibility()

    def _fit_scale(self) -> float:
        if self._pixmap_item.pixmap().isNull():
            return 1.0
        rect = self._pixmap_item.sceneBoundingRect()
        if rect.width() == 0 or rect.height() == 0:
            return 1.0
        vw, vh = self.viewport().width(), self.viewport().height()
        return min(vw / rect.width(), vh / rect.height())

    def _update_home_visibility(self) -> None:
        zoomed_in = self.transform().m11() > self._fit_scale() * 1.001
        self._home_btn.setVisible(zoomed_in)
        if zoomed_in:
            self._home_btn.raise_()

    def wheelEvent(self, event) -> None:
        zooming_in = event.angleDelta().y() > 0
        factor = self.ZOOM_STEP if zooming_in else 1 / self.ZOOM_STEP
        if not zooming_in:
            current = self.transform().m11()
            if current * factor <= self._fit_scale():
                self.fit()
                return
        self.scale(factor, factor)
        self._update_home_visibility()
```

- [ ] **Step 2: Smoke check imports**

Run: `QT_QPA_PLATFORM=offscreen python -c "from gallery_view.ui.zoomable_view import ZoomableImageView; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/ui/zoomable_view.py
git commit -m "Add ZoomableImageView (lifted from aion-explorer)"
```

---

## Task 17: Sources chip strip (`ui/sources_panel.py`)

**Files:**
- Create: `src/gallery_view/ui/sources_panel.py`

- [ ] **Step 1: Write `src/gallery_view/ui/sources_panel.py`**

```python
"""Top-of-window strip showing dropped source roots.

Each source is a chip with the basename, the acq count, and an × to remove.
The right-most ``+`` chip opens a folder picker (handled by the gallery
window via the ``add_requested`` signal)."""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)


class SourcesPanel(QScrollArea):
    remove_requested = Signal(str)  # source path
    add_requested = Signal(str)     # source path picked from dialog

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedHeight(38)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self._content = QWidget()
        self._row = QHBoxLayout(self._content)
        self._row.setContentsMargins(6, 4, 6, 4)
        self._row.setSpacing(6)
        self.setWidget(self._content)

        self._add_btn = QPushButton("+ Add folder…")
        self._add_btn.setCursor(Qt.PointingHandCursor)
        self._add_btn.setStyleSheet(
            "QPushButton { background-color: #333; color: #ccc; border: 1px dashed #555;"
            " border-radius: 12px; padding: 2px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #444; }"
        )
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._row.addWidget(self._add_btn)
        self._row.addStretch()

        self._chips: dict[str, QWidget] = {}

    def set_sources(self, sources: list[tuple[str, int]]) -> None:
        """Replace the chip set. ``sources`` is a list of ``(path, acq_count)``."""
        for chip in list(self._chips.values()):
            self._row.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()
        # Insert new chips before the trailing add button + stretch
        for i, (path, count) in enumerate(sources):
            chip = self._make_chip(path, count)
            self._row.insertWidget(i, chip)
            self._chips[path] = chip

    def _make_chip(self, path: str, count: int) -> QWidget:
        import os

        chip = QWidget()
        h = QHBoxLayout(chip)
        h.setContentsMargins(8, 2, 4, 2)
        h.setSpacing(4)
        chip.setStyleSheet(
            "QWidget { background-color: #2d5aa0; color: white; border-radius: 12px; }"
            "QLabel { color: white; font-size: 11px; }"
        )
        chip.setToolTip(path)

        label = QLabel(f"{os.path.basename(path) or path}  ({count})")
        h.addWidget(label)

        rm = QPushButton("×")
        rm.setFixedSize(16, 16)
        rm.setCursor(Qt.PointingHandCursor)
        rm.setStyleSheet(
            "QPushButton { color: white; background: transparent; border: none;"
            " font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { color: #ffcccc; }"
        )
        rm.clicked.connect(lambda _, p=path: self.remove_requested.emit(p))
        h.addWidget(rm)
        return chip

    def _on_add_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add folder")
        if path:
            self.add_requested.emit(path)
```

- [ ] **Step 2: Commit**

```bash
git add src/gallery_view/ui/sources_panel.py
git commit -m "Add sources chip strip with add/remove signals"
```

---

## Task 18: Gallery window scaffolding (`ui/gallery_window.py`)

**Files:**
- Create: `src/gallery_view/ui/gallery_window.py`

This task builds the window shell only: drag-drop, sources strip, filter row, display row, empty scroll area, status bar, loader thread wiring. **Row rendering, FOV picker, filtering logic, and Settings menu are added in Tasks 19-22.**

- [ ] **Step 1: Write the initial `src/gallery_view/ui/gallery_window.py`**

```python
"""Gallery window: drag-drop ingestion, sources strip, scroll area, loader."""

import os
from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import scan
from ..loader import Job, MipLoader
from ..types import Acquisition

THIN_Z_THRESHOLD = 5
THUMB_SIZE_PRESETS = [("Small", 80), ("Medium", 160), ("Large", 320)]
DEFAULT_THUMB_SIZE = 160


@dataclass
class Source:
    path: str
    acq_ids: list[int]


class GalleryWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gallery-view")
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            "QMainWindow { background-color: #1a1a1a; color: white; }"
            "QLabel { color: white; }"
        )

        self.acquisitions: list[Acquisition] = []
        self.sources: list[Source] = []

        # Loader thread (long-lived)
        self.loader = MipLoader()
        self.loader.mip_ready.connect(self._on_mip_ready)
        self.loader.progress.connect(self._on_progress)
        self.loader.idle.connect(self._on_idle)
        self.loader.start()

        self._build_ui()

    # ── ui scaffolding ──

    def _build_ui(self) -> None:
        from .sources_panel import SourcesPanel

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self.sources_panel = SourcesPanel()
        self.sources_panel.add_requested.connect(self._add_source)
        self.sources_panel.remove_requested.connect(self._remove_source)
        self.sources_panel.hide()  # appears after first source
        layout.addWidget(self.sources_panel)

        self._build_filter_row(layout)
        self._build_display_row(layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setSpacing(4)
        self.scroll_layout.addStretch()
        self.scroll.setWidget(self.scroll_content)
        layout.addWidget(self.scroll, stretch=1)

        self.empty_overlay = QLabel("Drop folders here to begin")
        self.empty_overlay.setAlignment(Qt.AlignCenter)
        self.empty_overlay.setStyleSheet(
            "QLabel { color: #666; font-size: 18px;"
            " border: 2px dashed #444; border-radius: 12px; padding: 60px; }"
        )
        self.scroll_layout.insertWidget(0, self.empty_overlay)

        self.status = QLabel("Drop folders to begin.")
        self.status.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
        layout.addWidget(self.status)

    def _build_filter_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        mag_label = QLabel("Magnification:")
        mag_label.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(mag_label)
        self.mag_row_layout = QHBoxLayout()
        self.mag_row_layout.setSpacing(6)
        row.addLayout(self.mag_row_layout)
        row.addSpacing(16)

        self.hide_thin_btn = QPushButton(f"Hide thin (Z<{THIN_Z_THRESHOLD})")
        self.hide_thin_btn.setCheckable(True)
        self.hide_thin_btn.setChecked(True)
        self.hide_thin_btn.setCursor(Qt.PointingHandCursor)
        self.hide_thin_btn.setStyleSheet(self._toggle_style())
        self.hide_thin_btn.toggled.connect(self._refresh_visibility)
        row.addWidget(self.hide_thin_btn)

        row.addStretch()
        layout.addLayout(row)

        self.mag_checkboxes: dict[int, QCheckBox] = {}

    def _build_display_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        proj_lbl = QLabel("Project:")
        proj_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(proj_lbl)

        self.axis_btn_group = QButtonGroup(self)
        self.axis_btn_group.setExclusive(True)
        self.axis_buttons: dict[str, QPushButton] = {}
        for ax, label in [("z", "XY"), ("y", "XZ"), ("x", "YZ")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedSize(36, 22)
            btn.setStyleSheet(self._toggle_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.setChecked(ax == "z")
            btn.clicked.connect(lambda _, a=ax: self._set_axis(a))
            self.axis_btn_group.addButton(btn)
            self.axis_buttons[ax] = btn
            row.addWidget(btn)
        self.view_axis: str = "z"

        row.addSpacing(16)
        size_lbl = QLabel("Thumbnail size:")
        size_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(size_lbl)
        self.size_combo = QComboBox()
        for label, size in THUMB_SIZE_PRESETS:
            self.size_combo.addItem(label, size)
        self.size_combo.setCurrentIndex(
            next(i for i, (_, s) in enumerate(THUMB_SIZE_PRESETS) if s == DEFAULT_THUMB_SIZE)
        )
        self.thumb_size: int = DEFAULT_THUMB_SIZE
        self.size_combo.currentIndexChanged.connect(
            lambda i: self._set_thumb_size(self.size_combo.itemData(i))
        )
        row.addWidget(self.size_combo)
        row.addStretch()
        layout.addLayout(row)

    @staticmethod
    def _toggle_style() -> str:
        return (
            "QPushButton { background-color: #333; color: #ccc; border: 1px solid #444;"
            " border-radius: 3px; font-size: 10px; padding: 2px 8px; }"
            "QPushButton:hover { background-color: #444; }"
            "QPushButton:checked { background-color: #2d5aa0; color: white; border-color: #3a6fc0; }"
        )

    # ── drag-drop ──

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local and os.path.isdir(local):
                self._add_source(local)
        event.acceptProposedAction()

    # ── source management ──

    def _add_source(self, path: str) -> None:
        real = os.path.realpath(path)
        if any(os.path.realpath(s.path) == real for s in self.sources):
            return  # already loaded
        new_acqs = scan.ingest(path)
        if not new_acqs:
            self.status.setText(f"No acquisitions found in {path}")
            return
        first_id = len(self.acquisitions)
        for acq in new_acqs:
            self.acquisitions.append(acq)
        ids = list(range(first_id, first_id + len(new_acqs)))
        self.sources.append(Source(path=path, acq_ids=ids))
        self.empty_overlay.hide()
        self.sources_panel.show()
        self._sync_sources_panel()
        self._rebuild_mag_filter()
        self._rebuild_rows()
        for acq_id, acq in zip(ids, new_acqs):
            self._enqueue_jobs_for_acq(acq_id, acq, acq.selected_fov)

    def _remove_source(self, path: str) -> None:
        # Find the source and cancel its acq jobs
        target = next((s for s in self.sources if s.path == path), None)
        if target is None:
            return
        for acq_id in target.acq_ids:
            self.loader.cancel(acq_id)
        # Mark acquisitions as removed by clearing them out of the list.
        # We keep the list dense by rebuilding (acq_ids in other sources stay
        # valid only if we DON'T renumber); instead we mark with None and
        # filter at render time.
        for acq_id in target.acq_ids:
            self.acquisitions[acq_id] = None  # type: ignore[assignment]
        self.sources.remove(target)
        self._sync_sources_panel()
        self._rebuild_mag_filter()
        self._rebuild_rows()
        if not self.sources:
            self.empty_overlay.show()
            self.sources_panel.hide()

    def _sync_sources_panel(self) -> None:
        self.sources_panel.set_sources([(s.path, len(s.acq_ids)) for s in self.sources])

    def _enqueue_jobs_for_acq(self, acq_id: int, acq: Acquisition, fov: str) -> None:
        for ch_idx, channel in enumerate(acq.channels):
            self.loader.enqueue(
                Job(acq_id=acq_id, acq=acq, fov=fov, channel=channel, ch_idx=ch_idx)
            )

    # ── stub hooks filled in by later tasks ──

    def _rebuild_mag_filter(self) -> None:
        # Filled in by Task 20.
        pass

    def _rebuild_rows(self) -> None:
        # Filled in by Task 19.
        pass

    def _refresh_visibility(self) -> None:
        # Filled in by Task 20.
        pass

    def _set_axis(self, axis: str) -> None:
        # Filled in by Task 19.
        self.view_axis = axis

    def _set_thumb_size(self, size: int) -> None:
        # Filled in by Task 19.
        self.thumb_size = size

    # ── loader callbacks (stubs filled in by Task 19) ──

    def _on_mip_ready(self, acq_id, fov, ch_idx, wavelength, channel_mips, shape) -> None:
        pass

    def _on_progress(self, done, queued, message) -> None:
        self.status.setText(f"{done}/{queued} channels — {message}")

    def _on_idle(self) -> None:
        loaded = sum(1 for a in self.acquisitions if a is not None)
        self.status.setText(f"{loaded} acquisitions loaded")

    # ── shutdown ──

    def closeEvent(self, event) -> None:  # noqa: N802
        self.loader.stop()
        self.loader.wait(2000)
        super().closeEvent(event)
```

- [ ] **Step 2: Smoke check imports**

Run: `QT_QPA_PLATFORM=offscreen python -c "from gallery_view.ui.gallery_window import GalleryWindow; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py
git commit -m "Add gallery window scaffolding (drag-drop, sources, loader wiring)"
```

---

## Task 19: Row rendering, thumbnails, FOV picker, axis switching

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py`

This task fills in `_rebuild_rows`, `_on_mip_ready`, `_set_axis`, `_set_thumb_size`, plus row helpers.

- [ ] **Step 1: Add a `_RowKey` class + per-row state at the top of `gallery_window.py`**

After the `Source` dataclass, add:

```python
@dataclass
class RowKey:
    """A row in the gallery: (acq_id, fov)."""
    acq_id: int
    fov: str


@dataclass
class RowWidgets:
    container: QWidget
    mag_lbl: "QLabel"
    time_lbl: "QLabel"
    name_lbl: "QLabel"
    thumb_labels: dict[int, "QLabel"]   # ch_idx -> data thumb
    thumb_columns: dict[str, "QLabel"]  # wavelength -> the cell currently rendered (data or placeholder)
    fov_combo: "QComboBox | None"
```

- [ ] **Step 2: Add the channel-color and physical-aspect imports near the top**

```python
from .colors import CHANNEL_ORDER, rgb_for
from ..mips import mip_to_rgba
import numpy as np
from qtpy.QtGui import QImage, QPixmap
```

- [ ] **Step 3: Add per-instance state to `__init__`**

After `self.sources: list[Source] = []` add:

```python
        self.row_keys: list[RowKey] = []
        self.row_widgets: dict[tuple[int, str], RowWidgets] = {}
        self.expanded_fov_mode: bool = False
        self.square_footprint: bool = False
        # (acq_id, fov, ch_idx, axis) -> AxisMip
        self.mip_data: dict[tuple[int, str, int, str], "AxisMip"] = {}
```

Also add the import:

```python
from ..types import AxisMip
```

- [ ] **Step 4: Implement `_rebuild_rows`, replacing the stub**

```python
    def _rebuild_rows(self) -> None:
        # Clear existing row widgets
        for rw in self.row_widgets.values():
            self.scroll_layout.removeWidget(rw.container)
            rw.container.deleteLater()
        self.row_widgets.clear()
        self.row_keys = []

        # Build the row key list per the current expansion mode
        for acq_id, acq in enumerate(self.acquisitions):
            if acq is None:
                continue
            if self.expanded_fov_mode:
                for fov in acq.fovs:
                    self.row_keys.append(RowKey(acq_id, fov))
            else:
                self.row_keys.append(RowKey(acq_id, acq.selected_fov))

        # Compute the active wavelength column set (drop any column no row uses)
        active_wls: list[str] = self._active_wavelengths()

        # Insert one row widget per RowKey, before the trailing stretch
        insert_at = self.scroll_layout.count() - 1  # before stretch
        for key in self.row_keys:
            acq = self.acquisitions[key.acq_id]
            row = self._make_row_widget(key, acq, active_wls)
            self.row_widgets[(key.acq_id, key.fov)] = row
            self.scroll_layout.insertWidget(insert_at, row.container)
            insert_at += 1

        # Re-render any thumbs we already have data for
        for (acq_id, fov, ch_idx, axis), ax_mip in list(self.mip_data.items()):
            if axis != self.view_axis:
                continue
            self._render_thumb(acq_id, fov, ch_idx, ax_mip)

        self._refresh_visibility()
        self._apply_label_sizes()

    def _active_wavelengths(self) -> list[str]:
        seen: set[str] = set()
        for acq in self.acquisitions:
            if acq is None:
                continue
            for ch in acq.channels:
                seen.add(ch.wavelength)
        ordered = [wl for wl in CHANNEL_ORDER if wl in seen]
        extras = sorted(
            [wl for wl in seen if wl not in CHANNEL_ORDER],
            key=lambda w: int(w) if w.isdigit() else 999,
        )
        return ordered + extras
```

- [ ] **Step 5: Implement `_make_row_widget` (the per-row builder)**

Add after `_active_wavelengths`:

```python
    def _make_row_widget(self, key, acq, active_wls):
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QLabel, QPushButton, QComboBox
        from ..sources._squid_common import parse_timestamp, parse_mag

        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(4)

        mag = parse_mag(acq.folder_name) or acq.params.get("mag") or "?"
        mag_lbl = QLabel(f"{mag}x" if isinstance(mag, int) else str(mag))
        mag_lbl.setFixedWidth(30)
        mag_lbl.setStyleSheet("color: #ccc; font-size: 11px; font-weight: bold;")
        h.addWidget(mag_lbl)

        ts = parse_timestamp(acq.folder_name)
        time_lbl = QLabel(f"{ts[0]}\n{ts[1]}" if ts else "")
        time_lbl.setFixedWidth(40)
        time_lbl.setAlignment(Qt.AlignCenter)
        time_lbl.setStyleSheet("color: #888; font-size: 9px;")
        h.addWidget(time_lbl)

        if self.expanded_fov_mode and len(acq.fovs) > 1:
            fov_lbl = QLabel(f"FOV {key.fov}")
            fov_lbl.setFixedWidth(50)
            fov_lbl.setStyleSheet("color: #888; font-size: 10px;")
            h.addWidget(fov_lbl)

        name_lbl = QLabel(acq.display_name)
        name_lbl.setFixedWidth(140)
        name_lbl.setToolTip(acq.path)
        name_lbl.setStyleSheet("color: #ccc; font-size: 10px;")
        h.addWidget(name_lbl)

        # One column per active wavelength
        thumb_labels: dict[int, QLabel] = {}
        thumb_columns: dict[str, QLabel] = {}
        ch_by_wl = {ch.wavelength: (i, ch) for i, ch in enumerate(acq.channels)}
        for wl in active_wls:
            col = QVBoxLayout()
            col.setSpacing(1)
            color = rgb_for(wl)
            ch_lbl = QLabel(wl if wl in ch_by_wl else "")
            ch_lbl.setAlignment(Qt.AlignCenter)
            ch_lbl.setFixedHeight(14)
            if wl in ch_by_wl:
                ch_lbl.setStyleSheet(
                    f"color: rgb({color[0]},{color[1]},{color[2]}); font-size: 9px;"
                )
            else:
                ch_lbl.setStyleSheet("color: transparent; font-size: 9px;")
            col.addWidget(ch_lbl)

            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setFixedSize(self.thumb_size, self.thumb_size)
            if wl in ch_by_wl:
                thumb.setStyleSheet(
                    "background-color: #222; border: 1px solid #2a2a2a; border-radius: 3px;"
                )
                thumb_labels[ch_by_wl[wl][0]] = thumb
            else:
                thumb.setStyleSheet("background-color: transparent; border: none;")
            thumb_columns[wl] = thumb
            col.addWidget(thumb)
            h.addLayout(col)

        h.addStretch()

        # FOV picker (default mode only)
        fov_combo: QComboBox | None = None
        if not self.expanded_fov_mode and len(acq.fovs) > 1:
            fov_combo = QComboBox()
            for fov in acq.fovs:
                fov_combo.addItem(f"FOV {fov}", fov)
            fov_combo.setCurrentIndex(acq.fovs.index(acq.selected_fov))
            fov_combo.currentIndexChanged.connect(
                lambda i, k=key, c=fov_combo: self._on_fov_changed(k, c.itemData(i))
            )
            h.addWidget(fov_combo)

        btn_3d = QPushButton("Open 3D View")
        btn_3d.setFixedSize(110, 26)
        btn_3d.setStyleSheet(
            "QPushButton { background-color: #2d5aa0; color: white; border-radius: 4px;"
            " font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3a6fc0; }"
        )
        btn_3d.clicked.connect(lambda _, k=key: self._open_napari(k))
        h.addWidget(btn_3d)

        btn_lut = QPushButton("Adjust Contrast")
        btn_lut.setFixedSize(110, 26)
        btn_lut.setStyleSheet(
            "QPushButton { background-color: #555; color: white; border-radius: 4px;"
            " font-size: 11px; }"
            "QPushButton:hover { background-color: #777; }"
        )
        btn_lut.clicked.connect(lambda _, k=key: self._adjust_lut(k))
        h.addWidget(btn_lut)

        return RowWidgets(
            container=container,
            mag_lbl=mag_lbl,
            time_lbl=time_lbl,
            name_lbl=name_lbl,
            thumb_labels=thumb_labels,
            thumb_columns=thumb_columns,
            fov_combo=fov_combo,
        )

    def _on_fov_changed(self, key, new_fov: str) -> None:
        acq = self.acquisitions[key.acq_id]
        acq.selected_fov = new_fov
        # Re-key the row widget; the easiest is to rebuild rows since rows are cheap
        self._rebuild_rows()
        # Enqueue jobs for the new FOV's channels (cache-aware; loader will skip cached)
        self._enqueue_jobs_for_acq(key.acq_id, acq, new_fov)
```

- [ ] **Step 6: Implement physical-aspect helper, label sizing, and `_render_thumb`**

```python
    def _phys_aspect(self, acq_id, fov, axis: str) -> float:
        """Return physical_height / physical_width for the given axis."""
        acq = self.acquisitions[acq_id]
        if acq is None:
            return 1.0
        sensor_pixel_um = acq.params.get("sensor_pixel_size_um", 6.5)
        from ..sources._squid_common import parse_mag
        mag = parse_mag(acq.folder_name) or acq.params.get("mag") or 1
        pixel_um = sensor_pixel_um / mag if mag else sensor_pixel_um
        dz_um = acq.params.get("dz(um)", pixel_um)
        shape = acq.shape_zyx
        if shape is None:
            return 1.0
        nz, ny, nx = shape
        if axis == "z":
            return (ny * pixel_um) / max(nx * pixel_um, 1e-9)
        if axis == "y":
            return (nz * dz_um) / max(nx * pixel_um, 1e-9)
        if axis == "x":
            return (nz * dz_um) / max(ny * pixel_um, 1e-9)
        return 1.0

    def _row_label_size(self, acq_id, fov) -> tuple[int, int]:
        if self.square_footprint:
            return self.thumb_size, self.thumb_size
        aspect = self._phys_aspect(acq_id, fov, self.view_axis)
        return self.thumb_size, max(20, int(round(self.thumb_size * aspect)))

    def _apply_label_sizes(self) -> None:
        for (acq_id, fov), rw in self.row_widgets.items():
            w, h = self._row_label_size(acq_id, fov)
            for thumb in rw.thumb_columns.values():
                thumb.setFixedSize(w, h)

    def _render_thumb(self, acq_id, fov, ch_idx, ax_mip) -> None:
        rw = self.row_widgets.get((acq_id, fov))
        if rw is None or ch_idx not in rw.thumb_labels:
            return
        acq = self.acquisitions[acq_id]
        wl = acq.channels[ch_idx].wavelength
        rgba = mip_to_rgba(ax_mip.mip, ax_mip.p1, ax_mip.p999, rgb_for(wl))
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        target_w, target_h = self._row_label_size(acq_id, fov)
        pixmap = QPixmap.fromImage(qimg).scaled(
            target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        rw.thumb_labels[ch_idx].setPixmap(pixmap)
        rw.thumb_labels[ch_idx].setStyleSheet(
            "background-color: #111; border: 1px solid #2a2a2a; border-radius: 3px;"
        )
```

- [ ] **Step 7: Replace `_on_mip_ready`, `_set_axis`, `_set_thumb_size`**

```python
    def _on_mip_ready(self, acq_id, fov, ch_idx, wavelength, channel_mips, shape) -> None:
        if acq_id >= len(self.acquisitions) or self.acquisitions[acq_id] is None:
            return
        acq = self.acquisitions[acq_id]
        if shape is not None and acq.shape_zyx is None:
            acq.shape_zyx = shape
        for axis, ax_mip in channel_mips.items():
            self.mip_data[(acq_id, fov, ch_idx, axis)] = ax_mip
        if self.view_axis in channel_mips:
            self._render_thumb(acq_id, fov, ch_idx, channel_mips[self.view_axis])
        # Aspect may now be known; re-size labels in this row
        self._apply_label_sizes()

    def _set_axis(self, axis: str) -> None:
        if axis == self.view_axis:
            return
        self.view_axis = axis
        self._apply_label_sizes()
        for (acq_id, fov, ch_idx, ax), ax_mip in self.mip_data.items():
            if ax != axis:
                continue
            self._render_thumb(acq_id, fov, ch_idx, ax_mip)

    def _set_thumb_size(self, size: int) -> None:
        self.thumb_size = size
        self._apply_label_sizes()
        for (acq_id, fov, ch_idx, axis), ax_mip in self.mip_data.items():
            if axis != self.view_axis:
                continue
            self._render_thumb(acq_id, fov, ch_idx, ax_mip)
```

- [ ] **Step 8: Add temporary stubs for `_open_napari` and `_adjust_lut`**

These are filled in by Tasks 22-23. Add at the bottom of the class:

```python
    def _open_napari(self, key) -> None:
        from qtpy.QtWidgets import QMessageBox
        QMessageBox.information(self, "Coming soon", "3D viewer wiring in Task 23")

    def _adjust_lut(self, key) -> None:
        from qtpy.QtWidgets import QMessageBox
        QMessageBox.information(self, "Coming soon", "LUT dialog wiring in Task 22")
```

- [ ] **Step 9: Smoke-launch with `QT_QPA_PLATFORM=offscreen` and a synthetic dataset**

Add a tiny smoke script `scripts/smoke.py` (we will delete it before commit; this is an interim verification only):

```python
import sys, os, tempfile, json, yaml
import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication
from gallery_view.ui.gallery_window import GalleryWindow

def make_acq(parent):
    folder = parent / "25x_A1_2026-04-26_12-00-00.000000"
    folder.mkdir()
    (folder / "acquisition parameters.json").write_text(json.dumps({"sensor_pixel_size_um": 6.5, "dz(um)": 2.0}))
    (folder / "acquisition_channels.yaml").write_text(yaml.safe_dump({"channels": [{"name": "Fluorescence_488_nm_Ex", "enabled": True, "camera_settings": {"exposure_time_ms": 100.0}, "illumination_settings": {"intensity": 25.0}}]}))
    (folder / "ome_tiff").mkdir()
    tifffile.imwrite(folder / "ome_tiff" / "current_0.ome.tiff",
                     np.zeros((4, 1, 8, 10), dtype=np.uint16),
                     metadata={"axes": "ZCYX"})
    return folder

if __name__ == "__main__":
    from pathlib import Path
    app = QApplication.instance() or QApplication(sys.argv)
    tmp = Path(tempfile.mkdtemp())
    folder = make_acq(tmp)
    w = GalleryWindow()
    w.show()
    w._add_source(str(tmp))
    print("OK; rows:", len(w.row_widgets))
```

Run: `QT_QPA_PLATFORM=offscreen python scripts/smoke.py`
Expected: prints `OK; rows: 1` (rows count is 1 after `_add_source`).

Then: `rm scripts/smoke.py`

- [ ] **Step 10: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py
git commit -m "Implement gallery row rendering, FOV picker, axis switching"
```

---

## Task 20: Filters (mag, hide-thin) and visibility

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py`

- [ ] **Step 1: Replace `_rebuild_mag_filter` and `_refresh_visibility`**

```python
    def _rebuild_mag_filter(self) -> None:
        from ..sources._squid_common import parse_mag

        # Collect distinct mags from current acquisitions
        mags = set()
        for acq in self.acquisitions:
            if acq is None:
                continue
            mag = parse_mag(acq.folder_name)
            if mag is not None:
                mags.add(mag)

        # Tear down old checkboxes
        for cb in list(self.mag_checkboxes.values()):
            self.mag_row_layout.removeWidget(cb)
            cb.deleteLater()
        self.mag_checkboxes.clear()

        if len(mags) <= 1:
            return  # hide the filter when only one mag is present

        for mag in sorted(mags):
            cb = QCheckBox(f"{mag}x")
            cb.setChecked(True)
            cb.setStyleSheet("QCheckBox { color: #ccc; font-size: 11px; }")
            cb.stateChanged.connect(self._refresh_visibility)
            self.mag_row_layout.addWidget(cb)
            self.mag_checkboxes[mag] = cb

    def _refresh_visibility(self) -> None:
        from ..sources._squid_common import parse_mag

        active_mags = {m for m, cb in self.mag_checkboxes.items() if cb.isChecked()}
        hide_thin = self.hide_thin_btn.isChecked()
        for (acq_id, fov), rw in self.row_widgets.items():
            acq = self.acquisitions[acq_id]
            visible = acq is not None
            if visible and self.mag_checkboxes:
                mag = parse_mag(acq.folder_name)
                visible = mag in active_mags if mag is not None else True
            if visible and hide_thin and acq.shape_zyx is not None:
                visible = acq.shape_zyx[0] >= THIN_Z_THRESHOLD
            rw.container.setVisible(visible)
```

- [ ] **Step 2: Smoke check no regressions**

Run: `pytest -q`
Expected: all existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py
git commit -m "Implement magnification filter and Z<5 visibility filter"
```

---

## Task 21: Settings menu (square footprint, expand FOVs, clear cache)

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py`

- [ ] **Step 1: Add a `_build_menus` method and call it in `_build_ui`**

In `_build_ui`, immediately after `self.setStyleSheet(...)`, add:

```python
        self._build_menus()
```

Add to the class:

```python
    def _build_menus(self) -> None:
        from qtpy.QtGui import QAction
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        menubar = self.menuBar()
        menubar.setStyleSheet(
            "QMenuBar { background-color: #2a2a2a; color: #ccc; }"
            "QMenuBar::item:selected { background-color: #444; }"
            "QMenu { background-color: #2a2a2a; color: #ccc; border: 1px solid #444; }"
            "QMenu::item:selected { background-color: #2d5aa0; color: white; }"
        )

        file_menu = menubar.addMenu("File")
        add_action = QAction("Add Folder…", self)
        add_action.setShortcut("Ctrl+O")
        add_action.triggered.connect(self._on_add_folder)
        file_menu.addAction(add_action)
        refresh_action = QAction("Refresh sources", self)
        refresh_action.triggered.connect(self._on_refresh_sources)
        file_menu.addAction(refresh_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        settings_menu = menubar.addMenu("Settings")

        self.square_action = QAction("Square footprint for XZ/YZ", self)
        self.square_action.setCheckable(True)
        self.square_action.toggled.connect(self._set_square_footprint)
        settings_menu.addAction(self.square_action)

        self.expand_action = QAction("Expand all FOVs as separate rows", self)
        self.expand_action.setCheckable(True)
        self.expand_action.toggled.connect(self._set_expanded_fov_mode)
        settings_menu.addAction(self.expand_action)

        settings_menu.addSeparator()
        clear_action = QAction("Clear MIP cache…", self)
        clear_action.triggered.connect(self._on_clear_cache)
        settings_menu.addAction(clear_action)

    def _on_add_folder(self) -> None:
        from qtpy.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Add folder")
        if path:
            self._add_source(path)

    def _on_refresh_sources(self) -> None:
        roots = [s.path for s in self.sources]
        # Remove + re-add each root: keeps API simple, may re-enqueue cached jobs
        # which the loader skips quickly via cache.load.
        for path in roots:
            self._remove_source(path)
        for path in roots:
            self._add_source(path)

    def _on_clear_cache(self) -> None:
        from qtpy.QtWidgets import QMessageBox
        from .. import cache

        reply = QMessageBox.question(
            self, "Clear MIP cache",
            f"Delete the MIP cache at {cache.CACHE_DIR}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            cache.clear_all()
            self.status.setText("MIP cache cleared.")

    def _set_square_footprint(self, checked: bool) -> None:
        self.square_footprint = checked
        self._apply_label_sizes()
        for (acq_id, fov, ch_idx, axis), ax_mip in self.mip_data.items():
            if axis != self.view_axis:
                continue
            self._render_thumb(acq_id, fov, ch_idx, ax_mip)

    def _set_expanded_fov_mode(self, checked: bool) -> None:
        self.expanded_fov_mode = checked
        self._rebuild_rows()
```

- [ ] **Step 2: Smoke check**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py
git commit -m "Add Settings menu (square footprint, expand FOVs, clear cache)"
```

---

## Task 22: LUT dialog with PNG export

**Files:**
- Create: `src/gallery_view/ui/lut_dialog.py`
- Modify: `src/gallery_view/ui/gallery_window.py` (replace `_adjust_lut` stub)

The dialog is large but mostly lifted from `aion-explorer/explorer_ovelle.py:1296-1657`.

- [ ] **Step 1: Write `src/gallery_view/ui/lut_dialog.py`**

```python
"""Per-channel LUT dialog with min/max sliders, dataset-wide actions, and PNG
export. Operates on the current projection axis; saves apply across all three
axes via the `.lut.json` sidecar."""

import os
import re
from typing import Callable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .. import cache
from ..mips import mip_to_rgba
from ..sources._squid_common import parse_mag
from ..types import Acquisition, AxisMip
from .colors import rgb_for
from .zoomable_view import ZoomableImageView

AXES = ("z", "y", "x")
PREVIEW_SIZE = 400
EXPORT_DPI = 600


def show_lut_dialog(
    parent,
    acq: Acquisition,
    fov: str,
    axis: str,
    mip_data: dict,
    refresh_thumb: Callable[[int, str, int, AxisMip], None],
    acq_id: int,
) -> None:
    """Open the LUT dialog. ``mip_data`` is the gallery window's
    ``{(acq_id, fov, ch_idx, axis): AxisMip}`` map; we mutate it in place.
    ``refresh_thumb(acq_id, fov, ch_idx, ax_mip)`` is called whenever a
    channel's LUT changes so the gallery thumbnail re-renders."""

    ch_keys = sorted(
        ci for (a, f, ci, ax) in mip_data
        if a == acq_id and f == fov and ax == axis
    )
    if not ch_keys:
        return

    axis_label = {"z": "Z (XY)", "y": "Y (XZ)", "x": "X (YZ)"}[axis]
    snapshot: dict[tuple[int, str, int, str], tuple[float, float]] = {}
    for ci in ch_keys:
        for ax in AXES:
            entry = mip_data.get((acq_id, fov, ci, ax))
            if entry is not None:
                snapshot[(acq_id, fov, ci, ax)] = (entry.p1, entry.p999)

    dlg = QDialog(parent)
    mag = parse_mag(acq.folder_name) or "?"
    dlg.setWindowTitle(
        f"LUT — {acq.display_name} | {mag}x | FOV {fov} | {axis_label}"
    )
    dlg.setStyleSheet("background-color: #1e1e1e; color: white;")

    def revert_unsaved() -> None:
        for k, (p1, p999) in snapshot.items():
            entry = mip_data.get(k)
            if entry is None:
                continue
            mip_data[k] = AxisMip(mip=entry.mip, p1=p1, p999=p999)
            _, _, ch_idx, ax = k
            if ax == axis:
                refresh_thumb(acq_id, fov, ch_idx, mip_data[k])

    dlg.rejected.connect(revert_unsaved)

    outer = QVBoxLayout(dlg)
    outer.setSpacing(8)

    channels_row = QHBoxLayout()
    channels_row.setSpacing(12)
    outer.addLayout(channels_row)

    sensor_pixel_um = acq.params.get("sensor_pixel_size_um", 6.5)
    pixel_um = sensor_pixel_um / mag if isinstance(mag, int) else sensor_pixel_um
    dz_um = acq.params.get("dz(um)", pixel_um)
    y_scale = 1.0 if axis == "z" else (dz_um / pixel_um if pixel_um > 0 else 1.0)

    min_slider_targets: list[tuple[QSlider, int]] = []

    for ch_idx in ch_keys:
        ax_mip = mip_data[(acq_id, fov, ch_idx, axis)]
        wl = acq.channels[ch_idx].wavelength
        color = rgb_for(wl)
        data_min, data_max = float(ax_mip.mip.min()), float(ax_mip.mip.max())

        col = QVBoxLayout()
        col.setSpacing(4)

        ch_lbl = QLabel(f"{wl} nm")
        ch_lbl.setAlignment(Qt.AlignCenter)
        ch_lbl.setStyleSheet(
            f"color: rgb({color[0]},{color[1]},{color[2]}); font-size: 12px; font-weight: bold;"
        )
        col.addWidget(ch_lbl)

        preview = ZoomableImageView(PREVIEW_SIZE)
        col.addWidget(preview)

        def make_render(prev_view, m, c, ys):
            def render(lo: float, hi: float) -> None:
                rgba = mip_to_rgba(m, lo, hi, c)
                h, w = rgba.shape[:2]
                qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
                prev_view.set_pixmap(QPixmap.fromImage(qimg), ys)
            return render

        render_fn = make_render(preview, ax_mip.mip, color, y_scale)
        render_fn(ax_mip.p1, ax_mip.p999)

        slider_lo = min(0, int(data_min))
        slider_hi = int(data_max)

        min_header = QHBoxLayout()
        min_header.setContentsMargins(0, 0, 0, 0)
        min_header.addWidget(QLabel("Min"))
        min_header.addStretch()
        btn_min_zero = QPushButton("→ 0")
        btn_min_zero.setFixedSize(38, 18)
        btn_min_zero.setStyleSheet(
            "QPushButton { background-color: #444; color: #ddd; border: 1px solid #555;"
            " border-radius: 3px; font-size: 9px; }"
            "QPushButton:hover { background-color: #666; }"
        )
        min_header.addWidget(btn_min_zero)
        col.addLayout(min_header)

        sl_min = QSlider(Qt.Horizontal)
        sl_min.setRange(slider_lo, slider_hi)
        sl_min.setValue(int(ax_mip.p1))
        col.addWidget(sl_min)
        btn_min_zero.clicked.connect(lambda _, s=sl_min: s.setValue(0))
        min_slider_targets.append((sl_min, int(data_min)))

        col.addWidget(QLabel("Max"))
        sl_max = QSlider(Qt.Horizontal)
        sl_max.setRange(slider_lo, slider_hi)
        sl_max.setValue(int(ax_mip.p999))
        col.addWidget(sl_max)

        val_lbl = QLabel(f"{int(ax_mip.p1)} — {int(ax_mip.p999)}")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet("color: #888; font-size: 10px;")
        col.addWidget(val_lbl)

        def make_handler(s_min, s_max, v_lbl, rfn, ci):
            def on_change() -> None:
                lo, hi = s_min.value(), s_max.value()
                if hi <= lo:
                    hi = lo + 1
                v_lbl.setText(f"{lo} — {hi}")
                rfn(float(lo), float(hi))
                # Apply to all three axes (synchronized)
                for ax in AXES:
                    key = (acq_id, fov, ci, ax)
                    entry = mip_data.get(key)
                    if entry is None:
                        continue
                    mip_data[key] = AxisMip(mip=entry.mip, p1=float(lo), p999=float(hi))
                refresh_thumb(acq_id, fov, ci, mip_data[(acq_id, fov, ci, axis)])
            return on_change

        handler = make_handler(sl_min, sl_max, val_lbl, render_fn, ch_idx)
        sl_min.valueChanged.connect(handler)
        sl_max.valueChanged.connect(handler)

        channels_row.addLayout(col)

    # Bottom bar
    bottom = QHBoxLayout()
    bottom.setSpacing(8)
    btn_min_reset = QPushButton("Min → Data Min (all channels)")
    btn_min_reset.setFixedHeight(30)
    btn_min_reset.setStyleSheet(
        "QPushButton { background-color: #555; color: white; border-radius: 4px;"
        " font-size: 11px; padding: 0 12px; }"
        "QPushButton:hover { background-color: #777; }"
    )
    btn_min_reset.clicked.connect(
        lambda: [s.setValue(d) for s, d in min_slider_targets]
    )
    bottom.addWidget(btn_min_reset)

    btn_export = QPushButton("Export PNG…")
    btn_export.setFixedHeight(30)
    btn_export.setStyleSheet(btn_min_reset.styleSheet())
    btn_export.clicked.connect(
        lambda: _export_png(dlg, acq, fov, axis, axis_label, ch_keys, mip_data, acq_id, dz_um, pixel_um)
    )
    bottom.addWidget(btn_export)
    bottom.addStretch()

    btn_save = QPushButton("Save")
    btn_save.setFixedSize(80, 30)
    btn_save.setStyleSheet(
        "QPushButton { background-color: #2d5aa0; color: white; border-radius: 4px;"
        " font-size: 12px; font-weight: bold; }"
        "QPushButton:hover { background-color: #3a6fc0; }"
    )

    def save_all() -> None:
        for ci in ch_keys:
            wl = acq.channels[ci].wavelength
            channel = acq.channels[ci]
            src, ch_id = acq.handler.cache_key(acq, fov, channel)
            axis_data = {}
            for ax in AXES:
                entry = mip_data.get((acq_id, fov, ci, ax))
                if entry is None:
                    continue
                axis_data[ax] = (entry.mip, entry.p1, entry.p999)
            if axis_data:
                cache.save_lut_only(src, ch_id, axis_data)
        dlg.accept()

    btn_save.clicked.connect(save_all)
    bottom.addWidget(btn_save)
    outer.addLayout(bottom)

    dlg.adjustSize()
    dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()


def _export_png(
    dlg, acq, fov, axis, axis_label, ch_keys, mip_data, acq_id, dz_um, pixel_um
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        QMessageBox.warning(
            dlg, "Missing dependency",
            "matplotlib is required to export. `pip install matplotlib`.",
        )
        return

    ts = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})", acq.folder_name)
    datetime_str = f"{ts.group(1)} {ts.group(2)}:{ts.group(3)}" if ts else ""
    ts_part = f"_{ts.group(1)}_{ts.group(2)}{ts.group(3)}" if ts else ""
    safe_name = acq.display_name.replace(" ", "_").replace("/", "_")
    default_name = f"{safe_name}_{axis}_fov{fov}{ts_part}.png"
    path, _ = QFileDialog.getSaveFileName(dlg, "Export view", default_name, "PNG (*.png)")
    if not path:
        return

    n_channels = len(ch_keys)
    fig, axes_grid = plt.subplots(
        2, n_channels,
        figsize=(5.0 * n_channels, 8.5),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="black",
    )
    if n_channels == 1:
        axes_grid = axes_grid.reshape(2, 1)

    imshow_aspect = 1.0 if axis == "z" else (
        dz_um / pixel_um if pixel_um > 0 else 1.0
    )
    text_pad_pt = 60 * 72 / EXPORT_DPI

    for col_i, ci in enumerate(ch_keys):
        entry = mip_data.get((acq_id, fov, ci, axis))
        if entry is None:
            continue
        wl = acq.channels[ci].wavelength
        color = rgb_for(wl)
        rgba = mip_to_rgba(entry.mip, entry.p1, entry.p999, color)
        color_norm = tuple(c / 255 for c in color)

        ax_img = axes_grid[0, col_i]
        ax_img.set_facecolor("black")
        ax_img.imshow(rgba, aspect=imshow_aspect, interpolation="none")
        ax_img.set_title(f"{wl} nm", color=color_norm, fontsize=11, pad=text_pad_pt)
        extras = acq.handler.channel_yaml_extras(acq, acq.channels[ci])
        parts = []
        if extras.get("exposure_ms") is not None:
            parts.append(f"{extras['exposure_ms']:.0f} ms")
        if extras.get("intensity") is not None:
            parts.append(f"{extras['intensity']:.0f}% laser")
        if parts:
            ax_img.set_xlabel(" · ".join(parts), color="#888",
                              fontsize=10, labelpad=text_pad_pt)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_color("#888")
            spine.set_linewidth(6 * 72 / EXPORT_DPI)

        ax_hist = axes_grid[1, col_i]
        ax_hist.set_facecolor("black")
        ax_hist.hist(entry.mip.ravel(), bins=256, color=color_norm, log=True)
        ax_hist.axvline(entry.p1, color="white", linestyle="--", linewidth=1,
                        label=f"min={entry.p1:.0f}")
        ax_hist.axvline(entry.p999, color="orange", linestyle="--", linewidth=1,
                        label=f"max={entry.p999:.0f}")
        leg = ax_hist.legend(fontsize=8, loc="upper right",
                             facecolor="black", labelcolor="white")
        leg.get_frame().set_edgecolor("#444")
        ax_hist.set_xlabel("intensity", color="white")
        ax_hist.set_ylabel("count (log)", color="white")
        ax_hist.tick_params(colors="white")
        for spine in ax_hist.spines.values():
            spine.set_color("#666")

    title_main = f"{acq.display_name} | FOV {fov} | {axis_label}"
    fig.text(0.5, 0.985, title_main, ha="center", va="top", fontsize=13, color="white")
    if datetime_str:
        fig.text(0.5, 0.96, datetime_str, ha="center", va="top", fontsize=11, color="#888")
        top_rect = 0.945
    else:
        top_rect = 0.97
    fig.tight_layout(rect=[0, 0, 1, top_rect], h_pad=0.6, w_pad=0.4)
    try:
        fig.savefig(path, dpi=EXPORT_DPI, facecolor="black",
                    bbox_inches="tight", pad_inches=0.15)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.warning(dlg, "Export failed", str(exc))
    finally:
        plt.close(fig)
```

- [ ] **Step 2: Wire it into the gallery window**

In `gallery_window.py`, replace the `_adjust_lut` stub with:

```python
    def _adjust_lut(self, key) -> None:
        from .lut_dialog import show_lut_dialog

        acq = self.acquisitions[key.acq_id]

        def refresh(acq_id, fov, ch_idx, ax_mip):
            self._render_thumb(acq_id, fov, ch_idx, ax_mip)

        show_lut_dialog(
            parent=self,
            acq=acq,
            fov=key.fov,
            axis=self.view_axis,
            mip_data=self.mip_data,
            refresh_thumb=refresh,
            acq_id=key.acq_id,
        )
```

- [ ] **Step 3: Smoke check imports**

Run: `QT_QPA_PLATFORM=offscreen python -c "from gallery_view.ui.lut_dialog import show_lut_dialog; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/gallery_view/ui/lut_dialog.py src/gallery_view/ui/gallery_window.py
git commit -m "Add LUT dialog with sliders, dataset-wide reset, and 600 dpi PNG export"
```

---

## Task 23: napari 3D viewer (`ui/viewer3d.py`)

**Files:**
- Create: `src/gallery_view/ui/viewer3d.py`
- Modify: `src/gallery_view/ui/gallery_window.py` (replace `_open_napari` stub)

- [ ] **Step 1: Write `src/gallery_view/ui/viewer3d.py`**

```python
"""Open the napari 3D viewer for a single acquisition (one FOV).

Loads each channel's full ZYX stack via the handler, scales axes to µm,
adds a 100 µm-tick bounding box overlay, and reuses LUT limits from the
gallery's currently displayed projection axis (with sensible fallbacks)."""

from typing import Callable

import numpy as np

from ..sources._squid_common import parse_mag, parse_timestamp
from ..types import Acquisition, AxisMip
from .colors import napari_cmap_for


def open_napari(
    acq: Acquisition,
    fov: str,
    lut_lookup: Callable[[int, str], tuple[float, float] | None],
) -> None:
    """``lut_lookup(ch_idx, axis)`` returns the gallery's contrast limits if
    they're known, else None. We try the current axis, then Z, then fall
    back to a fresh percentile pass on the loaded stack."""

    import napari

    sensor_pixel_um = acq.params.get("sensor_pixel_size_um", 6.5)
    mag = parse_mag(acq.folder_name) or 1
    pixel_um = sensor_pixel_um / mag if mag else sensor_pixel_um
    dz_um = acq.params.get("dz(um)", pixel_um)
    scale = (dz_um, pixel_um, pixel_um)

    ts = parse_timestamp(acq.folder_name)
    datetime_str = f"{ts[0]} {ts[1]}" if ts else ""
    title = f"{acq.display_name} | FOV {fov}"
    if datetime_str:
        title += f" | {datetime_str}"

    viewer = napari.Viewer(ndisplay=3, title=title)

    first_shape: tuple[int, int, int] | None = None
    for ch_idx, channel in enumerate(acq.channels):
        try:
            stack = acq.handler.load_full_stack(acq, fov, channel)
        except Exception:
            continue
        clim = lut_lookup(ch_idx, "current") or lut_lookup(ch_idx, "z")
        if clim is None:
            clim = (
                float(np.percentile(stack, 1)),
                float(np.percentile(stack, 99.9)),
            )
        viewer.add_image(
            stack,
            scale=scale,
            name=f"{channel.wavelength}nm",
            colormap=napari_cmap_for(channel.wavelength),
            blending="additive",
            contrast_limits=clim,
        )
        if first_shape is None:
            first_shape = stack.shape

    if first_shape is not None:
        _add_bounding_box(viewer, scale, first_shape)

    viewer.text_overlay.visible = True
    viewer.text_overlay.text = title
    viewer.text_overlay.font_size = 12
    viewer.text_overlay.color = "white"
    viewer.text_overlay.position = "top_center"
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"


def _add_bounding_box(viewer, scale, shape_zyx) -> None:
    nz, ny, nx = shape_zyx
    z_max = nz * scale[0]
    y_max = ny * scale[1]
    x_max = nx * scale[2]

    edges = [
        [[0, 0, 0], [0, 0, x_max]],
        [[0, 0, x_max], [0, y_max, x_max]],
        [[0, y_max, x_max], [0, y_max, 0]],
        [[0, y_max, 0], [0, 0, 0]],
        [[z_max, 0, 0], [z_max, 0, x_max]],
        [[z_max, 0, x_max], [z_max, y_max, x_max]],
        [[z_max, y_max, x_max], [z_max, y_max, 0]],
        [[z_max, y_max, 0], [z_max, 0, 0]],
        [[0, 0, 0], [z_max, 0, 0]],
        [[0, 0, x_max], [z_max, 0, x_max]],
        [[0, y_max, x_max], [z_max, y_max, x_max]],
        [[0, y_max, 0], [z_max, y_max, 0]],
    ]
    tick_len = min(z_max, y_max, x_max) * 0.02
    ticks: list[list[list[float]]] = []
    for x in np.arange(100, x_max, 100):
        ticks += [[[0, 0, x], [0, tick_len, x]], [[0, 0, x], [tick_len, 0, x]]]
    for y in np.arange(100, y_max, 100):
        ticks += [[[0, y, 0], [0, y, tick_len]], [[0, y, 0], [tick_len, y, 0]]]
    for z in np.arange(100, z_max, 100):
        ticks += [[[z, 0, 0], [z, tick_len, 0]], [[z, 0, 0], [z, 0, tick_len]]]
    viewer.add_shapes(
        [np.array(line) for line in edges + ticks],
        shape_type="line",
        edge_color="white",
        edge_width=2,
        name="Bounding Box (100µm ticks)",
    )
```

- [ ] **Step 2: Wire into gallery_window**

Replace the `_open_napari` stub:

```python
    def _open_napari(self, key) -> None:
        from .viewer3d import open_napari

        acq = self.acquisitions[key.acq_id]

        def lut_lookup(ch_idx, axis):
            ax = self.view_axis if axis == "current" else axis
            entry = self.mip_data.get((key.acq_id, key.fov, ch_idx, ax))
            if entry is None:
                return None
            return float(entry.p1), float(entry.p999)

        open_napari(acq, key.fov, lut_lookup)
```

- [ ] **Step 3: Smoke check imports**

Run: `QT_QPA_PLATFORM=offscreen python -c "from gallery_view.ui.viewer3d import open_napari; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/gallery_view/ui/viewer3d.py src/gallery_view/ui/gallery_window.py
git commit -m "Add napari 3D viewer with µm scaling and 100 µm-tick bounding box"
```

---

## Task 24: CLI + entry point

**Files:**
- Create: `src/gallery_view/cli.py`
- Create: `src/gallery_view/__main__.py`

- [ ] **Step 1: Write `src/gallery_view/cli.py`**

```python
"""argparse for ``python -m gallery_view``."""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gallery-view",
        description="Standalone gallery viewer for squid z-stack acquisitions.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to an acquisition folder or a parent containing many. "
             "Repeat for multiple sources.",
    )
    return parser.parse_args(argv)
```

- [ ] **Step 2: Write `src/gallery_view/__main__.py`**

```python
"""``python -m gallery_view`` entry point."""

import sys

from qtpy.QtWidgets import QApplication

from .cli import parse_args
from .ui.gallery_window import GalleryWindow


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = GalleryWindow()
    window.show()
    for path in args.source:
        window._add_source(path)
    return app.exec_() if hasattr(app, "exec_") else app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke check imports**

Run: `QT_QPA_PLATFORM=offscreen python -c "from gallery_view.__main__ import main; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/gallery_view/cli.py src/gallery_view/__main__.py
git commit -m "Add CLI argparse and python -m gallery_view entry point"
```

---

## Task 25: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest -q
```

- [ ] **Step 2: Verify locally**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI for pytest on ubuntu/macos × py3.11/3.12"
```

---

## Task 26: Manual smoke test against real data

**Files:** none (verification only)

- [ ] **Step 1: Run against a real squid acquisition**

```bash
python -m gallery_view --source "/Volumes/Extreme SSD/Ovelle/AION"
```

(Substitute the user's actual data path.)

- [ ] **Step 2: Verify acceptance criteria from the spec**

Walk the spec's §16 acceptance criteria:
- Empty window with drop target on launch with no `--source`.
- Drop a folder; rows appear immediately with `#222` cells; thumbnails fill in as MIPs compute.
- Re-launch: thumbnails appear instantly (warm cache).
- XY/XZ/YZ buttons swap projection across all visible rows.
- Magnification filter visible if dataset has multiple mags; toggling it hides/shows rows.
- "Hide thin (Z<5)" toggles correctly.
- LUT dialog opens, sliders update preview + thumbnails synchronously across all 3 axes; Save persists across restart; closing without saving reverts.
- PNG export writes a 600 dpi figure.
- napari 3D viewer opens with µm-scaled axes, scale bar, bounding box, contrast taken from the gallery.
- Settings → Square footprint, Expand all FOVs, Clear MIP cache all behave as described.

- [ ] **Step 3: Update `README.md` with anything you learned**

If anything is documented incorrectly, fix it.

- [ ] **Step 4: Tag v0.1.0**

```bash
git tag v0.1.0
```

---

## Self-review notes

The plan covers each spec section:

- §1 Overview, §2 UX → Tasks 18-21 (window scaffolding, rows, filters, settings menu) and Task 24 (CLI).
- §3 Repository layout → Task 1 + the per-module tasks.
- §4 Data model → Task 2.
- §5 FormatHandler protocol → Task 6.
- §6 Scanning & ingestion → Task 13.
- §7 MIP cache → Task 4.
- §8 Loader thread → Task 15.
- §9 LUT dialog + PNG export → Task 22.
- §10 3D viewer → Task 23.
- §11 Channel coloring → Task 14.
- §12 Testing → Tasks 3, 4, 5, 8, 9, 10, 11, 12, 13.
- §13 Dependencies → Task 1 (`pyproject.toml`).
- §14 Risks → addressed implicitly: enum scoping (we use `qtpy` everywhere); napari pinned in Task 1; `SQUID_TESTED_AGAINST` constant in Task 7; wavelength colormap fallback in Task 14; cache dir via `platformdirs` in Task 4.
- §15 Future work → out of scope.
- §16 Acceptance criteria → Task 26.
