# Region View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Region" view to the gallery as a peer to FOV view: each row is one well/region, each channel cell is a stitched XY MIP built from the per-FOV Z-MIPs already in the gallery cache plus `coordinates.csv`. A top-toolbar `View: [FOV] [Region]` toggle switches modes; Settings → "Expand all regions as separate rows" mirrors the existing FOV expanded toggle.

**Architecture:** Three new pieces and small extensions to existing ones. `Acquisition` gains `regions: list[str]` and `selected_region: str`. `SingleTiffHandler` populates them from filenames and lazily parses `coordinates.csv` into `acq.extra["coords_by_region"]`. A new `src/gallery_view/stitch.py` exposes a pure `stitch_region(fov_mips, coords, pixel_um, target_longest_px, flip_y) -> AxisMip` function (integer block-mean downsample, mean-blend overlaps, black for gaps). `MipLoader` learns a new `RegionStitchJob` type with a `region_mip_ready` signal. `GalleryWindow` gains `view_mode`, a toolbar button group, an Expand-regions Settings entry, a Region row combo, and FOV-readiness tracking that enqueues a stitch job once all FOV MIPs for a `(acq, region, channel)` have landed. The existing `cache.py` is unchanged except for one `load()` relaxation that allows partial axis sets (so region caches with only `"z"` round-trip).

**Tech Stack:** Python 3.11+, PyQt6 / qtpy, numpy, tifffile, pytest. Spec: `docs/superpowers/specs/2026-05-10-region-view-design.md`.

---

## File Structure

| Path | Action | Responsibility |
|------|--------|---------------|
| `src/gallery_view/types.py` | modify | `Acquisition.regions`, `selected_region`. |
| `src/gallery_view/stitch.py` | create | Pure `FovCoord` + `stitch_region(...)`. |
| `src/gallery_view/cache.py` | modify | `load()` accepts partial axis sets. |
| `src/gallery_view/sources/base.py` | modify | Add `cache_key_region` to the `FormatHandler` protocol. |
| `src/gallery_view/sources/single_tiff.py` | modify | Populate `acq.regions`; add `_load_coords` + `cache_key_region`. |
| `src/gallery_view/sources/ome_tiff.py` | modify | `cache_key_region` raises `NotImplementedError`. |
| `src/gallery_view/sources/stack_tiff.py` | modify | `cache_key_region` raises `NotImplementedError`. |
| `src/gallery_view/loader.py` | modify | `RegionStitchJob`, `region_mip_ready`, dispatch in `run()`. |
| `src/gallery_view/ui/gallery_window.py` | modify | Toolbar, Settings entry, view_mode branching in row build, readiness tracking. |
| `src/gallery_view/ui/lut_dialog.py` | modify | Accept a `key_fn` so it can target region caches; rename `fov` arg to `unit`. |
| `tests/conftest.py` | modify | Extend `make_squid_single_tiff_acq` to write `coordinates.csv` and support per-region FOV counts. |
| `tests/test_stitch.py` | create | Pure-function tests of `stitch_region`. |
| `tests/test_acquisition_regions.py` | create | `SingleTiffHandler` populates `acq.regions` correctly. |
| `tests/test_single_tiff_coords.py` | create | `_load_coords` parses CSV; handles missing/malformed. |
| `tests/test_cache_region.py` | create | Region-keyed cache round-trip with single `"z"` axis. |
| `tests/test_loader_region.py` | create | `RegionStitchJob` cache hit/miss/failure paths. |
| `tests/test_gallery_window_region.py` | create | Toolbar, expanded toggle, mode-switch row rebuild, stitch enqueue. |
| `README.md` | modify | Add region view to Features and Supported formats notes. |

---

## Task 1: Add `regions` to `Acquisition` and populate from filenames

**Files:**
- Modify: `src/gallery_view/types.py:40-61`
- Modify: `src/gallery_view/sources/single_tiff.py:59-87` (`build()`)
- Create: `tests/test_acquisition_regions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_acquisition_regions.py`:

```python
"""SingleTiffHandler populates Acquisition.regions / selected_region."""

from gallery_view.sources.single_tiff import SingleTiffHandler


def test_single_region_squid_folder_has_one_region(make_squid_single_tiff_acq):
    """A folder with FOVs all under region '0' yields regions == ['0']."""
    folder = make_squid_single_tiff_acq(regions=1, fovs_per_region=2)
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert acq is not None
    assert acq.regions == ["0"]
    assert acq.selected_region == "0"


def test_multi_region_numeric_regions_sort_numerically(make_squid_single_tiff_acq):
    """Numeric region ids sort as ints, not strings (so '10' comes after '2')."""
    folder = make_squid_single_tiff_acq(regions=12, fovs_per_region=1)
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert acq.regions == [str(i) for i in range(12)]


def test_legacy_folder_gets_single_region(make_single_tiff_acq):
    """Legacy current_<fov>_<z>_<channel>.tiff folders have regions == ['0']."""
    folder = make_single_tiff_acq()
    acq = SingleTiffHandler().build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert acq.regions == ["0"]
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_acquisition_regions.py -v
```

Expected: FAIL — `AttributeError: 'Acquisition' object has no attribute 'regions'`.

- [ ] **Step 3: Add fields to `Acquisition`**

In `src/gallery_view/types.py`, extend the `Acquisition` dataclass — add the two new fields after `selected_timepoint`:

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
    regions: list[str] = field(default_factory=lambda: ["0"])
    selected_region: str = "0"
    extra: dict = field(default_factory=dict)
```

- [ ] **Step 4: Compute `regions` in `SingleTiffHandler.build()`**

In `src/gallery_view/sources/single_tiff.py`, replace the `build()` body's `Acquisition(...)` constructor (around line 75-87) so that it derives a sorted region list from `fovs`. Replace with:

```python
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
        regions = self._regions_from_fovs(fovs)
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
            regions=regions,
            selected_region=regions[0],
            extra={"layout": layout},
        )
```

And add the `_regions_from_fovs` static method to `SingleTiffHandler` (place it next to `_fovs_for`):

```python
    @staticmethod
    def _regions_from_fovs(fovs: list[str]) -> list[str]:
        """Distinct region prefixes from composite '<region>_<fov>' strings,
        sorted: numeric regions as ints, alphanumeric (well names) using a
        digit-aware key so 'A2' precedes 'A10'."""
        seen: set[str] = set()
        for fov in fovs:
            region = fov.split("_", 1)[0] if "_" in fov else "0"
            seen.add(region)

        def sort_key(r: str) -> tuple:
            # Numeric regions sort first as ints; alphanumeric follow with a
            # (alpha_prefix, int_suffix) decomposition for natural ordering.
            if r.isdigit():
                return (0, int(r), "")
            m = re.match(r"^([A-Za-z]+)(\d+)$", r)
            if m:
                return (1, int(m.group(2)), m.group(1))
            return (2, 0, r)

        return sorted(seen, key=sort_key)
```

- [ ] **Step 5: Run test, expect pass**

```bash
pytest tests/test_acquisition_regions.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 6: Run full suite — no regressions**

```bash
pytest -q
```

Expected: PASS — existing 115 + 3 new = 118 tests.

- [ ] **Step 7: Commit**

```bash
git add src/gallery_view/types.py src/gallery_view/sources/single_tiff.py tests/test_acquisition_regions.py
git commit -m "Add Acquisition.regions populated from single_tiff filenames"
```

---

## Task 2: Extend `make_squid_single_tiff_acq` fixture to write `coordinates.csv`

The existing fixture builds multi-region squid folders but doesn't write `coordinates.csv`. Region-view tests need it.

**Files:**
- Modify: `tests/conftest.py:201-254`

- [ ] **Step 1: Write the failing test (covers the new fixture options + presence on disk)**

In `tests/test_acquisition_regions.py`, append:

```python
def test_fixture_writes_coordinates_csv(make_squid_single_tiff_acq):
    """The fixture writes coordinates.csv with one row per (region, fov, z)
    when ``write_coords=True``."""
    import csv

    folder = make_squid_single_tiff_acq(
        regions=2, fovs_per_region=3, nz=2,
        write_coords=True,
    )
    coords_path = folder / "0" / "coordinates.csv"
    assert coords_path.exists(), f"coordinates.csv missing under {folder}"

    with coords_path.open() as f:
        rows = list(csv.DictReader(f))
    # 2 regions * 3 fovs * 2 z = 12 rows
    assert len(rows) == 12
    assert {row["region"] for row in rows} == {"0", "1"}
    # FOVs within a region are 0..2; values come back as strings.
    assert {row["fov"] for row in rows} == {"0", "1", "2"}
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_acquisition_regions.py::test_fixture_writes_coordinates_csv -v
```

Expected: FAIL — `TypeError: _build() got an unexpected keyword argument 'write_coords'` (or the CSV doesn't exist).

- [ ] **Step 3: Extend the fixture**

In `tests/conftest.py`, modify `make_squid_single_tiff_acq` — extend the inner `_build` signature with two new kwargs (`write_coords`, `coord_grid_um`) and append the CSV-write block after the existing per-image-write loop. Replace the entire fixture body:

```python
@pytest.fixture
def make_squid_single_tiff_acq(tmp_path):
    """Build an acquisition in squid's per-image TIFF layout.

    ``<acq>/<t>/[<well>_]<region>_<fov>_<z>_<channel>.tiff``

    Optionally writes ``coordinates.csv`` with stage positions laid out as a
    deterministic grid — used by region-view tests.
    """
    import csv

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
        write_coords=False,
        coord_grid_um=(800.0, 600.0),  # (dx, dy) between FOV centers
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
        dx_um, dy_um = coord_grid_um
        for t in range(nt):
            t_dir = folder / str(t)
            t_dir.mkdir()
            for r in range(regions):
                for f in range(fovs_per_region):
                    for z in range(nz):
                        for c, wl in enumerate(wavelengths):
                            ch_stack = _gradient_3d(nz, ny, nx, c)
                            tifffile.imwrite(
                                t_dir / f"{prefix}{r}_{f}_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                                ch_stack[z],
                            )
            if write_coords:
                coords_path = t_dir / "coordinates.csv"
                with coords_path.open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["region", "fov", "z_level", "x (mm)", "y (mm)", "z (um)", "time"])
                    for r in range(regions):
                        for f in range(fovs_per_region):
                            x_mm = (f * dx_um) / 1000.0
                            y_mm = (r * dy_um) / 1000.0
                            for z in range(nz):
                                z_um = 3000.0 + z * dz_um
                                w.writerow([
                                    str(r), str(f), str(z),
                                    f"{x_mm:.6f}", f"{y_mm:.6f}",
                                    f"{z_um:.6f}", f"t{t}_z{z}",
                                ])
        return folder

    return _build
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/test_acquisition_regions.py::test_fixture_writes_coordinates_csv -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite — fixture default behavior unchanged**

```bash
pytest -q
```

Expected: PASS (118 tests; existing tests not passing `write_coords` get the default `False` and don't see the new CSV).

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_acquisition_regions.py
git commit -m "Extend squid fixture to optionally write coordinates.csv"
```

---

## Task 3: `SingleTiffHandler._load_coords` parses `coordinates.csv`

**Files:**
- Modify: `src/gallery_view/sources/single_tiff.py`
- Create: `tests/test_single_tiff_coords.py`

- [ ] **Step 1: Write the failing test**

`tests/test_single_tiff_coords.py`:

```python
"""SingleTiffHandler._load_coords reads coordinates.csv into
acq.extra['coords_by_region'] keyed by region, dedup'd to one row per
(region, fov) (z=0)."""

from gallery_view.sources.single_tiff import SingleTiffHandler


def test_load_coords_populates_extra(make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(
        regions=2, fovs_per_region=3, nz=2, write_coords=True,
    )
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    coords = handler._load_coords(acq)
    assert coords is not None
    assert acq.extra["coords_by_region"] is coords

    # 2 regions, 3 FOVs each — deduplicated across z.
    assert set(coords.keys()) == {"0", "1"}
    assert len(coords["0"]) == 3
    assert len(coords["1"]) == 3

    # Composite fov ids matching acq.fovs.
    fov_ids_r0 = {c.fov for c in coords["0"]}
    assert fov_ids_r0 == {"0_0", "0_1", "0_2"}

    # FOV 0 of region 0 is at (0, 0); fixture step is 800x600 um.
    c00 = next(c for c in coords["0"] if c.fov == "0_0")
    assert c00.x_mm == 0.0
    assert c00.y_mm == 0.0
    c02 = next(c for c in coords["0"] if c.fov == "0_2")
    assert c02.x_mm == 1.6  # 2 * 800 um = 1.6 mm


def test_load_coords_missing_file_returns_none(make_squid_single_tiff_acq):
    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=1, write_coords=False,
    )
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert handler._load_coords(acq) is None
    assert "coords_by_region" not in acq.extra


def test_load_coords_malformed_csv_returns_none(make_squid_single_tiff_acq, tmp_path):
    """Missing required columns -> None, no exception."""
    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=1, write_coords=True,
    )
    bad = folder / "0" / "coordinates.csv"
    bad.write_text("foo,bar\n1,2\n")  # overwrite with garbage
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    assert handler._load_coords(acq) is None


def test_load_coords_caches_result(make_squid_single_tiff_acq):
    """Second call returns the cached dict from acq.extra."""
    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=2, write_coords=True,
    )
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    first = handler._load_coords(acq)
    second = handler._load_coords(acq)
    assert first is second  # identity, not just equality
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_single_tiff_coords.py -v
```

Expected: FAIL — `AttributeError: 'SingleTiffHandler' object has no attribute '_load_coords'`.

- [ ] **Step 3: Add `FovCoord` to the stitch module (used by handler too)**

Create `src/gallery_view/stitch.py` with just the dataclass (algorithm goes in Task 6):

```python
"""Coordinate-based region stitcher: places per-FOV Z-MIPs by stage position
and mean-blends overlaps. Pure functions, no Qt, no thread state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FovCoord:
    """Stage position of one FOV center (in millimeters)."""
    fov: str   # composite '<region>_<fov>' matching acq.fovs
    x_mm: float
    y_mm: float
```

- [ ] **Step 4: Implement `_load_coords` on `SingleTiffHandler`**

In `src/gallery_view/sources/single_tiff.py`, add at the top:

```python
import csv

from ..stitch import FovCoord
```

Then add as a regular (non-static) method on `SingleTiffHandler`, after `cache_key`:

```python
    def _load_coords(self, acq: Acquisition) -> dict[str, list[FovCoord]] | None:
        """Parse coordinates.csv into acq.extra['coords_by_region'].

        Looks at ``<acq.path>/<selected_timepoint>/coordinates.csv`` first,
        then ``<acq.path>/coordinates.csv`` as fallback. Returns the parsed
        mapping (also stored on ``acq.extra``) or None when missing /
        malformed. Result is cached on ``acq.extra`` so repeated calls don't
        re-read the file.
        """
        cached = acq.extra.get("coords_by_region")
        if cached is not None:
            return cached

        candidates = [
            os.path.join(acq.path, acq.selected_timepoint, "coordinates.csv"),
            os.path.join(acq.path, "coordinates.csv"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return None

        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                required = {"region", "fov", "x (mm)", "y (mm)"}
                if not required.issubset(reader.fieldnames or []):
                    return None
                # Dedup to one row per (region, fov) — prefer z=0 when present.
                seen: dict[tuple[str, str], FovCoord] = {}
                for row in reader:
                    region = str(row["region"])
                    fov_idx = str(row["fov"])
                    key = (region, fov_idx)
                    if key in seen and row.get("z_level") not in ("0", 0, "0.0"):
                        continue
                    try:
                        x_mm = float(row["x (mm)"])
                        y_mm = float(row["y (mm)"])
                    except (TypeError, ValueError):
                        return None
                    composite = f"{region}_{fov_idx}"
                    seen[key] = FovCoord(fov=composite, x_mm=x_mm, y_mm=y_mm)
        except OSError:
            return None

        result: dict[str, list[FovCoord]] = {}
        for (region, _), coord in seen.items():
            result.setdefault(region, []).append(coord)
        # Sort each region's coords by composite fov string for determinism.
        for region in result:
            result[region].sort(key=lambda c: c.fov)
        acq.extra["coords_by_region"] = result
        return result
```

- [ ] **Step 5: Run test, expect pass**

```bash
pytest tests/test_single_tiff_coords.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 6: Run full suite**

```bash
pytest -q
```

Expected: PASS (122 tests).

- [ ] **Step 7: Commit**

```bash
git add src/gallery_view/stitch.py src/gallery_view/sources/single_tiff.py tests/test_single_tiff_coords.py
git commit -m "Add SingleTiffHandler._load_coords + FovCoord dataclass"
```

---

## Task 4: Relax `cache.load` to accept partial axis sets

Region mosaics only have a `"z"` axis. The current `cache.load()` aborts if any of `("z", "y", "x")` is missing — fix that.

**Files:**
- Modify: `src/gallery_view/cache.py:59-67`
- Create: `tests/test_cache_region.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cache_region.py`:

```python
"""Cache supports region mosaics (single 'z' axis, region-keyed channel_id)."""

import numpy as np
import pytest

from gallery_view import cache
from gallery_view.types import AxisMip


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def test_region_cache_roundtrip_single_z_axis():
    """A region mosaic saved with only 'z' loads back correctly."""
    mip = np.arange(24, dtype=np.float32).reshape(4, 6)
    data = {"z": AxisMip(mip=mip, p1=1.0, p999=22.0)}
    cache.save("/src/acq", "region:A5/t0/Fluorescence_488_nm_Ex", data)
    loaded, _ = cache.load("/src/acq", "region:A5/t0/Fluorescence_488_nm_Ex")
    assert loaded is not None
    assert set(loaded.keys()) == {"z"}
    np.testing.assert_array_equal(loaded["z"].mip, mip)
    assert loaded["z"].p1 == 1.0
    assert loaded["z"].p999 == 22.0


def test_fov_cache_still_requires_or_yields_all_axes():
    """Full FOV caches (all three axes saved) round-trip with all axes."""
    data = {
        ax: AxisMip(mip=np.zeros((3, 3), dtype=np.float32), p1=0.0, p999=1.0)
        for ax in ("z", "y", "x")
    }
    cache.save("/src/acq", "fov0/t0/Fluorescence_488_nm_Ex", data, (3, 3, 3))
    loaded, shape = cache.load("/src/acq", "fov0/t0/Fluorescence_488_nm_Ex")
    assert loaded is not None
    assert set(loaded.keys()) == {"z", "y", "x"}
    assert shape == (3, 3, 3)


def test_load_returns_none_when_no_axes_present(tmp_path, monkeypatch):
    """A .npz file with the right version but zero axes shouldn't masquerade
    as a valid cache."""
    import numpy as _np
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    (tmp_path / "mips").mkdir()
    path = cache._cache_path("/src/acq", "weird")
    _np.savez_compressed(path, version=_np.int32(cache.CACHE_VERSION))
    assert cache.load("/src/acq", "weird") == (None, None)


def test_region_and_fov_cache_keys_dont_collide(make_squid_single_tiff_acq):
    """region:<r>/... and fov<r>_<f>/... must hash to different files even
    when r happens to match."""
    fov_path = cache._cache_path("/src/acq", "fov0_0/t0/ch")
    region_path = cache._cache_path("/src/acq", "region:0/t0/ch")
    assert fov_path != region_path
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_cache_region.py -v
```

Expected: FAIL — `test_region_cache_roundtrip_single_z_axis` and `test_load_returns_none_when_no_axes_present` both fail because `load()` returns `(None, None)` whenever an axis is missing.

- [ ] **Step 3: Modify `cache.load()`**

In `src/gallery_view/cache.py`, replace the axis loop at lines 60-67:

```python
        # Skip missing axes instead of aborting: region mosaics save only the
        # 'z' axis, while FOV mosaics save all three. The new behaviour is
        # identical for full FOV caches and unlocks partial region caches.
        out: ChannelMips = {}
        for ax in AXES:
            if f"mip_{ax}" not in data.files:
                continue
            out[ax] = AxisMip(
                mip=np.asarray(data[f"mip_{ax}"]),
                p1=float(data[f"p1_{ax}"]),
                p999=float(data[f"p999_{ax}"]),
            )
        if not out:
            return None, None
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/test_cache_region.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Run full suite — existing cache tests must still pass**

```bash
pytest -q
```

Expected: PASS (126 tests).

- [ ] **Step 6: Commit**

```bash
git add src/gallery_view/cache.py tests/test_cache_region.py
git commit -m "Allow cache.load to return partial axis sets (for region mosaics)"
```

---

## Task 5: Add `cache_key_region` to handler protocol + implementations

**Files:**
- Modify: `src/gallery_view/sources/base.py` (`FormatHandler` protocol)
- Modify: `src/gallery_view/sources/single_tiff.py` (implement)
- Modify: `src/gallery_view/sources/ome_tiff.py` (raise NotImplementedError)
- Modify: `src/gallery_view/sources/stack_tiff.py` (raise NotImplementedError)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cache_region.py`:

```python
def test_single_tiff_cache_key_region(make_squid_single_tiff_acq):
    """SingleTiffHandler.cache_key_region produces a region: prefix key
    that differs by region, timepoint, and channel."""
    from gallery_view.sources.single_tiff import SingleTiffHandler

    folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=2, nt=2)
    handler = SingleTiffHandler()
    acq = handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    src_a, id_a = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="0")
    src_b, id_b = handler.cache_key_region(acq, "1", acq.channels[0], timepoint="0")
    src_c, id_c = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="1")
    src_d, id_d = handler.cache_key_region(acq, "0", acq.channels[1], timepoint="0")
    assert src_a == src_b == src_c == src_d == acq.path
    assert id_a.startswith("region:0/")
    assert id_b.startswith("region:1/")
    assert id_a != id_b != id_c != id_d


def test_ome_tiff_cache_key_region_not_implemented(make_ome_tiff_acq):
    from gallery_view.sources.ome_tiff import OmeTiffHandler

    handler = OmeTiffHandler()
    acq = handler.build(
        str(make_ome_tiff_acq()), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    with pytest.raises(NotImplementedError):
        handler.cache_key_region(acq, "0", acq.channels[0])


def test_stack_tiff_cache_key_region_not_implemented(make_stack_tiff_acq):
    from gallery_view.sources.stack_tiff import StackTiffHandler

    handler = StackTiffHandler()
    acq = handler.build(
        str(make_stack_tiff_acq()), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )
    with pytest.raises(NotImplementedError):
        handler.cache_key_region(acq, "current", acq.channels[0])
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/test_cache_region.py -v
```

Expected: FAIL — `AttributeError: ... has no attribute 'cache_key_region'`.

- [ ] **Step 3: Add the protocol method**

In `src/gallery_view/sources/base.py`, after `cache_key`:

```python
    def cache_key_region(
        self, acq: Acquisition, region: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        """Return ``(src_path_for_hash, channel_id)`` for the cached stitched
        region mosaic. Handlers that don't support region view raise
        ``NotImplementedError``.

        Only ``SingleTiffHandler`` implements this in v1; the UI gates the
        Region toolbar button on whether at least one source-handler supports
        it, so callers never reach the raise.
        """
        ...
```

- [ ] **Step 4: Implement on `SingleTiffHandler`**

In `src/gallery_view/sources/single_tiff.py`, add after `cache_key`:

```python
    def cache_key_region(
        self, acq: Acquisition, region: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        return acq.path, f"region:{region}/t{timepoint}/{channel.name}"
```

- [ ] **Step 5: Raise on the other two handlers**

In `src/gallery_view/sources/ome_tiff.py`, add a method to `OmeTiffHandler`:

```python
    def cache_key_region(
        self, acq: Acquisition, region: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        raise NotImplementedError(
            "OmeTiffHandler does not support region view"
        )
```

In `src/gallery_view/sources/stack_tiff.py`, add the same method to `StackTiffHandler`:

```python
    def cache_key_region(
        self, acq: Acquisition, region: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        raise NotImplementedError(
            "StackTiffHandler does not support region view"
        )
```

- [ ] **Step 6: Run tests, expect pass**

```bash
pytest tests/test_cache_region.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 7: Run full suite**

```bash
pytest -q
```

Expected: PASS (129 tests).

- [ ] **Step 8: Commit**

```bash
git add src/gallery_view/sources/base.py src/gallery_view/sources/single_tiff.py src/gallery_view/sources/ome_tiff.py src/gallery_view/sources/stack_tiff.py tests/test_cache_region.py
git commit -m "Add cache_key_region to handler protocol (single_tiff only impl)"
```

---

## Task 6: Implement `stitch_region` algorithm

**Files:**
- Modify: `src/gallery_view/stitch.py` (was created in Task 3 with only `FovCoord`)
- Create: `tests/test_stitch.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_stitch.py`:

```python
"""Pure-function tests of stitch_region.

Synthetic grids place tiny FOVs at known stage coordinates so we can assert
exact pixel layout, mean-blending, gap handling, and downsample math.
"""

import numpy as np
import pytest

from gallery_view.stitch import FovCoord, stitch_region


def _flat(value: float, ny: int = 4, nx: int = 4) -> np.ndarray:
    return np.full((ny, nx), value, dtype=np.float32)


def test_empty_input_returns_none():
    assert stitch_region({}, [], pixel_um=1.0) is None


def test_single_fov_returns_a_padded_version_of_it():
    """One FOV at the origin: mosaic equals that FOV (possibly downsampled).

    With nx=ny=4 and target_longest_px=1024, factor=1 so no downsampling."""
    mip = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = stitch_region(
        {"0_0": mip},
        [FovCoord("0_0", x_mm=0.0, y_mm=0.0)],
        pixel_um=1.0,
        target_longest_px=1024,
    )
    assert result is not None
    # Mosaic shape matches the single FOV (no padding when only one tile).
    assert result.mip.shape == (4, 4)
    np.testing.assert_array_equal(result.mip, mip)


def test_two_fovs_side_by_side_lay_out_horizontally():
    """Two 4x4 FOVs centered at x=0 and x=4 um (pixel_um=1) place adjacent."""
    left = _flat(10.0)
    right = _flat(20.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=4e-3, y_mm=0.0),  # 4 um right
    ]
    result = stitch_region(
        {"0_0": left, "0_1": right}, coords,
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    # Canvas width = 4 + 4 = 8 px; height = 4 px.
    assert result.mip.shape == (4, 8)
    np.testing.assert_array_equal(result.mip[:, :4], left)
    np.testing.assert_array_equal(result.mip[:, 4:], right)


def test_overlapping_fovs_mean_blend():
    """Two FOVs offset by half-width overlap by 50%; overlap pixels are mean."""
    a = _flat(10.0)   # 4x4
    b = _flat(20.0)   # 4x4
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=2e-3, y_mm=0.0),  # offset by 2 um (half width)
    ]
    result = stitch_region(
        {"0_0": a, "0_1": b}, coords,
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    # Canvas: 4 + 2 = 6 px wide. Cols 0-1 are A only, 2-3 are A+B blend
    # (mean=15), 4-5 are B only.
    assert result.mip.shape == (4, 6)
    np.testing.assert_array_equal(result.mip[:, 0:2], 10.0)
    np.testing.assert_allclose(result.mip[:, 2:4], 15.0)
    np.testing.assert_array_equal(result.mip[:, 4:6], 20.0)


def test_missing_fov_leaves_black_gap():
    """A FovCoord with no matching entry in fov_mips leaves that rect black."""
    a = _flat(10.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=4e-3, y_mm=0.0),
    ]
    result = stitch_region(
        {"0_0": a}, coords,    # 0_1 missing
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    assert result.mip.shape == (4, 8)
    np.testing.assert_array_equal(result.mip[:, :4], a)
    np.testing.assert_array_equal(result.mip[:, 4:], 0.0)


def test_heterogeneous_shapes_returns_none():
    """If FOV MIPs aren't the same shape we bail rather than guess."""
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=4e-3, y_mm=0.0),
    ]
    result = stitch_region(
        {"0_0": _flat(1.0, 4, 4), "0_1": _flat(1.0, 4, 5)},
        coords, pixel_um=1.0,
    )
    assert result is None


def test_integer_downsample_factor():
    """Big FOVs trigger integer downsample to fit target_longest_px.

    Two 8x8 FOVs placed side-by-side -> canvas 8x16. With target=8, factor=2
    -> mosaic shape (4, 8)."""
    a = _flat(10.0, 8, 8)
    b = _flat(20.0, 8, 8)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=8e-3, y_mm=0.0),
    ]
    result = stitch_region(
        {"0_0": a, "0_1": b}, coords,
        pixel_um=1.0, target_longest_px=8,
    )
    assert result is not None
    assert result.mip.shape == (4, 8)
    np.testing.assert_array_equal(result.mip[:, :4], 10.0)
    np.testing.assert_array_equal(result.mip[:, 4:], 20.0)


def test_negative_stage_coords_shift_to_origin():
    """Negative x_mm/y_mm shifts to a non-negative canvas origin."""
    a = _flat(10.0)
    coords = [FovCoord("0_0", x_mm=-1.0, y_mm=-1.0)]
    result = stitch_region(
        {"0_0": a}, coords, pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    np.testing.assert_array_equal(result.mip, a)


def test_flip_y_inverts_vertical_placement():
    """flip_y=True negates the y axis: y=0 ends up at the bottom of the canvas."""
    top_in_stage = _flat(10.0)
    bottom_in_stage = _flat(20.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=0.0, y_mm=4e-3),
    ]
    no_flip = stitch_region(
        {"0_0": top_in_stage, "0_1": bottom_in_stage},
        coords, pixel_um=1.0, target_longest_px=1024, flip_y=False,
    )
    with_flip = stitch_region(
        {"0_0": top_in_stage, "0_1": bottom_in_stage},
        coords, pixel_um=1.0, target_longest_px=1024, flip_y=True,
    )
    assert no_flip is not None and with_flip is not None
    # Without flip, y=0 (top_in_stage, value 10) sits at the top of the canvas.
    # With flip, the y=0 tile sits at the bottom instead.
    assert no_flip.mip[0, 0] == 10.0
    assert with_flip.mip[0, 0] == 20.0


def test_percentiles_ignore_black_gaps():
    """Auto-contrast computed on covered pixels only — black gaps don't pull
    p1 toward 0."""
    a = _flat(100.0)
    coords = [
        FovCoord("0_0", x_mm=0.0, y_mm=0.0),
        FovCoord("0_1", x_mm=8e-3, y_mm=0.0),  # 8 um gap from origin (FOV is 4 px)
    ]
    result = stitch_region(
        {"0_0": a}, coords,  # 0_1 missing -> big black gap
        pixel_um=1.0, target_longest_px=1024,
    )
    assert result is not None
    # p1 should reflect the covered pixels' percentile (all 100s), not 0.
    assert result.p1 == 100.0
    assert result.p999 == 100.0
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/test_stitch.py -v
```

Expected: FAIL — `ImportError: cannot import name 'stitch_region' from 'gallery_view.stitch'`.

- [ ] **Step 3: Implement `stitch_region`**

Replace `src/gallery_view/stitch.py` entirely:

```python
"""Coordinate-based region stitcher: places per-FOV Z-MIPs by stage position
and mean-blends overlaps. Pure functions, no Qt, no thread state."""

import math
from dataclasses import dataclass

import numpy as np

from .types import AxisMip


@dataclass(frozen=True)
class FovCoord:
    """Stage position of one FOV center (in millimeters)."""
    fov: str   # composite '<region>_<fov>' matching acq.fovs
    x_mm: float
    y_mm: float


def stitch_region(
    fov_mips: dict[str, np.ndarray],
    coords: list[FovCoord],
    pixel_um: float,
    target_longest_px: int = 1024,
    flip_y: bool = False,
) -> AxisMip | None:
    """Place each FOV's Z-MIP at its stage coordinate, downsample to fit
    ``target_longest_px`` on the longest axis, mean-blend overlaps.

    Returns ``None`` when ``fov_mips`` is empty, when FOV shapes differ, or
    when no FovCoord lines up with a key in ``fov_mips``. Auto-contrast
    percentiles are computed over the covered region only (so black gaps
    don't drag ``p1`` toward zero).
    """
    if not fov_mips:
        return None

    # Shape uniformity check — region view doesn't support heterogeneous FOVs.
    shapes = {arr.shape for arr in fov_mips.values()}
    if len(shapes) != 1:
        return None
    ny, nx = next(iter(shapes))

    # Filter coords to ones we have a tile for; bail if none match.
    usable = [c for c in coords if c.fov in fov_mips]
    if not usable:
        return None

    pixel_mm = pixel_um / 1000.0
    nx_mm = nx * pixel_mm
    ny_mm = ny * pixel_mm

    # FOV top-left corners in stage mm (CSV holds FOV centers).
    def corner(c: FovCoord) -> tuple[float, float]:
        x0 = c.x_mm - nx_mm / 2.0
        y0 = c.y_mm - ny_mm / 2.0
        return x0, y0

    corners = [corner(c) for c in usable]
    min_x = min(x for x, _ in corners)
    max_x = max(x for x, _ in corners) + nx_mm
    raw_min_y = min(y for _, y in corners)
    raw_max_y = max(y for _, y in corners) + ny_mm

    # Full-res canvas in pixels.
    W_full = max(nx, int(math.ceil((max_x - min_x) / pixel_mm)))
    H_full = max(ny, int(math.ceil((raw_max_y - raw_min_y) / pixel_mm)))

    # Integer downsample factor so the longest axis fits target_longest_px.
    factor = max(1, math.ceil(max(H_full, W_full) / max(target_longest_px, 1)))
    # Round FOV shape to a multiple of factor so block-mean is exact.
    ny_ds = max(1, ny // factor)
    nx_ds = max(1, nx // factor)
    H = max(ny_ds, H_full // factor)
    W = max(nx_ds, W_full // factor)

    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for c in usable:
        tile = fov_mips[c.fov]
        # Block-mean downsample (truncate to multiples of factor).
        ny_use, nx_use = ny_ds * factor, nx_ds * factor
        tile_ds = (
            tile[:ny_use, :nx_use]
            .reshape(ny_ds, factor, nx_ds, factor)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )

        x0, y0 = corner(c)
        # Downsampled pixel offsets.
        col = int(round((x0 - min_x) / pixel_mm / factor))
        if flip_y:
            # Flip stage Y so increasing y_mm moves down on the canvas.
            row = int(round((raw_max_y - (y0 + ny_mm)) / pixel_mm / factor))
        else:
            row = int(round((y0 - raw_min_y) / pixel_mm / factor))

        row = max(0, min(H - ny_ds, row))
        col = max(0, min(W - nx_ds, col))

        accum[row:row + ny_ds, col:col + nx_ds] += tile_ds
        weight[row:row + ny_ds, col:col + nx_ds] += 1.0

    mosaic = accum / np.maximum(weight, 1.0)
    covered = weight > 0

    if not covered.any():
        return None

    p1 = float(np.percentile(mosaic[covered], 0.5))
    p999 = float(np.percentile(mosaic[covered], 99.5))
    return AxisMip(mip=mosaic, p1=p1, p999=p999)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/test_stitch.py -v
```

Expected: PASS (9 tests).

- [ ] **Step 5: Run full suite**

```bash
pytest -q
```

Expected: PASS (138 tests).

- [ ] **Step 6: Commit**

```bash
git add src/gallery_view/stitch.py tests/test_stitch.py
git commit -m "Add stitch_region: coordinate-based mosaic with mean-blend"
```

---

## Task 7: Add `RegionStitchJob` to the loader

**Files:**
- Modify: `src/gallery_view/loader.py`
- Create: `tests/test_loader_region.py`

- [ ] **Step 1: Write the failing test**

`tests/test_loader_region.py`:

```python
"""MipLoader dispatches RegionStitchJob via stitch_region, caches the
result, and emits region_mip_ready. Failure path emits progress."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from unittest.mock import patch

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from gallery_view import cache
from gallery_view.loader import MipLoader, RegionStitchJob
from gallery_view.sources.single_tiff import SingleTiffHandler
from gallery_view.stitch import FovCoord
from gallery_view.types import AxisMip


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))
    return tmp_path / "mips"


def _run_loader_until(loader: MipLoader, predicate, timeout=2.0):
    """Spin the loader thread until ``predicate()`` is True or we time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _make_acq(make_squid_single_tiff_acq, regions=1, fovs_per_region=2):
    folder = make_squid_single_tiff_acq(
        regions=regions, fovs_per_region=fovs_per_region,
        nz=2, ny=4, nx=4, write_coords=True,
    )
    handler = SingleTiffHandler()
    return handler, handler.build(
        str(folder), {"dz(um)": 2.0, "sensor_pixel_size_um": 6.5}
    )


def test_region_stitch_job_emits_ready_on_cache_miss(qapp, make_squid_single_tiff_acq):
    handler, acq = _make_acq(make_squid_single_tiff_acq, regions=1, fovs_per_region=2)
    fov_mips = {
        "0_0": np.ones((4, 4), dtype=np.float32),
        "0_1": np.full((4, 4), 2.0, dtype=np.float32),
    }
    coords = [FovCoord("0_0", 0.0, 0.0), FovCoord("0_1", 4e-3, 0.0)]

    received: list = []
    loader = MipLoader()
    loader.region_mip_ready.connect(
        lambda *args: received.append(args)
    )
    loader.start()
    try:
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips=fov_mips, coords=coords,
        ))
        assert _run_loader_until(loader, lambda: bool(received))
        acq_id, t, region, ch_idx, wl, ax_mip = received[0]
        assert acq_id == 0 and t == "0" and region == "0" and ch_idx == 0
        assert isinstance(ax_mip, AxisMip)
        assert ax_mip.mip.shape == (4, 8)
    finally:
        loader.stop()
        loader.wait(2000)


def test_region_stitch_job_short_circuits_on_cache_hit(qapp, make_squid_single_tiff_acq):
    handler, acq = _make_acq(make_squid_single_tiff_acq)
    # Pre-populate cache.
    src, ch_id = handler.cache_key_region(acq, "0", acq.channels[0], timepoint="0")
    cached_mip = np.arange(16, dtype=np.float32).reshape(4, 4)
    cache.save(src, ch_id, {"z": AxisMip(mip=cached_mip, p1=0.0, p999=15.0)})

    received: list = []
    loader = MipLoader()
    loader.region_mip_ready.connect(lambda *args: received.append(args))
    loader.start()
    try:
        # Pass empty fov_mips — a cache miss would fail, but the cache hit
        # short-circuits before stitch_region is called.
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips={}, coords=[],
        ))
        assert _run_loader_until(loader, lambda: bool(received))
        ax_mip = received[0][-1]
        np.testing.assert_array_equal(ax_mip.mip, cached_mip)
    finally:
        loader.stop()
        loader.wait(2000)


def test_region_stitch_job_failure_emits_progress(qapp, make_squid_single_tiff_acq):
    handler, acq = _make_acq(make_squid_single_tiff_acq)

    progress_messages: list[str] = []
    loader = MipLoader()
    loader.progress.connect(lambda d, q, msg: progress_messages.append(msg))
    loader.start()
    try:
        # stitch_region returns None for empty inputs; loader treats None
        # as a failure and emits a progress message.
        loader.enqueue_region(RegionStitchJob(
            acq_id=0, acq=acq, region="0",
            channel=acq.channels[0], ch_idx=0,
            timepoint="0", fov_mips={}, coords=[],
        ))
        assert _run_loader_until(
            loader,
            lambda: any("region" in m and "failed" in m for m in progress_messages),
        )
    finally:
        loader.stop()
        loader.wait(2000)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/test_loader_region.py -v
```

Expected: FAIL — `ImportError: cannot import name 'RegionStitchJob'`.

- [ ] **Step 3: Add `RegionStitchJob` and dispatch in `loader.py`**

In `src/gallery_view/loader.py`, replace the imports and the `Job` dataclass area with both job types, and update `MipLoader`. Below is the full updated file content (the unchanged portions are preserved):

```python
"""Long-lived MIP loader thread.

Two job types share the queue: ``Job`` (per-FOV Z/Y/X MIPs) and
``RegionStitchJob`` (stitched per-region XY mosaic). ``cancel(acq_id)``
prunes pending jobs for that acq from the queue; in-flight jobs run to
completion. Emits ``mip_ready`` per finished channel, ``region_mip_ready``
per finished region, and ``progress`` after each step.
"""

import queue
from dataclasses import dataclass, field

import numpy as np
from qtpy.QtCore import QThread, Signal

from . import cache, mips, stitch
from .stitch import FovCoord
from .types import Acquisition, AxisMip, Channel, ChannelMips, ShapeZYX


@dataclass
class Job:
    acq_id: int
    acq: Acquisition
    fov: str
    channel: Channel
    ch_idx: int
    timepoint: str = "0"


@dataclass
class RegionStitchJob:
    acq_id: int
    acq: Acquisition
    region: str
    channel: Channel
    ch_idx: int
    timepoint: str
    fov_mips: dict[str, np.ndarray]    # composite fov_id -> (ny, nx) float32 Z-MIP
    coords: list[FovCoord]
    target_longest_px: int = 1024
    flip_y: bool = False


class MipLoader(QThread):
    # acq_id, timepoint, fov, ch_idx, wavelength, channel_mips, shape_zyx
    mip_ready = Signal(int, str, str, int, str, object, object)
    # acq_id, timepoint, region, ch_idx, wavelength, AxisMip
    region_mip_ready = Signal(int, str, str, int, str, object)
    # done_total, queued_total, message
    progress = Signal(int, int, str)
    idle = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue[Job | RegionStitchJob | None] = queue.Queue()
        self._done = 0
        self._enqueued = 0
        self._cancelled_acqs: set[int] = set()
        self._stop = False
        self._idle_emitted = False

    # ── public API (call from GUI thread) ──

    def enqueue(self, job: Job) -> None:
        self._enqueued += 1
        self._idle_emitted = False
        self._queue.put(job)

    def enqueue_region(self, job: RegionStitchJob) -> None:
        self._enqueued += 1
        self._idle_emitted = False
        self._queue.put(job)

    def cancel(self, acq_id: int) -> None:
        self._cancelled_acqs.add(acq_id)

    def stop(self) -> None:
        self._stop = True
        self._queue.put(None)

    # ── thread loop ──

    def run(self) -> None:
        while not self._stop:
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                if (
                    self._enqueued
                    and self._done >= self._enqueued
                    and not self._idle_emitted
                ):
                    self._idle_emitted = True
                    self.idle.emit()
                continue
            if job is None:
                break
            if job.acq_id in self._cancelled_acqs:
                self._done += 1
                self._emit_progress("skipped (cancelled)")
                continue
            try:
                if isinstance(job, RegionStitchJob):
                    self._process_region(job)
                else:
                    self._process(job)
            except Exception as exc:  # noqa: BLE001
                self._done += 1
                kind = "region" if isinstance(job, RegionStitchJob) else (
                    f"{job.channel.wavelength}nm"
                )
                self._emit_progress(
                    f"failed {kind} — {job.acq.display_name}: {exc}"
                )

    def _process(self, job: Job) -> None:
        # ... existing _process body unchanged ...
```

Keep the existing `_process(self, job: Job)` body exactly as it was. Add a new `_process_region` method after it:

```python
    def _process_region(self, job: RegionStitchJob) -> None:
        src, ch_id = job.acq.handler.cache_key_region(
            job.acq, job.region, job.channel, timepoint=job.timepoint
        )
        cached, _ = cache.load(src, ch_id)
        if cached is not None and "z" in cached:
            self.region_mip_ready.emit(
                job.acq_id, job.timepoint, job.region, job.ch_idx,
                job.channel.wavelength, cached["z"],
            )
            self._done += 1
            self._emit_progress(
                f"region {job.region} {job.channel.wavelength}nm cached — {job.acq.display_name}"
            )
            return

        self._emit_progress(
            f"stitching region {job.region} {job.channel.wavelength}nm — {job.acq.display_name}"
        )
        result = stitch.stitch_region(
            job.fov_mips, job.coords,
            pixel_um=_pixel_um_for(job.acq),
            target_longest_px=job.target_longest_px,
            flip_y=job.flip_y,
        )
        if result is None:
            self._done += 1
            self._emit_progress(
                f"failed region {job.region} {job.channel.wavelength}nm — {job.acq.display_name}: empty stitch"
            )
            return

        cache.save(src, ch_id, {"z": result})
        # Re-apply any saved LUT override (same pattern as _process).
        overrides = cache._load_lut_override(src, ch_id)
        if overrides and "z" in overrides:
            p1, p999 = overrides["z"]
            result = AxisMip(mip=result.mip, p1=p1, p999=p999)

        self.region_mip_ready.emit(
            job.acq_id, job.timepoint, job.region, job.ch_idx,
            job.channel.wavelength, result,
        )
        self._done += 1
        self._emit_progress(
            f"region {job.region} {job.channel.wavelength}nm stitched — {job.acq.display_name}"
        )
```

Add a helper at module level (below the imports, above the dataclasses):

```python
def _pixel_um_for(acq: Acquisition) -> float:
    """In-plane pixel size in micrometres. Mirrors the GalleryWindow logic."""
    from .sources._squid_common import resolve_mag

    sensor = acq.params.get("sensor_pixel_size_um", 6.5)
    mag = resolve_mag(acq.folder_name, acq.params) or 1
    return sensor / max(mag, 1)
```

Also keep the existing `_emit_ready` and `_emit_progress` methods unchanged.

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/test_loader_region.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Run full suite**

```bash
pytest -q
```

Expected: PASS (141 tests).

- [ ] **Step 6: Commit**

```bash
git add src/gallery_view/loader.py tests/test_loader_region.py
git commit -m "Add RegionStitchJob + region_mip_ready signal to MipLoader"
```

---

## Task 8: Add `View: [FOV] [Region]` toolbar and Settings entry (UI scaffolding only)

This task wires the toolbar widget and Settings checkbox but does **not** yet change row-building. Toggling the View buttons just sets `self.view_mode` and triggers a no-op rebuild. This isolates the visual surface change from the behavioral one.

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py:106-300` (`__init__`, `_build_display_row`, `_build_menus`)
- Create: `tests/test_gallery_window_region.py`

- [ ] **Step 1: Write the failing tests (scaffolding only — row content is checked later)**

`tests/test_gallery_window_region.py`:

```python
"""Region-view UI: toolbar wiring, button enable/disable rules, mode flag."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_view_toolbar_present(qapp):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        assert "fov" in win.view_buttons
        assert "region" in win.view_buttons
        assert win.view_mode == "fov"
        assert win.view_buttons["fov"].isChecked()
        assert not win.view_buttons["region"].isChecked()
    finally:
        win.close()


def test_region_button_disabled_when_no_multi_region_source(qapp, make_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        win._add_source(str(make_single_tiff_acq()))
        # Legacy folder -> regions == ["0"] -> Region button disabled.
        assert not win.view_buttons["region"].isEnabled()
    finally:
        win.close()


def test_region_button_enabled_with_multi_region_source(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=3, fovs_per_region=1)
        win._add_source(str(folder))
        assert win.view_buttons["region"].isEnabled()
    finally:
        win.close()


def test_switching_to_region_disables_axis_xz_yz(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=1)
        win._add_source(str(folder))
        win._set_view_mode("region")
        assert win.view_mode == "region"
        assert win.view_axis == "z"
        assert not win.axis_buttons["y"].isEnabled()
        assert not win.axis_buttons["x"].isEnabled()
        win._set_view_mode("fov")
        assert win.axis_buttons["y"].isEnabled()
        assert win.axis_buttons["x"].isEnabled()
    finally:
        win.close()


def test_settings_has_expand_all_regions(qapp):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        assert win.expand_region_action.isCheckable()
        assert not win.expand_region_action.isChecked()
    finally:
        win.close()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/test_gallery_window_region.py -v
```

Expected: FAIL — `AttributeError: 'GalleryWindow' object has no attribute 'view_buttons'`.

- [ ] **Step 3: Add state in `__init__`**

In `src/gallery_view/ui/gallery_window.py`, in `GalleryWindow.__init__` after the existing `self.expanded_fov_mode = False` line (around line 128), add:

```python
        self.view_mode: str = "fov"  # "fov" | "region"
        self.expanded_region_mode: bool = False
        # (acq_id, timepoint, region, ch_idx) -> AxisMip for stitched mosaics
        self.region_mip_data: dict[tuple[int, str, str, int], AxisMip] = {}
        # (acq_id, timepoint, region) -> set of (ch_idx, fov_id) we've seen,
        # used to detect "all FOV MIPs ready, time to stitch".
        self._region_fov_readiness: dict[
            tuple[int, str, str], set[tuple[int, str]]
        ] = {}
```

Also import `AxisMip` (already imported) — no change needed there. Add the import for `RegionStitchJob` at the top:

```python
from ..loader import Job, MipLoader, RegionStitchJob
```

- [ ] **Step 4: Build the toolbar group**

In `_build_display_row`, after the `Project:` button loop (around line 230) and before `row.addSpacing(16); size_lbl = ...`, insert the View group. Replace the `_build_display_row` method body:

```python
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
        view_lbl = QLabel("View:")
        view_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(view_lbl)

        self.view_btn_group = QButtonGroup(self)
        self.view_btn_group.setExclusive(True)
        self.view_buttons: dict[str, QPushButton] = {}
        for mode, label in [("fov", "FOV"), ("region", "Region")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedSize(54, 22)
            btn.setStyleSheet(self._toggle_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.setChecked(mode == "fov")
            btn.clicked.connect(lambda _, m=mode: self._set_view_mode(m))
            self.view_btn_group.addButton(btn)
            self.view_buttons[mode] = btn
            row.addWidget(btn)
        # Region button starts disabled; _refresh_region_button_enabled enables
        # it once a source with multi-region data is added.
        self.view_buttons["region"].setEnabled(False)
        self.view_buttons["region"].setToolTip(
            "No source supports region view"
        )

        row.addSpacing(16)
        size_lbl = QLabel("Thumbnail size:")
        size_lbl.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(size_lbl)
        self.size_combo = QComboBox()
        _style_combo(self.size_combo)
        for label, size in THUMB_SIZE_PRESETS:
            self.size_combo.addItem(label, size)
        _size_combo_to_contents(self.size_combo)
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
```

- [ ] **Step 5: Add `_set_view_mode` and `_refresh_region_button_enabled` methods**

After the existing `_set_axis` method (around line 856), add:

```python
    def _set_view_mode(self, mode: str) -> None:
        if mode == self.view_mode:
            return
        self.view_mode = mode
        # XZ/YZ stitched MIPs aren't supported; force XY in region view.
        if mode == "region":
            if self.view_axis != "z":
                self.view_axis = "z"
                self.axis_buttons["z"].setChecked(True)
            self.axis_buttons["y"].setEnabled(False)
            self.axis_buttons["x"].setEnabled(False)
        else:
            self.axis_buttons["y"].setEnabled(True)
            self.axis_buttons["x"].setEnabled(True)
        self._rebuild_rows()

    def _refresh_region_button_enabled(self) -> None:
        """Region button is enabled iff at least one loaded acquisition has
        more than one region."""
        any_multi_region = any(
            acq is not None and len(acq.regions) > 1
            for acq in self.acquisitions
        )
        self.view_buttons["region"].setEnabled(any_multi_region)
        if any_multi_region:
            self.view_buttons["region"].setToolTip("")
        else:
            self.view_buttons["region"].setToolTip(
                "No source supports region view"
            )
            if self.view_mode == "region":
                # Source removed; fall back to FOV view.
                self.view_buttons["fov"].setChecked(True)
                self._set_view_mode("fov")
```

In `_add_source` and `_remove_source`, after the existing call to `self._rebuild_mag_filter()`, add a call to `self._refresh_region_button_enabled()`:

In `_add_source` (around line 376), after `self._rebuild_mag_filter()`:
```python
        self._refresh_region_button_enabled()
```

In `_remove_source` (around line 392-393), after `self._rebuild_mag_filter()`:
```python
        self._refresh_region_button_enabled()
```

- [ ] **Step 6: Add the Settings entry**

In `_build_menus`, after `settings_menu.addAction(self.expand_action)` (around line 288), add:

```python
        self.expand_region_action = QAction(
            "Expand all regions as separate rows", self
        )
        self.expand_region_action.setCheckable(True)
        self.expand_region_action.toggled.connect(self._set_expanded_region_mode)
        settings_menu.addAction(self.expand_region_action)
```

And add a `_set_expanded_region_mode` method next to `_set_expanded_fov_mode`:

```python
    def _set_expanded_region_mode(self, checked: bool) -> None:
        self.expanded_region_mode = checked
        if self.view_mode == "region":
            self._rebuild_rows()
```

- [ ] **Step 7: Run tests, expect pass**

```bash
pytest tests/test_gallery_window_region.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 8: Run full suite — existing smoke test still passes**

```bash
pytest -q
```

Expected: PASS (146 tests).

- [ ] **Step 9: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py tests/test_gallery_window_region.py
git commit -m "Add View: [FOV] [Region] toolbar group and expand-regions setting"
```

---

## Task 9: Generalize row building for region view (combo, no 3D button, region rows)

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py` (`_fovs_for_row` → `_row_units_for`, `_make_row_widget`, `_make_source_group`, `_on_fov_changed`, mode-aware status)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gallery_window_region.py`:

```python
def test_region_view_compact_renders_one_row_per_acq_with_region_combo(
    qapp, make_squid_single_tiff_acq
):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=3, fovs_per_region=2)
        win._add_source(str(folder))
        win._set_view_mode("region")
        # One row per acquisition (compact mode).
        assert len(win.row_widgets) == 1
        key = next(iter(win.row_widgets))
        acq_id, t, unit = key
        rw = win.row_widgets[key]
        # Region combo is shown when multi-region; FOV combo is hidden.
        assert rw.fov_combo is not None  # repurposed for region in region view
        # Open 3D View is not rendered in region rows.
        # (Walk the button column inside the row container — easier: check
        # there's no button with that label.)
        from qtpy.QtWidgets import QPushButton
        buttons = rw.container.findChildren(QPushButton)
        assert not any(b.text() == "Open 3D View" for b in buttons)
    finally:
        win.close()


def test_region_view_expanded_renders_one_row_per_region(
    qapp, make_squid_single_tiff_acq
):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=3, fovs_per_region=2)
        win._add_source(str(folder))
        win._set_view_mode("region")
        win.expand_region_action.setChecked(True)  # triggers _set_expanded_region_mode
        # 3 regions = 3 rows.
        assert len(win.row_widgets) == 3
    finally:
        win.close()


def test_switching_back_to_fov_restores_fov_rows(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=3)
        win._add_source(str(folder))
        win._set_view_mode("region")
        assert len(win.row_widgets) == 1  # compact region view
        win._set_view_mode("fov")
        # FOV view compact: one row per acquisition.
        assert len(win.row_widgets) == 1
        # In FOV view, the row's unit slot is a composite fov, not a region.
        key = next(iter(win.row_widgets))
        _, _, unit = key
        assert "_" in unit  # composite "<region>_<fov>"
    finally:
        win.close()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/test_gallery_window_region.py -v
```

Expected: FAIL — `_set_view_mode` doesn't yet rebuild rows in a region-aware way.

- [ ] **Step 3: Generalize `_fovs_for_row` to `_row_units_for`**

In `src/gallery_view/ui/gallery_window.py`, replace `_fovs_for_row` (around line 495-499) with:

```python
    def _row_units_for(self, acq_id: int) -> list[str]:
        """Return the row-unit ids for this acquisition under the current view.

        FOV view: one or all FOVs (composite ``<region>_<fov>``).
        Region view: one or all regions.
        """
        acq = self.acquisitions[acq_id]
        if acq is None:
            return []
        if self.view_mode == "region":
            return acq.regions if self.expanded_region_mode else [acq.selected_region]
        return acq.fovs if self.expanded_fov_mode else [acq.selected_fov]
```

Search the file for any remaining `_fovs_for_row` references and replace with `_row_units_for`. There are two known callers:

- In `_refresh_visibility` (around line 471) — replace `self._fovs_for_row(acq_id)` with `self._row_units_for(acq_id)`.
- In `_make_source_group` (around line 561) — replace `for fov in self._fovs_for_row(acq_id):` with `for unit in self._row_units_for(acq_id):` and rename the loop variable from `fov` to `unit` throughout the loop body. Update the `RowKey` construction: `key = RowKey(acq_id, t, unit)`.

- [ ] **Step 4: Update `_make_row_widget` to branch on `view_mode`**

Replace `_make_row_widget` (around line 584-722) — key changes are:
- Use `unit` as the parameter name (replaces `key.fov`).
- In region view, the FOV combo becomes a Region combo.
- In region view, omit the `Open 3D View` button.

Replace the method:

```python
    def _make_row_widget(self, key, acq, active_wls):
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        h = QHBoxLayout(container)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(4)

        unit = key.fov  # in region view, this is the region id
        is_region_view = self.view_mode == "region"

        mag = resolve_mag(acq.folder_name, acq.params) or "?"
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

        # Per-row unit label (visible in expanded mode for either view).
        expanded = self.expanded_region_mode if is_region_view else self.expanded_fov_mode
        nunits = len(acq.regions if is_region_view else acq.fovs)
        if expanded and nunits > 1:
            label = f"Region {unit}" if is_region_view else f"FOV {display_fov(unit)}"
            unit_lbl = QLabel(label)
            unit_lbl.setFixedWidth(60)
            unit_lbl.setStyleSheet("color: #888; font-size: 10px;")
            h.addWidget(unit_lbl)

        name_lbl = QLabel(acq.display_name)
        name_lbl.setFixedWidth(80)
        name_lbl.setWordWrap(True)
        name_lbl.setToolTip(acq.path)
        name_lbl.setStyleSheet("color: #ccc; font-size: 9px; font-weight: bold;")
        h.addWidget(name_lbl)

        # One column per active wavelength.
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

        # Time picker (unchanged — both views respect timepoints).
        time_combo: QComboBox | None = None
        if len(acq.timepoints) > 1:
            time_combo = QComboBox()
            _style_combo(time_combo)
            for t in acq.timepoints:
                time_combo.addItem(f"t={t}", t)
            _size_combo_to_contents(time_combo)
            time_combo.setCurrentIndex(acq.timepoints.index(acq.selected_timepoint))
            time_combo.currentIndexChanged.connect(
                lambda i, k=key, c=time_combo: self._on_timepoint_changed(k, c.itemData(i))
            )
            h.addWidget(time_combo)

        # Unit picker (FOV combo in FOV view, Region combo in region view).
        unit_combo: QComboBox | None = None
        if not expanded:
            if is_region_view and len(acq.regions) > 1:
                unit_combo = QComboBox()
                _style_combo(unit_combo)
                for r in acq.regions:
                    unit_combo.addItem(f"Region {r}", r)
                _size_combo_to_contents(unit_combo)
                unit_combo.setCurrentIndex(acq.regions.index(acq.selected_region))
                unit_combo.currentIndexChanged.connect(
                    lambda i, k=key, c=unit_combo: self._on_region_changed(
                        k, c.itemData(i)
                    )
                )
                h.addWidget(unit_combo)
            elif (not is_region_view) and len(acq.fovs) > 1:
                unit_combo = QComboBox()
                _style_combo(unit_combo)
                for fov in acq.fovs:
                    unit_combo.addItem(f"FOV {display_fov(fov)}", fov)
                _size_combo_to_contents(unit_combo)
                unit_combo.setCurrentIndex(acq.fovs.index(acq.selected_fov))
                unit_combo.currentIndexChanged.connect(
                    lambda i, k=key, c=unit_combo: self._on_fov_changed(
                        k, c.itemData(i)
                    )
                )
                h.addWidget(unit_combo)

        # Action buttons — 'Open 3D View' suppressed in region view.
        btn_col = QVBoxLayout()
        btn_col.setSpacing(2)
        if not is_region_view:
            btn_3d = QPushButton("Open 3D View")
            btn_3d.setFixedSize(120, 30)
            btn_3d.setCursor(Qt.PointingHandCursor)
            btn_3d.setStyleSheet(
                "QPushButton { background-color: #2d5aa0; color: white; border-radius: 4px;"
                " font-size: 11px; font-weight: bold; }"
                "QPushButton:hover { background-color: #3a6fc0; }"
            )
            btn_3d.clicked.connect(lambda _, k=key: self._open_napari(k))
            btn_col.addWidget(btn_3d)

        btn_lut = QPushButton("Adjust Contrast")
        btn_lut.setFixedSize(120, 30)
        btn_lut.setCursor(Qt.PointingHandCursor)
        btn_lut.setStyleSheet(
            "QPushButton { background-color: #555; color: white; border-radius: 4px;"
            " font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background-color: #777; }"
        )
        btn_lut.clicked.connect(lambda _, k=key: self._adjust_lut(k))
        btn_col.addWidget(btn_lut)

        h.addLayout(btn_col)

        return RowWidgets(
            container=container,
            mag_lbl=mag_lbl,
            time_lbl=time_lbl,
            name_lbl=name_lbl,
            thumb_labels=thumb_labels,
            thumb_columns=thumb_columns,
            fov_combo=unit_combo,
            time_combo=time_combo,
        )
```

- [ ] **Step 5: Add `_on_region_changed`**

After `_on_fov_changed` (around line 724-732), add:

```python
    def _on_region_changed(self, key, new_region: str) -> None:
        acq = self.acquisitions[key.acq_id]
        acq.selected_region = new_region
        self._rebuild_rows()
        # Region view eagerly enqueues per-FOV MIPs for all FOVs in the
        # selected region; that path is added in Task 10.
        if self.view_mode == "region":
            self._enqueue_region_prereqs(
                key.acq_id, acq, acq.selected_timepoint, new_region
            )
```

For now stub `_enqueue_region_prereqs` so the smoke tests don't crash:

```python
    def _enqueue_region_prereqs(
        self, acq_id: int, acq, timepoint: str, region: str
    ) -> None:
        """Wired up in Task 10."""
        pass
```

- [ ] **Step 6: Run tests, expect pass**

```bash
pytest tests/test_gallery_window_region.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 7: Run full suite**

```bash
pytest -q
```

Expected: PASS (149 tests).

- [ ] **Step 8: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py tests/test_gallery_window_region.py
git commit -m "Region-aware row building: Region combo, drop 3D button in region rows"
```

---

## Task 10: Enqueue FOV prerequisites and dispatch `RegionStitchJob` when ready

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py` (`_enqueue_region_prereqs`, `_on_mip_ready`, `_on_region_mip_ready`, `_add_source`, `_render_region_thumb`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gallery_window_region.py`:

```python
def test_region_view_enqueues_fov_prereqs_then_stitch(
    qapp, make_squid_single_tiff_acq
):
    """Switching to region view should enqueue per-FOV MIP jobs for the
    selected region, and once all of them land we get a region stitch job.
    Verify by waiting for region_mip_ready to fire."""
    import time

    from gallery_view.ui.gallery_window import GalleryWindow

    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=2, nz=2, ny=4, nx=4, write_coords=True,
    )
    win = GalleryWindow()
    received: list = []
    win.loader.region_mip_ready.connect(lambda *args: received.append(args))
    try:
        win._add_source(str(folder))
        win._set_view_mode("region")
        # Switching modes triggers _rebuild_rows which (in region view)
        # enqueues prereqs and waits for them; the loader emits region_mip_ready
        # after stitch completes.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            QApplication.processEvents()
            if received:
                break
            time.sleep(0.05)
        assert received, "region_mip_ready never fired"
    finally:
        win.close()
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_gallery_window_region.py::test_region_view_enqueues_fov_prereqs_then_stitch -v
```

Expected: FAIL — no region_mip_ready emission.

- [ ] **Step 3: Replace `_enqueue_region_prereqs` stub with the real thing**

In `src/gallery_view/ui/gallery_window.py`, find the stub from Task 9 and replace with:

```python
    def _enqueue_region_prereqs(
        self, acq_id: int, acq, timepoint: str, region: str
    ) -> None:
        """For region view: enqueue per-FOV Z-MIP jobs for every FOV in
        ``region``, and register a readiness set we tick off as
        ``_on_mip_ready`` lands MIPs. When the set is complete we enqueue
        a ``RegionStitchJob`` per channel.

        Cache hits short-circuit through the loader as usual; this method is
        cheap to call repeatedly.
        """
        # FOVs in this region (composite '<region>_<fov>' filter on acq.fovs).
        fovs_in_region = [f for f in acq.fovs if f.split("_", 1)[0] == region]
        if not fovs_in_region:
            return
        key = (acq_id, timepoint, region)
        needed: set[tuple[int, str]] = set()
        for ch_idx in range(len(acq.channels)):
            for fov in fovs_in_region:
                needed.add((ch_idx, fov))
        self._region_fov_readiness[key] = needed.copy()
        # Track what's already in mip_data so the stitch trigger doesn't wait
        # for jobs that already completed before we switched modes.
        already = {
            (ci, f)
            for (a, t, f, ci, ax) in self.mip_data
            if a == acq_id and t == timepoint and ax == "z" and (
                ci, f
            ) in needed
        }
        self._region_fov_readiness[key] -= already

        # Enqueue any missing FOV jobs (loader skips cached ones in O(1)).
        for ch_idx, channel in enumerate(acq.channels):
            for fov in fovs_in_region:
                if (ch_idx, fov) in already:
                    continue
                self.loader.enqueue(Job(
                    acq_id=acq_id, acq=acq, fov=fov, channel=channel,
                    ch_idx=ch_idx, timepoint=timepoint,
                ))

        # If all FOV MIPs were already in memory, fire stitches immediately.
        if not self._region_fov_readiness[key]:
            for ch_idx, channel in enumerate(acq.channels):
                self._dispatch_region_stitch(
                    acq_id, acq, timepoint, region, ch_idx, channel
                )
```

- [ ] **Step 4: Add the dispatch helper and `_on_region_mip_ready`**

After `_enqueue_region_prereqs`, add:

```python
    def _dispatch_region_stitch(
        self, acq_id: int, acq, timepoint: str, region: str,
        ch_idx: int, channel,
    ) -> None:
        # Don't support region view for handlers that can't supply coords or
        # a region cache key.
        try:
            coords_map = acq.handler._load_coords(acq) if hasattr(
                acq.handler, "_load_coords"
            ) else None
        except AttributeError:
            coords_map = None
        if not coords_map or region not in coords_map:
            self.status.setText(
                f"{acq.display_name}: region view needs coordinates.csv"
            )
            return
        coords = coords_map[region]

        fov_mips = {
            f: self.mip_data[(acq_id, timepoint, f, ch_idx, "z")].mip
            for (a, t, f, ci, ax) in list(self.mip_data)
            if a == acq_id and t == timepoint and ci == ch_idx and ax == "z"
            and f.split("_", 1)[0] == region
        }
        if not fov_mips:
            return
        self.loader.enqueue_region(RegionStitchJob(
            acq_id=acq_id, acq=acq, region=region,
            channel=channel, ch_idx=ch_idx, timepoint=timepoint,
            fov_mips=fov_mips, coords=coords,
        ))

    def _on_region_mip_ready(
        self, acq_id, timepoint, region, ch_idx, wavelength, ax_mip,
    ) -> None:
        if acq_id >= len(self.acquisitions) or self.acquisitions[acq_id] is None:
            return
        self.region_mip_data[(acq_id, timepoint, region, ch_idx)] = ax_mip
        if self.view_mode == "region":
            self._render_region_thumb(acq_id, timepoint, region, ch_idx, ax_mip)
```

- [ ] **Step 5: Connect the new signal in `__init__`**

In `GalleryWindow.__init__`, after `self.loader.idle.connect(self._on_idle)` (around line 137), add:

```python
        self.loader.region_mip_ready.connect(self._on_region_mip_ready)
```

- [ ] **Step 6: Hook readiness ticking into `_on_mip_ready`**

Replace the existing `_on_mip_ready` (around line 824-841) with:

```python
    def _on_mip_ready(
        self, acq_id, timepoint, fov, ch_idx, wavelength, channel_mips, shape,
    ) -> None:
        if acq_id >= len(self.acquisitions) or self.acquisitions[acq_id] is None:
            return
        acq = self.acquisitions[acq_id]
        shape_was_unknown = acq.shape_zyx is None
        if shape is not None and shape_was_unknown:
            acq.shape_zyx = shape
        for axis, ax_mip in channel_mips.items():
            self.mip_data[(acq_id, timepoint, fov, ch_idx, axis)] = ax_mip
        if self.view_axis in channel_mips and self.view_mode == "fov":
            self._render_thumb(
                acq_id, timepoint, fov, ch_idx, channel_mips[self.view_axis]
            )
        if shape_was_unknown and acq.shape_zyx is not None:
            self._apply_label_sizes_for(acq_id, timepoint, fov)

        # Region-view readiness: tick this (ch_idx, fov) off the pending set;
        # if the set is now empty, enqueue a stitch.
        if self.view_mode == "region" and "z" in channel_mips:
            region = fov.split("_", 1)[0] if "_" in fov else "0"
            key = (acq_id, timepoint, region)
            pending = self._region_fov_readiness.get(key)
            if pending is not None:
                pending.discard((ch_idx, fov))
                if not pending:
                    # All FOV MIPs for this region are in — stitch every channel.
                    for ci, channel in enumerate(acq.channels):
                        self._dispatch_region_stitch(
                            acq_id, acq, timepoint, region, ci, channel
                        )
                    # Drop the readiness entry so we don't re-stitch.
                    self._region_fov_readiness.pop(key, None)
```

- [ ] **Step 7: Add `_render_region_thumb`**

After `_render_thumb` (around line 820), add:

```python
    def _render_region_thumb(
        self, acq_id, timepoint, region, ch_idx, ax_mip
    ) -> None:
        rw = self.row_widgets.get((acq_id, timepoint, region))
        if rw is None or ch_idx not in rw.thumb_labels:
            return
        acq = self.acquisitions[acq_id]
        wl = acq.channels[ch_idx].wavelength
        rgba = mip_to_rgba(ax_mip.mip, ax_mip.p1, ax_mip.p999, rgb_for(wl))
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        # Region thumbs scale to fit the cell, preserving the mosaic aspect.
        cell = self.thumb_size
        cell_aspect = 1.0
        aspect = h / max(w, 1)
        if aspect >= cell_aspect:
            img_w, img_h = max(1, int(round(cell / max(aspect, 1e-9)))), cell
        else:
            img_w, img_h = cell, max(1, int(round(cell * aspect)))
        pixmap = QPixmap.fromImage(qimg).scaled(
            img_w, img_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        rw.thumb_labels[ch_idx].setPixmap(pixmap)
        rw.thumb_labels[ch_idx].setStyleSheet(
            "background-color: #111; border: 1px solid #2a2a2a; border-radius: 3px;"
        )
```

- [ ] **Step 8: Trigger prereq enqueue on view-mode switch and on rebuild**

Replace `_set_view_mode` (added in Task 8) with the fuller version:

```python
    def _set_view_mode(self, mode: str) -> None:
        if mode == self.view_mode:
            return
        self.view_mode = mode
        if mode == "region":
            if self.view_axis != "z":
                self.view_axis = "z"
                self.axis_buttons["z"].setChecked(True)
            self.axis_buttons["y"].setEnabled(False)
            self.axis_buttons["x"].setEnabled(False)
        else:
            self.axis_buttons["y"].setEnabled(True)
            self.axis_buttons["x"].setEnabled(True)
        self._rebuild_rows()
        if mode == "region":
            for acq_id, acq in enumerate(self.acquisitions):
                if acq is None or len(acq.regions) <= 1:
                    # Single-region acquisitions have nothing to stitch.
                    continue
                for region in (
                    acq.regions if self.expanded_region_mode else [acq.selected_region]
                ):
                    self._enqueue_region_prereqs(
                        acq_id, acq, acq.selected_timepoint, region
                    )
        # Render any cached region thumbs we already have data for.
        if mode == "region":
            for (acq_id, t, region, ch_idx), ax_mip in self.region_mip_data.items():
                self._render_region_thumb(acq_id, t, region, ch_idx, ax_mip)
```

- [ ] **Step 9: Run tests, expect pass**

```bash
pytest tests/test_gallery_window_region.py -v
```

Expected: PASS (9 tests).

- [ ] **Step 10: Run full suite**

```bash
pytest -q
```

Expected: PASS (150 tests; the new region-view smoke test passes alongside existing ones).

- [ ] **Step 11: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py tests/test_gallery_window_region.py
git commit -m "Dispatch RegionStitchJob when all FOV MIPs for a region land"
```

---

## Task 11: Wire `Adjust Contrast` for region rows

`show_lut_dialog` currently takes `mip_data` keyed by `(acq_id, timepoint, fov, ch_idx, axis)` and calls `acq.handler.cache_key(...)` to write the LUT sidecar. Region rows need it to call `acq.handler.cache_key_region(...)` instead, with the region in place of fov.

**Files:**
- Modify: `src/gallery_view/ui/lut_dialog.py`
- Modify: `src/gallery_view/ui/gallery_window.py` (`_adjust_lut`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gallery_window_region.py`:

```python
def test_region_lut_sidecar_uses_region_cache_key(qapp, make_squid_single_tiff_acq, tmp_path, monkeypatch):
    """Saving a LUT override from a region row writes to the region-keyed
    sidecar path, not the FOV-keyed one."""
    from gallery_view import cache
    from gallery_view.types import AxisMip
    from gallery_view.ui.gallery_window import GalleryWindow

    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "mips"))

    folder = make_squid_single_tiff_acq(
        regions=1, fovs_per_region=1, write_coords=True,
    )
    win = GalleryWindow()
    try:
        win._add_source(str(folder))
        win._set_view_mode("region")
        # Synthesize a region mosaic in memory so the LUT dialog has something
        # to show without waiting for the loader thread.
        ax_mip = AxisMip(
            mip=__import__("numpy").zeros((4, 4), dtype="float32"),
            p1=0.0, p999=1.0,
        )
        win.region_mip_data[(0, "0", "0", 0)] = ax_mip

        # The LUT-only save path: directly call save_lut_only with the key
        # the window's _adjust_lut should produce.
        handler = win.acquisitions[0].handler
        src, ch_id = handler.cache_key_region(
            win.acquisitions[0], "0", win.acquisitions[0].channels[0]
        )
        cache.save_lut_only(src, ch_id, {"z": (ax_mip.mip, 0.1, 0.9)})

        sidecar = cache._lut_override_path(src, ch_id)
        assert sidecar.exists()
        # The sidecar path encodes 'region:' — distinct from the FOV path.
        fov_src, fov_id = handler.cache_key(
            win.acquisitions[0], win.acquisitions[0].fovs[0],
            win.acquisitions[0].channels[0],
        )
        assert cache._lut_override_path(fov_src, fov_id) != sidecar
    finally:
        win.close()
```

- [ ] **Step 2: Run test, expect pass right away (it doesn't exercise the dialog UI)**

The test verifies key-shape uniqueness, which Task 5 already established. Run it now to confirm it passes — it just documents the contract that the next step will fulfil in the dialog code.

```bash
pytest tests/test_gallery_window_region.py::test_region_lut_sidecar_uses_region_cache_key -v
```

Expected: PASS.

- [ ] **Step 3: Extend `show_lut_dialog` to take a `key_fn`**

In `src/gallery_view/ui/lut_dialog.py`, rewrite the `show_lut_dialog` signature so it accepts a `key_fn` callable instead of always calling `acq.handler.cache_key`. Search inside `show_lut_dialog` for any use of `acq.handler.cache_key(...)` and replace with `key_fn(acq, unit, channel, timepoint)`. Also rename the `fov` parameter to `unit` throughout.

Top of `lut_dialog.py`:

```python
def show_lut_dialog(
    parent,
    acq: Acquisition,
    unit: str,
    timepoint: str,
    axis: str,
    mip_data: dict,
    refresh_thumb: Callable[[int, str, str, int, AxisMip], None],
    acq_id: int,
    key_fn: Callable[[Acquisition, str, "Channel", str], tuple[str, str]] | None = None,
    unit_label: str = "FOV",
) -> None:
    """Open the LUT dialog. ``mip_data`` is the gallery window's
    ``{(acq_id, timepoint, unit, ch_idx, axis): AxisMip}`` map; we mutate
    it in place. ``unit`` is the row's third-slot id (a composite fov in
    FOV view, a region in region view). ``key_fn`` produces the
    ``(src, ch_id)`` tuple for cache writes — defaults to
    ``acq.handler.cache_key`` so FOV callers don't need to change.
    """
    if key_fn is None:
        key_fn = lambda a, u, c, t: a.handler.cache_key(a, u, c, timepoint=t)
```

Replace any `display_fov(fov)` in the window title with conditional logic:

```python
    title_unit = display_fov(unit) if unit_label == "FOV" else unit
    dlg.setWindowTitle(
        f"LUT — {acq.display_name} | {mag}x | {unit_label} {title_unit} | {axis_label}"
    )
```

Find every other `fov` reference in the function body and rename to `unit`. Specifically — when forming the dict lookup `mip_data.get((acq_id, timepoint, fov, ci, ax))`, change `fov` → `unit`. When calling `cache.save_lut_only(src, ch_id, ...)`, the `(src, ch_id)` comes from `key_fn(acq, unit, channel, timepoint)`.

- [ ] **Step 4: Update `_adjust_lut` in `gallery_window.py`**

Replace `_adjust_lut` (around line 886-903) with:

```python
    def _adjust_lut(self, key) -> None:
        from .lut_dialog import show_lut_dialog

        acq = self.acquisitions[key.acq_id]
        is_region = self.view_mode == "region"

        if is_region:
            mip_data = {
                (a, t, u, ci, "z"): ax_mip
                for (a, t, u, ci), ax_mip in self.region_mip_data.items()
            }
            key_fn = lambda a, u, c, t: a.handler.cache_key_region(
                a, u, c, timepoint=t
            )
            unit_label = "Region"
            refresh = self._render_region_thumb
        else:
            mip_data = self.mip_data
            key_fn = None
            unit_label = "FOV"
            refresh = self._render_thumb

        show_lut_dialog(
            parent=self,
            acq=acq,
            unit=key.fov,
            timepoint=key.timepoint,
            axis=self.view_axis,
            mip_data=mip_data,
            refresh_thumb=refresh,
            acq_id=key.acq_id,
            key_fn=key_fn,
            unit_label=unit_label,
        )

        # In region view, mip_data was a local copy; push p1/p999 changes
        # back to self.region_mip_data so the next thumb render picks them up.
        if is_region:
            for (a, t, u, ci, _), ax_mip in mip_data.items():
                self.region_mip_data[(a, t, u, ci)] = ax_mip
```

- [ ] **Step 5: Run all LUT-related tests**

```bash
pytest tests/test_lut_override.py tests/test_gallery_window_region.py -v
```

Expected: PASS — existing FOV LUT tests still pass; new region LUT key-shape test passes.

- [ ] **Step 6: Run full suite**

```bash
pytest -q
```

Expected: PASS (151 tests).

- [ ] **Step 7: Commit**

```bash
git add src/gallery_view/ui/lut_dialog.py src/gallery_view/ui/gallery_window.py tests/test_gallery_window_region.py
git commit -m "Wire Adjust Contrast for region rows via key_fn"
```

---

## Task 12: Mode-aware status bar copy and README update

**Files:**
- Modify: `src/gallery_view/ui/gallery_window.py` (`_refresh_visibility`, `_on_idle`)
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gallery_window_region.py`:

```python
def test_status_text_reports_regions_in_region_mode(qapp, make_squid_single_tiff_acq):
    from gallery_view.ui.gallery_window import GalleryWindow
    win = GalleryWindow()
    try:
        folder = make_squid_single_tiff_acq(regions=2, fovs_per_region=2)
        win._add_source(str(folder))
        win._set_view_mode("region")
        win.expand_region_action.setChecked(True)  # 2 region rows
        win._refresh_visibility()
        text = win.status.text()
        assert "region" in text.lower(), f"status was {text!r}"
    finally:
        win.close()
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_gallery_window_region.py::test_status_text_reports_regions_in_region_mode -v
```

Expected: FAIL — status text says "acquisitions visible" or "FOVs".

- [ ] **Step 3: Update `_refresh_visibility`**

In `src/gallery_view/ui/gallery_window.py`, locate the bottom of `_refresh_visibility` (around line 478-493). Replace the visibility-message block with:

```python
        unit_label = "regions" if self.view_mode == "region" else (
            "FOVs" if self.expanded_fov_mode else "acquisitions"
        )
        if total_rows and total_visible == 0:
            reasons = []
            if hide_thin:
                reasons.append('"Hide thin (Z<5)" is on')
            if self.mag_checkboxes and not active_mags:
                reasons.append("no magnification selected")
            why = " and ".join(reasons) if reasons else "filters"
            self.status.setText(
                f"All {total_rows} {unit_label} hidden by {why}."
            )
        elif total_rows:
            self.status.setText(
                f"{total_visible}/{total_rows} {unit_label} visible"
            )
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/test_gallery_window_region.py::test_status_text_reports_regions_in_region_mode -v
```

Expected: PASS.

- [ ] **Step 5: Update README**

In `README.md`, in the Features bullet list (after the multi-FOV bullet around line 25-28), insert:

```markdown
- **Region view** — multi-region squid acquisitions can be viewed as
  per-well stitched XY MIPs via the *View: [FOV] [Region]* toolbar toggle.
  Each region's MIP is a coordinate-based mosaic built from the per-FOV
  Z-MIPs the gallery already caches and `coordinates.csv`; FOVs are
  placed by stage position and overlaps are mean-blended. Use
  *Settings → Expand all regions as separate rows* to see every region
  at once. Requires `single_tiff` (squid) format with `coordinates.csv`.
```

- [ ] **Step 6: Run full suite**

```bash
pytest -q
```

Expected: PASS (152 tests).

- [ ] **Step 7: Commit**

```bash
git add src/gallery_view/ui/gallery_window.py tests/test_gallery_window_region.py README.md
git commit -m "Mode-aware status text and README update for region view"
```

---

## Task 13: Manual verification against the example dataset

This is a non-code checkpoint — no commit. Run the gallery against `/Volumes/Extreme SSD 1/DAPIageStudy/manu` and verify the mosaic orientation matches the physical sample.

- [ ] **Step 1: Launch the gallery on the example dataset**

```bash
python -m gallery_view --source "/Volumes/Extreme SSD 1/DAPIageStudy/manu"
```

- [ ] **Step 2: Wait for per-FOV MIP cache to populate**

The first run computes Z-MIPs for all FOVs across all 14 regions. Watch the status bar; this is the slow step (file I/O). On subsequent runs the gallery loads from cache.

- [ ] **Step 3: Switch to Region view**

Click `View: [Region]` in the top toolbar. The Region button must be enabled (the dataset has 14 regions).

- [ ] **Step 4: Visual check — orientation**

Click through the Region combo to inspect each well's mosaic. Compare against what a microscope user would expect:
- Adjacent FOVs should appear adjacent in the mosaic, not flipped or mirrored.
- The well with only 4 FOVs (B8 in this dataset) should render as a small 2×2-ish cluster against a black background, not a sparse pattern in a large canvas.

If the mosaic looks vertically mirrored: in `src/gallery_view/stitch.py`, in `stitch_region`, change the `flip_y: bool = False` default to `True`. Re-run and re-verify. Commit the flip with:

```bash
git commit -am "Stitcher: flip stage Y by default to match image-Y convention"
```

(Only if needed. If no flip is needed, no commit here.)

- [ ] **Step 5: Visual check — alignment quality**

Adjacent FOVs at the seams should align approximately (within a tile or two of pixels at the downsampled scale). The simple stitcher doesn't do sub-pixel registration, so don't expect perfect seams. If seams look off by more than ~10% of a tile width, suspect a pixel-size or coordinate-unit bug and dig in.

- [ ] **Step 6: Switch to expanded mode**

`Settings → Expand all regions as separate rows`. The gallery should render 14 rows, one per region. Scroll through them.

- [ ] **Step 7: Try 'Adjust Contrast' on a region row**

Click `Adjust Contrast`. Verify the dialog opens with the mosaic preview and that adjusting the slider re-renders the gallery thumbnail.

- [ ] **Step 8: Switch back to FOV view and confirm nothing broke**

Verify FOV view still works as it did before — per-FOV MIPs render, XZ/YZ toggles work, 3D View opens napari.

---

## Self-review notes

- All spec sections map to tasks: types/handler (1-3, 5), cache (4), stitcher (6), loader (7), UI toolbar (8), UI rows (9), readiness/dispatch (10), LUT (11), status copy/README (12), manual verification (13).
- No placeholder text in code blocks; every step shows actual code.
- Type consistency: `RegionStitchJob` fields, `region_mip_ready` signature, and window state are referenced consistently across Tasks 7, 8, 9, 10, 11.
- `key_fn` signature in Task 11 matches what `_adjust_lut` passes (`lambda a, u, c, t: a.handler.cache_key_region(a, u, c, timepoint=t)`).
- The `_render_region_thumb` aspect logic in Task 10 uses the mosaic's own shape, not `_phys_aspect` — region thumbs don't have a fixed FOV-shape aspect. The existing `_apply_label_sizes` path still sizes the row's cells based on `_phys_aspect`, which in region view defaults to 1.0 because `acq.shape_zyx` is the per-FOV shape. That's acceptable for v1; the actual rendered pixmap fits inside the cell preserving the mosaic aspect.
