# Region view: stitched per-region MIPs as a peer to FOV view

Status: approved (brainstorm)
Date: 2026-05-10

## Summary

Add a **Region view** to the gallery as a peer to the existing FOV view. In region view, each row corresponds to one well/region of a multi-region acquisition; each channel cell is a stitched XY MIP of that region built from the per-FOV Z-MIPs that the gallery already computes for FOV view. A `View: [FOV] [Region]` toggle in the top toolbar switches between the two; the existing axis (`Project: XY/XZ/YZ`) and thumbnail-size controls stay shared. The stitcher itself is a simple coordinate-based placement using `coordinates.csv` — no sub-pixel registration, no blending beyond mean-over-overlap. v1 is scoped to the `single_tiff` (squid) format. A higher-fidelity stitch via the Cephla-Lab TileFusion library is a separate, future enhancement (not implemented in v1).

## Why

- Acquisitions with many regions (the motivating example has 14 wells × 21 FOVs × 36 z-slices × 3 channels) are unwieldy in FOV view — a user has to step through every FOV per region to get the gestalt of a well. The natural unit of inspection is the whole well, not one FOV.
- The data needed to stitch a region (per-FOV Z-MIPs + per-FOV stage coordinates) is **already on disk**: FOV view computes and caches per-FOV Z-MIPs; squid writes `coordinates.csv` alongside the TIFFs with `(region, fov, x_mm, y_mm)`. Region view is a presentation layer over data the gallery already has, not a second I/O pipeline.
- The Cephla-Lab TileFusion stitcher (https://github.com/Cephla-Lab/stitcher) handles individual-TIFF + coordinates.csv natively. It's heavyweight (CuPy/cuCIM optional, zarr v3 output, sub-pixel registration). Pulling it in for gallery thumbnails is overkill — gallery thumbnails render at ≤ 320 px, sub-pixel accuracy is invisible. Building a simple coordinate-paste stitcher in-house keeps the dependency surface small and the per-region compute under a second once the per-FOV cache is warm.

## Non-goals (v1)

- **TileFusion integration.** The design leaves room to add a "high-quality stitch" option later (e.g. a Settings entry that calls TileFusion and caches its output), but v1 ships only the simple coord stitcher.
- **Region view for OME-TIFF and stack_tiff.** Only `single_tiff` (squid) acquisitions get region view in v1. Other formats either lack coordinates or lack multi-region semantics; the `Region` toolbar button is disabled when no source supports it.
- **Stitched XZ / YZ projections.** Stitching axial MIPs across a tile grid has no clean physical meaning; XZ/YZ buttons are disabled when in region view.
- **Open 3D View in region view.** The 3D viewer is per-stack; loading a whole region's worth of z-stacks into napari is out of scope. Button is hidden in region rows.
- **Live restitch as per-FOV MIPs trickle in.** Stitching happens once when all per-FOV MIPs for a region are available; no progressive paint.
- **Pluggable view-mode abstraction (RowProducer).** Tracked separately at https://github.com/hongquanli/gallery-view/issues/6.

## Design

### Data model — `Acquisition` (`src/gallery_view/types.py`)

Two new fields:

```python
regions: list[str] = field(default_factory=lambda: ["0"])
selected_region: str = "0"
```

`fovs` (composite `"<region>_<fov>"` strings) stays untouched. `regions` is the deduplicated, sorted list of distinct region ids drawn from `fovs`. Acquisitions with a single region get `regions = ["0"]` and the toolbar's `Region` button stays disabled for them.

`Acquisition.extra` gains one optional key, populated lazily by handlers:

```python
extra["coords_by_region"]: dict[str, list[FovCoord]]
```

where `FovCoord = (fov: str, x_mm: float, y_mm: float)`. A handler that supports region view populates this on first read of `coordinates.csv` and caches the result on `acq.extra` to avoid re-parsing.

### Handler support — `SingleTiffHandler` (`src/gallery_view/sources/single_tiff.py`)

Two additions:

1. **Populate `acq.regions`** in `build()` by extracting the distinct region prefixes from the parsed FOVs (already discovered by `_fovs_for`). Order: numeric ids first (sorted as int), then alphanumeric (well names like `A5`, `B10`, sorted lexicographically with a digit-aware key so `A2` comes before `A10`).

2. **`coordinates.csv` reader.** A small helper, e.g. `_load_coords(acq)`, that:
   - Resolves the CSV at `<acq.path>/<selected_timepoint>/coordinates.csv` (or `<acq.path>/coordinates.csv` as fallback — both layouts exist in real squid data).
   - Parses with `csv` stdlib (no pandas dependency for gallery-view core; tilefusion uses pandas, but the gallery doesn't need it).
   - Groups rows by `region`, picks one row per `(region, fov)` (z_level=0 is sufficient — XY position is z-invariant in squid).
   - Stores `dict[region, list[FovCoord]]` in `acq.extra["coords_by_region"]`.
   - Returns `None` (and leaves `extra` unset) if the file is missing or malformed; callers fall back gracefully (see Error Handling).

No protocol change: `coords_by_region` lives in `extra`, which is by design a per-handler escape hatch. Other handlers don't need to know about it.

### New module — `src/gallery_view/stitch.py`

Pure functions, no Qt, no thread state. Roughly 200 lines.

**Public API:**

```python
@dataclass(frozen=True)
class FovCoord:
    fov: str           # composite "<region>_<fov>" matching acq.fovs
    x_mm: float        # stage X of FOV center
    y_mm: float        # stage Y of FOV center

def stitch_region(
    fov_mips: dict[str, np.ndarray],   # composite fov_id -> (ny, nx) float32 Z-MIP
    coords: list[FovCoord],            # FOVs in this region, with stage positions
    pixel_um: float,                   # in-plane pixel size in micrometres
    target_longest_px: int = 1024,
    flip_y: bool = False,              # see "Coordinate orientation" below
) -> AxisMip | None:
    """Place each FOV's full-res MIP at its stage coordinate, downsample to fit
    ``target_longest_px`` on the longest axis, mean-blend overlaps. Returns
    ``None`` when ``fov_mips`` is empty or shapes are inconsistent."""
```

**Algorithm:**

1. Establish per-FOV shape `(ny, nx)` from any FOV MIP (the loader already rejects heterogeneous shapes; we assert uniform here and bail with `None` if not).
2. For each `FovCoord` whose `fov` is present in `fov_mips`:
   - Top-left corner in mm: `(x_mm - nx * pixel_um / 2000, y_mm - ny * pixel_um / 2000)`.
3. Bounding box across all corners → canvas origin = `(min_x_mm, min_y_mm)`.
4. Full-res canvas size in pixels: `(H_full, W_full) = ceil((max - min) / (pixel_um / 1000)) + (ny, nx)`.
5. Downsample factor `f = max(1, ceil(max(H_full, W_full) / target_longest_px))`. Integer for cheap block-mean.
6. Downsampled canvas size: `(H, W) = (H_full // f, W_full // f)`. Allocate two `float32` arrays: `accum[H, W]` and `weight[H, W]`.
7. For each FOV: block-mean downsample its MIP by `f` (using `numpy` reshape-and-mean — `skimage.transform.downscale_local_mean` is a nicer fallback if available, but the numpy version has no dep). Compute its pixel offset on the downsampled canvas. Add into `accum`; add `1.0` into `weight` over the same rectangle.
8. `mosaic = accum / np.maximum(weight, 1.0)` — pixels covered by no FOV remain `0.0`, the "black for missing positions" behaviour.
9. Auto-contrast percentiles (`p1` = 0.5th, `p999` = 99.5th) computed on `mosaic[weight > 0]` so black gaps don't pull `p1` down.

**Coordinate orientation.** Squid's stage Y convention vs image Y convention is a known unknown. The stitcher exposes `flip_y` as a parameter; v1 defaults to `False` and we verify against the motivating dataset (`/Volumes/Extreme SSD 1/DAPIageStudy/manu`). If the mosaic comes out vertically mirrored relative to expectation, flip the default and document why. A single-line constant change.

**Why integer downsample factor.** Non-integer downsampling forces interpolation, which makes thumbnail rendering slower and adds aliasing for no perceptual gain. Integer block-mean is exact, fast, and the resulting mosaic looks identical at gallery thumbnail sizes.

### Loader — `src/gallery_view/loader.py`

New job type alongside `Job`:

```python
@dataclass
class RegionStitchJob:
    acq_id: int
    acq: Acquisition
    region: str
    channel: Channel
    ch_idx: int
    timepoint: str
    fov_mips: dict[str, np.ndarray]    # supplied by the window when all FOV MIPs are ready
```

A new `region_mip_ready` signal mirrors `mip_ready`:

```python
region_mip_ready = Signal(
    int,    # acq_id
    str,    # timepoint
    str,    # region
    int,    # ch_idx
    str,    # wavelength
    object, # AxisMip
)
```

`MipLoader.run()` handles both job types via type dispatch. For a `RegionStitchJob`:
1. Compute the cache key via `acq.handler.cache_key_region(acq, region, channel, timepoint)` (new method on `FormatHandler`; default protocol implementation just raises `NotImplementedError` and is overridden only by `SingleTiffHandler`).
2. Cache hit ⇒ emit `region_mip_ready` with the cached `AxisMip`.
3. Cache miss ⇒ call `stitch.stitch_region(...)`. Save the resulting single-axis `ChannelMips` to disk via the existing `cache.save` (the cache module is FOV-agnostic; it doesn't care that the channel_id encodes a region instead of a FOV). Emit `region_mip_ready`.

The window enqueues a `RegionStitchJob` only after observing that every FOV's Z-MIP for that `(acq, region, channel)` has landed in `self.mip_data`. The stitch job carries the FOV MIPs directly (already in memory on the GUI thread), so the loader doesn't have to reach back into window state — clean producer/consumer.

**Handler protocol addition — `cache_key_region`:**

```python
def cache_key_region(
    self, acq: Acquisition, region: str, channel: Channel, timepoint: str = "0"
) -> tuple[str, str]:
    """Return (src_path, channel_id) for the stitched region mosaic cache."""
```

`SingleTiffHandler` returns `(acq.path, f"region:{region}/t{timepoint}/{channel.name}")`. Other handlers raise `NotImplementedError`; the window never calls this on unsupported handlers because the Region toolbar button is disabled.

### Cache — `src/gallery_view/cache.py`

**One small relaxation in `load()`.** The key scheme `_cache_path(src_path, channel_id) = MD5(src_path|channel_id)[:16].npz` is opaque to channel_id format, so region mosaics ride on it via the `"region:<region>/t<t>/<channel>"` channel_id from `cache_key_region`. `cache.save()` already iterates `axis_data.items()` — saving a single-axis `{"z": AxisMip(...)}` works unmodified. `cache.load()`, however, today returns `(None, None)` if any of the three axes is missing from the `.npz`. That's a no-op for FOV caches (which always have all three) but would block region caches (which have only `"z"`). The fix is a one-liner: skip missing axes instead of aborting:

```python
# was:
for ax in AXES:
    if f"mip_{ax}" not in data.files:
        return None, None
    out[ax] = AxisMip(...)
# becomes:
for ax in AXES:
    if f"mip_{ax}" not in data.files:
        continue
    out[ax] = AxisMip(...)
if not out:
    return None, None
```

FOV callers don't see a behaviour change — their caches always have all three axes. Region callers get a single-axis result, which the new code paths handle.

LUT-override sidecars (`.lut.json`) work the same way — different channel_id ⇒ different sidecar file. "Adjust Contrast" in region view writes a region-keyed sidecar without conflicting with FOV LUT overrides.

**No `CACHE_VERSION` bump needed.** Region entries have channel_ids prefixed with `region:` and won't collide with FOV entries. Old caches keep working.

### UI — `src/gallery_view/ui/gallery_window.py`

**New state:**

```python
self.view_mode: Literal["fov", "region"] = "fov"
self.expanded_region_mode: bool = False
# Composite key now is (acq_id, timepoint, fov_or_region) — RowKey keeps the
# same shape; the third slot's meaning depends on view_mode.
self.region_mip_data: dict[tuple[int, str, str, int], "AxisMip"] = {}
# (acq_id, timepoint, region) -> set of fov ids whose MIPs we've stored
# (used to detect "all FOV MIPs ready, time to enqueue a stitch")
self._region_fov_readiness: dict[tuple[int, str, str], set[tuple[int, str]]] = {}
```

**New toolbar (extension to `_build_display_row`):**

```
Project: [XY] [XZ] [YZ]    View: [FOV] [Region]    Thumbnail size: [Medium ▾]
```

`View: [FOV] [Region]` is a second `QButtonGroup` with the same `_toggle_style()`. Region is `setEnabled(False)` whenever no loaded acquisition has `len(acq.regions) > 1`. Clicking `Region`:
1. Sets `self.view_mode = "region"`.
2. Forces `self.view_axis = "z"` and disables the XZ/YZ buttons.
3. Calls `self._rebuild_rows()`.

Clicking `FOV` reverses these.

**Settings menu addition:**

```python
self.expand_region_action = QAction("Expand all regions as separate rows", self)
self.expand_region_action.setCheckable(True)
self.expand_region_action.toggled.connect(self._set_expanded_region_mode)
```

Same shape as `expand_action` for FOVs. Only has visible effect in region view.

**Row building (`_make_row_widget` and helpers):**

`_fovs_for_row(acq_id) -> list[str]` is generalised to `_row_units_for(acq_id) -> list[str]`:

- FOV view, compact: `[acq.selected_fov]`.
- FOV view, expanded: `acq.fovs`.
- Region view, compact: `[acq.selected_region]`.
- Region view, expanded: `acq.regions`.

The per-row combo logic changes by mode:

- FOV view, compact, `len(acq.fovs) > 1` ⇒ render the existing FOV combo.
- Region view, compact, `len(acq.regions) > 1` ⇒ render a Region combo (`A5`, `A6`, … — well-name display) wired to `_on_region_changed`.
- Either view, expanded mode ⇒ no combo; the unit appears as a row label (the existing "FOV X" label generalises to "Region A5").

Per-row buttons:

- FOV view: `Open 3D View` + `Adjust Contrast` (unchanged).
- Region view: `Adjust Contrast` only. `Open 3D View` is omitted from the layout (not just disabled — keeps the row width tight).

**Enqueuing in region view (`_enqueue_jobs_for_acq` extension):**

When `view_mode == "region"`:
- For each FOV in the region: enqueue a per-FOV Z-MIP `Job` (loader skips cached ones in O(1)).
- Track the FOV set in `self._region_fov_readiness[(acq_id, t, region)]`.

On each `_on_mip_ready` while in region view, if the new MIP completes the readiness set for `(acq_id, t, region, channel)`, the window builds a `dict[fov_id, ndarray]` from `self.mip_data` (axis `"z"` only) and enqueues a `RegionStitchJob`.

**Rendering a region thumb (`_on_region_mip_ready`):**

Mirrors `_on_mip_ready` but writes to `self.region_mip_data` and renders via the existing `mip_to_rgba` + `_render_thumb` path (the latter is already axis-agnostic — it just takes an `AxisMip`).

**Aspect:** `_phys_aspect` in region view uses the canvas dimensions from the stitched mosaic (`mosaic.shape[0] / mosaic.shape[1]`) instead of the FOV's `(ny, nx)`. Same plumbing; different inputs.

**Status bar:** `_refresh_visibility` reports "X/Y regions visible" when `view_mode == "region"`, "X/Y FOVs visible" or "X/Y acquisitions visible" otherwise (existing logic). One-line label swap.

### LUT contrast for region view

`show_lut_dialog` (in `src/gallery_view/ui/lut_dialog.py`) takes a `mip_data` dict and a `(acq_id, timepoint, fov, ch_idx, axis)` lookup. The window passes `self.region_mip_data` and a `(acq_id, timepoint, region, ch_idx, "z")` key when in region view. The dialog itself is axis-agnostic — only XY is shown, which it already supports as one of three axes. The sidecar save path uses the region-keyed channel_id automatically since it's derived from `acq.handler.cache_key_region`.

## Error handling

| Case | Behavior |
|---|---|
| `coordinates.csv` missing for a multi-region single_tiff acquisition | `coords_by_region` stays unset. Window detects this on enqueue and shows `"<acq>: region view needs coordinates.csv"` in the status bar; that acquisition's region rows render as empty placeholders. FOV view works normally. |
| `coordinates.csv` missing for a single-region acquisition (`regions == ["0"]`) | Region button stays disabled for that source; no change. |
| `coordinates.csv` malformed (missing `region`/`fov`/`x (mm)`/`y (mm)` columns) | Same as missing — set nothing; warn once. |
| Region has 1 FOV | `stitch_region` returns the single FOV's MIP unchanged (after downsample to fit `target_longest_px`). FOV view is the better tool here, but region view still works. |
| FOV referenced in `coordinates.csv` but missing from `fov_mips` (file missing on disk, or load failed) | Skip in stitch; the rect stays black. |
| `pixel_um` unknown (no `mag`, no `sensor_pixel_size_um` in params) | Fall back to placing FOVs in a uniform grid (sorted by FOV index, row-major over an inferred grid shape). Flag in status: `"<acq>: pixel size unknown, using grid layout"`. |
| Heterogeneous FOV shapes within a region | `stitch_region` returns `None`. Loader's `_process` catches as it does today and emits a failure progress message. Region row renders empty. |
| Stitch raises an unexpected exception | Caught by the loader's existing try/except in `_process`. Message: `"failed region <X> channel <wl> — <reason>"`. |

## Testing

| Test file | Coverage |
|---|---|
| `tests/test_stitch.py` (new) | `stitch_region` against a synthetic 2×2 grid (4 FOVs of known shapes/positions); overlapping FOVs mean-blend; missing FOV leaves a black rect at the right pixel offset; integer downsample factor scales correctly; coord origin shift (negative stage coords); `flip_y` flips the Y component; empty `fov_mips` returns `None`; heterogeneous shapes returns `None`. |
| `tests/test_acquisition_regions.py` (new) | `SingleTiffHandler.build()` populates `acq.regions` correctly: numeric-only regions sort as ints; mixed alphanumeric regions sort A1..A10..B1..B10; single-region squid folders get `regions = ["0"]`; legacy folders get `regions = ["0"]`. |
| `tests/test_single_tiff_coords.py` (new) | `_load_coords` parses `coordinates.csv` into `coords_by_region`; missing file returns `None`; missing columns return `None`; rows with same `(region, fov)` deduplicate to one (z=0). |
| `tests/test_cache_region.py` (new) | `cache.save` then `cache.load` of a region-keyed entry (single `"z"` axis) round-trips the mosaic correctly; verifies the load-skips-missing-axes change doesn't regress full-axis FOV loads; doesn't collide with a same-acq FOV cache entry. |
| `tests/test_loader_region.py` (new) | `RegionStitchJob` on cache miss calls `stitch_region` and emits `region_mip_ready`; cache hit short-circuits to emit; stitcher exception is caught and surfaces as a failure progress message. |
| `tests/test_gallery_window_region.py` (new) | `View` toggle disables XZ/YZ when Region is selected and re-enables them on FOV; Region button disabled when no source has multi-region; "Expand all regions" rebuilds rows; switching modes doesn't leak Qt widgets; LUT dialog from a region row writes to a region-keyed sidecar path. |

Existing tests untouched — FOV view code paths don't change in shape, only branch on `view_mode` for row building.

## Build sequence

The implementation plan (next step, written by the `writing-plans` skill) will sequence the work roughly as follows; this is a preview, not the plan itself:

1. **Types + handler** — add `regions` / `selected_region` to `Acquisition`; populate `regions` in `SingleTiffHandler.build()`; add `_load_coords` + `cache_key_region` to the handler; add `cache_key_region` to the `FormatHandler` protocol with `NotImplementedError` defaults on OME-TIFF and stack_tiff. Tests-first.
2. **Stitcher** — `stitch.py` with `stitch_region` and `FovCoord`. Synthetic-grid tests. No UI integration yet.
3. **Loader** — `RegionStitchJob`, `region_mip_ready` signal, dispatch in `run()`. Tests with a mocked stitcher.
4. **UI toolbar + settings** — `View: [FOV] [Region]` button group; XZ/YZ disable logic; Settings → "Expand all regions". No row-building changes yet; toggling the button just no-ops with a TODO. Smoke test.
5. **UI row building** — generalise `_fovs_for_row` to `_row_units_for`; render Region combo when applicable; hide `Open 3D View` in region rows; wire `_on_region_mip_ready`; track FOV readiness and enqueue stitch jobs; report mode-aware status.
6. **LUT dialog wiring** — pass region-keyed `mip_data` + lookup when in region view. Verify sidecar path.
7. **Manual verification against `/Volumes/Extreme SSD 1/DAPIageStudy/manu`** — confirm orientation (set `flip_y` if needed); spot-check tile alignment; tune `target_longest_px` if 1024 feels off.

Each step is independently testable. Steps 1–3 land without any UI changes (region view literally cannot be activated from the UI until step 4). Steps 4–6 land the UI behind a feature that's already exercised by unit tests. Step 7 is a manual checkpoint, not a code step.
