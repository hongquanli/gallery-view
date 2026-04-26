# gallery-view — design

**Status:** approved (brainstorming complete)
**Date:** 2026-04-26
**Author:** Hongquan Li (with Claude Code)

## 1. Overview

`gallery-view` is a standalone PyQt6 desktop application for browsing z-stack
acquisitions written by [squid](https://github.com/cephla-lab/squid). The user
drops one or more folders onto the window — either single acquisition folders
or parent folders containing many — and the app builds an instant gallery of
per-channel max-intensity projections that can be re-projected along Z, Y, or X,
recolored with a LUT dialog, exported as a 600 dpi figure, or opened in napari
for 3D inspection.

The app is derived from `explorer_ovelle.py` in
[`aion-explorer`](https://github.com/cephla-lab/aion-explorer) but is
deliberately simpler: it does not group by well, does not do multi-device
comparison, and does not apply any device-specific image transforms (e.g. the
Cicero 180° rotation). It targets squid's output as the single supported input
format family.

### 1.1 Goals

- Browse-session feel must be instant: full-resolution Z/Y/X MIPs are
  pre-computed and cached on first sight; thereafter axis switches and contrast
  tweaks render entirely from RAM.
- Drag-and-drop input. Users never need to edit source files to point at data.
- One file owns each squid format detail; adding a new format (e.g. OME-Zarr)
  is a single new file plus a registry entry — no edits elsewhere.
- Code small enough to hold in your head: target ~200-400 lines per module,
  ~10 modules total.

### 1.2 Non-goals

- **Multi-FOV unrolling by default.** Each row is one acquisition with a per-row
  FOV dropdown. A settings toggle (off by default) expands to one row per
  `(acquisition, FOV)` pair grouped under the acquisition.
- **Time series.** Out of scope for v1; planned in §15.
- **Multi-device comparison UX.** No well grouping, no device filter, no per-device
  rotation. squid is the producer; one logical input shape.
- **PyPI release in v1.** Install via `pip install -e .` from the repo.
- **GUI tests.** pytest covers logic only.
- **Concurrent / parallel MIP loader.** Single-threaded suffices today; documented
  as future work in §15.

### 1.3 What "connected to squid" means

`gallery-view` is an independent repository that vendors its own copy of
squid-format parsing logic. It does **not** import or depend on the squid
package. The parsing module is shaped so that, if and when squid publishes a
stable Python I/O API, the swap is one file.

The format-parsing module records `SQUID_TESTED_AGAINST = "<commit>"` in a
docstring so format drift is auditable.

## 2. User experience

### 2.1 Launch and ingestion

- Launch with no sources loaded: window shows a centered "Drop folders here"
  target.
- Drop one or more folders → each is ingested.
  - If a dropped folder is itself a recognized squid acquisition, it is added
    directly.
  - Otherwise, the app walks its subfolders and ingests every recognized
    acquisition found. Walks are bounded: the dropped folder is depth 0; the
    walker descends until it reaches an acquisition or hits depth 3 (i.e.
    drop → child → grandchild → great-grandchild).
- Hidden folders (`.*`) and symlinks are skipped. Duplicate ingest of the same
  realpath is a no-op.
- A "Sources" chip strip at the top of the window shows each dropped root with
  the count of acquisitions found and an `×` to remove. Removing a root removes
  all of its acquisitions from the gallery and cancels any pending MIP loads
  for them; on-disk cache files are kept untouched.
- `File → Add Folder…` (Cmd-O) opens a folder picker; same code path as drop.
- `File → Refresh sources` re-walks every loaded root and ingests anything new.
- CLI: `python -m gallery_view --source PATH [--source PATH …]` is equivalent
  to launching empty and dropping each path.

### 2.2 Gallery layout

```
┌──────────────────────────────────────────────────────────┐
│ MenuBar:  File   View   Settings                         │
├──────────────────────────────────────────────────────────┤
│ Sources: [/path/A (3 acqs) ×]  [/path/B (1 acq) ×]  [+]  │
├──────────────────────────────────────────────────────────┤
│ Magnification: [10x] [25x]    Hide thin (Z<5)            │
│ Project: [XY][XZ][YZ]   Thumb size: Medium ▾             │
├──────────────────────────────────────────────────────────┤
│ ┌─ Scroll area ────────────────────────────────────────┐ │
│ │ 25x  04-18  AION_25x_H3   [405][488][561][638]       │ │
│ │            00:14  FOV:0 ▾  [Open 3D] [Adjust LUT]    │ │
│ │ 10x  04-19  AION_10x_F4   [   ][488][   ][638]       │ │
│ │            12:32  FOV:0 ▾  [Open 3D] [Adjust LUT]    │ │
│ └──────────────────────────────────────────────────────┘ │
│ Status: 12/47 channels — computing 488 nm MIP — 25x H3   │
└──────────────────────────────────────────────────────────┘
```

**Filter row (top of toolbar, below Sources strip):**

- **Magnification.** Checkboxes, one per distinct mag in the current dataset,
  all checked by default. Hidden if the dataset has only one mag.
- **Hide thin (Z<5).** Toggle, on by default.

**Display row (below filters):**

- **Project.** XY / XZ / YZ exclusive buttons. Toggle is global; re-renders all
  visible rows from RAM (no recomputation, no I/O).
- **Thumb size.** Small (80 px) / Medium (160 px) / Large (320 px) preset combo.

**Per-row content (default mode, one row per acquisition):**

- Magnification (30 px column).
- Date/time (40 px column, two lines `MM-DD` / `HH:MM`, parsed from the trailing
  timestamp in the folder name).
- Display name (~120 px column, ellipsized; tooltip = full path). The display
  name defaults to the acquisition folder's basename.
- Channel thumbnails — fixed-width columns in the dataset's *active* wavelength
  order: start from `[405, 488, 561, 638, 730]`, then any additional wavelengths
  in numerical order, **dropping any wavelength no visible row uses**. Empty
  cells are transparent placeholders that maintain column alignment. Width is
  `thumb_size`; height is `thumb_size × physical_aspect(acq, axis)` so that XZ
  and YZ rows are squat when the Z extent is small relative to the XY field.
- FOV picker — compact `QComboBox`, hidden when the acquisition has only one
  FOV. Changing it sets `acq.selected_fov`, enqueues any not-yet-loaded MIP
  jobs for that FOV, and re-renders the row's thumbs from cache when ready.
- Action buttons — `Open 3D View`, `Adjust Contrast`.

**Expanded-FOV mode (Settings → "Expand all FOVs as separate rows"):**

When toggled on, the flat list becomes grouped: one `QGroupBox` per acquisition
with header `<display_name>  (3 FOVs)`. Inside, one row per FOV, with the FOV
picker replaced by a small `FOV n` label. Filters and project axis still apply
globally. Toggling back off collapses each group to its `selected_fov` row.

Internally, the gallery is modeled as a list of `(acq, fov)` row keys; in
default mode the list contains one entry per acquisition with
`fov = acq.selected_fov`, in expanded mode it contains one entry per
`(acq, fov)` pair. Toggle rebuilds the list.

**Settings menu items:**

- **Square footprint for XZ/YZ** — keeps row heights constant on axis switch,
  preventing layout reflow.
- **Expand all FOVs as separate rows** — default off; described above.
- **Clear MIP cache…** — confirm dialog, deletes the cache directory.

**File menu items:** `Add Folder…`, `Refresh sources`, `Quit`.

### 2.3 Sorting

Default sort is by source root in drop order, then by folder name
lexicographically (squid timestamps make this chronological). No
user-controllable sort in v1.

### 2.4 Empty / loading / error states

- Empty: full-window drop target.
- Loading: rows appear immediately with each not-yet-loaded *data* thumb cell
  rendered as a dark `#222` square (visually distinct from the *transparent*
  placeholders used for wavelengths a row simply doesn't have — see §2.2).
  As MIPs complete, the dark cells render. Status bar shows progress.
- Failed rows (handler raised) render a `⚠` indicator with a tooltip showing
  the error. Always visible in v1; "Hide failed" toggle is YAGNI until it
  bothers a user.

## 3. Repository layout

```
gallery-view/
├── pyproject.toml
├── README.md
├── .github/workflows/ci.yml
├── src/gallery_view/
│   ├── __init__.py               # __version__
│   ├── __main__.py               # `python -m gallery_view`
│   ├── cli.py                    # argparse: --source PATH (repeatable)
│   ├── types.py                  # Acquisition, Channel, AxisMip, …
│   ├── scan.py                   # walk dropped folders, dispatch to handlers
│   ├── mips.py                   # MIP math
│   ├── cache.py                  # .npz cache + .lut.json sidecar I/O
│   ├── loader.py                 # MipLoader QThread
│   ├── sources/
│   │   ├── __init__.py           # HANDLERS registry, detect()
│   │   ├── base.py               # FormatHandler Protocol
│   │   ├── ome_tiff.py
│   │   ├── multi_channel_tiff.py
│   │   └── single_channel_tiff.py
│   └── ui/
│       ├── __init__.py
│       ├── gallery_window.py     # main window, drop, filters, scroll area
│       ├── sources_panel.py      # chip strip
│       ├── zoomable_view.py      # ZoomableImageView
│       ├── lut_dialog.py         # LUT sliders + PNG export
│       └── viewer3d.py           # napari opener
└── tests/
    ├── conftest.py               # synthetic acq fixtures
    ├── test_scan.py
    ├── test_cache.py
    ├── test_mips.py
    ├── test_lut_override.py
    └── sources/
        ├── test_ome_tiff.py
        ├── test_multi_channel_tiff.py
        └── test_single_channel_tiff.py
```

## 4. Data model (`types.py`)

```python
@dataclass(frozen=True)
class Channel:
    name: str           # e.g. "Fluorescence_488_nm_Ex"
    wavelength: str     # "488", or "unknown"

@dataclass(frozen=True)
class AxisMip:
    mip: np.ndarray         # 2-D float32, full resolution
    p1: float               # auto-contrast low (0.5 percentile)
    p999: float             # auto-contrast high (99.5 percentile)

# {z, y, x} -> AxisMip for one channel
ChannelMips = dict[Literal["z", "y", "x"], AxisMip]

@dataclass
class Acquisition:
    handler: "FormatHandler"        # owns format-specific behavior
    path: str                       # folder path
    folder_name: str
    display_name: str               # what the row header shows
    params: dict                    # parsed acquisition parameters.json
    channels: list[Channel]
    shape_zyx: tuple[int, int, int] | None  # populated lazily
    fovs: list[str]                 # ["0"] for single-FOV
    selected_fov: str = "0"         # picker state (mutable)
    extra: dict = field(default_factory=dict)  # handler-private fields
```

`extra` is a deliberate escape hatch so handlers can stash format-private data
(`ome_path`, `channel_paths`, etc.) without forcing them into the shared
schema. Outside the handler, no one reads it.

`selected_fov` is the only mutable field on `Acquisition`. The FOV dropdown
writes here and triggers a re-render.

## 5. `FormatHandler` protocol (`sources/base.py`)

```python
class FormatHandler(Protocol):
    name: str  # "ome_tiff" / "multi_channel_tiff" / "single_channel_tiff"

    def detect(self, folder: str) -> bool: ...
    def build(self, folder: str, params: dict) -> Acquisition | None: ...
    def list_fovs(self, acq: Acquisition) -> list[str]: ...
    def read_shape(self, acq: Acquisition, fov: str) -> tuple[int, int, int] | None: ...
    def cache_key(self, acq: Acquisition, fov: str, channel: Channel) -> tuple[str, str]:
        # returns (source_path_for_hash, channel_id).
        # The cache key in cache.py is MD5("source|fov|channel_id"), so what
        # each handler returns must be:
        #   - stable across runs for the same on-disk acquisition,
        #   - distinct between channels of the same acquisition,
        #   - distinct between acquisitions even if folders are renamed (use
        #     the realpath of the file or folder that owns the channel).
        # Recommended values per format:
        #   ome_tiff:             (acq.extra["ome_path"], f"wl_{channel.wavelength}")
        #   multi_channel_tiff:   (acq.path, channel.name)
        #   single_channel_tiff:  (acq.extra["channel_paths"][channel.wavelength],
        #                          f"Fluorescence_{channel.wavelength}_nm_Ex")
        ...
    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]: ...
    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray: ...
    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        # exposure_ms, intensity, etc., for PNG export footer; {} if unknown.
        ...
```

`sources/__init__.py` defines:

```python
HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    MultiChannelTiffHandler(),
    SingleChannelTiffHandler(),
]

def detect(folder: str) -> FormatHandler | None:
    for h in HANDLERS:
        if h.detect(folder):
            return h
    return None
```

Handlers are stateless singletons; first match wins.

The MIP loader, the napari opener, the LUT dialog's save path, and the PNG
export footer all delegate via `acq.handler.<method>`. **Nothing outside
`sources/` ever branches on `acq.handler.name`.**

## 6. Scanning & ingestion (`scan.py`)

```python
def ingest(path: str, *, _seen: set[str] | None = None,
           _depth: int = 0, _max_depth: int = 3) -> list[Acquisition]:
    """Walk `path` and return every recognized acquisition.

    Skips hidden folders, symlinks, and any realpath already in `_seen`.
    Bounded depth keeps walks predictable.
    """
```

The gallery window owns `self.acquisitions: list[Acquisition]` and
`self.sources: list[Source]` where `Source` records the dropped root path and
the list of acquisition IDs originating from it. Both are mutable at runtime.

## 7. MIP cache (`cache.py`)

- **Location.** `platformdirs.user_cache_dir("gallery-view")/mips/`. (Uses
  `platformdirs` so Windows works without a special case.)
- **Per-channel `.npz`.** Keyed by an MD5 of `<source_path>|<fov>|<channel_id>`,
  truncated to 16 hex chars. The FOV in the key is new vs. `explorer_ovelle.py`
  and is what makes multi-FOV correct.
- **Contents:** `version: int32`, `mip_z`, `mip_y`, `mip_x` (float32, full-res),
  `p1_z`, `p999_z`, `p1_y`, `p999_y`, `p1_x`, `p999_x`, `nz_orig`, `ny_orig`,
  `nx_orig`. Saved with `np.savez_compressed`.
- **Sidecar.** `<hash>.lut.json` holds user-saved contrast overrides as
  `{axis: {p1, p999}}`. Lets a "Save" in the LUT dialog complete in milliseconds
  without recompressing the heavy MIP arrays.
- **`CACHE_VERSION = 1`** — fresh, separate from the explorer's `v3`. Bump on
  any schema change; older entries are silently re-computed.

Public surface:

```python
def load(src_path: str, channel_id: str) -> tuple[ChannelMips | None, ShapeZYX | None]: ...
def save(src_path: str, channel_id: str, mips: ChannelMips, shape_zyx: ShapeZYX) -> None: ...
def save_lut_only(src_path: str, channel_id: str, axis_data: dict) -> None: ...
def clear_all() -> None: ...  # for "Clear MIP cache…" menu item
```

`save` removes any pre-existing `.lut.json` (a fresh compute invalidates user
overrides). `save_lut_only` writes the sidecar and never touches the `.npz`.
Nothing outside `cache.py` opens these paths.

## 8. Loader thread (`loader.py`)

`MipLoader(QThread)` is long-lived and queue-driven:

- Constructed once; reads `(acq_id, acq, fov, channel)` jobs from a thread-safe
  `queue.Queue`. The gallery window enqueues whenever new acquisitions arrive
  via drag-drop or the FOV dropdown asks for a not-yet-cached FOV.
- `cancel(acq_id)` removes any pending jobs for that acq from the queue.
  In-flight jobs run to completion (one channel of one FOV, typically a few
  seconds for a 50-150 slice z-stack on local disk). The cancel API is for
  cleaning up the queue, not interrupting work in progress.
- Signals:
  - `mip_ready(acq_id, fov, ch_idx, wavelength, channel_mips, shape_zyx)`
  - `progress(done, queued, message)` — `queued` is the current queue length,
    so it grows as the user drops more folders. Status bar shows
    "12/47 channels — computing 488 nm MIP — 25x H3".
  - `idle()` — emitted whenever the queue empties; the gallery window updates
    the footer.

Per-job pseudocode:

```python
src, ch_id = acq.handler.cache_key(acq, fov, channel)
cached, shape = cache.load(src, ch_id)
if cached is not None:
    emit_mip_ready(...); advance()
else:
    state = mips.new_axis_state()
    for slice_yx in acq.handler.iter_z_slices(acq, fov, channel):
        mips.accumulate_axes(slice_yx, state)
    mips_per_axis = mips.finalize(state)
    cache.save(src, ch_id, mips_per_axis, state.shape_zyx)
    emit_mip_ready(...); advance()
```

The MIP math (`new_axis_state`, `accumulate_axes`, `finalize`,
`mip_to_rgba`, percentile helpers) lives in `mips.py` as pure functions. They
are lifted from `explorer_ovelle.py` essentially as-is and are trivially
unit-testable.

Single-threaded by design in v1. Concurrent loader is §15 future work.

## 9. LUT dialog (`ui/lut_dialog.py`)

Behavior is lifted from `explorer_ovelle.py`'s `_adjust_lut`:

- Per-channel column with a large `ZoomableImageView` preview at the cached
  full resolution; mouse-wheel zoom (anchored under cursor, clamped to fit
  on zoom-out); click-drag pan.
- For XZ / YZ axes, the preview applies a non-uniform Y stretch
  (`dz_um / pixel_um`) so the image is shown at correct physical aspect while
  preserving every cached pixel.
- Per-channel min/max sliders; per-channel `→ 0` button; dataset-wide
  `Min → Data Min (all channels)`.
- LUT changes apply across all three axes simultaneously (XY/XZ/YZ stay
  synchronized).
- **Save** writes the small `<hash>.lut.json` sidecar via
  `cache.save_lut_only(...)` and never touches the `.npz`.
- **Closing without saving** reverts in-memory LUT changes to a snapshot taken
  at dialog open time.
- **Export PNG…** writes a 600 dpi figure using matplotlib's `Agg` backend:
  one column per channel showing the full-resolution image (with correct
  aspect) plus a log-scale intensity histogram with min/max marker lines.
  Channel exposure / laser intensity (when the handler returns them via
  `channel_yaml_extras`) appear under each image. Title shows display name +
  axis label + datetime.

The structural change from `explorer_ovelle.py` is that **the dialog is
format-agnostic**: the save path calls `cache.save_lut_only(*acq.handler.cache_key(...))`,
and the PNG export's per-channel info comes from
`acq.handler.channel_yaml_extras(...)`. No `if format == ...` branches inside
the dialog.

## 10. 3D viewer (`ui/viewer3d.py`)

One function: `open_napari(acq, fov, current_axis_lut_lookup)`. Pseudocode:

```python
def open_napari(acq, fov, lut_lookup):
    pixel_um = effective_pixel_um(acq)
    dz_um = acq.params["dz(um)"]
    scale = (dz_um, pixel_um, pixel_um)

    viewer = napari.Viewer(ndisplay=3, title=f"{acq.display_name} | FOV {fov}")
    for ch_idx, channel in enumerate(acq.channels):
        stack = acq.handler.load_full_stack(acq, fov, channel)
        clim = lut_lookup(acq, fov, ch_idx) or fallback_percentile(stack)
        cmap = NAPARI_COLORMAPS.get(channel.wavelength, "gray")
        viewer.add_image(stack, scale=scale, name=f"{channel.wavelength}nm",
                         colormap=cmap, blending="additive",
                         contrast_limits=clim)

    add_bounding_box_with_100um_ticks(viewer, scale, stack.shape)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = f"{acq.display_name} | FOV {fov}"
```

`lut_lookup` is a callable supplied by the gallery window that returns the
contrast limits for the given `(acq, fov, ch_idx)` tuple at the currently
displayed projection axis (falling back to Z-axis MIP percentiles, then to a
fresh percentile pass on the loaded stack).

## 11. Channel coloring

Hardcoded wavelength → RGB map for the canonical 5 wavelengths (`405`, `488`,
`561`, `638`, `730`); out-of-distribution wavelengths fall back to neutral
gray. This matches `explorer_ovelle.py` and matches squid users' actual data.
Reading OME `Channel.Color` metadata is §15 future work.

## 12. Testing

### 12.1 Synthetic data fixtures (`tests/conftest.py`)

Three fixture functions build squid-shaped acquisitions in a `tmp_path`:

```python
def make_ome_tiff_acq(tmp_path, channels, nz, ny, nx) -> Path: ...
def make_multi_channel_tiff_acq(tmp_path, channels, nz, ny, nx) -> Path: ...
def make_single_channel_tiff_acq(tmp_path, channels, nz, ny, nx) -> Path: ...
```

Each writes the format's exact on-disk layout (including
`acquisition parameters.json` and, where applicable, `acquisition_channels.yaml`)
populated with predictable synthetic data — e.g. a `z + y * 0.1 + x * 0.01`
gradient — so MIP results are deterministic and round-trips are checkable.

### 12.2 Unit tests

- **Per-handler.** `detect` returns true on its own format, false on the other
  two. `build` populates channels and parses scale. `read_shape` matches the
  written shape. `iter_z_slices` yields the right number of arrays in the right
  order. `cache_key` is stable across calls. `load_full_stack` shape is
  `(nz, ny, nx)`.
- **Scanner.** Depth-2 walk discovers nested acquisitions. Symlink loops are
  skipped. Duplicate `ingest()` calls are deduped on `realpath`. Mixed-format
  parents work.
- **MIP math.** `accumulate_axes` + `finalize` produces the same arrays as
  `np.max` along each axis on the full stack.
- **Cache.** Save/load round-trip preserves arrays exactly. `CACHE_VERSION`
  mismatch invalidates. LUT-sidecar overrides apply on top of cached defaults.
  Full save removes any pre-existing sidecar.
- **LUT override.** JSON round-trip; partial axis coverage is fine; malformed
  JSON returns `None` cleanly.

### 12.3 No GUI tests in v1

`pytest-qt` is added when a UI regression bites.

### 12.4 CI

GitHub Actions matrix: `ubuntu-latest` + `macos-latest`, Python 3.11 + 3.12.
Runs `pytest -q`. No napari/Qt initialization in CI — those imports happen only
inside `gallery_view.ui`, and tests import only from `gallery_view.scan`,
`gallery_view.cache`, `gallery_view.mips`, `gallery_view.sources.*`.

## 13. Dependencies

```toml
[project]
dependencies = [
    "qtpy",
    "PyQt6",
    "numpy",
    "tifffile",
    "pyyaml",
    "napari",
    "matplotlib",
    "platformdirs",
]
```

`napari` is pinned to a known-good range to insulate against its frequent
minor-version API changes; the exact range is set when v1 is cut.

## 14. Risks and known unknowns

| # | Risk | Mitigation |
|---|---|---|
| 1 | PyQt6 enum scoping (`Qt.AlignCenter` etc.) leaks past `qtpy`'s shim in odd places. | Smoke-launch the app once early; treat `qtpy` warnings as errors during dev. |
| 2 | napari API drift across minor versions. | Pin `napari>=X,<Y` in `pyproject.toml`. |
| 3 | squid format changes silently in a future commit. | Each `sources/<format>.py` carries a `SQUID_TESTED_AGAINST = "<commit>"` constant in its docstring; fixtures pin format-as-of-that-commit. CI catches drift only when fixtures are updated. |
| 4 | Wavelength → color map is hardcoded for the canonical 5. | Out-of-distribution wavelengths fall back to gray. Acceptable for v1. |
| 5 | Long-lived loader thread + drag-drop could starve the GUI on a huge initial drop. | Mostly mitigated by the queue + signal-per-channel design. If it bites, §15 item 1 is the fix. |
| 6 | OS-specific drag-drop (URLs vs paths, network mounts). | Use `QMimeData.urls()` and `QUrl.toLocalFile()`. Manual smoke test on macOS + Linux is part of v1 acceptance. |

## 15. Future work (explicitly **not** in v1)

1. **Concurrent MIP loader.** Replace the single-thread `MipLoader` with a
   `concurrent.futures` thread pool of size `min(4, os.cpu_count())`. Job queue
   gains priorities so visible rows compute first. In-flight cancellation
   propagates via per-job `Event`. Revisit when a single drag-drop session
   produces > ~50 acquisitions and the wait becomes user-visible.
2. **OME-Zarr handler.** New `sources/ome_zarr.py` implementing the same
   `FormatHandler` protocol. Cache layer is unchanged.
3. **Time-series support.** A `selected_t` field on `Acquisition` analogous to
   `selected_fov`, plus a row-level T picker.
4. **Multi-tile / mosaic stitching.** Out of v1 scope; needs its own design pass.
5. **Filesystem watching.** Auto-detect new acquisitions as squid writes them.
   Requires `watchdog` and careful debouncing during in-progress acquisitions.
6. **GUI tests.** Add `pytest-qt` once a UI regression hurts.
7. **Channel coloring from OME `Channel.Color` metadata.** When non-canonical
   wavelengths get used.
8. **PyPI release.** Once API stabilizes.

## 16. Acceptance criteria for v1

- `python -m gallery_view` opens an empty window with a drop target.
- Dropping a folder of squid acquisitions (or a parent containing several)
  populates the gallery — empty thumb cells appear immediately and fill in as
  MIPs compute (cache cold), or render instantly (cache warm).
- All three projection axes (XY/XZ/YZ) render correctly; switching is instant
  on warm cache.
- LUT dialog adjusts contrast across all three axes synchronously, save
  persists across restart, close-without-save reverts.
- PNG export writes a 600 dpi figure with histograms and per-channel info.
- napari 3D viewer opens with µm-scaled axes, bounding box, scale bar, and
  reuses LUT from the gallery.
- `pytest -q` passes on macOS + Linux for Python 3.11 and 3.12.
