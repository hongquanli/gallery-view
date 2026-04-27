# gallery-view

Standalone gallery viewer for z-stack microscopy acquisitions written by
[squid](https://github.com/cephla-lab/squid).

Browse many acquisitions side-by-side as false-color MIP thumbnails grouped by
well, switch between XY / XZ / YZ projections, and pop a single acquisition
open in a 3D napari viewer for closer inspection.

## Features

- **Gallery view** — per-acquisition rows, one column per fluorescence
  wavelength (405 / 488 / 561 / 638 / 730 nm), false-colored and additively
  blended for a multi-channel overview.
- **Three projection axes** — toggle XY (MIP along Z), XZ (along Y), or YZ
  (along X) without reloading the data.
- **3D napari viewer** — *Open 3D View* on any acquisition launches a napari
  volume viewer with µm-scaled axes, a 100 µm-tick bounding box, and the
  current LUT limits carried over from the gallery.
- **LUT controls** — *Adjust Contrast* opens a per-channel histogram dialog;
  changes are persisted alongside the MIP cache.
- **Background loading + disk cache** — full-resolution MIPs are computed
  once on first sight and cached as `.npz` files under the OS cache dir;
  subsequent loads are instant.
- **Multi-FOV and multi-timepoint acquisitions** — multi-FOV rows expose a
  per-row FOV picker (or "Expand all FOVs as separate rows" via the
  Settings menu); multi-timepoint acquisitions expose a per-row Time
  picker that switches the displayed timepoint without reloading.
- **Multiple sources** — drag folders onto the window or pass `--source PATH`
  on the command line; sources can be acquisition folders or parents
  containing many.

## Supported formats

The scanner auto-detects two squid output layouts:

| Format        | Layout                                                                |
|---------------|-----------------------------------------------------------------------|
| `ome_tiff`    | `<acq>/ome_tiff/current_0.ome.tiff` — one OME-TIFF holding every (z, channel). |
| `single_tiff` (squid)  | `<acq>/<t>/<region>_<fov>_<z>_<channel>.tiff` — one TIFF per (timepoint, region, fov, z, channel). `<region>` is any non-underscore string (numeric id, well like `A1`, etc.); matches `cephla-lab/ndviewer_light`'s filename convention. Multi-timepoint acquisitions surface a per-row Time picker. |
| `single_tiff` (legacy) | `<acq>/0/current_<fov>_<z>_<channel>.tiff` — older squid output, single timepoint. Still supported. |

Channel metadata is read from `<acq>/acquisition_channels.yaml` when present
and inferred from filenames otherwise.

## Requirements

- Python ≥ 3.11
- macOS or Linux
- Dependencies (installed automatically): PyQt6, napari ≥ 0.4.18 < 0.6,
  tifffile, numpy, matplotlib, pyyaml, platformdirs

## Install

```bash
pip install -e '.[dev]'
```

(Quote the extras spec — zsh treats unquoted `[dev]` as a glob.)

## Run

```bash
python -m gallery_view
# or with one or more sources preloaded:
python -m gallery_view --source /path/to/acquisitions
python -m gallery_view --source /path/A --source /path/B
```

Then drag acquisition folders (or parent folders containing many) onto the
window. Use the *File* menu to add or refresh folders, the axis buttons in
the toolbar to switch projection, and *Settings → Clear MIP cache…* to
recompute from scratch.

The MIP cache lives under:

- macOS: `~/Library/Caches/gallery-view/mips/`
- Linux: `~/.cache/gallery-view/mips/`

## Tests

```bash
pytest -q
```
