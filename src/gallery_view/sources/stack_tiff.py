"""Stack-per-FOV TIFF handler.

Layout: ``<acq>/<t>/<region>_<fov>_stack.tiff`` — one multi-page TIFF per
FOV per timepoint, with all (z, channel) planes interleaved as pages.

Two metadata flavours are handled:

  * **per_page_meta** (current squid): each page carries a JSON
    ``ImageDescription`` with ``z_level``, ``channel``, ``channel_index``,
    ``region_id``, ``fov`` — authoritative for channel order and z mapping.
    All future acquisitions take this path.
  * **implicit** (legacy — early Dragonfly builds, before per-page metadata
    was added): ``ImageDescription`` only has ``shape``. We derive
    ``Nc = total_pages / Nz`` from ``acquisition parameters.json`` and read
    channel names from ``configurations.xml``. Page order is z-major:
    ``page = z * Nc + c``. This path is only here to read existing legacy
    data; once those datasets are reprocessed or aged out, the path can
    be removed.

``configurations.xml`` also provides per-channel exposure/intensity for
the export footer; there is no ``acquisition_channels.yaml``.
"""

import functools
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Iterator

import numpy as np
from tifffile import TiffFile

from ..types import Acquisition, Channel, ShapeZYX
from . import _squid_common as common

# {channel_name: [(z_level, page_index), …]}, z-sorted. Stored in
# ``acq.extra`` so iteration paths don't re-parse the TIFF.
_CHANNEL_PAGES = "channel_pages"

_STACK_RE = re.compile(
    r"^(?P<region>[^_]+)_(?P<fov>\d+)_stack\.tiff?$", re.IGNORECASE
)


def parse_stack_filename(name: str) -> dict | None:
    """Parse ``<region>_<fov>_stack.tiff`` -> ``{"region", "fov"}`` or None."""
    m = _STACK_RE.match(name)
    return m.groupdict() if m else None


class StackTiffHandler:
    name = "stack_tiff"

    def detect(self, folder: str) -> bool:
        return self._first_stack_path(folder) is not None

    def build(self, folder: str, params: dict) -> Acquisition | None:
        first_stack = self._first_stack_path(folder)
        if first_stack is None:
            return None
        timepoints = self._timepoints_for(folder)
        if not timepoints:
            return None
        fovs = self._fovs_for(folder, timepoints[0])
        if not fovs:
            return None
        channels, channel_pages = _discover_channels_and_pages(
            first_stack, folder, params
        )
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
            extra={_CHANNEL_PAGES: channel_pages},
        )

    def read_shape(self, acq: Acquisition, fov: str) -> ShapeZYX | None:
        path = self._stack_path(acq, fov, acq.selected_timepoint)
        if path is None:
            return None
        # nz comes from the precomputed page map; only ny/nx require I/O.
        pages = acq.extra.get(_CHANNEL_PAGES, {})
        any_channel_pages = next(iter(pages.values()), [])
        nz = max(len(any_channel_pages), 1)
        try:
            with TiffFile(path) as tif:
                ny, nx = tif.pages[0].shape
        except (OSError, ValueError, IndexError):
            return None
        return (nz, ny, nx)

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        return acq.path, f"fov{fov}/t{timepoint}/wl_{channel.wavelength}"

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> Iterator[np.ndarray]:
        path = self._stack_path(acq, fov, timepoint)
        if path is None:
            return
        order = acq.extra.get(_CHANNEL_PAGES, {}).get(channel.name, [])
        with TiffFile(path) as tif:
            for _z, page_idx in order:
                yield tif.pages[page_idx].asarray().astype(np.float32)

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> np.ndarray:
        path = self._stack_path(acq, fov, timepoint)
        if path is None:
            raise FileNotFoundError(
                f"No stack TIFF for fov={fov!r} t={timepoint!r} in {acq.path}"
            )
        order = acq.extra.get(_CHANNEL_PAGES, {}).get(channel.name, [])
        with TiffFile(path) as tif:
            return np.stack([tif.pages[p].asarray() for _z, p in order])

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str, timepoint: str = "0"
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        path = self._stack_path(acq, fov, timepoint)
        if path is None:
            return
        pages = acq.extra.get(_CHANNEL_PAGES, {})
        with TiffFile(path) as tif:
            for channel in acq.channels:
                order = pages.get(channel.name, [])
                yield channel, np.stack([
                    tif.pages[p].asarray() for _z, p in order
                ])

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        for mode in _parse_modes_xml(acq.path):
            if mode["name"] == channel.name:
                return {
                    "exposure_ms": mode["exposure_ms"],
                    "intensity": mode["intensity"],
                }
        return {}

    # ── helpers ──

    @staticmethod
    def _first_stack_path(folder: str) -> str | None:
        """Return any one stack TIFF under ``<folder>/<digit>/`` or None."""
        if not os.path.isdir(folder):
            return None
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            return None
        for entry in entries:
            if not entry.isdigit():
                continue
            t_dir = os.path.join(folder, entry)
            if not os.path.isdir(t_dir):
                continue
            for f in sorted(_glob_stack_files(t_dir)):
                if parse_stack_filename(os.path.basename(f)) is not None:
                    return f
        return None

    @staticmethod
    def _timepoints_for(folder: str) -> list[str]:
        try:
            entries = os.listdir(folder)
        except OSError:
            return []
        return sorted(
            (e for e in entries
             if e.isdigit() and os.path.isdir(os.path.join(folder, e))),
            key=int,
        )

    @staticmethod
    def _fovs_for(folder: str, timepoint: str) -> list[str]:
        t_dir = os.path.join(folder, timepoint)
        seen: set[tuple[str, str]] = set()
        for f in _glob_stack_files(t_dir):
            p = parse_stack_filename(os.path.basename(f))
            if p is not None:
                seen.add((p["region"], p["fov"]))

        def sort_key(pair: tuple[str, str]) -> tuple:
            r, f = pair
            return (0 if r.isdigit() else 1, int(r) if r.isdigit() else r, int(f))

        return [f"{r}_{f}" for r, f in sorted(seen, key=sort_key)]

    @staticmethod
    def _stack_path(
        acq: Acquisition, fov: str, timepoint: str
    ) -> str | None:
        region, fov_idx = fov.split("_", 1) if "_" in fov else ("0", fov)
        t_dir = os.path.join(acq.path, timepoint)
        for ext in ("tiff", "tif"):
            p = os.path.join(t_dir, f"{region}_{fov_idx}_stack.{ext}")
            if os.path.exists(p):
                return p
        return None


# ── module-level helpers (no class state needed) ──


def _glob_stack_files(t_dir: str) -> list[str]:
    """List `*_stack.tiff` and `*_stack.tif` under a single directory."""
    return (
        glob.glob(os.path.join(t_dir, "*_stack.tiff"))
        + glob.glob(os.path.join(t_dir, "*_stack.tif"))
    )


def _page_meta(page) -> dict:
    """Parse the JSON ``ImageDescription`` on a TIFF page; ``{}`` if missing."""
    tag = page.tags.get("ImageDescription")
    if tag is None:
        return {}
    try:
        return json.loads(tag.value)
    except (TypeError, ValueError):
        return {}


def _discover_channels_and_pages(
    stack_path: str, folder: str, params: dict
) -> tuple[list[Channel], dict[str, list[tuple[int, int]]]]:
    """Return ``(channels, {channel_name: [(z, page_idx)…]})``.

    Walks the TIFF *once*, branching on whether per-page metadata is
    present. For the implicit flavour, the page→(z, c) mapping is
    derived arithmetically from ``params["Nz"]`` and the
    ``configurations.xml`` channel list (reversed; see
    :py:func:`_selected_fluorescence_modes`).

    Returns ``([], {})`` if neither flavour yields a usable layout.
    """
    try:
        with TiffFile(stack_path) as tif:
            metas = [_page_meta(p) for p in tif.pages]
    except (OSError, ValueError):
        return [], {}
    if not metas:
        return [], {}

    if metas[0].get("channel") and "z_level" in metas[0]:
        return _from_per_page_meta(metas)
    return _from_implicit_layout(metas, folder, params)


def _from_per_page_meta(
    metas: list[dict],
) -> tuple[list[Channel], dict[str, list[tuple[int, int]]]]:
    by_idx: dict[int, str] = {}
    pages: dict[str, list[tuple[int, int]]] = {}
    for page_idx, meta in enumerate(metas):
        name = meta.get("channel")
        c_idx = meta.get("channel_index")
        z = meta.get("z_level")
        if not name or c_idx is None or z is None:
            continue
        by_idx.setdefault(int(c_idx), name)
        pages.setdefault(name, []).append((int(z), page_idx))
    if not by_idx:
        return [], {}
    for v in pages.values():
        v.sort(key=lambda zp: zp[0])
    channels = [common.make_channel_from_name(by_idx[i]) for i in sorted(by_idx)]
    return channels, pages


def _from_implicit_layout(
    metas: list[dict], folder: str, params: dict,
) -> tuple[list[Channel], dict[str, list[tuple[int, int]]]]:
    try:
        nz = int(params.get("Nz", 1))
    except (TypeError, ValueError):
        return [], {}
    total = len(metas)
    if nz <= 0 or total <= 0 or total % nz != 0:
        return [], {}
    nc = total // nz
    names = _selected_fluorescence_modes(folder)
    if len(names) != nc:
        return [], {}
    channels = [common.make_channel_from_name(n) for n in names]
    pages: dict[str, list[tuple[int, int]]] = {n: [] for n in names}
    for page_idx in range(total):
        z, c = divmod(page_idx, nc)
        pages[names[c]].append((z, page_idx))
    return channels, pages


@functools.lru_cache(maxsize=64)
def _parse_modes_xml(folder: str) -> list[dict]:
    """Parse ``<folder>/configurations.xml`` into a list of mode dicts in
    XML doc order. Each entry has ``name``, ``selected``, ``exposure_ms``,
    ``intensity``. Returns ``[]`` on any failure.

    Cached because both channel discovery (every ingest) and the PNG-export
    footer (per channel per export) need it.
    """
    path = os.path.join(folder, "configurations.xml")
    if not os.path.exists(path):
        return []
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    return [
        {
            "name": mode.get("Name", ""),
            "selected": mode.get("Selected", "").lower() == "true",
            "exposure_ms": _safe_float(mode.get("ExposureTime")),
            "intensity": _safe_float(mode.get("IlluminationIntensity")),
        }
        for mode in root.findall("mode")
    ]


def _selected_fluorescence_modes(folder: str) -> list[str]:
    """Selected fluorescence-mode names in disk order (reverse XML doc order).

    Used **only by the legacy implicit-layout path**. The current squid
    writes per-page metadata that pins each page to a specific channel by
    name, so XML order doesn't matter there. This helper exists to read
    older Dragonfly data that pre-dates per-page metadata.

    Squid writes channels to those legacy TIFFs in *reverse* of the XML's
    selected-mode order — verified empirically: a Widefield acquisition
    whose configurations.xml selected modes were 405, 488, 730 produced
    per-page metadata pinning ``channel_index`` 0,1,2 to 730,488,405 nm.
    If a future legacy file ever surfaces with a different convention,
    this function is the single place to revisit. New acquisitions are
    immune.
    """
    selected = [
        m["name"] for m in _parse_modes_xml(folder)
        if m["selected"] and m["name"].startswith("Fluorescence")
    ]
    return list(reversed(selected))


def _safe_float(s) -> float | None:
    try:
        return float(s) if s is not None else None
    except (TypeError, ValueError):
        return None
