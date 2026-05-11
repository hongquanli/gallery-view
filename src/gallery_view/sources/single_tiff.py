"""Single TIFF (individual-image-per-file) handler for squid output.

Layout: ``<acq>/0/current_0_<z>_<channel_name>.tiff`` — one TIFF per Z slice
per channel, all in a single FOV directory.
"""

import csv
import functools
import glob
import os
import re
from typing import Iterator

import numpy as np
from tifffile import TiffFile, imread

from ..stitch import FovCoord
from ..types import Acquisition, Channel, ShapeZYX
from . import _squid_common as common

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


def _fov_sort_key(coord) -> tuple:
    """Natural sort key for FovCoord composites: '0_2' < '0_10' < '1_0'."""
    parts = coord.fov.split("_")
    return tuple(int(p) if p.isdigit() else p for p in parts)


class SingleTiffHandler:
    name = "single_tiff"

    def detect(self, folder: str) -> bool:
        return self._detect_layout(folder) is not None

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
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        return acq.path, f"fov{fov}/t{timepoint}/{channel.name}"

    def cache_key_region(
        self, acq: Acquisition, region: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        return acq.path, f"region:{region}/t{timepoint}/{channel.name}"

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
                    if key in seen and row.get("z_level") not in ("0", "0.0"):
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
        if not result:
            return None
        # Sort each region's coords by composite fov for determinism.
        for region in result:
            result[region].sort(key=_fov_sort_key)
        acq.extra["coords_by_region"] = result
        return result

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> Iterator[np.ndarray]:
        for f in self._tiffs_for(acq, fov, channel, timepoint=timepoint):
            yield imread(f).astype(np.float32)

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> np.ndarray:
        tiffs = self._tiffs_for(acq, fov, channel, timepoint=timepoint)
        if not tiffs:
            raise FileNotFoundError(
                f"No TIFFs for {channel.name!r} at fov={fov!r} t={timepoint!r}"
            )
        return np.stack([imread(f) for f in tiffs])

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str, timepoint: str = "0"
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        # Each channel lives in its own set of TIFFs, so per-channel loading
        # is already memory-efficient — no shared parent buffer to worry
        # about. Just delegate to load_full_stack.
        for channel in acq.channels:
            yield channel, self.load_full_stack(acq, fov, channel, timepoint=timepoint)

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        return common.channel_extras_from_yaml(acq.path, channel)

    # ── helpers ──

    @staticmethod
    def _tiffs_for(
        acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0",
    ) -> list[str]:
        """Return z-sorted TIFF paths for one (timepoint, fov, channel).

        ``fov`` is the composite ``"<region>_<fov>"`` string from
        ``acq.fovs``; for legacy folders the region is always ``"0"``.
        """
        layout = acq.extra.get("layout", "legacy")
        region, fov_idx = fov.split("_", 1) if "_" in fov else ("0", fov)
        pattern = channel.name.replace(" ", "_")
        t_dir = os.path.join(acq.path, timepoint)
        if layout == "squid":
            parser = parse_squid_filename
            # Glob anchors the channel suffix; the leading prefix may include
            # an optional well, so we filter by parsing each match to confirm
            # (region, fov) and discard collisions.
            files = []
            for f in glob.glob(os.path.join(t_dir, f"*_{pattern}.tiff")):
                p = parser(os.path.basename(f))
                if p is None:
                    continue
                if p["region"] == region and p["fov"] == fov_idx:
                    files.append(f)
        else:
            files = glob.glob(
                os.path.join(t_dir, f"current_{fov_idx}_*_{pattern}.tiff")
            )

        def z_of(path: str) -> int:
            base = os.path.basename(path)
            p = (
                parse_squid_filename(base)
                if layout == "squid"
                else parse_legacy_filename(base)
            )
            return int(p["z"]) if p else 0

        files.sort(key=z_of)
        return files

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _detect_layout(folder: str) -> str | None:
        """Return ``"squid"`` or ``"legacy"`` if folder matches; else None.

        Tries legacy first because the squid regex's ``[^_]+`` region
        group also matches a ``current`` prefix — order matters.

        ``glob.glob`` results are sorted so the "first matching file wins"
        outcome is deterministic across filesystems (the glob's natural
        order is filesystem-dependent on macOS/Linux).

        Cached because ``scan.ingest`` calls ``handler.detect`` then
        ``handler.build``, and ``build`` calls this again — without
        memoization that's two filesystem walks per acquisition for every
        drop. Cache lives for the process and is keyed by the raw folder
        string; ``scan.ingest`` already deduplicates paths via realpath.
        """
        if not os.path.isdir(folder):
            return None
        # Legacy + squid both put files under <folder>/0/ when t=0.
        zero_dir = os.path.join(folder, "0")
        if os.path.isdir(zero_dir):
            for f in sorted(glob.glob(os.path.join(zero_dir, "*.tiff"))):
                base = os.path.basename(f)
                if parse_legacy_filename(base) is not None:
                    return "legacy"
                if parse_squid_filename(base) is not None:
                    return "squid"
        # Squid layout may also have <t>/ dirs other than 0 (multi-timepoint).
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
            for f in sorted(glob.glob(os.path.join(t_dir, "*.tiff"))):
                if parse_squid_filename(os.path.basename(f)) is not None:
                    return "squid"
        return None

    @staticmethod
    def _timepoints_for(folder: str, layout: str) -> list[str]:
        if layout == "legacy":
            return ["0"]
        return sorted(
            (e for e in os.listdir(folder)
             if e.isdigit() and os.path.isdir(os.path.join(folder, e))),
            key=int,
        )

    @staticmethod
    def _fovs_for(folder: str, layout: str, timepoint: str) -> list[str]:
        t_dir = os.path.join(folder, timepoint)
        parser = (
            parse_squid_filename if layout == "squid" else parse_legacy_filename
        )
        seen: set[tuple[str, str]] = set()
        for f in glob.glob(os.path.join(t_dir, "*.tiff")):
            p = parser(os.path.basename(f))
            if p is None:
                continue
            region = p.get("region", "0")
            seen.add((region, p["fov"]))
        # Sort numerically when possible; fall back to lexicographic for
        # well-as-region cases like "A1".
        def sort_key(pair: tuple[str, str]) -> tuple:
            r, f = pair
            return (0 if r.isdigit() else 1, int(r) if r.isdigit() else r, int(f))
        return [f"{r}_{f}" for r, f in sorted(seen, key=sort_key)]

    @staticmethod
    def _regions_from_fovs(fovs: list[str]) -> list[str]:
        """Distinct region prefixes from composite '<region>_<fov>' strings,
        preserving the order established by ``_fovs_for`` (single source of
        truth — keeps regions and fovs from drifting apart on plates with
        well-name regions like A1, A2, ...)."""
        seen: dict[str, None] = {}   # dict for insertion-ordered dedup
        for fov in fovs:
            region = fov.split("_", 1)[0] if "_" in fov else "0"
            seen[region] = None
        return list(seen)

    @staticmethod
    def _channels_from_filenames(
        folder: str, layout: str, timepoint: str,
    ) -> list[Channel]:
        t_dir = os.path.join(folder, timepoint)
        parser = (
            parse_squid_filename if layout == "squid" else parse_legacy_filename
        )
        names: set[str] = set()
        for f in glob.glob(os.path.join(t_dir, "*.tiff")):
            p = parser(os.path.basename(f))
            if p is None:
                continue
            names.add(p["channel"])
        out = [common.make_channel_from_name(name) for name in sorted(names)]
        out.sort(
            key=lambda c: int(c.wavelength) if c.wavelength.isdigit() else 999
        )
        return out
