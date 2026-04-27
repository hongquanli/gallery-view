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

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        # Each channel lives in its own set of TIFFs, so per-channel loading
        # is already memory-efficient — no shared parent buffer to worry
        # about. Just delegate to load_full_stack.
        for channel in acq.channels:
            yield channel, self.load_full_stack(acq, fov, channel)

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
