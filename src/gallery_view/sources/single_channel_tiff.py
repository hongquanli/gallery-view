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

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        # Each channel lives in its own folder of TIFFs — no shared parent
        # buffer concern; per-channel loading is already efficient.
        for channel in acq.channels:
            yield channel, self.load_full_stack(acq, fov, channel)

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
