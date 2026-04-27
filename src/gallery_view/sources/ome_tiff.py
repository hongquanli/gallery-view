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

    def read_shape(self, acq: Acquisition, fov: str) -> ShapeZYX | None:
        try:
            with TiffFile(acq.extra["ome_path"]) as tif:
                s = tif.series[0]
                axes, shape = s.axes, s.shape
                if axes == "CYX":
                    return (1, shape[1], shape[2])
                if axes in ("ZYX", "TYX"):
                    # Single-channel z-stack; tifffile drops the size-1 C axis.
                    return (shape[0], shape[1], shape[2])
                if axes == "YX":
                    return (1, shape[0], shape[1])
                if axes in ("ZCYX", "TCYX"):
                    return (shape[0], shape[2], shape[3])
        except (OSError, ValueError, IndexError):
            return None
        return None

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        return acq.extra["ome_path"], f"fov{fov}/t{timepoint}/wl_{channel.wavelength}"

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> Iterator[np.ndarray]:
        ome_path = acq.extra["ome_path"]
        ch_idx = self._channel_index(acq, channel)
        with TiffFile(ome_path) as tif:
            s = tif.series[0]
            axes, shape = s.axes, s.shape
            if axes == "CYX":
                yield tif.pages[ch_idx].asarray().astype(np.float32)
                return
            if axes == "YX":
                yield tif.pages[0].asarray().astype(np.float32)
                return
            # tifffile drops size-1 C axes on read, so a (nz, 1, ny, nx) ZCYX
            # file comes back as ZYX. Handle the single-channel case explicitly.
            if axes in ("ZYX", "TYX"):
                nz = shape[0]
                for z in range(nz):
                    yield tif.pages[z].asarray().astype(np.float32)
                return
            # TCYX is treated identically to ZCYX: squid v1 only writes
            # single-timepoint TCYX, where the T-axis is effectively the
            # Z-axis. Multi-timepoint OME-TIFFs are out of scope.
            if axes in ("ZCYX", "TCYX"):
                nz, nc = shape[0], shape[1]
                for z in range(nz):
                    yield tif.pages[z * nc + ch_idx].asarray().astype(np.float32)
                return
            raise ValueError(f"Unsupported OME-TIFF axes: {axes}")

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> np.ndarray:
        ch_idx = self._channel_index(acq, channel)
        with TiffFile(acq.extra["ome_path"]) as tif:
            s = tif.series[0]
            return self._read_channel_stack(tif, s.axes, s.shape, ch_idx)

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str, timepoint: str = "0"
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        """Yield ``(Channel, ZYX-stack)`` for each channel as an
        **independent** contiguous array.

        Reading per-channel (rather than ``series.asarray()`` + a sliced
        view) means each napari layer holds its own (nz, ny, nx) buffer.
        When the user closes the 3D viewer, layers can be garbage-
        collected individually instead of all four pinning a single
        shared ~5 GB parent until the last reference drops.
        """
        with TiffFile(acq.extra["ome_path"]) as tif:
            s = tif.series[0]
            axes, shape = s.axes, s.shape
            for ch_idx, channel in enumerate(acq.channels):
                yield channel, self._read_channel_stack(tif, axes, shape, ch_idx)

    @staticmethod
    def _read_channel_stack(
        tif: TiffFile, axes: str, shape: tuple, ch_idx: int
    ) -> np.ndarray:
        if axes == "CYX":
            return tif.pages[ch_idx].asarray()[np.newaxis, :, :]
        if axes == "YX":
            return tif.pages[0].asarray()[np.newaxis, :, :]
        if axes in ("ZYX", "TYX"):
            nz = shape[0]
            return np.stack([tif.pages[z].asarray() for z in range(nz)])
        if axes in ("ZCYX", "TCYX"):
            nz, nc = shape[0], shape[1]
            return np.stack([
                tif.pages[z * nc + ch_idx].asarray() for z in range(nz)
            ])
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
                if axes in ("ZCYX", "TCYX"):
                    nc = shape[1]
                elif axes == "CYX":
                    nc = shape[0]
                elif axes in ("YX", "ZYX", "TYX"):
                    nc = 1  # tifffile collapsed a size-1 C axis on read
                else:
                    nc = 0
        except (OSError, ValueError, IndexError):
            return []
        return [Channel(name=f"channel_{i}", wavelength="unknown") for i in range(nc)]
