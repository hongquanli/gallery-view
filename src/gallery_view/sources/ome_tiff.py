"""OME-TIFF handler for squid output (axes ZCYX, TCYX, or CYX)."""

from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np
from tifffile import TiffFile

from ..types import Acquisition, Channel, ShapeZYX
from . import _squid_common as common

OME_PATH = ("ome_tiff", "current_0.ome.tiff")


def _build_page_map(ome_xml: str | None) -> dict[tuple[int, int, int], int] | None:
    """Parse OME ``Plane`` tags into a ``(t, z, c) → page_idx`` map.

    The OME spec lets a writer declare a ``DimensionOrder`` (e.g. ``XYCZT``)
    while actually storing pages in some other order — every page carries
    explicit ``TheT``/``TheZ``/``TheC`` attributes that authoritatively
    locate it. squid (via tifffile) emits ``DimensionOrder="XYCZT"`` for
    multi-channel stacks but the XLight V3 path stores pages C-major
    (per-channel blocks of Z slices), so a ``page = z*nc + c`` formula
    reads from the wrong channel.

    Returns ``None`` when the XML is absent or has no Plane tags; callers
    must fall back to the DimensionOrder-implied formula.
    """
    if not ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None
    out: dict[tuple[int, int, int], int] = {}
    idx = 0
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]  # strip namespace
        if tag != "Plane":
            continue
        try:
            t = int(elem.attrib.get("TheT", "0"))
            z = int(elem.attrib.get("TheZ", "0"))
            c = int(elem.attrib.get("TheC", "0"))
        except ValueError:
            return None
        out[(t, z, c)] = idx
        idx += 1
    return out or None


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

        # Build the (t, z, c) -> page_idx map once per acquisition; iter and
        # full-stack readers use it to find each plane's actual storage page.
        page_map: dict[tuple[int, int, int], int] | None = None
        try:
            with TiffFile(ome_path) as tif:
                page_map = _build_page_map(tif.ome_metadata)
        except (OSError, ValueError, IndexError):
            page_map = None

        extra: dict = {"ome_path": ome_path}
        if page_map is not None:
            extra["page_map"] = page_map

        folder_name = os.path.basename(folder)
        return Acquisition(
            handler=self,
            path=folder,
            folder_name=folder_name,
            display_name=common.display_name_for(folder_name),
            params=params,
            channels=channels,
            fovs=["0"],
            extra=extra,
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
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> tuple[str, str]:
        return acq.extra["ome_path"], f"fov{fov}/wl_{channel.wavelength}"

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]:
        ome_path = acq.extra["ome_path"]
        ch_idx = self._channel_index(acq, channel)
        page_map = acq.extra.get("page_map")
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
                    page = self._page_index(page_map, z, ch_idx, nc)
                    yield tif.pages[page].asarray().astype(np.float32)
                return
            raise ValueError(f"Unsupported OME-TIFF axes: {axes}")

    @staticmethod
    def _page_index(
        page_map: dict[tuple[int, int, int], int] | None,
        z: int,
        c: int,
        nc: int,
    ) -> int:
        """Return the storage-page index for ``(t=0, z, c)``.

        Prefers the OME ``Plane``-tag map when present; falls back to the
        ``DimensionOrder="XYCZT"`` formula (channels-fast within Z) used by
        most squid OME-TIFFs that lack Plane tags.
        """
        if page_map is not None:
            page = page_map.get((0, z, c))
            if page is not None:
                return page
        return z * nc + c

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray:
        # Single-channel callers: ``series.asarray()`` reads the OME Plane
        # metadata internally and reshapes pages into logical ZCYX/TCYX
        # order, which matters for files like XLight V3 that declare
        # ``DimensionOrder="XYCZT"`` but store pages C-major. Multi-channel
        # callers should use :py:meth:`iter_full_channel_stacks` to avoid
        # reloading the full file once per channel.
        ch_idx = self._channel_index(acq, channel)
        with TiffFile(acq.extra["ome_path"]) as tif:
            s = tif.series[0]
            data = s.asarray()
            axes = s.axes
        return self._slice_channel(data, axes, ch_idx)

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        """Load the OME-TIFF once, then yield each channel as a ZYX view.

        ``data[:, ch_idx, :, :]`` returns a numpy view that shares memory
        with the parent ``data`` array — so the four channel views in a
        4-channel acquisition all point at the same ~5GB buffer rather
        than each holding its own copy.
        """
        with TiffFile(acq.extra["ome_path"]) as tif:
            s = tif.series[0]
            data = s.asarray()
            axes = s.axes
        for ch_idx, channel in enumerate(acq.channels):
            yield channel, self._slice_channel(data, axes, ch_idx)

    @staticmethod
    def _slice_channel(data: np.ndarray, axes: str, ch_idx: int) -> np.ndarray:
        if axes == "CYX":
            return data[ch_idx][np.newaxis, :, :]
        if axes == "YX":
            return data[np.newaxis, :, :]
        if axes in ("ZYX", "TYX"):
            return data
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
