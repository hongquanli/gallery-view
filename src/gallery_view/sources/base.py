"""FormatHandler protocol.

Each squid on-disk format gets one handler. A handler is stateless and is
constructed once at import time; the global registry in ``__init__.py``
holds the singletons in priority order.

The cache module is FOV-agnostic; handlers fold the FOV into the
``channel_id`` they return from ``cache_key``.
"""

from typing import Iterator, Protocol

import numpy as np

from ..types import Acquisition, Channel, ShapeZYX


class FormatHandler(Protocol):
    name: str

    def detect(self, folder: str) -> bool:
        """Return True iff ``folder`` is in this handler's format."""
        ...

    def build(self, folder: str, params: dict) -> Acquisition | None:
        """Construct an Acquisition from the folder. Returns None if the
        folder is unparseable for any reason (channels missing, etc.)."""
        ...

    def read_shape(
        self, acq: Acquisition, fov: str
    ) -> ShapeZYX | None:
        """Return ``(nz, ny, nx)`` for one FOV without reading the full stack."""
        ...

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> tuple[str, str]:
        """Return ``(src_path_for_hash, channel_id)`` for ``cache.load/save``.

        ``channel_id`` MUST be unique per (acq, fov, channel, timepoint).
        Encode the FOV (and timepoint, where applicable) into the string
        when the handler can have more than one. Handlers whose data is
        partitioned by timepoint use this argument; OME-TIFF (which packs
        all timepoints into a single file) ignores it.
        """
        ...

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

    def load_region_coords(
        self, acq: Acquisition
    ) -> "dict[str, list] | None":
        """Return per-region stage coordinates for region-view stitching.

        Returns the parsed mapping (keyed by region id) or ``None`` when the
        handler doesn't support region view or coordinates aren't available.
        """
        ...

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> Iterator[np.ndarray]:
        """Yield each Z-slice as a YX float32 array, in Z-order.

        Handlers whose data is partitioned by timepoint use ``timepoint``
        to pick the per-timepoint subdirectory; OME-TIFF (which packs all
        timepoints into a single file) ignores it.
        """
        ...

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel, timepoint: str = "0"
    ) -> np.ndarray:
        """Return the full ZYX stack (float32 or native dtype) for napari.

        Handlers whose data is partitioned by timepoint use ``timepoint``;
        OME-TIFF ignores it.
        """
        ...

    def iter_full_channel_stacks(
        self, acq: Acquisition, fov: str, timepoint: str = "0"
    ) -> Iterator[tuple[Channel, np.ndarray]]:
        """Yield ``(Channel, ZYX-stack)`` pairs for every channel in ``fov``.

        Default handlers can implement this as a per-channel loop on top of
        :py:meth:`load_full_stack`, but formats that bundle all channels in
        one file (notably OME-TIFF) should override it to load the file
        ONCE and slice — calling ``load_full_stack`` per channel for an
        OME-TIFF would re-load the entire (potentially multi-GB) file
        each time. Handlers whose data is partitioned by timepoint use
        ``timepoint``; OME-TIFF ignores it.
        """
        ...

    def channel_yaml_extras(
        self, acq: Acquisition, channel: Channel
    ) -> dict:
        """Per-channel extras for the PNG export footer.

        Returns ``{}`` when the metadata isn't available. Recognized keys:
          - ``exposure_ms`` (float)
          - ``intensity`` (float, 0-100% laser power)
        """
        ...
