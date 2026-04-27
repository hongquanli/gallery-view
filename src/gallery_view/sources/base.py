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

    def list_fovs(self, acq: Acquisition) -> list[str]:
        """List of FOV ids available in the acquisition. v1: always ['0']."""
        ...

    def read_shape(
        self, acq: Acquisition, fov: str
    ) -> ShapeZYX | None:
        """Return ``(nz, ny, nx)`` for one FOV without reading the full stack."""
        ...

    def cache_key(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> tuple[str, str]:
        """Return ``(src_path_for_hash, channel_id)`` for ``cache.load/save``.

        ``channel_id`` MUST be unique per (acq, fov, channel). Encode the FOV
        into the string when the handler can have more than one FOV.
        """
        ...

    def iter_z_slices(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> Iterator[np.ndarray]:
        """Yield each Z-slice as a YX float32 array, in Z-order."""
        ...

    def load_full_stack(
        self, acq: Acquisition, fov: str, channel: Channel
    ) -> np.ndarray:
        """Return the full ZYX stack (float32 or native dtype) for napari."""
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
