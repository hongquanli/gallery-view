"""Handler registry. ``detect()`` walks ``HANDLERS`` in priority order and
returns the first one whose ``detect()`` returns True (or None).

Order matters: ``ome_tiff`` is checked first because it's identified by a
specific file path; ``stack_tiff`` matches a ``<region>_<fov>_stack.tiff``
filename pattern that ``single_tiff`` would otherwise miss; ``single_tiff``
catches the per-image-per-z layout.
"""

from .base import FormatHandler
from .ome_tiff import OmeTiffHandler
from .single_tiff import SingleTiffHandler
from .stack_tiff import StackTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    StackTiffHandler(),
    SingleTiffHandler(),
]


def detect(folder: str) -> FormatHandler | None:
    """First-match-wins format detection. Returns None if no handler claims
    the folder."""
    for h in HANDLERS:
        if h.detect(folder):
            return h
    return None
