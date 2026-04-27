"""Handler registry. ``detect()`` walks ``HANDLERS`` in priority order and
returns the first one whose ``detect()`` returns True (or None).

Order matters: ``ome_tiff`` is checked first because it's identified by a
specific file path; ``single_tiff`` matches per-image-TIFF folders by
filename pattern.
"""

from .base import FormatHandler
from .ome_tiff import OmeTiffHandler
from .single_tiff import SingleTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    SingleTiffHandler(),
]


def detect(folder: str) -> FormatHandler | None:
    """First-match-wins format detection. Returns None if no handler claims
    the folder."""
    for h in HANDLERS:
        if h.detect(folder):
            return h
    return None
