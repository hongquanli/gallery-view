"""Handler registry. ``detect()`` walks ``HANDLERS`` in priority order and
returns the first one whose ``detect()`` returns True (or None).

Order matters: ``ome_tiff`` is checked first because it's identified by a
specific file path; ``multi_channel_tiff`` matches per-image-TIFF folders by
filename pattern.
"""

from .base import FormatHandler
from .multi_channel_tiff import MultiChannelTiffHandler
from .ome_tiff import OmeTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    MultiChannelTiffHandler(),
]


def detect(folder: str) -> FormatHandler | None:
    """First-match-wins format detection. Returns None if no handler claims
    the folder."""
    for h in HANDLERS:
        if h.detect(folder):
            return h
    return None
