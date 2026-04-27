"""Handler registry. ``detect()`` walks ``HANDLERS`` in priority order and
returns the first one whose ``detect()`` returns True (or None).

Order matters: ``ome_tiff`` is checked first because it's identified by a
specific file path; ``multi_channel_tiff`` and ``single_channel_tiff`` both
look at ``./0/current_0_0_*.tiff`` filename patterns and are distinguished
by the *count* of distinct channels at z=0.
"""

from .base import FormatHandler
from .multi_channel_tiff import MultiChannelTiffHandler
from .ome_tiff import OmeTiffHandler
from .single_channel_tiff import SingleChannelTiffHandler

HANDLERS: list[FormatHandler] = [
    OmeTiffHandler(),
    MultiChannelTiffHandler(),
    SingleChannelTiffHandler(),
]


def detect(folder: str) -> FormatHandler | None:
    """First-match-wins format detection. Returns None if no handler claims
    the folder."""
    for h in HANDLERS:
        if h.detect(folder):
            return h
    return None
