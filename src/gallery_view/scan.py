"""Folder ingestion: walk dropped paths and dispatch to handlers."""

import os

from . import sources
from .sources import _squid_common as common
from .types import Acquisition

MAX_DEPTH = 3


def ingest(
    path: str,
    *,
    _seen: set[str] | None = None,
    _depth: int = 0,
) -> list[Acquisition]:
    """Walk ``path`` (depth-bounded) and return every recognized acquisition.

    - Depth 0 is the dropped folder itself; the walker descends until it finds
      an acquisition or hits ``MAX_DEPTH``.
    - Hidden folders (``.*``) and symlinks are skipped.
    - Duplicate ingestion of the same realpath is suppressed via ``_seen``.
    """
    if _seen is None:
        _seen = set()
    if not os.path.isdir(path) or os.path.islink(path) or _is_hidden(path):
        return []
    real = os.path.realpath(path)
    if real in _seen:
        return []
    _seen.add(real)

    handler = sources.detect(path)
    if handler is not None:
        params = common.parse_acquisition_params(path) or {}
        acq = handler.build(path, params)
        return [acq] if acq is not None else []

    if _depth >= MAX_DEPTH:
        return []

    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return []

    out: list[Acquisition] = []
    for entry in entries:
        sub = os.path.join(path, entry)
        if _is_hidden(sub) or not os.path.isdir(sub) or os.path.islink(sub):
            continue
        sub_handler = sources.detect(sub)
        if sub_handler is None:
            out.extend(ingest(sub, _seen=_seen, _depth=_depth + 1))
            continue
        params = common.parse_acquisition_params(sub) or {}
        acq = sub_handler.build(sub, params)
        if acq is not None:
            out.append(acq)
    return out


def _is_hidden(path: str) -> bool:
    return os.path.basename(path).startswith(".")
