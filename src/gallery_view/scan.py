"""Folder ingestion: walk dropped paths, dispatch to handlers, merge
single-channel siblings into logical multi-channel acquisitions."""

import os
from collections import defaultdict
from typing import Iterable

from . import sources
from .sources import _squid_common as common
from .types import Acquisition, Channel

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
    - Sibling single-channel folders of matching ``(mag, well)`` are merged
      into one ``Acquisition`` with multiple channels.
    """
    if _seen is None:
        _seen = set()
    real = os.path.realpath(path)
    if real in _seen:
        return []
    _seen.add(real)

    if not os.path.isdir(path) or os.path.islink(path) or _is_hidden(path):
        return []

    handler = sources.detect(path)
    if handler is not None and handler.name != "single_channel_tiff":
        params = common.parse_acquisition_params(path) or {}
        acq = handler.build(path, params)
        return [acq] if acq is not None else []

    if _depth >= MAX_DEPTH:
        return []

    out: list[Acquisition] = []
    single_channel_buckets: dict[tuple[int, str], list[str]] = defaultdict(list)

    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if _is_hidden(sub) or not os.path.isdir(sub) or os.path.islink(sub):
            continue
        sub_handler = sources.detect(sub)
        if sub_handler is None:
            out.extend(ingest(sub, _seen=_seen, _depth=_depth + 1))
            continue
        if sub_handler.name == "single_channel_tiff":
            mag_well = common.parse_mag_well_wl(os.path.basename(sub))
            if mag_well is None:
                # Treat as a normal one-channel acquisition
                params = common.parse_acquisition_params(sub) or {}
                acq = sub_handler.build(sub, params)
                if acq is not None:
                    out.append(acq)
                continue
            mag, well, _wl = mag_well
            single_channel_buckets[(mag, well)].append(sub)
        else:
            params = common.parse_acquisition_params(sub) or {}
            acq = sub_handler.build(sub, params)
            if acq is not None:
                out.append(acq)

    out.extend(_merge_single_channel_siblings(single_channel_buckets))
    return out


def _merge_single_channel_siblings(
    buckets: dict[tuple[int, str], list[str]],
) -> Iterable[Acquisition]:
    handler = next(
        h for h in sources.HANDLERS if h.name == "single_channel_tiff"
    )
    for (mag, well), folders in buckets.items():
        # One acquisition per (mag, well); pick the latest folder per
        # wavelength when multiple acquisitions exist for the same channel.
        by_wl: dict[str, str] = {}
        for f in folders:
            mag_well = common.parse_mag_well_wl(os.path.basename(f))
            if mag_well is None:
                continue
            _, _, wl = mag_well
            # "Latest" by lexicographic folder name (squid timestamps sort).
            if wl not in by_wl or os.path.basename(f) > os.path.basename(by_wl[wl]):
                by_wl[wl] = f
        if not by_wl:
            continue
        first_folder = sorted(by_wl.values())[0]
        params = common.parse_acquisition_params(first_folder) or {}
        channels = [
            Channel(name=f"Fluorescence_{wl}_nm_Ex", wavelength=wl)
            for wl in sorted(by_wl, key=int)
        ]
        channel_paths = {wl: by_wl[wl] for wl in by_wl}
        folder_name = os.path.basename(first_folder)
        merged = Acquisition(
            handler=handler,
            path=first_folder,
            folder_name=folder_name,
            display_name=f"{mag}x {well}",
            params=params,
            channels=channels,
            fovs=["0"],
            extra={"channel_paths": channel_paths, "merged_mag": mag, "merged_well": well},
        )
        yield merged


def _is_hidden(path: str) -> bool:
    return os.path.basename(path).startswith(".")
