# gallery-view

Standalone gallery viewer for z-stack microscopy acquisitions written by [squid](https://github.com/cephla-lab/squid).

## Install

```bash
pip install -e .[dev]
```

## Run

```bash
python -m gallery_view
# or with one or more sources preloaded:
python -m gallery_view --source /path/to/acquisitions
```

Drop folders (acquisition folders or parent folders containing many) onto the
window. The first time an acquisition is seen, full-resolution Z/Y/X MIPs are
computed and cached under `~/Library/Caches/gallery-view/` (macOS) or
`~/.cache/gallery-view/` (Linux). Subsequent loads are instant.

## Tests

```bash
pytest -q
```
