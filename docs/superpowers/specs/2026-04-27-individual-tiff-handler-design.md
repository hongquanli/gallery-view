# Adopt squid's per-image TIFF layout (and rename `multi_channel_tiff` → `individual_tiff`)

Status: approved (brainstorm)
Date: 2026-04-27

## Summary

Two changes shipped together:

1. **Drop `SingleChannelTiffHandler`.** The AION single-channel-per-folder format is no longer in scope; the handler, its tests, its conftest fixture, and the README mention are removed. Folders that previously matched it now either go unmatched or fall through to the renamed `individual_tiff` handler if their filenames happen to fit the legacy `current_<fov>_<z>_<channel>.tiff` shape.

2. **Rename `MultiChannelTiffHandler` → `IndividualTiffHandler` and teach it the squid layout.** The handler now recognises `<acq>/<t>/<region>_<fov>_<z>_<channel>.tiff` (with an optional `<well>_` prefix) in addition to the legacy `<acq>/0/current_<fov>_<z>_<channel>.tiff`, gains explicit timepoint awareness, and reports composite `<region>_<fov>` FOV identifiers. The gallery row UI gains a Time combo (hidden when a single timepoint is present).

OME-Zarr is **out of scope** for this design — squid's three zarr modes (HCS, per-FOV, 6D) warrant their own brainstorm with a concrete dataset in hand.

## Why

- Real squid output uses the new filename layout (`{file_id}_{channel}.tiff` where `file_id = "<region>_<fov>_<z>"`, optionally prefixed with the well). Gallery-view currently only reads the older `current_…` form, so it can't open recent acquisitions without renaming on disk.
- `multi_channel_tiff` was a misleading name — there is one TIFF per `(z, channel)`; the "multi" referred to multiple channel names appearing in one folder. `individual_tiff` describes the layout accurately and avoids confusion with OME-TIFF (which actually packs channels into a single multi-page file).
- `SingleChannelTiffHandler` existed to merge AION sibling folders into one logical acquisition. The user has confirmed AION isn't using TIFF output any more (OME-TIFF only) — keeping the handler is dead weight.
- The legacy `current_…` layout has to keep working: the existing test fixtures and the user's on-disk data still use it.

## Non-goals

- OME-Zarr support (deferred).
- Multi-region as a separate axis. Region is folded into the composite FOV label.
- Multi-timepoint expansion in the gallery (i.e. one row per timepoint). The Time combo is the only multi-T affordance for now.
- Migration tooling for old gallery caches. Cache keys change shape; old entries are evicted via a `CACHE_VERSION` bump and recomputed on first run.

## Design

### Data model — `Acquisition` (`src/gallery_view/types.py`)

Two new fields:

```python
timepoints: list[str] = field(default_factory=lambda: ["0"])
selected_timepoint: str = "0"
```

OME-TIFF and legacy folders leave both at the defaults — no per-handler change needed for them.

### Handler protocol — `FormatHandler` (`src/gallery_view/sources/base.py`)

`iter_z_slices`, `load_full_stack`, `iter_full_channel_stacks`, and `cache_key` gain a `timepoint: str = "0"` keyword argument. Existing handlers (`OmeTiffHandler`) accept the parameter and ignore it; only `IndividualTiffHandler` consumes it to pick the `<t>/` subdirectory.

### Handler implementation — `IndividualTiffHandler`

**Detection.** A folder is `individual_tiff` if either:

- (legacy) `<folder>/0/current_0_0_*.tiff` exists with at least one channel name, or
- (squid) any numeric `<folder>/<t>/` subdirectory contains a file matching the squid filename regex.

**Filename parser.** Two regexes, tried in order during detection so a folder commits to one layout:

```
squid:  ^(?:(?P<well>[A-Z]+\d+)_)?(?P<region>\d+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)\.tiff?$
legacy: ^current_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)\.tiff?$
```

Legacy matches synthesise `region = "0"` so the composite FOV id is `"0_<fov>"`, identical in shape to squid's. The detected layout is recorded in `acq.extra["layout"] = "squid" | "legacy"` and used by the iteration helpers.

**`build()`.**

1. List numeric `<t>/` subdirectories under `<acq>` (`["0", "1", …]`); legacy folders → `["0"]`.
2. For the first timepoint encountered, glob all matching files, parse, collect distinct `(region, fov)` pairs → `acq.fovs = ["{region}_{fov}", …]` sorted lexicographically.
3. Channels: keep the existing `acquisition_channels.yaml` logic; fall back to the distinct channel-name set parsed from filenames.
4. `selected_timepoint = timepoints[0]`, `selected_fov = fovs[0]`.

**`_tiffs_for(acq, fov, channel, timepoint)`.** Single iteration helper used by `iter_z_slices` / `load_full_stack` / `iter_full_channel_stacks`. Builds the appropriate glob:

```
squid:   <acq>/<t>/<region>_<fov>_*_<channel_pattern>.tiff
legacy:  <acq>/<t>/current_<fov>_*_<channel_pattern>.tiff
```

(The legacy form's `<t>` is always `"0"`.) Sort results by parsed `<z>`.

### Cache (`src/gallery_view/cache.py`)

`cache_key()` returns `(source_path, "fov{fov}/t{t}/wl_{wavelength}")` — one new path segment. Bump `CACHE_VERSION` so any pre-existing cache entries get evicted on first read and recomputed.

### Gallery UI (`src/gallery_view/ui/gallery_window.py`)

- `RowKey` becomes `(acq_id, timepoint, fov)`.
- `mip_data` keys become `(acq_id, timepoint, fov, ch_idx, axis)`.
- New per-row Time combo box, populated from `acq.timepoints`. Hidden when `len(acq.timepoints) == 1`.
- `_enqueue_jobs_for_acq(acq_id, acq, timepoint, fov)` — both selectors are passed through.
- "Expand all FOVs as separate rows" stays FOV-only; timepoints are not expanded.

### MIP loader (`src/gallery_view/loader.py`)

`Job` gains a `timepoint: str = "0"` field. The worker passes `timepoint` to `handler.iter_z_slices`.

### Sources registry (`src/gallery_view/sources/__init__.py`)

`HANDLERS = [OmeTiffHandler(), IndividualTiffHandler()]`. `SingleChannelTiffHandler` is removed.

### Backward compatibility

- The legacy `current_<fov>_<z>_<channel>.tiff` layout under `<acq>/0/` keeps working — same detection, same iteration, just routed through the renamed handler.
- The MIP cache is invalidated by the `CACHE_VERSION` bump; on first run, MIPs recompute. There is no migration path for the AION single-channel cache entries — those acquisitions don't load anymore.

## Files touched

```
src/gallery_view/types.py                          # +timepoints, +selected_timepoint
src/gallery_view/sources/base.py                   # +timepoint kwarg in protocol
src/gallery_view/sources/__init__.py               # drop single, rename multi -> individual
src/gallery_view/sources/multi_channel_tiff.py     # git mv -> individual_tiff.py + rewrite
src/gallery_view/sources/single_channel_tiff.py    # delete
src/gallery_view/sources/ome_tiff.py               # accept+ignore timepoint kwarg
src/gallery_view/cache.py                          # cache key shape, CACHE_VERSION bump
src/gallery_view/loader.py                         # Job.timepoint, plumb through
src/gallery_view/ui/gallery_window.py              # Time combo, RowKey/mip_data shape
README.md                                          # format table updated
tests/conftest.py                                  # +make_squid_individual_tiff_acq, rename multi
tests/sources/test_single_channel_tiff.py          # delete
tests/sources/test_multi_channel_tiff.py           # git mv -> test_individual_tiff.py + add cases
tests/sources/test_registry.py                     # handler list
tests/sources/test_ome_tiff.py                     # drop single-channel cross-refs if any
tests/test_scan.py                                 # drop AION case, add squid case
tests/test_handler_cache_integration.py            # cache-key shape, no single-channel
tests/test_cache.py                                # CACHE_VERSION assertion
```

## Tests

**Delete.** `tests/sources/test_single_channel_tiff.py`; `make_single_channel_tiff_acq` from conftest; any single-channel references in other test files.

**Rename via `git mv`.** `tests/sources/test_multi_channel_tiff.py` → `test_individual_tiff.py`. Conftest fixture `make_multi_channel_tiff_acq` → `make_individual_tiff_acq` (still writes the legacy layout — existing tests assert backward compatibility through it).

**New conftest fixture.** `make_squid_individual_tiff_acq(nt=1, regions=1, fovs_per_region=1, wavelengths=…, nz=…, with_well_prefix=False)`.

**New cases in `test_individual_tiff.py`.**

1. `test_detect_squid_layout` — folder with `<t>/<region>_<fov>_<z>_<channel>.tiff` is detected.
2. `test_detect_squid_layout_with_well_prefix` — same, with the optional `<well>_` prefix.
3. `test_build_discovers_multiple_timepoints` — `nt=3` → `acq.timepoints == ["0","1","2"]`, `selected_timepoint == "0"`.
4. `test_build_discovers_composite_fovs` — `regions=2, fovs_per_region=2` → `acq.fovs == ["0_0","0_1","1_0","1_1"]`.
5. `test_iter_z_slices_uses_timepoint_arg` — slices for `timepoint="1"` come from `<acq>/1/`, not `<acq>/0/`.
6. `test_load_full_stack_with_timepoint` — same, full-stack path.
7. `test_legacy_layout_still_detects_and_loads` — keeps existing assertions under the renamed test.
8. `test_filename_regex_rejects_non_squid` — `garbage_…tiff` is not matched.

**Updated cases.**

- `test_scan.py` — drop AION single-channel case; add a squid-layout scan that exercises multi-timepoint discovery.
- `test_handler_cache_integration.py` — assert cache key includes `t<t>` and per-timepoint entries don't collide.
- `test_registry.py` — `HANDLERS == [OmeTiffHandler, IndividualTiffHandler]`.
- `test_cache.py` — `CACHE_VERSION` is the bumped value; old-version cache entries are not loaded.

**Out of scope.** GUI tests for the Time combo. The combo is wired identically to the existing FOV combo; manual verification in the running app covers it.

## Open questions

None. (OME-Zarr is intentionally deferred.)
