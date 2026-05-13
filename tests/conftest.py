"""Synthetic squid-format acquisition fixtures.

Each generator writes a real on-disk acquisition into ``tmp_path`` so handler
tests exercise the actual file-layout detection and reading code paths.
The synthetic data is a deterministic ``z*100 + y*10 + x`` gradient so MIP
results are predictable.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml


def _gradient_3d(nz: int, ny: int, nx: int, channel_offset: int = 0) -> np.ndarray:
    z, y, x = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    return (z * 100 + y * 10 + x + channel_offset * 1000).astype(np.uint16)


def _write_params(folder: Path, params: dict) -> None:
    (folder / "acquisition parameters.json").write_text(json.dumps(params))


def _write_channels_yaml(folder: Path, channels: list[dict]) -> None:
    (folder / "acquisition_channels.yaml").write_text(
        yaml.safe_dump({"channels": channels})
    )


@pytest.fixture
def make_ome_tiff_acq(tmp_path):
    """Build an OME-TIFF acquisition with axes ZCYX."""

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
        folder_name="25x_A1_2026-04-26_12-00-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
        mag=25,
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        _write_params(folder, {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
        })
        _write_channels_yaml(folder, [
            {
                "name": f"Fluorescence_{wl}_nm_Ex",
                "enabled": True,
                "camera_settings": {"exposure_time_ms": 100.0 * (i + 1)},
                "illumination_settings": {"intensity": 25.0 * (i + 1)},
            }
            for i, wl in enumerate(wavelengths)
        ])
        ome_dir = folder / "ome_tiff"
        ome_dir.mkdir()
        nc = len(wavelengths)
        # Build a (nz, nc, ny, nx) ZCYX volume from per-channel 3-D gradients.
        zcyx = np.stack([_gradient_3d(nz, ny, nx, c) for c in range(nc)], axis=1)
        assert zcyx.shape == (nz, nc, ny, nx), zcyx.shape
        tifffile.imwrite(
            ome_dir / "current_0.ome.tiff",
            zcyx,
            metadata={"axes": "ZCYX"},
        )
        return folder

    return _build


@pytest.fixture
def make_single_tiff_acq(tmp_path):
    """Build a single-TIFF acquisition: one image per (z, channel),
    laid out as ``./0/current_0_<z>_<chname>.tiff``."""

    def _build(
        wavelengths=("488", "561"),
        nz=4,
        ny=8,
        nx=10,
        folder_name="25x_B2_2026-04-26_12-30-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
        mag=25,
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        _write_params(folder, {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
        })
        _write_channels_yaml(folder, [
            {
                "name": f"Fluorescence_{wl}_nm_Ex",
                "enabled": True,
                "camera_settings": {"exposure_time_ms": 100.0 * (i + 1)},
                "illumination_settings": {"intensity": 25.0 * (i + 1)},
            }
            for i, wl in enumerate(wavelengths)
        ])
        fov_dir = folder / "0"
        fov_dir.mkdir()
        for c, wl in enumerate(wavelengths):
            stack = _gradient_3d(nz, ny, nx, c)
            for z in range(nz):
                tifffile.imwrite(
                    fov_dir / f"current_0_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                    stack[z],
                )
        return folder

    return _build


@pytest.fixture
def make_stack_tiff_acq(tmp_path):
    """Build a stack-per-FOV acquisition: ``<acq>/<t>/<region>_<fov>_stack.tiff``,
    each TIFF holding all (z, channel) planes z-major with a JSON
    ImageDescription tag per page.
    """

    def _build(
        wavelengths=("730", "488", "405"),
        nz=3,
        ny=4,
        nx=5,
        nt=1,
        region="current",
        n_fovs=1,
        folder_name="_2026-05-08_18-50-05.640535",
        sensor_pixel_size_um=6.5,
        dz_um=1.0,
        mag=60,
        write_configurations_xml=True,
        per_page_meta=True,
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        params = {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
            "objective": {"magnification": float(mag)},
        }
        if not per_page_meta:
            # Implicit layout needs Nz in params so the handler can derive Nc.
            params["Nz"] = nz
        _write_params(folder, params)
        if write_configurations_xml:
            # Squid stores channels in *reverse* XML doc order (see
            # ``_selected_fluorescence_modes`` for why). To make
            # ``wavelengths`` represent the on-disk page order — i.e., the
            # order tests will see in ``acq.channels`` — we write the XML
            # modes in the reverse of ``wavelengths``.
            xml_order_wavelengths = list(reversed(wavelengths))
            modes = "".join(
                f'<mode ID="{i}" Name="Fluorescence {wl} nm Ex" '
                f'ExposureTime="{50.0 + i}" IlluminationIntensity="{25.0 + i}" '
                f'Selected="true">0</mode>'
                for i, wl in enumerate(xml_order_wavelengths)
            )
            (folder / "configurations.xml").write_text(f"<modes>{modes}</modes>")
        for t in range(nt):
            t_dir = folder / str(t)
            t_dir.mkdir()
            for fov_idx in range(n_fovs):
                # tifffile.imwrite only accepts one description; per-page
                # ImageDescription tags need TiffWriter + .write per page.
                with tifffile.TiffWriter(
                    t_dir / f"{region}_{fov_idx}_stack.tiff"
                ) as tw:
                    for z in range(nz):
                        for c, wl in enumerate(wavelengths):
                            page = _gradient_3d(nz, ny, nx, c)[z]
                            if per_page_meta:
                                desc = json.dumps({
                                    "z_level": z,
                                    "channel": f"Fluorescence {wl} nm Ex",
                                    "channel_index": c,
                                    "region_id": region,
                                    "fov": fov_idx,
                                    "shape": [ny, nx],
                                })
                            else:
                                desc = json.dumps({"shape": [ny, nx]})
                            tw.write(page, description=desc, contiguous=False)
        return folder

    return _build


@pytest.fixture
def make_squid_single_tiff_acq(tmp_path):
    """Build an acquisition in squid's per-image TIFF layout.

    ``<acq>/<t>/[<well>_]<region>_<fov>_<z>_<channel>.tiff``

    Optionally writes ``coordinates.csv`` with stage positions laid out as a
    deterministic grid — used by region-view tests.
    """
    import csv

    def _build(
        wavelengths=("488", "561"),
        nz=3,
        ny=8,
        nx=10,
        nt=1,
        regions=1,
        fovs_per_region=1,
        with_well_prefix=False,
        well="A1",
        folder_name="25x_A1_2026-04-26_12-00-00.000000",
        sensor_pixel_size_um=6.5,
        dz_um=2.0,
        mag=25,
        write_coords=False,
        coord_grid_um=(800.0, 600.0),  # (dx, dy) between FOV centers
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        _write_params(folder, {
            "sensor_pixel_size_um": sensor_pixel_size_um,
            "dz(um)": dz_um,
        })
        _write_channels_yaml(folder, [
            {
                "name": f"Fluorescence_{wl}_nm_Ex",
                "enabled": True,
                "camera_settings": {"exposure_time_ms": 100.0 * (i + 1)},
                "illumination_settings": {"intensity": 25.0 * (i + 1)},
            }
            for i, wl in enumerate(wavelengths)
        ])
        prefix = f"{well}_" if with_well_prefix else ""
        dx_um, dy_um = coord_grid_um
        for t in range(nt):
            t_dir = folder / str(t)
            t_dir.mkdir()
            for r in range(regions):
                for f in range(fovs_per_region):
                    for z in range(nz):
                        for c, wl in enumerate(wavelengths):
                            ch_stack = _gradient_3d(nz, ny, nx, c)
                            tifffile.imwrite(
                                t_dir / f"{prefix}{r}_{f}_{z}_Fluorescence_{wl}_nm_Ex.tiff",
                                ch_stack[z],
                            )
            if write_coords:
                coords_path = t_dir / "coordinates.csv"
                with coords_path.open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["region", "fov", "z_level", "x (mm)", "y (mm)", "z (um)", "time"])
                    for r in range(regions):
                        for f in range(fovs_per_region):
                            x_mm = (f * dx_um) / 1000.0
                            y_mm = (r * dy_um) / 1000.0
                            for z in range(nz):
                                z_um = 3000.0 + z * dz_um
                                w.writerow([
                                    str(r), str(f), str(z),
                                    f"{x_mm:.6f}", f"{y_mm:.6f}",
                                    f"{z_um:.6f}", f"t{t}_z{z}",
                                ])
        return folder

    return _build
