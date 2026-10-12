"""Checks for worldcover.py and the overlay key in the image payload.

The overlay rasters are optional per image, so the contract under test is:
missing/corrupt file -> no worldcover_png key (checkbox disables), present
file -> an RGBA PNG colourised with the ESA legend where nodata (0) and
unknown values stay transparent. The payload is recomputed per get_image()
call (no caching), so a freshly dropped overlay appears on the next call.
"""

import base64
import io
import os

import numpy as np
import pytest
import rasterio
from PIL import Image

from worldcover import (WORLDCOVER_ALPHA, WORLDCOVER_COLORS,
                        read_worldcover, worldcover_overlay_b64,
                        worldcover_path)

REAL_TRANSFORM = rasterio.Affine(10.0, 0, 500000, 0, -10, 5000000)


def _write_overlay(path, data):
    """A tiny uint8 GeoTIFF shaped like the ones in worldcover/."""
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
        "count": 1, "dtype": "uint8", "crs": "EPSG:32633",
        "transform": REAL_TRANSFORM, "nodata": 0, "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _decode_png(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))


def test_missing_overlay_reads_as_none(tmp_path):
    assert read_worldcover(str(tmp_path / "nope.tif")) is None
    assert worldcover_overlay_b64(str(tmp_path / "nope.tif")) is None


def test_overlay_png_uses_the_legend_and_keeps_nodata_transparent(tmp_path):
    # One pixel per legend class plus nodata and an unknown value.
    values = [0, 10, 40, 80, 100, 77]  # 77: not in the legend, must be transparent
    path = str(tmp_path / "t.tif")
    _write_overlay(path, np.array([values], dtype=np.uint8))

    img = _decode_png(worldcover_overlay_b64(path))
    assert img.shape == (1, len(values), 4)
    for x, value in enumerate(values):
        r, g, b, a = img[0, x]
        if value in WORLDCOVER_COLORS:
            assert (r, g, b) == WORLDCOVER_COLORS[value], value
            assert a == WORLDCOVER_ALPHA, value
        else:
            assert (r, g, b, a) == (0, 0, 0, 0), value


def test_a_two_band_file_is_not_an_overlay(tmp_path):
    path = str(tmp_path / "t.tif")
    profile = {
        "driver": "GTiff", "height": 4, "width": 4, "count": 2, "dtype": "uint8",
        "crs": "EPSG:32633", "transform": REAL_TRANSFORM,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((2, 4, 4), dtype=np.uint8))
    assert read_worldcover(path) is None


def test_image_payload_carries_the_overlay_only_when_it_exists(scratch_project):
    from api import Api

    tmp, name = scratch_project
    api = Api(project_root=tmp)
    payload = api.get_image()
    assert payload["name"] == name
    assert "worldcover_png" not in payload

    # Drop a full-size overlay in: the NEXT get_image carries it, without a
    # reload (the payload is recomputed per call).
    os.makedirs(os.path.join(tmp, "worldcover"))
    _write_overlay(worldcover_path(tmp, name),
                   np.full((payload["height"], payload["width"]), 80, np.uint8))
    payload = api.get_image()
    assert "worldcover_png" in payload
    img = _decode_png(payload["worldcover_png"])
    assert img.shape == (payload["height"], payload["width"], 4)
    # Every pixel is class 80 (water): legend blue at the overlay alpha.
    assert tuple(img[0, 0]) == (*WORLDCOVER_COLORS[80], WORLDCOVER_ALPHA)


def test_real_project_overlays_cover_every_input_image():
    """The dataset invariant: every image in input_images/ has a matching
    overlay in worldcover/ with identical grid and legend-valid values.

    Skips when the project dataset is absent (fresh clone) -- input_images/
    and worldcover/ are both gitignored data folders.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(root, "input_images")
    wc_dir = os.path.join(root, "worldcover")
    if not os.path.isdir(images_dir) or not os.path.isdir(wc_dir):
        pytest.skip("dataset folders not present")
    names = sorted(n for n in os.listdir(images_dir)
                   if n.lower().endswith((".tif", ".tiff")))
    if not names:
        pytest.skip("no images in the project")
    missing = [n for n in names
               if not os.path.exists(os.path.join(wc_dir, n))]
    assert not missing, f"{len(missing)} images lack an overlay: {missing[:5]}"
    # Grid match + legend-valid values on a sample (checking all 167 pairs
    # costs seconds of I/O per test run; every 20th file keeps it honest
    # and cheap).
    for name in names[::20]:
        with rasterio.open(os.path.join(images_dir, name)) as img, \
                rasterio.open(os.path.join(wc_dir, name)) as wc:
            assert img.crs == wc.crs, name
            assert (img.width, img.height) == (wc.width, wc.height), name
            assert img.transform == wc.transform, name
            data = wc.read(1)
        valid = {0, *WORLDCOVER_COLORS}
        assert set(np.unique(data)) <= valid, (name, np.unique(data))
