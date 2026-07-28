"""Round-trip checks for label-mask I/O. Run: python backend/test_masks.py

The app works in uint8 label space (0..254 classes, 255 unlabeled) while this
project's masks live on disk as int32 GeoTIFFs with nodata=-9999. The old
reader did `src.read(1).astype(np.uint8)`, which wraps -9999 to 49 -- a class
id that does not exist -- so every unlabeled pixel loaded as bogus ground
truth. These checks pin the conversion in both directions against the real
label files.
"""

import os
import sys
import tempfile

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import UNLABELED, read_label_mask, write_label_mask  # noqa: E402

LABEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "label_raster")
INT32_FMT = {"dtype": "int32", "nodata": -9999}
# A real (non-identity) transform: rasterio warns that GDAL may drop an
# identity geotransform, and the synthetic fixtures below have no reason to
# trip that.
SYNTHETIC_TRANSFORM = rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)


def _fail(message):
    raise AssertionError(message)


def test_nodata_reads_as_unlabeled_not_49():
    """-9999 must become UNLABELED. The uint8 cast bug turned it into 49."""
    path = os.path.join(LABEL_DIR, "00.tif")
    with rasterio.open(path) as src:
        raw = src.read(1)
    mask = read_label_mask(path)
    nodata_px = raw == -9999
    if not nodata_px.any():
        _fail("fixture 00.tif has no -9999 pixels; it cannot exercise this case")
    if not (mask[nodata_px] == UNLABELED).all():
        _fail("nodata pixels did not read as UNLABELED")
    if (mask == 49).any() and not (raw == 49).any():
        _fail("nodata leaked through as class 49 (the uint8 wrap bug)")


def test_class_values_survive_the_read():
    path = os.path.join(LABEL_DIR, "00.tif")
    with rasterio.open(path) as src:
        raw = src.read(1)
    mask = read_label_mask(path)
    labeled = raw != -9999
    if not (mask[labeled] == raw[labeled].astype(np.uint8)).all():
        _fail("class ids changed while reading")


def test_missing_file_reads_as_none():
    if read_label_mask(os.path.join(LABEL_DIR, "does-not-exist.tif")) is not None:
        _fail("a missing mask must read as None so callers fall back to unlabeled")


def test_round_trip_preserves_every_real_label_file():
    """Read -> write -> read must reproduce dtype, nodata and every pixel."""
    names = sorted(n for n in os.listdir(LABEL_DIR) if n.endswith(".tif"))
    if not names:
        _fail(f"no label files found in {LABEL_DIR}")
    for name in names:
        path = os.path.join(LABEL_DIR, name)
        with rasterio.open(path) as src:
            original = src.read(1)
            crs, transform = src.crs, src.transform
            src_dtype, src_nodata = src.dtypes[0], src.nodata

        mask = read_label_mask(path)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, name)
            write_label_mask(out, mask, crs, transform, INT32_FMT)
            with rasterio.open(out) as dst:
                written = dst.read(1)
                if dst.dtypes[0] != src_dtype:
                    _fail(f"{name}: dtype {dst.dtypes[0]} != original {src_dtype}")
                if dst.nodata != src_nodata:
                    _fail(f"{name}: nodata {dst.nodata} != original {src_nodata}")
                if dst.crs != crs or dst.transform != transform:
                    _fail(f"{name}: georeferencing changed")
            if not np.array_equal(written, original):
                diff = int((written != original).sum())
                _fail(f"{name}: {diff} pixels changed through the round trip")


def test_uint8_format_still_writes_255_nodata():
    """Existing binary projects must keep producing uint8/255 masks."""
    mask = np.array([[0, 1, UNLABELED]], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.tif")
        write_label_mask(out, mask, None, SYNTHETIC_TRANSFORM, None)
        with rasterio.open(out) as src:
            if src.dtypes[0] != "uint8":
                _fail(f"default format wrote {src.dtypes[0]}, expected uint8")
            if src.nodata != UNLABELED:
                _fail(f"default format wrote nodata {src.nodata}, expected {UNLABELED}")
        if not np.array_equal(read_label_mask(out), mask):
            _fail("uint8 round trip changed the mask")


def test_out_of_range_values_read_as_unlabeled():
    """A stray value outside [0, 254] is not a class and must not become one."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.tif")
        profile = {"driver": "GTiff", "height": 1, "width": 4, "count": 1,
                   "dtype": "int32", "nodata": -9999,
                   "transform": SYNTHETIC_TRANSFORM}
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(np.array([[3, -1, 300, -9999]], dtype=np.int32), 1)
        mask = read_label_mask(out)
        expected = np.array([[3, UNLABELED, UNLABELED, UNLABELED]], dtype=np.uint8)
        if not np.array_equal(mask, expected):
            _fail(f"out-of-range handling wrong: got {mask.tolist()}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - surface the real error
            failures += 1
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
