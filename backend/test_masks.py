"""Checks for data.py: label-mask I/O, and the display transform.

The app works in uint8 label space (0..254 classes, 255 unlabeled) while this
project's masks live on disk as int32 GeoTIFFs with nodata=-9999. The old
reader did `src.read(1).astype(np.uint8)`, which wraps -9999 to 49 -- a class
id that does not exist -- so every unlabeled pixel loaded as bogus ground
truth. These checks pin the conversion in both directions against the real
label files.

The display half covers the two helpers the viewer and the thumbnail strip now
share. They used to be a copy each, so "thumbnails match the panes" held only
by coincidence; these pin the properties both callers depend on.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio

from data import (UNLABELED, display_band_indices, read_label_mask,
                  stretch_limits, stretch_to_unit, write_label_mask)

LABEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "label_raster")
# A real (non-identity) transform: rasterio warns that GDAL may drop an
# identity geotransform, and the synthetic fixtures below have no reason to
# trip that.
SYNTHETIC_TRANSFORM = rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)


def _label_files():
    """Every real mask file in the project, sorted by name."""
    if not os.path.isdir(LABEL_DIR):
        return []
    return sorted(n for n in os.listdir(LABEL_DIR) if n.endswith(".tif"))


def _first_label_with_nodata_and_classes():
    """A real mask carrying both -9999 nodata and class pixels.

    Found by scan rather than pinned to a filename: the dataset in
    input_images/label_raster is swapped between projects, and a hard-coded
    "00.tif" broke the moment one swap renamed the tiles. The conversion these
    tests pin is a property of any real mask, not of a particular file.
    """
    for name in _label_files():
        path = os.path.join(LABEL_DIR, name)
        with rasterio.open(path) as src:
            raw = src.read(1)
            nodata = src.nodata
        if nodata is not None and (raw == nodata).any() \
                and (raw != nodata).any():
            return path
    pytest.skip("no label file with both nodata and class pixels to test against")


def test_nodata_reads_as_unlabeled_not_49():
    """-9999 must become UNLABELED. The uint8 cast bug turned it into 49."""
    path = _first_label_with_nodata_and_classes()
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata
    mask = read_label_mask(path)
    nodata_px = raw == nodata
    if not nodata_px.any():
        pytest.fail("fixture has no nodata pixels; it cannot exercise this case")
    if not (mask[nodata_px] == UNLABELED).all():
        pytest.fail("nodata pixels did not read as UNLABELED")
    if (mask == 49).any() and not (raw == 49).any():
        pytest.fail("nodata leaked through as class 49 (the uint8 wrap bug)")


def test_class_values_survive_the_read():
    path = _first_label_with_nodata_and_classes()
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata
    mask = read_label_mask(path)
    labeled = raw != nodata
    if not (mask[labeled] == raw[labeled].astype(np.uint8)).all():
        pytest.fail("class ids changed while reading")


def test_missing_file_reads_as_none():
    if read_label_mask(os.path.join(LABEL_DIR, "does-not-exist.tif")) is not None:
        pytest.fail("a missing mask must read as None so callers fall back to unlabeled")


def test_round_trip_preserves_every_real_label_file():
    """Read -> write -> read must reproduce the labeling and the format.

    Each file round-trips in its own on-disk format. The invariant that
    matters is the LABEL SEMANTICS: re-reading the written file gives the same
    class id at every labeled pixel and UNLABELED at every nodata pixel -- that
    is what an edit/save/reopen cycle must not disturb. Class-pixel values are
    asserted byte-identical, and the raw nodata sentinel is too, but only for
    files whose declared nodata is the value they actually store.

    That last caveat is a real property of this dataset, not a dodge: the masks
    mix int32/-9999, int16/-9999 and int32/255, and a subset of the int32/255
    files declare nodata 255 while storing -9999 in the pixels. read_label_mask
    keys off the value range rather than the declared sentinel, so it reads
    them correctly; write_label_mask then normalizes the sentinel to the
    declared value (correct -- it honours the format it is handed). Those files
    are reported at the end rather than silently accepted, so a future producer
    that fixes the inconsistency makes this test stricter on its own.
    """
    names = _label_files()
    if not names:
        pytest.skip(f"no label files found in {LABEL_DIR}")
    inconsistent = []
    for name in names:
        path = os.path.join(LABEL_DIR, name)
        with rasterio.open(path) as src:
            original = src.read(1)
            crs, transform = src.crs, src.transform
            src_dtype, src_nodata = src.dtypes[0], src.nodata

        # The file's own storage format, so a round trip is faithful to it.
        fmt = {"dtype": src_dtype, "nodata": src_nodata}
        mask = read_label_mask(path)

        # Is the declared nodata the sentinel actually present in the data?
        non_class = np.unique(original[(original < 0) | (original > 254)])
        self_consistent = (non_class.size == 0
                           or set(non_class.tolist()) == {src_nodata})

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, name)
            write_label_mask(out, mask, crs, transform, fmt)
            with rasterio.open(out) as dst:
                written = dst.read(1)
                if dst.dtypes[0] != src_dtype:
                    pytest.fail(f"{name}: dtype {dst.dtypes[0]} != original {src_dtype}")
                if dst.nodata != src_nodata:
                    pytest.fail(f"{name}: nodata {dst.nodata} != original {src_nodata}")
                if dst.crs != crs or dst.transform != transform:
                    pytest.fail(f"{name}: georeferencing changed")
                reread = read_label_mask(out)

        # The labeling survives exactly: same class ids, same UNLABELED set.
        if not np.array_equal(reread, mask):
            diff = int((reread != mask).sum())
            pytest.fail(f"{name}: {diff} label pixels changed through the round trip")

        # Class-pixel VALUES are byte-identical whatever the nodata convention.
        is_class = (original >= 0) & (original <= 254)
        if not np.array_equal(written[is_class], original[is_class]):
            pytest.fail(f"{name}: class pixel values changed through the round trip")

        if self_consistent:
            if not np.array_equal(written, original):
                diff = int((written != original).sum())
                pytest.fail(f"{name}: {diff} pixels changed through the round trip")
        else:
            inconsistent.append(name)

    if inconsistent:
        print(f"\n{len(inconsistent)} label files declare a nodata they do not "
              f"store (e.g. {inconsistent[:3]}); the sentinel is normalized to "
              f"the declared value on write, class pixels preserved")


def test_uint8_format_still_writes_255_nodata():
    """Existing binary projects must keep producing uint8/255 masks."""
    mask = np.array([[0, 1, UNLABELED]], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.tif")
        write_label_mask(out, mask, None, SYNTHETIC_TRANSFORM, None)
        with rasterio.open(out) as src:
            if src.dtypes[0] != "uint8":
                pytest.fail(f"default format wrote {src.dtypes[0]}, expected uint8")
            if src.nodata != UNLABELED:
                pytest.fail(f"default format wrote nodata {src.nodata}, expected {UNLABELED}")
        if not np.array_equal(read_label_mask(out), mask):
            pytest.fail("uint8 round trip changed the mask")


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
            pytest.fail(f"out-of-range handling wrong: got {mask.tolist()}")


# ---------- the display transform ----------

def test_display_bands_always_yield_three_usable_indices():
    """Every caller indexes a raster with the result, so a short or
    out-of-range list has to come back usable rather than three frames later
    as a shape error."""
    for count in range(1, 13):
        for raw in (None, [], [0], [0, 1], [0, 1, 2], [2, 1, 0], [9, 5, 0],
                    [99, -3, 1], [0, 1, 2, 3], (1, 1, 1)):
            idx = display_band_indices(raw, count)
            if len(idx) != 3:
                pytest.fail(f"count={count} raw={raw}: got {len(idx)} indices, {idx}")
            if not all(isinstance(i, int) and 0 <= i < count for i in idx):
                pytest.fail(f"count={count} raw={raw}: out of range {idx}")


def test_the_stretch_is_bounded_and_blanks_nodata():
    stack = np.stack([
        np.linspace(-50, 5000, 64, dtype=np.float32).reshape(8, 8),
        np.full((8, 8), 7.0, dtype=np.float32),        # constant band
        np.zeros((8, 8), dtype=np.float32),
    ])
    valid = np.ones((8, 8), dtype=bool)
    valid[0, :] = False  # a nodata row
    out = stretch_to_unit(stack, valid)
    if not np.isfinite(out).all():
        pytest.fail("a constant band produced non-finite output")
    if out.min() < 0.0 or out.max() > 1.0:
        pytest.fail(f"output left [0, 1]: {out.min()}..{out.max()}")
    if out[:, 0, :].any():
        pytest.fail("nodata pixels are not blank")


def test_the_stretch_ignores_nodata_when_choosing_its_limits():
    """The reason the limits take a mask at all: a -9999 fill would drag the
    low percentile to the floor and flatten the real ground into the top of
    the range."""
    band = np.full((1, 16, 16), 1000.0, dtype=np.float32)
    band[0, :4, :] = -9999.0
    valid = band[0] != -9999.0
    lo_masked, hi_masked = stretch_limits(band, valid)
    lo_all, _ = stretch_limits(band, np.ones_like(valid))
    if lo_masked[0] <= lo_all[0]:
        pytest.fail(f"nodata did not drag the unmasked limit down: "
                    f"masked {lo_masked[0]}, unmasked {lo_all[0]}")
    if not np.isclose(lo_masked[0], 1000.0):
        pytest.fail(f"masked limits should see only the real ground, got {lo_masked[0]}")


def test_a_degenerate_band_does_not_divide_by_zero():
    flat = np.full((1, 4, 4), 3.0, dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    lo, hi = stretch_limits(flat, valid)
    if hi[0] <= lo[0]:
        pytest.fail("the hi > lo guard did not fire on a constant band")
    if not np.isfinite(stretch_to_unit(flat, valid)).all():
        pytest.fail("a constant band still produced non-finite output")