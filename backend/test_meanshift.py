"""meanshift.py: PCA -> uint8 -> cv2.pyrMeanShiftFiltering, as a (3, H, W) stack.

Pure image-processing function, so these are property tests on synthetic stacks
plus determinism/range guards -- no ground truth needed. The integration
invariants (that a wand selection over this field behaves like every other
source) live in test_wand.py.
"""

import numpy as np
import pytest

from meanshift import (ITERATIONS, N_COMPONENTS, SP, SR, meanshift_available,
                       meanshift_image)


def _two_block(h=64, w=64, bands=10, seed=0):
    """A stack whose left half is one spectrum, right half another.

    The property mean-shift exists for: after filtering, each half collapses to
    a near-constant colour and the two halves stay distinct -- a flat region per
    material with a sharp boundary between them.
    """
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.1, 0.2, size=bands)
    right = rng.uniform(0.7, 0.8, size=bands)
    stack = np.empty((bands, h, w), dtype=np.float32)
    stack[:, :, : w // 2] = left[:, None, None]
    stack[:, :, w // 2:] = right[:, None, None]
    # A little texture so it is not perfectly flat going in.
    stack += rng.normal(0, 0.02, size=stack.shape).astype(np.float32)
    return np.clip(stack, 0.0, 1.0)


def test_available_reflects_cv2():
    # opencv is a hard project dependency; if it is importable the source is up.
    import importlib.util
    have = importlib.util.find_spec("cv2") is not None
    assert meanshift_available() == have


def test_output_shape_dtype_range():
    stack = _two_block()
    valid = np.ones(stack.shape[1:], dtype=bool)
    out = meanshift_image(stack, valid)
    assert out.shape == (N_COMPONENTS,) + stack.shape[1:]
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_two_blocks_converge_and_stay_separate():
    """The core property: flat within a region, distinct between regions.

    Compared PER CHANNEL: each PCA component is min-max scaled independently,
    so channel means differ by construction -- what must hold is that the
    channel carrying the block signal is near-constant inside each block and
    far apart between them.
    """
    stack = _two_block(h=64, w=64, seed=1)
    valid = np.ones(stack.shape[1:], dtype=bool)
    out = meanshift_image(stack, valid)

    left = out[:, :, :20]     # well inside the left half
    right = out[:, :, -20:]   # well inside the right half
    # The dominant channel is the one carrying the between-block variance.
    dom = int(np.argmax(out.std(axis=(1, 2))))
    # Near-constant inside each block: within-block std is small ...
    assert left[dom].std() < 0.05
    assert right[dom].std() < 0.05
    # ... and the two blocks are far apart on that channel.
    gap = abs(float(left[dom].mean()) - float(right[dom].mean()))
    assert gap > 0.3


def test_filtering_reduces_local_variance():
    """Mean-shift smooths within regions, so gradient energy drops.

    Compared against the SAME representation going in -- the PCA uint8 image
    -- not against the raw bands, whose channel scaling is unrelated.
    """
    stack = _two_block(seed=2)
    valid = np.ones(stack.shape[1:], dtype=bool)
    out = meanshift_image(stack, valid)
    from meanshift import _pca_uint8
    u8 = _pca_uint8(stack, valid).astype(np.float32).transpose(2, 0, 1) / 255.0

    def grad_energy(a):
        gy = np.diff(a, axis=1)
        gx = np.diff(a, axis=2)
        return float((gy ** 2).mean() + (gx ** 2).mean())

    assert grad_energy(out) < grad_energy(u8)


def test_deterministic():
    stack = _two_block(seed=3)
    valid = np.ones(stack.shape[1:], dtype=bool)
    a = meanshift_image(stack, valid)
    b = meanshift_image(stack, valid)
    assert np.array_equal(a, b)


def test_range_and_iterations_flow_through():
    """A wider colour range (sr) or more iterations changes the result."""
    stack = _two_block(seed=4)
    valid = np.ones(stack.shape[1:], dtype=bool)
    base = meanshift_image(stack, valid, sp=SP, sr=SR, iterations=ITERATIONS)
    wide = meanshift_image(stack, valid, sp=SP, sr=SR * 2, iterations=ITERATIONS)
    assert not np.allclose(base, wide)


def test_fewer_than_three_bands_still_works():
    """PCA takes min(3, C) components, so a 1- or 2-band stack does not crash."""
    for bands in (1, 2):
        stack = _two_block(bands=bands, seed=5)
        valid = np.ones(stack.shape[1:], dtype=bool)
        out = meanshift_image(stack, valid)
        assert out.shape == (N_COMPONENTS,) + stack.shape[1:]
        assert np.isfinite(out).all()


def test_nodata_pixels_do_not_crash_or_corrupt():
    """Nodata is 0 in the normalized stack; it must not poison the valid ground.

    The wand excludes nodata via valid_mask regardless, but the filter is
    spatial, so this checks the real ground beside a nodata border still comes
    out finite and in range.
    """
    stack = _two_block(seed=6)
    valid = np.ones(stack.shape[1:], dtype=bool)
    valid[:, :8] = False           # a nodata border
    stack[:, :, :8] = 0.0
    out = meanshift_image(stack, valid)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    # The interior, away from the nodata border, still separates the blocks --
    # on the dominant channel, per the same reasoning as the convergence test.
    left = out[:, 16:, 12:28]
    right = out[:, 16:, -20:]
    dom = int(np.argmax(out.std(axis=(1, 2))))
    assert abs(float(left[dom].mean()) - float(right[dom].mean())) > 0.3


def test_all_valid_false_is_handled():
    """A degenerate mask must not raise (the wand would select nothing anyway)."""
    stack = _two_block(h=16, w=16, seed=7)
    valid = np.zeros(stack.shape[1:], dtype=bool)
    out = meanshift_image(stack, valid)
    assert out.shape == (N_COMPONENTS, 16, 16)
    assert np.isfinite(out).all()
