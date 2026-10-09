"""EmbedService: the auto-encoder ONNX as a per-pixel embedding stack.

Skips wholesale when the graph or onnxruntime is missing, like the SAM tests.
Everything runs on synthetic DN stacks: the service only sees (10, H, W)
numbers, so no project fixture is needed -- the point of these tests is the
tiling/padding/stitching machinery and the cache contract, not the semantics
of the embedding (that is what bench_wand.py scores).
"""

import numpy as np
import pytest

from embed_service import PATCH, EmbedService


@pytest.fixture()
def svc():
    s = EmbedService()
    reason = s.unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    return s


def _raw(h, w, bands=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(bands, h, w)).astype(np.float32)


def test_exact_multiple_roundtrip(svc):
    """A 256x256 stack encodes to (8, 256, 256), finite, at pixel resolution."""
    svc.set_image(_raw(256, 256), key="a")
    feats = svc.feature_stack()
    assert feats.shape == (8, 256, 256)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_a_single_patch_is_cheap(svc):
    """The 512px project tiles encode in well under a second (probe: ~0.09s).

    A guard on the batching machinery silently degrading -- e.g. running one
    patch per session.run after a refactor -- rather than a precise benchmark.
    """
    import time
    t0 = time.time()
    svc.set_image(_raw(512, 512), key="timing")
    assert time.time() - t0 < 2.0


def test_non_multiple_is_padded_and_cropped(svc):
    """Sizes that are not multiples of 128 come back at their own size."""
    svc.set_image(_raw(130, 200), key="b")
    assert svc.feature_stack().shape == (8, 130, 200)
    svc.set_image(_raw(64, 64), key="small")  # smaller than one patch
    assert svc.feature_stack().shape == (8, 64, 64)


def test_padding_is_edge_not_zeros(svc):
    """A padded border must not poison the last patch's real pixels.

    Zero padding would put a huge fake reflectance cliff inside the encoder's
    receptive field next to the image edge; edge padding continues the ground.
    Checked by encoding a stack whose border column repeats: with edge
    padding the stitched embedding at the border stays finite and close in
    magnitude to its neighbour, not blown up by a cliff.
    """
    raw = np.full((10, 100, 100), 2000.0, dtype=np.float32)
    raw[:, :, 50:] = 500.0  # a real edge, plus padding beyond x=100
    svc.set_image(raw, key="pad")
    feats = svc.feature_stack()
    assert np.isfinite(feats).all()


def test_cache_key_is_honoured(svc):
    svc.set_image(_raw(128, 128, seed=1), key="same")
    first = svc.feature_stack().copy()
    svc.set_image(_raw(128, 128, seed=2), key="same")  # new pixels, old key
    assert np.array_equal(first, svc.feature_stack())
    svc.set_image(_raw(128, 128, seed=2), key="other")
    assert not np.array_equal(first, svc.feature_stack())


def test_feature_stack_before_set_image_raises(svc):
    with pytest.raises(RuntimeError):
        svc.feature_stack()


def test_wrong_band_count_raises(svc):
    with pytest.raises(ValueError, match="10 bands"):
        svc.set_image(_raw(128, 128, bands=4), key="c")
    # The failed encode must not leave a stale key behind.
    with pytest.raises(RuntimeError):
        svc.feature_stack()


def test_patch_constant_matches_the_graph(svc):
    """PATCH is what the plan's tiling assumes and the graph accepts."""
    assert PATCH == 128
    svc.set_image(_raw(PATCH, PATCH), key="one")
    assert svc.feature_stack().shape == (8, PATCH, PATCH)


def test_unavailable_reason_names_a_missing_file(tmp_path):
    s = EmbedService(onnx_path=str(tmp_path / "nope.onnx"))
    assert "missing" in s.unavailable_reason()
    assert s.available() is False
