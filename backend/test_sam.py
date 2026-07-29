"""SAM2 must produce the same masks after dropping torch.

Run: python3 backend/test_sam.py

The preprocessing and the final mask upscale used torchvision and
torch.nn.functional.interpolate; they are numpy and scipy.ndimage.zoom now.
That is a behaviour-preserving rewrite, not a redesign, so the fixtures in
testdata/sam_golden.npz were captured from the torch implementation and this
compares against them. The failure mode being guarded is a silent half-pixel
shift: scipy's default zoom aligns outermost sample centres, torch's
align_corners=False aligns cell centres, and only `grid_mode=True` reconciles
them. Everything still looks plausible if that is wrong -- the masks are just
subtly off.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testlib import fail, main  # noqa: E402

from data import SentinelImage  # noqa: E402
from sam_service import SamService  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "testdata", "sam_golden.npz")
IMAGE = os.path.join(ROOT, "input_images", "00.tiff")

# The click sets the fixtures were captured with, as (x, y, label).
CLICKS = {
    "single_fg": [(256, 256, 1)],
    "two_fg": [(120, 120, 1), (140, 160, 1)],
    "fg_then_bg": [(300, 300, 1), (320, 320, 0)],
    "corner": [(40, 40, 1)],
    "three_mixed": [(200, 380, 1), (210, 390, 1), (260, 420, 0)],
}

# A handful of boundary pixels may legitimately flip: the two bilinear
# implementations agree to float rounding, and a logit sitting within an ulp of
# the 0 threshold can land either way. A shifted resize moves thousands.
MAX_DIFFERING_PIXELS = 16


def _service_and_rgb():
    svc = SamService()
    reason = svc.unavailable_reason()
    if reason:
        fail(f"SAM2 unavailable: {reason}")
    image = SentinelImage(IMAGE, expected_bands=10)
    return svc, image.rgb_composite([0, 1, 2])


def test_no_torch_is_imported():
    """The point of the rewrite: SAM2 must not pull torch in, even lazily."""
    svc, rgb = _service_and_rgb()
    svc.set_image(rgb, key="test")
    svc.predict_points([(256, 256, 1)])
    for module in ("torch", "torchvision"):
        if module in sys.modules:
            fail(f"{module} was imported during a SAM2 round trip")


def test_preprocessing_matches_torchvision():
    svc, rgb = _service_and_rgb()
    svc._ensure_loaded()
    pre = svc._preprocess(np.ascontiguousarray(rgb))
    golden = np.load(GOLDEN)

    if list(pre.shape) != list(golden["pre_shape"]):
        fail(f"preprocessed shape {pre.shape} != {tuple(golden['pre_shape'])}")
    stats = np.array([pre.min(), pre.max(), pre.mean(), pre.std()],
                     dtype=np.float64)
    if not np.allclose(stats, golden["pre_stats"], atol=1e-4):
        fail(f"preprocessed stats {stats} != {golden['pre_stats']}")
    if not np.allclose(pre[0, :, :4, :4], golden["pre_corner"], atol=1e-4):
        fail("preprocessed corner pixels differ — the resize is misaligned")


def test_masks_match_the_torch_implementation():
    svc, rgb = _service_and_rgb()
    svc.set_image(rgb, key="test")
    golden = np.load(GOLDEN)
    problems = []
    for name, points in CLICKS.items():
        mask, score = svc.predict_points(points)
        want_mask = golden[f"mask_{name}"]
        want_score = float(golden[f"score_{name}"])
        if mask.shape != want_mask.shape:
            problems.append(f"{name}: shape {mask.shape} != {want_mask.shape}")
            continue
        differing = int((mask != want_mask).sum())
        if differing > MAX_DIFFERING_PIXELS:
            problems.append(
                f"{name}: {differing:,} px differ (limit {MAX_DIFFERING_PIXELS}); "
                f"got {int(mask.sum()):,} px, want {int(want_mask.sum()):,}")
        # The score comes straight out of the decoder and never touches the
        # rewritten code, so any drift here means something else moved.
        if abs(score - want_score) > 1e-5:
            problems.append(f"{name}: score {score} != {want_score}")
    if problems:
        fail("; ".join(problems))




if __name__ == "__main__":
    sys.exit(main(globals()))
