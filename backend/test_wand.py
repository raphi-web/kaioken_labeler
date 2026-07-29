"""Invariants of the magic wand. Run: python3 backend/test_wand.py

These properties were each established by hand during the wand's development
and re-derived on every subsequent change, which is exactly the sort of thing
that belongs in a file. They are also easy to break silently: a selection that
is slightly wrong still looks like a selection.

Everything runs against a scratch copy of the project, never label_raster/ --
see test_no_torch._scratch_project for why.
"""

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wand import _WAND_CONNECTIVITY  # noqa: E402
from testlib import fail, main, with_scratch_api  # noqa: E402

# A tolerance loose enough that regions are large enough to be interesting, and
# a budget small enough that the cap actually bites at it.
TOL = 0.25
BUDGET = 3200
A, B, C = (256, 256), (120, 400), (400, 120)


def _mask(result):
    """The bit-packed mask a caller gets back, as a bool array."""
    import base64
    if not result.get("ok"):
        fail(f"wand_select failed: {result.get('error')}")
    bits = np.unpackbits(np.frombuffer(base64.b64decode(result["mask"]),
                                       dtype=np.uint8))
    n = result["width"] * result["height"]
    return bits[:n].reshape(result["height"], result["width"]).astype(bool)


def _select(api, samples, **options):
    opts = {"tolerance": TOL, "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)




# ---------- the options object ----------

def test_options_are_passed_as_one_object():
    def check(api):
        r = _select(api, [A], tolerance=0.13, max_pixels=1000, sample_size=9)
        if not r.get("ok"):
            fail(f"options object rejected: {r.get('error')}")
        if r["count"] > 1000:
            fail(f"max_pixels ignored: {r['count']} > 1000")
    with_scratch_api(check)


def test_missing_options_use_defaults():
    def check(api):
        r = api.wand_select([[256, 256]], {"tolerance": TOL})
        if not r.get("ok"):
            fail(f"a bare tolerance should suffice: {r.get('error')}")
    with_scratch_api(check)


def test_bad_input_returns_an_error_not_an_exception():
    def check(api):
        for opts, why in (({"tolerance": "x"}, "non-numeric tolerance"),
                          ({"tolerance": TOL, "max_pixels": 0}, "zero budget"),
                          ({"tolerance": TOL, "max_pixels": "x"}, "bad budget"),
                          ({"tolerance": TOL, "sample_size": "x"}, "bad sample size")):
            r = api.wand_select([[10, 10]], opts)
            if r.get("ok") or not r.get("error"):
                fail(f"{why} should return an error, got {r}")
        if api.wand_select([], {"tolerance": TOL}).get("ok"):
            fail("an empty sample list should return an error")
    with_scratch_api(check)


# ---------- the invariants ----------

def test_adding_a_sample_never_removes_a_pixel():
    """The property the tool's predictability rests on."""
    def check(api):
        for tol in (0.08, 0.13, 0.20, 0.30):
            for budget in (BUDGET, 18000, None):
                masks = [_mask(_select(api, s, tolerance=tol, max_pixels=budget))
                         for s in ([A], [A, B], [A, B, C])]
                for i in range(len(masks) - 1):
                    lost = int((masks[i] & ~masks[i + 1]).sum())
                    if lost:
                        fail(f"tol={tol} budget={budget}: adding sample "
                              f"{i + 2} removed {lost} px")
    with_scratch_api(check)


def test_a_sample_region_is_the_same_alone_or_alongside_others():
    def check(api):
        solo = _mask(_select(api, [B], max_pixels=BUDGET))
        both = _mask(_select(api, [A, B], max_pixels=BUDGET))
        missing = int((solo & ~both).sum())
        if missing:
            fail(f"{missing} px of B's own region vanished when A was added")
    with_scratch_api(check)


def test_the_budget_is_exact_and_per_sample():
    def check(api):
        for budget in (777, 3333):
            one = _select(api, [A], max_pixels=budget)
            if one["count"] != budget:
                fail(f"budget {budget}: got {one['count']} px, expected exactly it")
            three = _select(api, [A, B, C], max_pixels=budget)
            if three["count"] != 3 * budget:
                fail(f"budget {budget} x 3 samples: got {three['count']}, "
                      f"expected {3 * budget}")
    with_scratch_api(check)


def test_a_sample_is_always_inside_its_own_selection():
    """Otherwise a click can appear to do nothing at all."""
    def check(api):
        m = _mask(_select(api, [A, B], tolerance=0.0, max_pixels=1))
        for x, y in (A, B):
            if not m[y, x]:
                fail(f"sample ({x}, {y}) is not in its own selection")
    with_scratch_api(check)


def test_a_capped_selection_stays_connected():
    def check(api):
        m = _mask(_select(api, [A], max_pixels=8000))
        pieces = ndimage.label(m, structure=_WAND_CONNECTIVITY)[1]
        if pieces != 1:
            fail(f"a clipped single-sample selection broke into {pieces} pieces")
    with_scratch_api(check)


def test_nodata_is_never_selected():
    def check(api):
        invalid = ~api._image.valid_mask
        if not invalid.any():
            print("     (this image has no nodata — skipping)")
            return
        m = _mask(_select(api, [A], tolerance=0.9))
        if (m & invalid).any():
            fail(f"{int((m & invalid).sum())} nodata px were selected")
    # 14.tiff is the one image in this dataset that carries nodata (6,204 px).
    with_scratch_api(check, prefer="14.tiff")


def test_global_reaches_disconnected_material():
    """The point of the mode: these classes come in dozens of fragments."""
    def check(api):
        local = _mask(_select(api, [A], max_pixels=None))
        everywhere = _mask(_select(api, [A], max_pixels=None, **{"global": True}))
        if int(everywhere.sum()) <= int(local.sum()):
            fail("global mode selected no more than the connected piece")
        if (local & ~everywhere).any():
            fail("global mode dropped pixels the connected selection had")
        pieces = ndimage.label(everywhere, structure=_WAND_CONNECTIVITY)[1]
        if pieces < 2:
            fail("global mode returned a single piece — connectivity still bounding it")
    with_scratch_api(check)


def test_global_still_honours_the_budget():
    def check(api):
        for budget in (500, 5000):
            r = _select(api, [A], max_pixels=budget, **{"global": True})
            if r["count"] != budget:
                fail(f"global budget {budget}: got {r['count']} px")
    with_scratch_api(check)


def test_a_capped_global_selection_keeps_the_nearest_pixels():
    """With no path to walk, the cap spends on straight-line nearness."""
    def check(api):
        budget = 2000
        m = _mask(_select(api, [A], max_pixels=budget, **{"global": True}))
        full = _mask(_select(api, [A], max_pixels=None, **{"global": True}))
        ys, xs = np.nonzero(m)
        kept = np.sqrt((ys - A[1]) ** 2 + (xs - A[0]) ** 2).max()
        fy, fx = np.nonzero(full & ~m)
        if len(fy):
            dropped = np.sqrt((fy - A[1]) ** 2 + (fx - A[0]) ** 2).min()
            if dropped < kept - 1.5:  # ties on the ring are arbitrary
                fail(f"a pixel at {dropped:.0f} px was dropped while one at "
                      f"{kept:.0f} px was kept")
    with_scratch_api(check)


def test_global_is_still_strictly_additive():
    def check(api):
        masks = [_mask(_select(api, s, max_pixels=3200, **{"global": True}))
                 for s in ([A], [A, B], [A, B, C])]
        for i in range(len(masks) - 1):
            lost = int((masks[i] & ~masks[i + 1]).sum())
            if lost:
                fail(f"global: adding sample {i + 2} removed {lost} px")
    with_scratch_api(check)


def test_global_and_protect_compose():
    def check(api):
        labels = api._labels
        if labels is None or (labels != 255).sum() == 0:
            print("     (nothing labeled in this image — skipping)")
            return
        m = _mask(_select(api, [A], protect=True, **{"global": True}))
        if (m & (labels != 255)).any():
            fail("global + protect selected labeled pixels")
    with_scratch_api(check)


def test_a_negative_shrinks_the_selection():
    def check(api):
        plain = _mask(_select(api, [A], max_pixels=None))
        # Exclude something the selection actually contains, so it must bite.
        ys, xs = np.nonzero(plain)
        j = len(ys) // 2
        cut = _mask(_select(api, [A], max_pixels=None,
                            negatives=[[int(xs[j]), int(ys[j])]]))
        if int(cut.sum()) >= int(plain.sum()):
            fail("a negative click did not shrink the selection")
        if (cut & ~plain).any():
            fail("a negative click ADDED pixels")
    with_scratch_api(check)


def test_a_negative_reaches_every_fragment():
    """The reason negatives subtract a colour rather than a region."""
    def check(api):
        opts = {"max_pixels": None, "global": True}
        plain = _mask(_select(api, [A], **opts))
        ys, xs = np.nonzero(plain)
        j = len(ys) // 2
        neg = [[int(xs[j]), int(ys[j])]]
        cut = _mask(_select(api, [A], negatives=neg, **opts))
        removed = plain & ~cut
        if not removed.any():
            fail("the negative removed nothing")
        pieces = ndimage.label(removed, structure=_WAND_CONNECTIVITY)[1]
        if pieces < 2:
            fail("the negative only removed one connected piece; it should "
                  "subtract that material wherever it appears")
    with_scratch_api(check)


def test_a_sample_survives_its_own_negative_neighbourhood():
    """A positive click must stay selected however the negatives fall."""
    def check(api):
        m = _mask(_select(api, [A], max_pixels=None,
                          negatives=[[B[0], B[1]], [C[0], C[1]]]))
        if not m[A[1], A[0]]:
            fail("the positive sample was excluded from its own selection")
    with_scratch_api(check)


def test_negatives_compose_with_protect_and_global():
    def check(api):
        labels = api._labels
        r = _select(api, [A], protect=True, negatives=[[B[0], B[1]]],
                    max_pixels=5000, **{"global": True})
        if not r.get("ok"):
            fail(f"the combination was rejected: {r.get('error')}")
        m = _mask(r)
        if labels is not None and (m & (labels != 255)).any():
            fail("protect stopped applying once negatives were present")
        if r["count"] > 5000:
            fail("the budget stopped applying once negatives were present")
    with_scratch_api(check)


def test_protect_never_selects_an_already_labeled_pixel():
    def check(api):
        labels = api._labels
        if labels is None or (labels != 255).sum() == 0:
            print("     (nothing labeled in this image — skipping)")
            return
        m = _mask(_select(api, [A, B], protect=True))
        clashes = int((m & (labels != 255)).sum())
        if clashes:
            fail(f"{clashes} already-labeled px were selected with protect on")
    with_scratch_api(check)


def test_protect_reports_what_it_withheld():
    def check(api):
        off = _select(api, [A], protect=False)
        on = _select(api, [A], protect=True)
        if on["count"] + on["protected"] != off["count"]:
            fail(f"{on['count']} kept + {on['protected']} protected "
                  f"!= {off['count']} unprotected")
        if on["count"] > off["count"]:
            fail("protection selected MORE than no protection")
    with_scratch_api(check)


def test_protect_is_a_subset_not_a_different_selection():
    """Protection must only remove pixels, never move the selection."""
    def check(api):
        off = _mask(_select(api, [A, B], protect=False))
        on = _mask(_select(api, [A, B], protect=True))
        if (on & ~off).any():
            fail(f"{int((on & ~off).sum())} px appear only when protected")
    with_scratch_api(check)


def test_protection_conducts_through_labeled_pixels():
    """A labeled pixel blocks painting, not the fill's path through it.

    Masking the match set instead of the finished selection would sever a
    region at the first labeled pixel it meets, so a fill could not reach
    unlabeled ground on the far side of a labeled strip.
    """
    def check(api):
        labels = api._labels
        if labels is None or (labels != 255).sum() == 0:
            print("     (nothing labeled in this image — skipping)")
            return
        on = _mask(_select(api, [A], protect=True, max_pixels=None))
        off = _mask(_select(api, [A], protect=False, max_pixels=None))
        # Everything protection keeps must be exactly the unlabeled part of
        # the unprotected selection -- no more, and crucially no less.
        expected = off & (labels == 255)
        if not np.array_equal(on, expected):
            fail(f"{int((expected & ~on).sum())} unlabeled px were lost, "
                  f"{int((on & ~expected).sum())} extra px appeared")
    with_scratch_api(check)


def test_the_reported_count_matches_the_mask():
    def check(api):
        r = _select(api, [A, B], max_pixels=BUDGET)
        if int(_mask(r).sum()) != r["count"]:
            fail(f"count {r['count']} != {int(_mask(r).sum())} px in the mask")
    with_scratch_api(check)


def test_an_under_budget_region_is_not_reported_as_capped():
    def check(api):
        r = _select(api, [A], tolerance=0.05, max_pixels=100000)
        if r["capped"]:
            fail("a region smaller than the budget was reported as capped")
    with_scratch_api(check)


# ---------- the SAM2 embedding source ----------
#
# Wand-SAM keeps every invariant above and changes only what distance means, so
# what these check is that the substitution really is that clean: the same
# additivity, the same option handling, and a field that is a cosine distance
# at pixel resolution rather than a resized approximation of one.

# A cosine distance against the image's mean feature, not an RMS band
# difference -- see Api._sam_features and LabelerPane's SAM_TOL_*. These are
# the UI's defaults: the fine level, which is the one that produces usable
# selections on Sentinel-2, at the tolerance where a click takes a region
# rather than a speck or the whole scene.
SAM_TOL = 0.8
SAM_LEVEL = "fine"


def _field(api, point, source, level=SAM_LEVEL, size=9):
    """One sample's raw distance field, straight from the Wand.

    Goes through the algorithm rather than the bridge because these checks are
    about the field itself, not about what survives base64. set_image is what
    Api.wand_select would have called; it is cheap and idempotent.
    """
    from wand import parse_options
    api._wand.set_image(api._image, api._sam_image_key(), api._display_bands())
    options = parse_options({"tolerance": SAM_TOL, "sample_size": size,
                             "source": source, "level": level})
    return api._wand.fields([point], options)[0]


def _sam_ready(api):
    """Whether the encoder can actually run here; these tests skip if not."""
    reason = api._sam.unavailable_reason()
    if reason is not None:
        print(f"     (SAM2 unavailable: {reason} — skipping)")
        return False
    return True


def _sam_select(api, samples, **options):
    opts = {"source": "sam", "level": SAM_LEVEL, "tolerance": SAM_TOL,
            "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)


def test_sam_source_is_rejected_cleanly_when_misconfigured():
    """A bad source or level is a message, not a traceback across the bridge."""
    def check(api):
        for opts, why in (({"tolerance": SAM_TOL, "source": "nonsense"}, "bad source"),
                          ({"tolerance": SAM_TOL, "source": "sam",
                            "level": "nonsense"}, "bad level")):
            r = api.wand_select([[10, 10]], opts)
            if r.get("ok") or not r.get("error"):
                fail(f"{why} should return an error, got {r}")
    with_scratch_api(check)


def test_the_embedding_field_is_a_cosine_distance_at_pixel_resolution():
    """Bounded in [0, 2], and the size of the image.

    The field is computed on the encoder's own grid and upsampled, so its
    shape is the one thing that could silently drift from the label buffer --
    everything downstream indexes both with the same flat offsets.

    Deliberately no assertion that the click sits at distance zero. It does on
    the feature grid, by construction, but the field is read at pixel
    resolution: a bilinear upsample blends the reference cell with neighbours
    that may be far away, so the exact zero survives nowhere. Measured at
    0.09 (fine) to 0.44 (deep) at the click itself. The wand does not depend on
    it either -- wand_select seeds `within[y, x] = True` so that a click is
    never a no-op regardless of its own distance.
    """
    def check(api):
        if not _sam_ready(api):
            return
        for level in ("fine", "mid", "deep"):
            _, field = _field(api, A, "sam", level)
            if field.shape != (api._image.height, api._image.width):
                fail(f"{level}: field is {field.shape}, image is "
                      f"{(api._image.height, api._image.width)}")
            if not np.isfinite(field).all():
                fail(f"{level}: field has non-finite values")
            if field.min() < -1e-5 or field.max() > 2 + 1e-5:
                fail(f"{level}: cosine distance out of [0, 2]: "
                      f"{field.min()}..{field.max()}")
    with_scratch_api(check)


def test_the_default_tolerance_selects_a_region_and_not_the_scene():
    """The UI's default lands in the band where the level is usable at all.

    The band is narrow and does not start at zero -- at the default level a
    click takes under 0.15% of the image below 0.7 and over 7% by 1.0 -- so a
    default outside it makes the tool look broken rather than mistuned. This
    fails if a re-exported encoder moves the distances, which is the point:
    the number in LabelerPane would then need re-sighting.

    Deliberately generous bounds. This pins "the default is inside the useful
    band", not a particular selection size, which varies by what was clicked.
    """
    def check(api):
        if not _sam_ready(api):
            return
        n = api._image.height * api._image.width
        for point in (A, B, C):
            share = 100 * _sam_select(api, [point], protect=False)["count"] / n
            if not 0.05 <= share <= 25:
                fail(f"at tolerance {SAM_TOL} a click at {point} selected "
                      f"{share:.2f}% of the image — outside the band the "
                      f"default is meant to sit in")
    with_scratch_api(check)


def test_the_two_sources_measure_different_things():
    """Guards against the embedding source quietly falling back to spectral.

    Compares the fields rather than the masks: two different measures can
    happen to select the same region at some tolerance, but they cannot
    produce the same distance at every pixel.
    """
    def check(api):
        if not _sam_ready(api):
            return
        _, spectral = _field(api, A, "image")
        _, embedding = _field(api, A, "sam")
        if np.allclose(spectral, embedding):
            fail("the embedding field is identical to the spectral one")
    with_scratch_api(check)


def test_adding_a_sample_never_removes_a_pixel_on_embeddings():
    """The wand's central property, on the other source."""
    def check(api):
        if not _sam_ready(api):
            return
        one = _mask(_sam_select(api, [A], protect=False))
        two = _mask(_sam_select(api, [A, B], protect=False))
        lost = int((one & ~two).sum())
        if lost:
            fail(f"a second sample dropped {lost} px the first had selected")
    with_scratch_api(check)


def test_the_embedding_selection_is_deterministic_and_respects_the_budget():
    """Same clicks, same mask -- and the cap still bites the same way."""
    def check(api):
        if not _sam_ready(api):
            return
        first = _mask(_sam_select(api, [A], protect=False))
        again = _mask(_sam_select(api, [A], protect=False))
        if not np.array_equal(first, again):
            fail("two identical embedding selections disagree")
        capped = _sam_select(api, [A], protect=False, max_pixels=BUDGET)
        if capped["count"] > BUDGET:
            fail(f"budget ignored on embeddings: {capped['count']} > {BUDGET}")
        # Connectivity still applies: every selected pixel is reachable from
        # the click, exactly as with the spectral source.
        pieces = ndimage.label(_mask(capped), structure=_WAND_CONNECTIVITY)[1]
        if pieces > 1:
            fail(f"a connected embedding selection came back in {pieces} pieces")
    with_scratch_api(check)




if __name__ == "__main__":
    sys.exit(main(globals()))
