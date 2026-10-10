"""Invariants of the magic wand.

These properties were each established by hand during the wand's development
and re-derived on every subsequent change, which is exactly the sort of thing
that belongs in a file. They are also easy to break silently: a selection that
is slightly wrong still looks like a selection.

Everything runs against a scratch copy of the project via the `scratch_api`
fixture in backend/conftest.py, never label_raster/.
"""

import base64

import numpy as np
import pytest
from scipy import ndimage

from wand import _WAND_CONNECTIVITY

# A tolerance loose enough that regions are large enough to be interesting, and
# a budget small enough that the cap actually bites at it.
TOL = 0.25
BUDGET = 3200
A, B, C = (256, 256), (120, 400), (400, 120)


def _mask(result):
    """The bit-packed mask a caller gets back, as a bool array."""
    if not result.get("ok"):
        pytest.fail(f"wand_select failed: {result.get('error')}")
    bits = np.unpackbits(np.frombuffer(base64.b64decode(result["mask"]),
                                       dtype=np.uint8))
    n = result["width"] * result["height"]
    return bits[:n].reshape(result["height"], result["width"]).astype(bool)


def _select(api, samples, **options):
    opts = {"tolerance": TOL, "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)


# ---------- the options object ----------

def test_options_are_passed_as_one_object(scratch_api):
    api = scratch_api
    r = _select(api, [A], tolerance=0.13, max_pixels=1000, sample_size=9)
    if not r.get("ok"):
        pytest.fail(f"options object rejected: {r.get('error')}")
    if r["count"] > 1000:
        pytest.fail(f"max_pixels ignored: {r['count']} > 1000")


def test_missing_options_use_defaults(scratch_api):
    api = scratch_api
    r = api.wand_select([[256, 256]], {"tolerance": TOL})
    if not r.get("ok"):
        pytest.fail(f"a bare tolerance should suffice: {r.get('error')}")


def test_bad_input_returns_an_error_not_an_exception(scratch_api):
    api = scratch_api
    for opts, why in (({"tolerance": "x"}, "non-numeric tolerance"),
                      ({"tolerance": TOL, "max_pixels": 0}, "zero budget"),
                      ({"tolerance": TOL, "max_pixels": "x"}, "bad budget"),
                      ({"tolerance": TOL, "sample_size": "x"}, "bad sample size")):
        r = api.wand_select([[10, 10]], opts)
        if r.get("ok") or not r.get("error"):
            pytest.fail(f"{why} should return an error, got {r}")
    if api.wand_select([], {"tolerance": TOL}).get("ok"):
        pytest.fail("an empty sample list should return an error")


# ---------- the invariants ----------

def test_adding_a_sample_never_removes_a_pixel(scratch_api):
    """The property the tool's predictability rests on."""
    api = scratch_api
    for tol in (0.08, 0.13, 0.20, 0.30):
        for budget in (BUDGET, 18000, None):
            masks = [_mask(_select(api, s, tolerance=tol, max_pixels=budget))
                     for s in ([A], [A, B], [A, B, C])]
            for i in range(len(masks) - 1):
                lost = int((masks[i] & ~masks[i + 1]).sum())
                if lost:
                    pytest.fail(f"tol={tol} budget={budget}: adding sample "
                                f"{i + 2} removed {lost} px")


def test_a_sample_region_is_the_same_alone_or_alongside_others(scratch_api):
    api = scratch_api
    solo = _mask(_select(api, [B], max_pixels=BUDGET))
    both = _mask(_select(api, [A, B], max_pixels=BUDGET))
    missing = int((solo & ~both).sum())
    if missing:
        pytest.fail(f"{missing} px of B's own region vanished when A was added")


def test_the_budget_is_exact_and_per_sample(scratch_api):
    api = scratch_api
    for budget in (777, 3333):
        one = _select(api, [A], max_pixels=budget)
        if one["count"] != budget:
            pytest.fail(f"budget {budget}: got {one['count']} px, expected exactly it")
        three = _select(api, [A, B, C], max_pixels=budget)
        if three["count"] != 3 * budget:
            pytest.fail(f"budget {budget} x 3 samples: got {three['count']}, "
                        f"expected {3 * budget}")


def test_a_sample_is_always_inside_its_own_selection(scratch_api):
    """Otherwise a click can appear to do nothing at all."""
    api = scratch_api
    m = _mask(_select(api, [A, B], tolerance=0.0, max_pixels=1))
    for x, y in (A, B):
        if not m[y, x]:
            pytest.fail(f"sample ({x}, {y}) is not in its own selection")


def test_a_capped_selection_stays_connected(scratch_api):
    api = scratch_api
    m = _mask(_select(api, [A], max_pixels=8000))
    pieces = ndimage.label(m, structure=_WAND_CONNECTIVITY)[1]
    if pieces != 1:
        pytest.fail(f"a clipped single-sample selection broke into {pieces} pieces")


@pytest.mark.parametrize("scratch_project", ["001.tif"], indirect=True)
def test_nodata_is_never_selected(scratch_api, scratch_project):
    api = scratch_api
    invalid = ~api._image.valid_mask
    if not invalid.any():
        pytest.skip("this image has no nodata")
    m = _mask(_select(api, [A], tolerance=0.9))
    if (m & invalid).any():
        pytest.fail(f"{int((m & invalid).sum())} nodata px were selected")
    # 001.tif is an edge tile of this dataset that carries nodata (8,443 px).
    # The tile name is the only dataset-specific pin here; if a swap drops it,
    # the test skips rather than passing vacuously on a nodata-free image.


def test_global_reaches_disconnected_material(scratch_api):
    """The point of the mode: these classes come in dozens of fragments."""
    api = scratch_api
    local = _mask(_select(api, [A], max_pixels=None))
    everywhere = _mask(_select(api, [A], max_pixels=None, **{"global": True}))
    if int(everywhere.sum()) <= int(local.sum()):
        pytest.fail("global mode selected no more than the connected piece")
    if (local & ~everywhere).any():
        pytest.fail("global mode dropped pixels the connected selection had")
    pieces = ndimage.label(everywhere, structure=_WAND_CONNECTIVITY)[1]
    if pieces < 2:
        pytest.fail("global mode returned a single piece — connectivity still bounding it")


def test_global_still_honours_the_budget(scratch_api):
    api = scratch_api
    for budget in (500, 5000):
        r = _select(api, [A], max_pixels=budget, **{"global": True})
        if r["count"] != budget:
            pytest.fail(f"global budget {budget}: got {r['count']} px")


def test_a_capped_global_selection_keeps_the_nearest_pixels(scratch_api):
    """With no path to walk, the cap spends on straight-line nearness."""
    api = scratch_api
    budget = 2000
    m = _mask(_select(api, [A], max_pixels=budget, **{"global": True}))
    full = _mask(_select(api, [A], max_pixels=None, **{"global": True}))
    ys, xs = np.nonzero(m)
    kept = np.sqrt((ys - A[1]) ** 2 + (xs - A[0]) ** 2).max()
    fy, fx = np.nonzero(full & ~m)
    if len(fy):
        dropped = np.sqrt((fy - A[1]) ** 2 + (fx - A[0]) ** 2).min()
        if dropped < kept - 1.5:  # ties on the ring are arbitrary
            pytest.fail(f"a pixel at {dropped:.0f} px was dropped while one at "
                        f"{kept:.0f} px was kept")


def test_global_is_still_strictly_additive(scratch_api):
    api = scratch_api
    masks = [_mask(_select(api, s, max_pixels=3200, **{"global": True}))
             for s in ([A], [A, B], [A, B, C])]
    for i in range(len(masks) - 1):
        lost = int((masks[i] & ~masks[i + 1]).sum())
        if lost:
            pytest.fail(f"global: adding sample {i + 2} removed {lost} px")


def test_global_and_protect_compose(scratch_api):
    api = scratch_api
    labels = api._labels
    if labels is None or (labels != 255).sum() == 0:
        pytest.skip("nothing labeled in this image")
    m = _mask(_select(api, [A], protect=True, **{"global": True}))
    if (m & (labels != 255)).any():
        pytest.fail("global + protect selected labeled pixels")


def test_a_negative_shrinks_the_selection(scratch_api):
    api = scratch_api
    plain = _mask(_select(api, [A], max_pixels=None))
    # Exclude something the selection actually contains, so it must bite.
    ys, xs = np.nonzero(plain)
    j = len(ys) // 2
    cut = _mask(_select(api, [A], max_pixels=None,
                        negatives=[[int(xs[j]), int(ys[j])]]))
    if int(cut.sum()) >= int(plain.sum()):
        pytest.fail("a negative click did not shrink the selection")
    if (cut & ~plain).any():
        pytest.fail("a negative click ADDED pixels")


def test_a_negative_reaches_every_fragment(scratch_api):
    """The reason negatives subtract a colour rather than a region."""
    api = scratch_api
    opts = {"max_pixels": None, "global": True}
    plain = _mask(_select(api, [A], **opts))
    ys, xs = np.nonzero(plain)
    j = len(ys) // 2
    neg = [[int(xs[j]), int(ys[j])]]
    cut = _mask(_select(api, [A], negatives=neg, **opts))
    removed = plain & ~cut
    if not removed.any():
        pytest.fail("the negative removed nothing")
    pieces = ndimage.label(removed, structure=_WAND_CONNECTIVITY)[1]
    if pieces < 2:
        pytest.fail("the negative only removed one connected piece; it should "
                    "subtract that material wherever it appears")


def test_a_sample_survives_its_own_negative_neighbourhood(scratch_api):
    """A positive click must stay selected however the negatives fall."""
    api = scratch_api
    m = _mask(_select(api, [A], max_pixels=None,
                      negatives=[[B[0], B[1]], [C[0], C[1]]]))
    if not m[A[1], A[0]]:
        pytest.fail("the positive sample was excluded from its own selection")


def test_negatives_compose_with_protect_and_global(scratch_api):
    api = scratch_api
    labels = api._labels
    r = _select(api, [A], protect=True, negatives=[[B[0], B[1]]],
                max_pixels=5000, **{"global": True})
    if not r.get("ok"):
        pytest.fail(f"the combination was rejected: {r.get('error')}")
    m = _mask(r)
    if labels is not None and (m & (labels != 255)).any():
        pytest.fail("protect stopped applying once negatives were present")
    if r["count"] > 5000:
        pytest.fail("the budget stopped applying once negatives were present")


def test_protect_never_selects_an_already_labeled_pixel(scratch_api):
    api = scratch_api
    labels = api._labels
    if labels is None or (labels != 255).sum() == 0:
        pytest.skip("nothing labeled in this image")
    m = _mask(_select(api, [A, B], protect=True))
    clashes = int((m & (labels != 255)).sum())
    if clashes:
        pytest.fail(f"{clashes} already-labeled px were selected with protect on")


def test_protect_reports_what_it_withheld(scratch_api):
    api = scratch_api
    off = _select(api, [A], protect=False)
    on = _select(api, [A], protect=True)
    if on["count"] + on["protected"] != off["count"]:
        pytest.fail(f"{on['count']} kept + {on['protected']} protected "
                    f"!= {off['count']} unprotected")
    if on["count"] > off["count"]:
        pytest.fail("protection selected MORE than no protection")


def test_protect_is_a_subset_not_a_different_selection(scratch_api):
    """Protection must only remove pixels, never move the selection."""
    api = scratch_api
    off = _mask(_select(api, [A, B], protect=False))
    on = _mask(_select(api, [A, B], protect=True))
    if (on & ~off).any():
        pytest.fail(f"{int((on & ~off).sum())} px appear only when protected")


def test_protection_conducts_through_labeled_pixels(scratch_api):
    """A labeled pixel blocks painting, not the fill's path through it.

    Masking the match set instead of the finished selection would sever a
    region at the first labeled pixel it meets, so a fill could not reach
    unlabeled ground on the far side of a labeled strip.
    """
    api = scratch_api
    labels = api._labels
    if labels is None or (labels != 255).sum() == 0:
        pytest.skip("nothing labeled in this image")
    on = _mask(_select(api, [A], protect=True, max_pixels=None))
    off = _mask(_select(api, [A], protect=False, max_pixels=None))
    # Everything protection keeps must be exactly the unlabeled part of
    # the unprotected selection -- no more, and crucially no less.
    expected = off & (labels == 255)
    if not np.array_equal(on, expected):
        pytest.fail(f"{int((expected & ~on).sum())} unlabeled px were lost, "
                    f"{int((on & ~expected).sum())} extra px appeared")


def test_the_reported_count_matches_the_mask(scratch_api):
    api = scratch_api
    r = _select(api, [A, B], max_pixels=BUDGET)
    if int(_mask(r).sum()) != r["count"]:
        pytest.fail(f"count {r['count']} != {int(_mask(r).sum())} px in the mask")


def test_an_under_budget_region_is_not_reported_as_capped(scratch_api):
    api = scratch_api
    r = _select(api, [A], tolerance=0.05, max_pixels=100000)
    if r["capped"]:
        pytest.fail("a region smaller than the budget was reported as capped")


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
        pytest.skip(f"SAM2 unavailable: {reason}")
    return True


def _sam_select(api, samples, **options):
    opts = {"source": "sam", "level": SAM_LEVEL, "tolerance": SAM_TOL,
            "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)


def test_sam_source_is_rejected_cleanly_when_misconfigured(scratch_api):
    """A bad source or level is a message, not a traceback across the bridge."""
    api = scratch_api
    for opts, why in (({"tolerance": SAM_TOL, "source": "nonsense"}, "bad source"),
                      ({"tolerance": SAM_TOL, "source": "sam",
                        "level": "nonsense"}, "bad level")):
        r = api.wand_select([[10, 10]], opts)
        if r.get("ok") or not r.get("error"):
            pytest.fail(f"{why} should return an error, got {r}")


def test_the_embedding_field_is_a_cosine_distance_at_pixel_resolution(scratch_api):
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
    api = scratch_api
    if not _sam_ready(api):
        return
    for level in ("fine", "mid", "deep"):
        _, field = _field(api, A, "sam", level)
        if field.shape != (api._image.height, api._image.width):
            pytest.fail(f"{level}: field is {field.shape}, image is "
                        f"{(api._image.height, api._image.width)}")
        if not np.isfinite(field).all():
            pytest.fail(f"{level}: field has non-finite values")
        if field.min() < -1e-5 or field.max() > 2 + 1e-5:
            pytest.fail(f"{level}: cosine distance out of [0, 2]: "
                        f"{field.min()}..{field.max()}")


@pytest.mark.parametrize("scratch_project", ["003.tif"], indirect=True)
def test_the_default_tolerance_selects_a_region_and_not_the_scene(scratch_api, scratch_project):
    """The UI's default lands in the band where the level is usable at all.

    The band is narrow and does not start at zero -- at the default level a
    click takes under 0.15% of the image below 0.7 and over 7% by 1.0 -- so a
    default outside it makes the tool look broken rather than mistuned. This
    fails if a re-exported encoder moves the distances, which is the point:
    the number in LabelerPane would then need re-sighting.

    Pinned to 003.tif, a tile where all three of A/B/C land inside the band at
    the default (measured 0.98% / 4.22% / 5.56%). Deliberately generous bounds:
    this pins "the default is inside the useful band", not a particular
    selection size, which varies by what was clicked.
    """
    api = scratch_api
    if not _sam_ready(api):
        return
    n = api._image.height * api._image.width
    for point in (A, B, C):
        share = 100 * _sam_select(api, [point], protect=False)["count"] / n
        if not 0.05 <= share <= 25:
            pytest.fail(f"at tolerance {SAM_TOL} a click at {point} selected "
                        f"{share:.2f}% of the image — outside the band the "
                        f"default is meant to sit in")


def test_the_two_sources_measure_different_things(scratch_api):
    """Guards against the embedding source quietly falling back to spectral.

    Compares the fields rather than the masks: two different measures can
    happen to select the same region at some tolerance, but they cannot
    produce the same distance at every pixel.
    """
    api = scratch_api
    if not _sam_ready(api):
        return
    _, spectral = _field(api, A, "image")
    _, embedding = _field(api, A, "sam")
    if np.allclose(spectral, embedding):
        pytest.fail("the embedding field is identical to the spectral one")


def test_adding_a_sample_never_removes_a_pixel_on_embeddings(scratch_api):
    """The wand's central property, on the other source."""
    api = scratch_api
    if not _sam_ready(api):
        return
    one = _mask(_sam_select(api, [A], protect=False))
    two = _mask(_sam_select(api, [A, B], protect=False))
    lost = int((one & ~two).sum())
    if lost:
        pytest.fail(f"a second sample dropped {lost} px the first had selected")


def test_the_embedding_selection_is_deterministic_and_respects_the_budget(scratch_api):
    """Same clicks, same mask -- and the cap still bites the same way."""
    api = scratch_api
    if not _sam_ready(api):
        return
    first = _mask(_sam_select(api, [A], protect=False))
    again = _mask(_sam_select(api, [A], protect=False))
    if not np.array_equal(first, again):
        pytest.fail("two identical embedding selections disagree")
    capped = _sam_select(api, [A], protect=False, max_pixels=BUDGET)
    if capped["count"] > BUDGET:
        pytest.fail(f"budget ignored on embeddings: {capped['count']} > {BUDGET}")
    # Connectivity still applies: every selected pixel is reachable from
    # the click, exactly as with the spectral source.
    pieces = ndimage.label(_mask(capped), structure=_WAND_CONNECTIVITY)[1]
    if pieces > 1:
        pytest.fail(f"a connected embedding selection came back in {pieces} pieces")
# ---------- the auto-encoder embedding source ----------
#
# Same substitution test as the SAM section: every invariant above must hold
# unchanged when distance means "8-dim land-cover embedding" instead of bands
# or SAM2 features. Two things are genuinely new. The embed stack is at pixel
# resolution, so the click's own distance survives -- there is no upsample to
# blur it -- and the source must ignore the SAM-only level option rather than
# rejecting it.

# Provisional: the usable band is measured by bench_wand.py (sources mode)
# before the frontend default is set; these tests score fields and invariants,
# which do not depend on where the band sits.
EMBED_TOL = 0.2


def _embed_ready(api):
    """Whether the embedding encoder can run here; these tests skip if not."""
    reason = api._embed.unavailable_reason()
    if reason is not None:
        pytest.skip(f"embed unavailable: {reason}")
    if api._image.bands != 10:
        pytest.skip("embed needs a 10-band stack")
    return True


def _embed_select(api, samples, **options):
    opts = {"source": "embed", "tolerance": EMBED_TOL, "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)


def test_embed_ignores_the_sam_only_level_option(scratch_api):
    """A level travels with every request; only "sam" gets to reject a bad one."""
    api = scratch_api
    if not _embed_ready(api):
        return
    r = _embed_select(api, [A], level="nonsense")
    if not r.get("ok"):
        pytest.fail(f"embed should ignore the level option, got: {r.get('error')}")


def test_the_embed_field_is_a_cosine_distance_at_pixel_resolution(scratch_api):
    api = scratch_api
    if not _embed_ready(api):
        return
    _, field = _field(api, A, "embed")
    if field.shape != (api._image.height, api._image.width):
        pytest.fail(f"field is {field.shape}, image is "
                    f"{(api._image.height, api._image.width)}")
    if not np.isfinite(field).all():
        pytest.fail("field has non-finite values")
    if field.min() < -1e-5 or field.max() > 2 + 1e-5:
        pytest.fail(f"cosine distance out of [0, 2]: {field.min()}..{field.max()}")


def test_the_embed_field_is_exact_at_the_click(scratch_api):
    """Pixel-resolution grid: unlike SAM, nothing upsamples the zero away.

    sample_size=1 makes the reference the clicked pixel's own vector, so its
    distance is exactly 0 by construction. This is the property that lets an
    embed selection hug boundaries the SAM field's half-pixel blur cannot.
    """
    api = scratch_api
    if not _embed_ready(api):
        return
    _, field = _field(api, A, "embed", size=1)
    if field[A[1], A[0]] > 1e-4:
        pytest.fail(f"click not at distance ~0 on the embed grid: "
                    f"{field[A[1], A[0]]}")


def test_embed_measures_something_of_its_own(scratch_api):
    """Guards against the new source quietly aliasing an existing one."""
    api = scratch_api
    if not _embed_ready(api):
        return
    _, spectral = _field(api, A, "image")
    _, embed = _field(api, A, "embed")
    if np.allclose(spectral, embed):
        pytest.fail("the embed field is identical to the spectral one")
    if api._sam.unavailable_reason() is None:
        _, sam = _field(api, A, "sam")
        if np.allclose(sam, embed):
            pytest.fail("the embed field is identical to the SAM one")


def test_adding_a_sample_never_removes_a_pixel_on_embed(scratch_api):
    """The wand's central property, on the third source."""
    api = scratch_api
    if not _embed_ready(api):
        return
    one = _mask(_embed_select(api, [A], protect=False))
    two = _mask(_embed_select(api, [A, B], protect=False))
    lost = int((one & ~two).sum())
    if lost:
        pytest.fail(f"a second embed sample dropped {lost} px the first had selected")


def test_embed_selection_is_deterministic_and_respects_the_budget(scratch_api):
    api = scratch_api
    if not _embed_ready(api):
        return
    first = _mask(_embed_select(api, [A], protect=False))
    again = _mask(_embed_select(api, [A], protect=False))
    if not np.array_equal(first, again):
        pytest.fail("two identical embed selections disagree")
    capped = _embed_select(api, [A], protect=False, max_pixels=BUDGET)
    if capped["count"] > BUDGET:
        pytest.fail(f"budget ignored on embed: {capped['count']} > {BUDGET}")
    pieces = ndimage.label(_mask(capped), structure=_WAND_CONNECTIVITY)[1]
    if pieces > 1:
        pytest.fail(f"a connected embed selection came back in {pieces} pieces")


@pytest.mark.parametrize("scratch_project", ["026.tif"], indirect=True)
def test_the_default_embed_tolerance_selects_a_region_and_not_the_scene(scratch_api, scratch_project):
    """The frontend's EMBED_TOL_DEFAULT=50 maps to 0.08 -- the bench peak.

    Same shape as the SAM default test: pins "the shipped default lands in
    the band where the source is usable" (measured over 470 regions: peak
    0.328 mean IoU at 0.08, band 0.04..0.12), so a re-exported encoder that
    moves the distances fails here instead of shipping a mistuned slider.

    The click is the most interior pixel of the image's largest labeled
    region -- bench_wand.py's own methodology. An arbitrary point can land
    on a boundary or inside a tiny field parcel, where a tight and entirely
    correct selection fails a share band that would then say nothing. Pinned
    to 026.tif, a class-rich tile where that interior click selects 7.41% at
    the default.
    """
    api = scratch_api
    if not _embed_ready(api):
        return
    labels = api._labels
    if labels is None or (labels != 255).sum() == 0:
        pytest.skip("nothing labeled in this image")
    best = None
    for cls in np.unique(labels):
        if cls == 255:
            continue
        comp, count = ndimage.label(labels == cls)
        if not count:
            continue
        sizes = ndimage.sum(np.ones_like(comp), comp, range(1, count + 1))
        idx = int(np.argmax(sizes))
        if best is None or sizes[idx] > best[0]:
            dist = ndimage.distance_transform_edt(comp == (idx + 1))
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            best = (int(sizes[idx]), int(x), int(y))
    _, x, y = best
    n = api._image.height * api._image.width
    share = 100 * _embed_select(api, [(x, y)], protect=False,
                                tolerance=0.08)["count"] / n
    if not 0.05 <= share <= 25:
        pytest.fail(f"at tolerance 0.08 an interior click selected "
                    f"{share:.2f}% of the image -- outside the band the "
                    f"default is meant to sit in")


def test_embed_protect_and_global_compose(scratch_api):
    api = scratch_api
    if not _embed_ready(api):
        return
    labels = api._labels
    r = _embed_select(api, [A], protect=True, max_pixels=5000, **{"global": True})
    if not r.get("ok"):
        pytest.fail(f"embed + protect + global rejected: {r.get('error')}")
    m = _mask(r)
    if labels is not None and (m & (labels != 255)).any():
        pytest.fail("protect stopped applying on the embed source")
    if r["count"] > 5000:
        pytest.fail("the budget stopped applying on the embed source")


def test_embed_select_fails_as_a_message_not_a_traceback(scratch_api, monkeypatch):
    """A missing model must reach the frontend as {ok: false, error: ...}."""
    api = scratch_api
    monkeypatch.setattr(api._embed, "unavailable_reason",
                        lambda: "embedding model missing (embedding.onnx)")
    r = api.wand_select([[10, 10]], {"tolerance": EMBED_TOL, "source": "embed"})
    if r.get("ok") or "missing" not in r.get("error", ""):
        pytest.fail(f"expected a clean 'missing' error, got {r}")


def test_embed_available_reports_the_model_state(scratch_api):
    api = scratch_api
    r = api.embed_available()
    reason = api._embed.unavailable_reason()
    bands = api._image.bands if api._image is not None else None
    expected = reason is None and bands == 10
    if r["available"] != expected:
        pytest.fail(f"embed_available says {r}, expected available={expected} "
                    f"(reason={reason}, bands={bands})")

# ---------- the mean-shift source ----------
#
# The spectral measure on a mean-shift-filtered PCA image. It shares the RMS
# units with "image", so the invariants that matter here are (a) it is really
# a different field, (b) it is sharper -- the whole reason it exists -- and
# (c) the shared wand machinery still behaves over it.

MS_TOL = 0.10  # provisional until the bench measures the band; see implementation_log


def _ms_ready(api):
    from meanshift import meanshift_available
    if not meanshift_available():
        pytest.skip("OpenCV unavailable")
    return True


def _ms_select(api, samples, **options):
    opts = {"source": "meanshift", "tolerance": MS_TOL, "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)


def test_the_meanshift_field_is_rms_at_pixel_resolution(scratch_api):
    api = scratch_api
    if not _ms_ready(api):
        return
    _, field = _field(api, A, "meanshift", size=1)
    if field.shape != (api._image.height, api._image.width):
        pytest.fail(f"field is {field.shape}")
    if not np.isfinite(field).all():
        pytest.fail("field has non-finite values")
    # RMS of 3 components in [0,1]: bounded by 1, and exact-zero at the click
    # with a one-pixel sample (pixel resolution, like embed, unlike SAM).
    if field.min() < -1e-6 or field.max() > 1 + 1e-6:
        pytest.fail(f"RMS distance out of [0, 1]: {field.min()}..{field.max()}")
    if field[A[1], A[0]] > 1e-5:
        pytest.fail(f"click not at distance 0: {field[A[1], A[0]]}")


def test_meanshift_measures_something_of_its_own(scratch_api):
    """Not an alias of the spectral field, despite sharing its units."""
    api = scratch_api
    if not _ms_ready(api):
        return
    _, spectral = _field(api, A, "image")
    _, ms = _field(api, A, "meanshift")
    if np.allclose(spectral, ms):
        pytest.fail("the mean-shift field is identical to the spectral one")


def test_meanshift_is_sharper_than_spectral(scratch_api):
    """The property the source exists for, measured on this project's labels.

    Same-label pixels must sit closer together, relative to different-label
    pixels, than they do in the raw bands -- otherwise the filter is
    decoration. Uses the image's own user labels; skips if there are none.
    A generous margin (ratio must improve by >10%) keeps it a regression
    guard rather than a knife-edge number.
    """
    api = scratch_api
    if not _ms_ready(api):
        return
    labels = api._labels
    if labels is None:
        pytest.skip("no labels")
    classes = [c for c in np.unique(labels)
               if c != 255 and (labels == c).sum() >= 400]
    if len(classes) < 2:
        pytest.skip("fewer than two labeled classes")
    rng = np.random.default_rng(0)

    def ratio(field):
        coords = {c: np.argwhere(labels == c) for c in classes}
        intra = []
        for c in classes:
            p = coords[c]
            idx = rng.choice(len(p), 300, replace=len(p) < 300)
            pts = p[idx]
            v = field[pts[:, 0], pts[:, 1]]
            intra.append(float(v.mean()))
        inter = []
        for i, a in enumerate(classes):
            for b in classes[i + 1:]:
                pa = coords[a][rng.choice(len(coords[a]), 300, replace=len(coords[a]) < 300)]
                pb = coords[b][rng.choice(len(coords[b]), 300, replace=len(coords[b]) < 300)]
                inter.append(float(np.sqrt(((field[pa[:, 0], pa[:, 1]].mean()
                                             - field[pb[:, 0], pb[:, 1]].mean()) ** 2))))
        return float(np.mean(intra)) / max(float(np.mean(inter)), 1e-9)

    _, spectral = _field(api, A, "image")
    _, ms = _field(api, A, "meanshift")
    r_spec, r_ms = ratio(spectral), ratio(ms)
    if r_ms > 0.9 * r_spec:
        pytest.fail(f"mean-shift did not sharpen the labels: intra/inter "
                    f"{r_ms:.3f} vs spectral {r_spec:.3f}")


def test_meanshift_keeps_the_additivity_invariant(scratch_api):
    api = scratch_api
    if not _ms_ready(api):
        return
    one = _mask(_ms_select(api, [A], protect=False))
    two = _mask(_ms_select(api, [A, B], protect=False))
    lost = int((one & ~two).sum())
    if lost:
        pytest.fail(f"a second mean-shift sample dropped {lost} px the first had selected")


def test_meanshift_selection_is_deterministic_and_respects_the_budget(scratch_api):
    api = scratch_api
    if not _ms_ready(api):
        return
    first = _mask(_ms_select(api, [A], protect=False))
    again = _mask(_ms_select(api, [A], protect=False))
    if not np.array_equal(first, again):
        pytest.fail("two identical mean-shift selections disagree")
    capped = _ms_select(api, [A], protect=False, max_pixels=BUDGET)
    if capped["count"] > BUDGET:
        pytest.fail(f"budget ignored on mean-shift: {capped['count']} > {BUDGET}")


def test_meanshift_ignores_the_sam_only_level_option(scratch_api):
    api = scratch_api
    if not _ms_ready(api):
        return
    r = _ms_select(api, [A], level="nonsense")
    if not r.get("ok"):
        pytest.fail(f"meanshift should ignore the level option, got: {r.get('error')}")
