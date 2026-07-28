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

from api import _WAND_CONNECTIVITY  # noqa: E402
from test_no_torch import _scratch_project  # noqa: E402

# A tolerance loose enough that regions are large enough to be interesting, and
# a budget small enough that the cap actually bites at it.
TOL = 0.25
BUDGET = 3200
A, B, C = (256, 256), (120, 400), (400, 120)


def _fail(message):
    raise AssertionError(message)


def _api(prefer=None):
    from api import Api
    tmp, _ = _scratch_project(prefer)
    if tmp is None:
        return None, None
    return Api(project_root=tmp), tmp


def _mask(result):
    """The bit-packed mask a caller gets back, as a bool array."""
    import base64
    if not result.get("ok"):
        _fail(f"wand_select failed: {result.get('error')}")
    bits = np.unpackbits(np.frombuffer(base64.b64decode(result["mask"]),
                                       dtype=np.uint8))
    n = result["width"] * result["height"]
    return bits[:n].reshape(result["height"], result["width"]).astype(bool)


def _select(api, samples, **options):
    opts = {"tolerance": TOL, "sample_size": 9}
    opts.update(options)
    return api.wand_select([[x, y] for x, y in samples], opts)


def _run(check, prefer=None):
    """Give a check an Api on a scratch project, and clean up after it."""
    import shutil
    api, tmp = _api(prefer)
    if api is None:
        print("     (no project to copy — skipping)")
        return
    try:
        check(api)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- the options object ----------

def test_options_are_passed_as_one_object():
    def check(api):
        r = _select(api, [A], tolerance=0.13, max_pixels=1000, sample_size=9)
        if not r.get("ok"):
            _fail(f"options object rejected: {r.get('error')}")
        if r["count"] > 1000:
            _fail(f"max_pixels ignored: {r['count']} > 1000")
    _run(check)


def test_missing_options_use_defaults():
    def check(api):
        r = api.wand_select([[256, 256]], {"tolerance": TOL})
        if not r.get("ok"):
            _fail(f"a bare tolerance should suffice: {r.get('error')}")
    _run(check)


def test_bad_input_returns_an_error_not_an_exception():
    def check(api):
        for opts, why in (({"tolerance": "x"}, "non-numeric tolerance"),
                          ({"tolerance": TOL, "max_pixels": 0}, "zero budget"),
                          ({"tolerance": TOL, "max_pixels": "x"}, "bad budget"),
                          ({"tolerance": TOL, "sample_size": "x"}, "bad sample size")):
            r = api.wand_select([[10, 10]], opts)
            if r.get("ok") or not r.get("error"):
                _fail(f"{why} should return an error, got {r}")
        if api.wand_select([], {"tolerance": TOL}).get("ok"):
            _fail("an empty sample list should return an error")
    _run(check)


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
                        _fail(f"tol={tol} budget={budget}: adding sample "
                              f"{i + 2} removed {lost} px")
    _run(check)


def test_a_sample_region_is_the_same_alone_or_alongside_others():
    def check(api):
        solo = _mask(_select(api, [B], max_pixels=BUDGET))
        both = _mask(_select(api, [A, B], max_pixels=BUDGET))
        missing = int((solo & ~both).sum())
        if missing:
            _fail(f"{missing} px of B's own region vanished when A was added")
    _run(check)


def test_the_budget_is_exact_and_per_sample():
    def check(api):
        for budget in (777, 3333):
            one = _select(api, [A], max_pixels=budget)
            if one["count"] != budget:
                _fail(f"budget {budget}: got {one['count']} px, expected exactly it")
            three = _select(api, [A, B, C], max_pixels=budget)
            if three["count"] != 3 * budget:
                _fail(f"budget {budget} x 3 samples: got {three['count']}, "
                      f"expected {3 * budget}")
    _run(check)


def test_a_sample_is_always_inside_its_own_selection():
    """Otherwise a click can appear to do nothing at all."""
    def check(api):
        m = _mask(_select(api, [A, B], tolerance=0.0, max_pixels=1))
        for x, y in (A, B):
            if not m[y, x]:
                _fail(f"sample ({x}, {y}) is not in its own selection")
    _run(check)


def test_a_capped_selection_stays_connected():
    def check(api):
        m = _mask(_select(api, [A], max_pixels=8000))
        pieces = ndimage.label(m, structure=_WAND_CONNECTIVITY)[1]
        if pieces != 1:
            _fail(f"a clipped single-sample selection broke into {pieces} pieces")
    _run(check)


def test_nodata_is_never_selected():
    def check(api):
        invalid = ~api._image.valid_mask
        if not invalid.any():
            print("     (this image has no nodata — skipping)")
            return
        m = _mask(_select(api, [A], tolerance=0.9))
        if (m & invalid).any():
            _fail(f"{int((m & invalid).sum())} nodata px were selected")
    # 14.tiff is the one image in this dataset that carries nodata (6,204 px).
    _run(check, prefer="14.tiff")


def test_global_reaches_disconnected_material():
    """The point of the mode: these classes come in dozens of fragments."""
    def check(api):
        local = _mask(_select(api, [A], max_pixels=None))
        everywhere = _mask(_select(api, [A], max_pixels=None, **{"global": True}))
        if int(everywhere.sum()) <= int(local.sum()):
            _fail("global mode selected no more than the connected piece")
        if (local & ~everywhere).any():
            _fail("global mode dropped pixels the connected selection had")
        pieces = ndimage.label(everywhere, structure=_WAND_CONNECTIVITY)[1]
        if pieces < 2:
            _fail("global mode returned a single piece — connectivity still bounding it")
    _run(check)


def test_global_still_honours_the_budget():
    def check(api):
        for budget in (500, 5000):
            r = _select(api, [A], max_pixels=budget, **{"global": True})
            if r["count"] != budget:
                _fail(f"global budget {budget}: got {r['count']} px")
    _run(check)


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
                _fail(f"a pixel at {dropped:.0f} px was dropped while one at "
                      f"{kept:.0f} px was kept")
    _run(check)


def test_global_is_still_strictly_additive():
    def check(api):
        masks = [_mask(_select(api, s, max_pixels=3200, **{"global": True}))
                 for s in ([A], [A, B], [A, B, C])]
        for i in range(len(masks) - 1):
            lost = int((masks[i] & ~masks[i + 1]).sum())
            if lost:
                _fail(f"global: adding sample {i + 2} removed {lost} px")
    _run(check)


def test_global_and_protect_compose():
    def check(api):
        labels = api._labels
        if labels is None or (labels != 255).sum() == 0:
            print("     (nothing labeled in this image — skipping)")
            return
        m = _mask(_select(api, [A], protect=True, **{"global": True}))
        if (m & (labels != 255)).any():
            _fail("global + protect selected labeled pixels")
    _run(check)


def test_a_negative_shrinks_the_selection():
    def check(api):
        plain = _mask(_select(api, [A], max_pixels=None))
        # Exclude something the selection actually contains, so it must bite.
        ys, xs = np.nonzero(plain)
        j = len(ys) // 2
        cut = _mask(_select(api, [A], max_pixels=None,
                            negatives=[[int(xs[j]), int(ys[j])]]))
        if int(cut.sum()) >= int(plain.sum()):
            _fail("a negative click did not shrink the selection")
        if (cut & ~plain).any():
            _fail("a negative click ADDED pixels")
    _run(check)


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
            _fail("the negative removed nothing")
        pieces = ndimage.label(removed, structure=_WAND_CONNECTIVITY)[1]
        if pieces < 2:
            _fail("the negative only removed one connected piece; it should "
                  "subtract that material wherever it appears")
    _run(check)


def test_a_sample_survives_its_own_negative_neighbourhood():
    """A positive click must stay selected however the negatives fall."""
    def check(api):
        m = _mask(_select(api, [A], max_pixels=None,
                          negatives=[[B[0], B[1]], [C[0], C[1]]]))
        if not m[A[1], A[0]]:
            _fail("the positive sample was excluded from its own selection")
    _run(check)


def test_negatives_compose_with_protect_and_global():
    def check(api):
        labels = api._labels
        r = _select(api, [A], protect=True, negatives=[[B[0], B[1]]],
                    max_pixels=5000, **{"global": True})
        if not r.get("ok"):
            _fail(f"the combination was rejected: {r.get('error')}")
        m = _mask(r)
        if labels is not None and (m & (labels != 255)).any():
            _fail("protect stopped applying once negatives were present")
        if r["count"] > 5000:
            _fail("the budget stopped applying once negatives were present")
    _run(check)


def test_protect_never_selects_an_already_labeled_pixel():
    def check(api):
        labels = api._labels
        if labels is None or (labels != 255).sum() == 0:
            print("     (nothing labeled in this image — skipping)")
            return
        m = _mask(_select(api, [A, B], protect=True))
        clashes = int((m & (labels != 255)).sum())
        if clashes:
            _fail(f"{clashes} already-labeled px were selected with protect on")
    _run(check)


def test_protect_reports_what_it_withheld():
    def check(api):
        off = _select(api, [A], protect=False)
        on = _select(api, [A], protect=True)
        if on["count"] + on["protected"] != off["count"]:
            _fail(f"{on['count']} kept + {on['protected']} protected "
                  f"!= {off['count']} unprotected")
        if on["count"] > off["count"]:
            _fail("protection selected MORE than no protection")
    _run(check)


def test_protect_is_a_subset_not_a_different_selection():
    """Protection must only remove pixels, never move the selection."""
    def check(api):
        off = _mask(_select(api, [A, B], protect=False))
        on = _mask(_select(api, [A, B], protect=True))
        if (on & ~off).any():
            _fail(f"{int((on & ~off).sum())} px appear only when protected")
    _run(check)


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
            _fail(f"{int((expected & ~on).sum())} unlabeled px were lost, "
                  f"{int((on & ~expected).sum())} extra px appeared")
    _run(check)


def test_the_reported_count_matches_the_mask():
    def check(api):
        r = _select(api, [A, B], max_pixels=BUDGET)
        if int(_mask(r).sum()) != r["count"]:
            _fail(f"count {r['count']} != {int(_mask(r).sum())} px in the mask")
    _run(check)


def test_an_under_budget_region_is_not_reported_as_capped():
    def check(api):
        r = _select(api, [A], tolerance=0.05, max_pixels=100000)
        if r["capped"]:
            _fail("a region smaller than the budget was reported as capped")
    _run(check)


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
