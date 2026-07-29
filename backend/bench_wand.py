"""Measure the wand against the project's own labels.

    python3 backend/bench_wand.py tolerance   # per-class tolerance sweep
    python3 backend/bench_wand.py sources     # spectral vs SAM2 features

Both take connected regions of the saved user masks as ground truth, click the
most interior pixel of each, and score the selection by IoU against that
region. That is the method every default in LabelerPane's wand constants came
from, and this is the script they came from -- kept so the numbers can be
re-derived rather than trusted, and so the open question can be answered: the
SAM-features source is expected to do better on higher-resolution imagery than
on 10 m/px Sentinel-2, and `sources` is how you would find out.

Runs against a throwaway copy of the project. Nothing here writes to
label_raster/.

Options: --images N (default 20), --regions N per image (default 25),
--min-region PX (default 150).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import UNLABELED  # noqa: E402

# "Synthetic Nodata" is a bookkeeping class, not a material anyone wands at.
NODATA_CLASS_NAME = "synthetic nodata"


def scratch_project(n_images):
    """A throwaway copy of the first n labeled images of the real project.

    A copy rather than the real thing for the same reason the tests use one:
    Api.load_image autosaves the active mask on every switch, and a benchmark
    must not be able to touch the ground truth it is scoring against.
    """
    import project as project_mod
    from test_no_torch import ROOT

    config_path = os.path.join(ROOT, project_mod.CONFIG_NAME)
    if not os.path.exists(config_path):
        return None
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    src_images = os.path.join(ROOT, config["paths"]["images_folder"])
    src_masks = os.path.join(ROOT, config["paths"]["masks_user"])
    names = sorted(n for n in os.listdir(src_images)
                   if n.lower().endswith(project_mod.IMAGE_EXTENSIONS))
    tmp = tempfile.mkdtemp(prefix="kaioken-bench-")
    os.makedirs(os.path.join(tmp, "images"))
    os.makedirs(os.path.join(tmp, "masks"))
    taken = 0
    for name in names:
        stem = os.path.splitext(name)[0] + ".tif"
        src_mask = os.path.join(src_masks, stem)
        if not os.path.exists(src_mask):
            continue  # no ground truth to score against
        shutil.copy2(os.path.join(src_images, name),
                     os.path.join(tmp, "images", name))
        shutil.copy2(src_mask, os.path.join(tmp, "masks", stem))
        taken += 1
        if taken >= n_images:
            break
    config["paths"] = {"images_folder": "images", "masks_user": "masks",
                       "masks_ai": "masks_ai"}
    with open(os.path.join(tmp, project_mod.CONFIG_NAME), "w",
              encoding="utf-8") as f:
        json.dump(config, f)
    return tmp


def unpack(result, height, width):
    """The bit-packed mask a wand_select returns, as a bool array."""
    import base64
    bits = np.unpackbits(np.frombuffer(base64.b64decode(result["mask"]),
                                       dtype=np.uint8))
    return bits[:height * width].reshape(height, width).astype(bool)


def regions(labels, skip_ids, min_region, limit):
    """(truth, class id, x, y, size) per connected labeled region worth a click.

    The seed is the pixel furthest from the region's edge: a fair, reproducible
    stand-in for where someone would actually click, and stable across runs in
    a way a random interior pixel is not.
    """
    out = []
    for cls in np.unique(labels):
        if cls == UNLABELED or int(cls) in skip_ids:
            continue
        comp, count = ndimage.label(labels == cls)
        if not count:
            continue
        sizes = ndimage.sum(np.ones_like(comp), comp, range(1, count + 1))
        for idx in np.argsort(sizes)[::-1]:
            if sizes[idx] < min_region or len(out) >= limit:
                break
            truth = comp == (idx + 1)
            dist = ndimage.distance_transform_edt(truth)
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            out.append((truth, int(cls), int(x), int(y), int(sizes[idx])))
    return out


def iou(selection, truth):
    union = int((selection | truth).sum())
    return int((selection & truth).sum()) / union if union else 0.0


def collect(api, args, sweep):
    """Score every region under every setting in `sweep`.

    `sweep` is [(label, options), ...]; returns rows of
    {cls, size, scores: {label: iou}}.
    """
    rows = []
    names = {c["id"]: c["name"] for c in api._project.config["classes"]}
    skip = {i for i, n in names.items() if n.strip().lower() == NODATA_CLASS_NAME}
    for name in api._project.list_images()[:args.images]:
        api.load_image(name)
        labels = api._labels
        if labels is None:
            continue
        height, width = labels.shape
        found = regions(labels, skip, args.min_region, args.regions)
        for truth, cls, x, y, size in found:
            scores = {}
            for label, options in sweep:
                result = api.wand_select([[x, y]], {**options, "protect": False})
                if not result.get("ok"):
                    raise SystemExit(f"wand_select failed: {result['error']}")
                scores[label] = iou(unpack(result, height, width), truth)
            rows.append({"cls": cls, "size": size, "scores": scores})
        print(f"{name}: {len(rows)} regions so far", flush=True)
    return rows, names


def report_tolerance(rows, names, tols):
    """Per-class tolerance curves, and what per-class defaults would be worth."""
    print(f"\n{len(rows)} regions\n")
    print(f"{'class':>22s} {'n':>4s} {'best':>6s} {'IoU@best':>9s} "
          f"{'IoU@0.13':>9s} {'gain':>7s}")
    best_by_class = {}
    for cls in sorted({r["cls"] for r in rows}):
        mine = [r for r in rows if r["cls"] == cls]
        curve = {t: float(np.mean([r["scores"][t] for r in mine])) for t in tols}
        best = max(curve, key=curve.get)
        best_by_class[cls] = best
        print(f"{names.get(cls, cls):>22s} {len(mine):4d} {best:6.2f} "
              f"{curve[best]:9.3f} {curve[0.13]:9.3f} {curve[best] - curve[0.13]:+7.3f}")

    flat = float(np.mean([r["scores"][0.13] for r in rows]))
    per_class = float(np.mean([r["scores"][best_by_class[r["cls"]]] for r in rows]))
    oracle = float(np.mean([max(r["scores"].values()) for r in rows]))
    print(f"\nmean IoU, one global tolerance (0.13):   {flat:.3f}")
    print(f"mean IoU, best tolerance per class:      {per_class:.3f} "
          f"({100 * (per_class / flat - 1):+.0f}%)")
    print(f"mean IoU, best tolerance per region:     {oracle:.3f}  (oracle)")
    print("\nSeeds for LabelerPane's WAND_TOL_SEEDS, in slider units:")
    for cls, tol in sorted(best_by_class.items()):
        print(f"  {names.get(cls, cls):>22s}  {round(tol * 200)}")


def report_sources(rows, names, spectral, embedding):
    """Whether the two distance sources are redundant or complementary."""
    best = {k: np.array([max(r["scores"][t] for t in labels) for r in rows])
            for k, labels in (("spectral", spectral), ("embedding", embedding))}
    print(f"\n{len(rows)} regions\n")
    for key, values in best.items():
        print(f"{key:>10s} @ its own best tolerance: mean {values.mean():.3f}  "
              f"median {np.median(values):.3f}  IoU>=0.5 "
              f"{100 * (values >= 0.5).mean():4.1f}%")
    wins = int((best["embedding"] > best["spectral"]).sum())
    union = np.maximum(best["spectral"], best["embedding"])
    print(f"\nembedding wins on {wins}/{len(rows)} regions")
    print(f"correlation of the two: {np.corrcoef(best['spectral'], best['embedding'])[0, 1]:.3f}")
    print(f"picking the better per region: {union.mean():.3f} "
          f"({100 * (union.mean() / best['spectral'].mean() - 1):+.0f}% over spectral)")
    print("\nA high correlation means the second source is redundant; a low one "
          "with a real union gain\nmeans they fail on different regions, which "
          "is the only thing that justifies keeping both.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tolerance", "sources"))
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--regions", type=int, default=25)
    parser.add_argument("--min-region", type=int, default=150)
    args = parser.parse_args()

    tmp = scratch_project(args.images)
    if tmp is None:
        raise SystemExit("no project at the repo root to copy")
    from api import Api
    api = Api(project_root=tmp)
    try:
        if args.mode == "tolerance":
            tols = [round(0.03 + 0.01 * i, 3) for i in range(30)]
            sweep = [(t, {"tolerance": t, "sample_size": 9}) for t in tols]
            rows, names = collect(api, args, sweep)
            report_tolerance(rows, names, tols)
        else:
            if api._sam.unavailable_reason():
                raise SystemExit(f"SAM2 unavailable: {api._sam.unavailable_reason()}")
            spectral = [round(0.05 + 0.05 * i, 3) for i in range(9)]
            embedding = [round(0.5 + 0.05 * i, 3) for i in range(11)]
            sweep = ([(t, {"tolerance": t, "sample_size": 9}) for t in spectral]
                     + [(f"sam{t}", {"tolerance": t, "sample_size": 9,
                                     "source": "sam", "level": "fine"})
                        for t in embedding])
            rows, names = collect(api, args, sweep)
            report_sources(rows, names, spectral, [f"sam{t}" for t in embedding])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
