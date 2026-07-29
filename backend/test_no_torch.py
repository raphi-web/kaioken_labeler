"""The app must run with PyTorch uninstallable, not merely uninstalled.

Run: python3 backend/test_no_torch.py

torch is still importable in most dev environments, so "we removed the import"
is not something a normal run can check -- a stray `import torch` inside a
function would sit there working fine until someone deployed to a machine
without it. This installs a meta path finder that makes torch, torchvision and
the other training-only packages raise ImportError, then drives a real labeling
session: open the project, load an image, paint, autosave, read back, and run a
SAM2 snap.

Anything that reaches for torch fails loudly here instead of quietly bloating
the next PyInstaller bundle by ~2 GB.
"""

import base64
import importlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testlib import fail, main as run_tests  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def _scratch_project(prefer=None):
    """A throwaway one-image project, copied out of the real one.

    The labeling round trip below writes a mask and restores it. Doing that in
    the real project would mutate the ground truth this tool exists to protect,
    and would race the running app's autosave in both directions if anyone
    happens to be labeling. A copy costs one image and stays entirely out of
    the way.

    `prefer` picks a specific image by name when a check needs one with
    particular properties (nodata, say); otherwise the first is copied.

    Returns (tmpdir, image_name) or (None, None) if the real project has
    nothing to copy.
    """
    import project as project_mod

    config_path = os.path.join(ROOT, project_mod.CONFIG_NAME)
    if not os.path.exists(config_path):
        return None, None
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    src_images = os.path.join(ROOT, config["paths"]["images_folder"])
    src_masks = os.path.join(ROOT, config["paths"]["masks_user"])
    names = sorted(n for n in os.listdir(src_images)
                   if n.lower().endswith(project_mod.IMAGE_EXTENSIONS))
    if not names:
        return None, None
    name = prefer if prefer in names else names[0]
    stem = os.path.splitext(name)[0] + ".tif"

    tmp = tempfile.mkdtemp(prefix="kaioken-test-")
    os.makedirs(os.path.join(tmp, "images"))
    os.makedirs(os.path.join(tmp, "masks"))
    shutil.copy2(os.path.join(src_images, name), os.path.join(tmp, "images", name))
    src_mask = os.path.join(src_masks, stem)
    if os.path.exists(src_mask):
        shutil.copy2(src_mask, os.path.join(tmp, "masks", stem))

    config["paths"] = {"images_folder": "images", "masks_user": "masks",
                       "masks_ai": "masks_ai"}
    with open(os.path.join(tmp, project_mod.CONFIG_NAME), "w",
              encoding="utf-8") as f:
        json.dump(config, f)
    return tmp, name

BANNED = ("torch", "torchvision", "segmentation_models_pytorch", "timm",
          "albumentations")


class _BannedFinder:
    """Meta path finder that refuses the training-only packages."""

    def find_module(self, fullname, path=None):  # legacy API, harmless
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BANNED:
            raise ImportError(
                f"{fullname} is not available in the labeling build "
                f"(blocked by test_no_torch)")
        return None


def test_backend_modules_import_without_torch():
    for name in ("data", "project", "sam_service", "api"):
        sys.modules.pop(name, None)
        importlib.import_module(name)
    for name in BANNED:
        if name in sys.modules:
            fail(f"{name} got imported while loading the backend")


def test_labeling_round_trip_without_torch():
    """Open a project, paint, autosave, and read the mask back off disk.

    Runs against a scratch copy, never the real label_raster/ -- see
    _scratch_project.
    """
    from api import Api
    from data import UNLABELED, read_label_mask

    tmp, _ = _scratch_project()
    if tmp is None:
        print("     (no project to copy — skipping)")
        return
    try:
        _round_trip(Api(project_root=tmp), UNLABELED, read_label_mask)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for module in BANNED:
        if module in sys.modules:
            fail(f"{module} got imported during a labeling round trip")


def _round_trip(api, UNLABELED, read_label_mask):
    project = api.get_project()
    if not project:
        fail("no project opened")
    image = api.get_image()
    if not image:
        fail("no image loaded")

    name = image["name"]
    mask_path = api._project.mask_paths(name)[0]
    original = read_label_mask(mask_path)
    if original is None:
        fail(f"could not read the existing mask for {name}")

    if True:
        # Paint a 10x10 block of class 2 into the top-left corner and persist it
        # exactly the way a brush stroke does.
        labels = original.copy()
        labels[:10, :10] = 2
        api.set_labels(base64.b64encode(labels.tobytes()).decode("ascii"))
        saved = api.autosave_user_mask()
        if not saved.get("ok"):
            fail(f"autosave failed: {saved}")

        reloaded = read_label_mask(mask_path)
        if reloaded is None:
            fail("mask could not be read back after autosave")
        if not np.array_equal(reloaded, labels):
            fail(f"{int((reloaded != labels).sum())} px differ after a save/load cycle")
        if not (reloaded[:10, :10] == 2).all():
            fail("the painted block did not survive the round trip")
        # The rest of the image must be untouched by the edit.
        rest = np.ones(original.shape, dtype=bool)
        rest[:10, :10] = False
        if not np.array_equal(reloaded[rest], original[rest]):
            fail("painting one corner changed pixels elsewhere")
        if UNLABELED in labels[:10, :10]:
            fail("painted block still contains unlabeled pixels")
        # Restoring is still checked: it is the same write path the app uses
        # when you undo, and it runs against the scratch copy.
        api.set_labels(base64.b64encode(original.tobytes()).decode("ascii"))
        api.autosave_user_mask()
        restored = read_label_mask(mask_path)
        if restored is None or not np.array_equal(restored, original):
            fail("the mask did not restore to its original contents")


def test_sam_snap_without_torch():
    from api import Api

    tmp, _ = _scratch_project()
    if tmp is None:
        print("     (no project to copy — skipping)")
        return
    try:
        _sam_snap(Api(project_root=tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _sam_snap(api):
    if not api.sam_available().get("available"):
        print("     (SAM2 unavailable — skipping snap check)")
        return
    result = api.sam_snap([[256, 256, 1]])
    if not result.get("ok"):
        fail(f"sam_snap failed: {result.get('error')}")
    if not result.get("polygons"):
        fail("sam_snap returned no polygons")
    for name in BANNED:
        if name in sys.modules:
            fail(f"{name} got imported during a SAM2 snap")


def main():
    for name in BANNED:
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, _BannedFinder())

    # Prove the block itself works, or every test below is vacuous.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        print("FAIL: the import block is not working; results would be meaningless")
        return 1

    return run_tests(globals())


if __name__ == "__main__":
    sys.exit(main())
