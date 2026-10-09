"""The app must run with PyTorch uninstallable, not merely uninstalled.

torch is still importable in most dev environments, so "we removed the import"
is not something a normal run can check -- a stray `import torch` inside a
function would sit there working fine until someone deployed to a machine
without it. The block itself is installed by backend/conftest.py at session
start, before this file is imported, so any test below is meaningful.

Anything that reaches for torch fails loudly here instead of quietly dragging
~2 GB of training-only dependencies back into the install.
"""

import base64
import importlib

import numpy as np
import pytest

from data import UNLABELED, read_label_mask

BANNED = ("torch", "torchvision", "segmentation_models_pytorch", "timm",
          "albumentations")


def test_backend_modules_import_without_torch():
    for name in ("data", "project", "sam_service", "embed_service", "api"):
        importlib.import_module(name)
    for name in BANNED:
        if name in __import__("sys").modules:
            pytest.fail(f"{name} got imported while loading the backend")


def test_labeling_round_trip_without_torch(scratch_api):
    """Open a project, paint, autosave, and read the mask back off disk.

    Runs against a scratch copy, never the real label_raster/ -- see the
    `scratch_api` fixture in backend/conftest.py.
    """
    api = scratch_api
    project = api.get_project()
    if not project:
        pytest.fail("no project opened")
    image = api.get_image()
    if not image:
        pytest.fail("no image loaded")

    name = image["name"]
    mask_path = api._project.mask_paths(name)[0]
    original = read_label_mask(mask_path)
    if original is None:
        pytest.fail(f"could not read the existing mask for {name}")

    # Paint a 10x10 block of class 2 into the top-left corner and persist it
    # exactly the way a brush stroke does.
    labels = original.copy()
    labels[:10, :10] = 2
    api.set_labels(base64.b64encode(labels.tobytes()).decode("ascii"))
    saved = api.autosave_user_mask()
    if not saved.get("ok"):
        pytest.fail(f"autosave failed: {saved}")

    reloaded = read_label_mask(mask_path)
    if reloaded is None:
        pytest.fail("mask could not be read back after autosave")
    if not np.array_equal(reloaded, labels):
        pytest.fail(f"{int((reloaded != labels).sum())} px differ after a save/load cycle")
    if not (reloaded[:10, :10] == 2).all():
        pytest.fail("the painted block did not survive the round trip")
    # The rest of the image must be untouched by the edit.
    rest = np.ones(original.shape, dtype=bool)
    rest[:10, :10] = False
    if not np.array_equal(reloaded[rest], original[rest]):
        pytest.fail("painting one corner changed pixels elsewhere")
    if UNLABELED in labels[:10, :10]:
        pytest.fail("painted block still contains unlabeled pixels")
    # Restoring is still checked: it is the same write path the app uses
    # when you undo, and it runs against the scratch copy.
    api.set_labels(base64.b64encode(original.tobytes()).decode("ascii"))
    api.autosave_user_mask()
    restored = read_label_mask(mask_path)
    if restored is None or not np.array_equal(restored, original):
        pytest.fail("the mask did not restore to its original contents")

    for module in BANNED:
        if module in __import__("sys").modules:
            pytest.fail(f"{module} got imported during a labeling round trip")


def test_sam_snap_without_torch(scratch_api):
    api = scratch_api
    if not api.sam_available().get("available"):
        pytest.skip("SAM2 unavailable")
    result = api.sam_snap([[256, 256, 1]])
    if not result.get("ok"):
        pytest.fail(f"sam_snap failed: {result.get('error')}")
    if not result.get("polygons"):
        pytest.fail("sam_snap returned no polygons")
    for name in BANNED:
        if name in __import__("sys").modules:
            pytest.fail(f"{name} got imported during a SAM2 snap")