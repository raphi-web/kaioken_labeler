"""Shared fixtures for the backend tests.

The scratch project is a throwaway copy of the real one, used by every test
that touches labels. The labeling round-trip writes masks; doing that in the
real project would mutate the ground truth the tool exists to protect.

The SAM2 import-block lives here too: it has to be installed before any test
module is imported, and conftest.py is the earliest place pytest gives us.
A stray `import torch` inside a function would otherwise sit there working
fine until someone deployed to a machine without it.
"""
import json
import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- SAM2 / torch import block ----------
#
# This is what the test_no_torch file used to install at the top of its own
# main(). Moving it here means pytest loads it once at session start, before
# any test module is imported -- which is the only window that matters, since
# a banned package leaking in during pytest's own startup is not what this
# guards against (the block runs against the project's modules, not pytest's).

BANNED = ("torch", "torchvision", "segmentation_models_pytorch", "timm",
          "albumentations")


class _BannedFinder:
    """Meta path finder that refuses the training-only packages."""

    def find_module(self, fullname, path=None):  # legacy API, harmless
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BANNED:
            raise ImportError(
                f"{fullname} is not available in the labeling build "
                f"(blocked by conftest._BannedFinder)")
        return None


# A test that torch really is unblocked before this conftest loaded would be
# vacuous -- the block itself might have been installed too late. Prove it
# works here, at conftest load, so the rest of the suite can rely on it.
for name in BANNED:
    sys.modules.pop(name, None)
sys.meta_path.insert(0, _BannedFinder())

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    raise RuntimeError(
        "the torch import block is not working; the SAM2 round-trip tests "
        "would be meaningless")

# ---------- scratch project fixtures ----------


def _scratch_project(prefer=None):
    """(tmpdir, image_name) for a throwaway copy of the project's first image,
    or (None, None) if the project has nothing to copy.

    A copy rather than the real thing for the same reason the tests use one:
    Api.load_image autosaves the active mask on every switch, and a test
    writing to label_raster/ would mutate the ground truth it exists to
    protect and race the running app's autosave besides.
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
    shutil.copy2(os.path.join(src_images, name),
                 os.path.join(tmp, "images", name))
    src_mask = os.path.join(src_masks, stem)
    if os.path.exists(src_mask):
        shutil.copy2(src_mask, os.path.join(tmp, "masks", stem))

    config["paths"] = {"images_folder": "images", "masks_user": "masks",
                       "masks_ai": "masks_ai"}
    with open(os.path.join(tmp, project_mod.CONFIG_NAME), "w",
              encoding="utf-8") as f:
        json.dump(config, f)
    return tmp, name


@pytest.fixture
def scratch_project(request):
    """A (tmpdir, image_name) pair, copied from the real project. Skips if
    the project at the repo root is missing; cleans up after the test.

    `request.param` may be a string naming an image to copy; default is the
    first image in the images folder. Pytest's `parametrize` mechanism is
    available too, but the bare call site `scratch_project` (no params)
    remains the most common shape.
    """
    prefer = getattr(request, "param", None)
    tmp, name = _scratch_project(prefer=prefer)
    if tmp is None:
        pytest.skip("no project at the repo root to copy")
    try:
        yield tmp, name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def scratch_api(scratch_project):
    """An Api on a throwaway project copy. Builds on `scratch_project` so
    cleanup runs once. Inherits the `prefer` argument via indirect
    parametrization on `scratch_project`."""
    from api import Api
    tmp, _ = scratch_project
    yield Api(project_root=tmp)