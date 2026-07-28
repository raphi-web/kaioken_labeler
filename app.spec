# PyInstaller spec for the Kaio-ken Labeler desktop app (onedir).
#
# Build:  ./build_app.sh   (or: venv/bin/python -m PyInstaller app.spec --noconfirm)
# Result: dist/kaioken-labeler/  (executable + _internal/).
#
# Bundles backend/ (pywebview host) together with the built frontend/dist/
# (React UI). Run `cd frontend && npm run build` first -- build_app.sh does
# this automatically if dist/ is missing.
#
# This build has no torch: SAM2 runs on onnxruntime and everything else is
# numpy/scipy/rasterio. cv2, onnxruntime, scipy and pywebview all ship their
# own PyInstaller hooks, so only rasterio -- which has none -- needs a manual
# collect_all.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("rasterio")

datas += [("frontend/dist", "frontend/dist")]

# backend/*.py import each other as top-level modules (main.py adds
# backend/ to sys.path at runtime); pathex below lets Analysis find them,
# but list them explicitly too since they're imported dynamically by api.py.
hiddenimports += ["api", "data", "project", "sam_service"]

a = Analysis(
    ["backend/main.py"],
    pathex=["backend"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Nothing here should pull torch in. Excluding it outright is a tripwire:
    # a stray import would otherwise grow the bundle by ~2 GB silently, which
    # is exactly the regression this build exists to avoid. The test modules
    # are dev-only and would drag their fixtures in.
    excludes=["torch", "torchvision", "segmentation_models_pytorch", "timm",
              "albumentations", "test_masks", "test_sam"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="kaioken-labeler",
    console=True,  # keep stdout for logs/tracebacks
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="kaioken-labeler",
)
