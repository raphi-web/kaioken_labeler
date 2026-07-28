#!/usr/bin/env bash
# Build the Kaio-ken Labeler desktop app with PyInstaller.
#
# Usage: ./build_app.sh
#
# Result: dist/kaioken-labeler/  (executable + _internal/)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_PYTHON="venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: venv python not found at $VENV_PYTHON" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
    echo "==> Installing PyInstaller into venv"
    "$VENV_PYTHON" -m pip install pyinstaller
fi

if [[ ! -f frontend/dist/index.html ]]; then
    echo "==> Frontend not built, running npm install && npm run build"
    (cd frontend && npm install && npm run build)
fi

echo "==> Removing previous build/ and dist/"
rm -rf build dist

echo "==> Running PyInstaller"
"$VENV_PYTHON" -m PyInstaller app.spec --noconfirm

# The SAM2 graphs are the only copy -- this build has no exporter -- so a
# missing file here is a packaging error rather than an optional extra.
if [[ -f sam2/onnx/sam2.1_hiera_tiny.encoder.onnx ]]; then
    echo "==> Bundling sam2/onnx for in-app SAM2 assist"
    mkdir -p dist/kaioken-labeler/sam2
    cp -r sam2/onnx dist/kaioken-labeler/sam2/onnx
else
    echo "error: sam2/onnx/sam2.1_hiera_tiny.encoder.onnx is missing." >&2
    echo "       Restore it from version control; it cannot be regenerated here." >&2
    exit 1
fi

echo "==> Build complete: $(pwd)/dist/kaioken-labeler/"
du -sh dist/kaioken-labeler
