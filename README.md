# Kaio-ken Labeler

## A label tool for Remote Sensing, supports GeoTiffs with more than just 3 Bands!

![screenshot.png](readme_assets/screenshot.png)

## What this build is

This is the **labeling-only** fork of the Kaio-ken Segmenter, set up for one
fixed 11-class land-cover dataset. Everything to do with the model is gone:
no training, no inference, no accuracy report, no model or ONNX export, and
**no PyTorch dependency** — see [Removed from this build](#removed-from-this-build).

What remains is the part you actually label with: the class palette, brush,
polygon and SAM2 click-assist tools, the thumbnail strip, and mask I/O that
round-trips your GeoTIFFs without touching their format.

Train on these labels with the full app, which reads the same
`project_config.json`.

## Setup & run

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install "pywebview[qt]"   # bundled Qt renderer, no system GTK needed
cd frontend && npm install && npm run build && cd ..
venv/bin/python backend/main.py [path/to/image.tiff | path/to/project_dir]
```

With no argument it opens the project described by the `project_config.json` at
the repo root. There is no CUDA wheel to fetch and no multi-gigabyte torch
install — the whole dependency set is numpy, rasterio, pywebview, pillow,
onnxruntime, scipy and opencv.

SAM2 click-assist needs `sam2/onnx/*.onnx`. Those graphs are **checked in and
are the only copy**: this build has no exporter, so treat them as source, not
as a build artifact. Without them the SAM2 button stays disabled and brush and
polygon still work.

## The dataset

`project_config.json` points at `input_images/` (34 ten-band GeoTIFFs) and
`label_raster/`.

| Index | Class | | Index | Class |
|-------|-------|-|-------|-------|
| 0 | Urban | | 6 | Shrub |
| 1 | Road | | 7 | Water |
| 2 | Agriculture-Green | | 8 | Snow |
| 3 | Raw-Earth/Agriculture | | 9 | Wetland |
| 4 | Meadow | | 10 | Synthetic Nodata |
| 5 | Forest | | | |

Pick a class from the palette left of the labeling pane, or with `1`–`9`, `0`,
`-`; `E` toggles the eraser. The palette shows a live painted-pixel count per
class and highlights the ones still at zero — a class with no pixels is one the
model can never learn.

Labels autosave a second after the last stroke, and again whenever you switch
images, so the "Labeled" pixel count in the toolbar doubles as the "your work is
on disk" readout.

**Mask format.** The label rasters are `int32` with `nodata = -9999`, while the
app works internally in `uint8` (class ids `0..N-1`, `255` unlabeled). The
`mask_format` key in the project config drives the conversion both ways, so
edited masks are written back as `int32`/`-9999` and stay byte-compatible with
the surrounding GIS pipeline. `backend/test_masks.py` round-trips all 34 real
files to keep it that way — run it after touching mask I/O.

`input_images/` and `label_raster/` are gitignored (~140 MB) and the app edits
the labels **in place**, so back them up outside version control.

## Sharing a project with the full app

`project_config.json` is a shared schema. This build validates all of it but
only manages part of it: the training split, the model patch size and the
PointRend flag are read, checked and **written back untouched**. Saving settings
here never quietly reconfigures a training run set up over there.

## How it works

### 1. Data & Classes

- Classes: one id per entry in the project config's `classes` list, each with a
  name and a colour. Two by default; eleven in this dataset.
- Unlabeled (255): the "not annotated" value; also what the `-9999` nodata in
  the label rasters maps to on read.
- Preprocessing: per-band 2nd–98th percentile scaling to [0, 1] for display,
  ignoring nodata.

### 2. SAM2 Assist

- Efficient-SAM2 as two prebuilt ONNX graphs, run with onnxruntime.
- The encoder runs once per image (~seconds on CPU) and is cached; each click
  replays the decoder over those cached embeddings, which is fast.
- The surrounding resize/normalize is numpy and `scipy.ndimage.zoom`. It used
  torchvision and `torch.nn.functional.interpolate` — the last reason this build
  would have needed PyTorch, for a resize and a normalize.
  `backend/test_sam.py` pins the rewrite against masks captured from the torch
  implementation; they match pixel for pixel.

## Removed from this build

| Gone | Why |
|------|-----|
| `backend/model.py`, `train.py`, `report.py` | the U-Net, the training loop and the accuracy report |
| `backend/export_sam_onnx.py` | the SAM2 → ONNX export chain; the graphs are checked in instead |
| `standalone/`, `pretraining/` | the standalone predictor and the pretrained weights |
| Train / Stop / Epochs, Adopt Prediction, Reset Model | training controls |
| The inference pane and uncertainty overlay | there is no prediction to show |
| Generate Accuracy Report | scoring needs a model |
| Export Model / ONNX / Executable / Mask | `export_mask` merged labels over a prediction; labels already autosave to the masks folder |
| The validation split and its thumbnail UI | it exists to hold images out of training |

## Tests

```bash
python3 backend/test_masks.py   # mask round-trip over all 34 real label files
python3 backend/test_sam.py     # SAM2 still matches the torch implementation
```

## Frontend development

```bash
cd frontend && npm run dev   # hot-reload UI (bridge calls need the pywebview host)
npm run build                # rebuild dist/ used by backend/main.py
```

## Disclaimer

Originally built as a prototype for human-in-the-loop semantic segmentation in
remote sensing. **I also used AI code generation in this project to get
started.** **At the current state I can't guarantee that it works on every
machine!**
