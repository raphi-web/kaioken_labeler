# Magic Wand Embedding ONNX

**File:** `runs/ce/embedding.onnx` (61 KB, single-file)
**Source:** `export_ce_onnx.py` (re-run after retraining)
**Checkpoint:** `runs/ce/best.pt` — epoch 33, intra/inter ratio 0.16

## Contract

| | name | shape | dtype | description |
|---|---|---|---|---|
| input | `raw_image` | `(B, 10, 128, 128)` | int16 | raw Sentinel-2 L2A DN, exactly as `rasterio.read()` returns |
| output | `embedding` | `(B, 8, 128, 128)` | float32 | per-pixel 8-dim embedding |

Batch dim (`B`) is dynamic; spatial dims are fixed 128×128.

## Preprocessing

**None.** Normalisation is baked into the graph:

```
z = (raw_DN - mean * 10000) / (std * 10000)
```

The constants are stored as graph initializers and were computed over all
1814 tiles in the dataset:

| band | mean (refl) | std (refl) | mean (DN) | std (DN) |
|------|-------------|------------|-----------|----------|
| 1 | 0.1932 | 0.1729 | 1932 | 1729 |
| 2 | 0.1919 | 0.1657 | 1919 | 1657 |
| 3 | 0.1694 | 0.1677 | 1694 | 1677 |
| 4 | 0.3712 | 0.1512 | 3712 | 1512 |
| 5 | 0.2347 | 0.1657 | 2347 | 1657 |
| 6 | 0.3262 | 0.1492 | 3262 | 1492 |
| 7 | 0.3581 | 0.1459 | 3581 | 1459 |
| 8 | 0.3777 | 0.1437 | 3777 | 1437 |
| 9 | 0.2802 | 0.1060 | 2802 | 1060 |
| 10 | 0.2140 | 0.0949 | 2140 | 949 |

## Usage

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("runs/ce/embedding.onnx")

# tile_dn: (10, 128, 128) int16, from rasterio.open("tile.tif").read()
# Add batch dim before calling:
emb = session.run(None, {"raw_image": tile_dn[np.newaxis]})[0]
# emb.shape == (1, 8, 128, 128)
```

## Magic wand with cosine similarity

Replace raw 10-band spectral similarity with 8-dim embedding similarity:

```python
embedding = emb[0]  # (8, 128, 128)

# Clicked pixel at (y, x)
query = embedding[:, y, x]  # (8,)

# Cosine similarity across the whole image
query_norm = query / np.linalg.norm(query)
dot = np.tensordot(query_norm, embedding, axes=([0], [0]))  # (128, 128)
# dot ≈ cosine similarity (embeddings are roughly unit-norm after training)

# Threshold to select similar pixels
mask = dot > 0.95  # adjust threshold as needed
```

Or with L2 distance (no normalisation needed):

```python
dists = np.linalg.norm(embedding - query[:, np.newaxis, np.newaxis], axis=0)
mask = dists < 0.3  # adjust threshold as needed
```

## What the embedding represents

The 8-dim per-pixel embedding was trained to cluster by land-cover class:

| metric | value |
|---|---|
| intra-class cosine distance | 0.15 |
| inter-class cosine distance | 0.91 |
| **intra/inter ratio** | **0.16** (6× separation) |

Within a class, the mean cosine distance is 0.15; between different classes,
it's 0.91. This means two pixels of the same class are ~6× closer in
embedding space than two pixels of different classes.

The embedding was trained on 8 land-cover classes (0–4, 6–8). No pixels of
class 5 exist in the dataset. Pixels labeled as void (−9999) were excluded
from the contrastive loss but contributed to reconstruction.

## Re-export

After retraining with `ce_train.py`:

```
python export_ce_onnx.py --checkpoint runs/ce/best.pt --out runs/ce/embedding.onnx
```

The script loads the checkpoint, wraps the encoder with normalisation,
exports at batch=2 (required for dynamo in torch ≥ 2.10), collapses the
external-data sidecar, and verifies bit-exact against PyTorch. No manual
preprocessing needed in the consumer.