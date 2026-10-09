"""Per-pixel land-cover embeddings from the auto-encoder ONNX graph.

Runs auto_encoder/embedding.onnx (contract: auto_encoder/ONNX.md): raw
Sentinel-2 DN in, an 8-dim per-pixel embedding out, normalisation baked into
the graph. The graph is fixed at 128x128 spatially with a dynamic batch axis,
so an image of any size is cut into 128x128 patches, encoded in batches and
stitched back to (8, H, W) at pixel resolution -- unlike the SAM2 features,
the wand gets a cell per pixel and no upsample is involved.

Same service shape as SamService (lazy session, key-cached encode, one lock)
because it does the same job for a different graph, and the wand already
knows how to talk to that shape.
"""

import os
import threading

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(_PROJECT_ROOT, "auto_encoder", "embedding.onnx")

PATCH = 128        # the graph's fixed spatial side
INPUT_BANDS = 10   # the graph's fixed channel count
_MAX_BATCH = 64    # patches per session.run; bounds peak memory on big images


class EmbedService:
    """Lazy onnxruntime wrapper around the embedding encoder."""

    def __init__(self, onnx_path=ONNX_PATH):
        self._path = onnx_path
        self._sess = None
        self._feats = None       # cached (8, H, W) float32
        self._image_key = None   # identifies the cached image
        self._lock = threading.Lock()

    # ---------- availability ----------

    def unavailable_reason(self):
        """None if the encoder can run here, else a short human-readable reason."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "onnxruntime is not installed"
        if not os.path.exists(self._path):
            return (f"embedding model missing ({os.path.basename(self._path)}) "
                    f"-- restore auto_encoder/embedding.onnx")
        return None

    def available(self):
        return self.unavailable_reason() is None

    # ---------- sessions / embeddings ----------

    def _ensure_loaded(self):
        if self._sess is not None:
            return
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(self._path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])

    def set_image(self, raw, key):
        """Cache the embedding stack for a raw (10, H, W) DN array under `key`.

        Recomputes only when `key` changes, so repeated wand clicks on one
        image reuse the encoder pass. `raw` is SentinelImage.raw: the bands
        exactly as rasterio read them, before any normalization -- the graph
        carries its own dataset mean/std. `key` should capture everything
        that changes the pixels (project, image name); unlike the SAM service
        it must NOT carry the display bands, since the encoder consumes every
        band and the display mapping cannot change its output.
        """
        with self._lock:
            self._ensure_loaded()
            if key is not None and key == self._image_key:
                return
            self._image_key = None  # a failed encode must not leave a stale key
            self._feats = self._encode(np.asarray(raw))
            self._image_key = key

    def feature_stack(self):
        """The cached (8, H, W) float32 embedding; set_image() must have run."""
        with self._lock:
            if self._feats is None or self._image_key is None:
                raise RuntimeError("embedding features are not set")
            return self._feats

    def _encode(self, raw):
        """(10, H, W) DN -> (8, H, W) embedding: pad, tile, batch-run, stitch."""
        if raw.ndim != 3:
            raise ValueError(f"expected a (bands, H, W) stack, got shape {raw.shape}")
        bands, height, width = raw.shape
        if bands != INPUT_BANDS:
            raise ValueError(f"the embedding encoder needs {INPUT_BANDS} bands, "
                             f"got {bands}")
        # The graph wants int16 DN. Real Sentinel-2 DN fits; clipping guards a
        # float stack from another sensor wrapping around on the cast.
        dn = np.clip(raw, -32768, 32767).astype(np.int16)
        pad_h = (-height) % PATCH
        pad_w = (-width) % PATCH
        if pad_h or pad_w:
            # Edge padding, not zeros: a zero border would put a fake
            # reflectance cliff inside the encoder's receptive field beside
            # every edge pixel.
            dn = np.pad(dn, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        rows, cols = dn.shape[1] // PATCH, dn.shape[2] // PATCH
        patches = np.stack([dn[:, y * PATCH:(y + 1) * PATCH,
                               x * PATCH:(x + 1) * PATCH]
                            for y in range(rows) for x in range(cols)])
        outs = []
        for i in range(0, len(patches), _MAX_BATCH):
            outs.append(self._sess.run(
                None, {"raw_image": patches[i:i + _MAX_BATCH]})[0])
        out = np.concatenate(outs, axis=0)  # (P, C, PATCH, PATCH)
        channels = out.shape[1]
        feats = np.empty((channels, rows * PATCH, cols * PATCH),
                         dtype=np.float32)
        i = 0
        for y in range(rows):
            for x in range(cols):
                feats[:, y * PATCH:(y + 1) * PATCH,
                      x * PATCH:(x + 1) * PATCH] = out[i]
                i += 1
        return feats[:, :height, :width]  # drop the padding
