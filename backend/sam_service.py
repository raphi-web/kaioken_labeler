"""Efficient-SAM2 "snap-to-edge" click segmentation, running on ONNX Runtime.

The SAM2 image predictor is shipped as two prebuilt ONNX graphs in `sam2/onnx/`
and run here with onnxruntime, so this feature needs neither PyTorch-SAM2 nor
hydra:

  * encoder graph: preprocessed 1x3x1024x1024 image -> image_embed + 2 high-res
    feature maps. Runs once per image (~seconds on CPU); the outputs are cached.
  * decoder graph: those feature maps + clicks (foreground/background) ->
    low-res mask logits + IoU. Runs per click (fast) on the cached embeddings.

Only the trivial resize/normalize and the final mask upscale/threshold happen
outside ONNX; the encoder and decoder network passes are pure ONNX.

Those two steps used torchvision and `torch.nn.functional.interpolate`, which
were the last reason this labeling-only build would have needed PyTorch at all
-- for a resize and a normalize. They are numpy and `scipy.ndimage.zoom` now,
so nothing here imports torch. See `resize_bilinear` for why `grid_mode=True`
is the load-bearing argument.

The graphs in `sam2/onnx/` are checked in and are the only copy: this build has
no exporter, so treat them as source rather than as a build artifact.
"""

import os
import sys
import threading

import numpy as np
from scipy import ndimage

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a project dependency
    cv2 = None

# Source tree: sam2/onnx next to backend/ (see export_sam_onnx.py).
# Frozen app: sam2/onnx next to this executable -- build_app.sh copies it
# there when present, mirroring how it bundles the standalone predictor.
if getattr(sys, "frozen", False):
    _ONNX_DIR = os.path.join(os.path.dirname(sys.executable), "sam2", "onnx")
else:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ONNX_DIR = os.path.join(_PROJECT_ROOT, "sam2", "onnx")

# Model size -> filename stem of its exported encoder/decoder graphs.
_MODELS = {"tiny": "sam2.1_hiera_tiny", "small": "sam2.1_hiera_small"}

# SAM2 preprocessing constants (ImageNet mean/std, matching SAM2Transforms).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# The encoder's three outputs, named by how much detail they keep rather than by
# their position in the graph. The decoder consumes all three; the embedding
# wand (see Api._embedding_field) matches on one of them at a time, which is
# what these names are for.
#
#   fine  high_res_feat0   32 ch at 256x256  -- sharpest, most texture-like
#   mid   high_res_feat1   64 ch at 128x128
#   deep  image_embed     256 ch at  64x64   -- most semantic, coarsest grid
FEATURE_LEVELS = {"fine": 1, "mid": 2, "deep": 0}


def resize_bilinear(arr, out_hw):
    """Bilinear resize over the trailing two axes of a float array.

    Stands in for torchvision's Resize and for
    torch.nn.functional.interpolate(mode="bilinear", align_corners=False).

    `grid_mode=True` is what makes those equivalent: it tells scipy to treat
    samples as unit-width cells centred at i + 0.5, which is exactly
    align_corners=False's convention. Without it scipy aligns the outermost
    sample centres instead, shifting the result by up to half a pixel and
    moving every mask boundary with it. `mode="nearest"` clamps at the edges
    the way torch does, and order=1 is plain bilinear (a spline prefilter would
    be a no-op at that order, so it is switched off).

    SAM2 only ever upscales here (512 -> 1024 for the encoder input, 256 -> 512
    for the mask), so antialiasing does not arise; torchvision's antialias
    default affects downscaling only.
    """
    src_h, src_w = arr.shape[-2:]
    out_h, out_w = out_hw
    src = np.asarray(arr, dtype=np.float32)
    if (src_h, src_w) == (out_h, out_w):
        return src
    factors = [1.0] * (src.ndim - 2) + [out_h / src_h, out_w / src_w]
    return ndimage.zoom(src, factors, order=1, grid_mode=True, mode="nearest",
                        prefilter=False).astype(np.float32, copy=False)


class SamService:
    """Lazy onnxruntime wrapper around the exported SAM2 encoder/decoder."""

    def __init__(self, model_size="tiny"):
        stem = _MODELS[model_size]
        self._encoder_path = os.path.join(_ONNX_DIR, f"{stem}.encoder.onnx")
        self._decoder_path = os.path.join(_ONNX_DIR, f"{stem}.decoder.onnx")
        self._enc = None  # onnxruntime encoder session
        self._dec = None  # onnxruntime decoder session
        self._res = 1024  # encoder input resolution (read from the graph on load)
        self._mask_res = 256  # decoder mask_input side (read from the graph on load)
        self._feats = None  # cached (image_embed, high_res_feat0, high_res_feat1)
        self._orig_hw = None  # (H, W) of the cached image
        self._image_key = None  # identifies the image whose embeddings are cached
        self._lock = threading.Lock()  # serializes load + encode + decode

    # ---------- availability ----------

    def unavailable_reason(self):
        """None if SAM2 can run here, else a short human-readable reason."""
        if cv2 is None:
            return "OpenCV (cv2) is not installed"
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "onnxruntime is not installed"
        try:
            import scipy  # noqa: F401
        except ImportError:
            return "scipy is not installed"
        for path in (self._encoder_path, self._decoder_path):
            if not os.path.exists(path):
                return (f"SAM2 ONNX model missing ({os.path.basename(path)}) "
                        f"from sam2/onnx/ -- restore it from version control")
        return None

    def available(self):
        return self.unavailable_reason() is None

    # ---------- sessions / embeddings ----------

    def _ensure_loaded(self):
        if self._enc is not None:
            return
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # hide the benign Hiera shape-merge warning
        self._enc = ort.InferenceSession(self._encoder_path, sess_options=opts,
                                         providers=providers)
        self._dec = ort.InferenceSession(self._decoder_path, sess_options=opts,
                                         providers=providers)
        shape = self._enc.get_inputs()[0].shape  # [1, 3, R, R]
        if isinstance(shape[-1], int):
            self._res = shape[-1]
        # Side of the decoder's mask_input grid (its own low-res logits, fed
        # back on refinement); read from the graph so a re-export can change it.
        mask_shape = {i.name: i.shape for i in self._dec.get_inputs()}["mask_input"]
        if isinstance(mask_shape[-1], int):
            self._mask_res = mask_shape[-1]

    def _preprocess(self, rgb):
        """HWC uint8 RGB -> 1x3xRxR float32, matching SAM2Transforms.

        The three steps are torchvision's ToTensor (scale to [0,1] and move the
        channel axis first), Resize, then Normalize, in that order -- resizing
        before normalizing, as SAM2 does.
        """
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        resized = resize_bilinear(chw, (self._res, self._res))
        return ((resized - _MEAN) / _STD)[np.newaxis]

    def set_image(self, rgb, key):
        """Cache encoder embeddings for an (H, W, 3) uint8 RGB image under `key`.

        Recomputes only when `key` changes, so repeated clicks on the same image
        reuse the encoder pass. `key` should capture everything that changes the
        pixels (project, image name, display bands).
        """
        with self._lock:
            self._ensure_loaded()
            if key is not None and key == self._image_key:
                return
            self._image_key = None  # a failed encode must not leave a stale key
            inp = self._preprocess(np.ascontiguousarray(rgb))
            image_embed, hrf0, hrf1 = self._enc.run(None, {"image": inp})
            self._feats = (image_embed, hrf0, hrf1)
            self._orig_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
            self._image_key = key

    def feature_stack(self, level="fine"):
        """One cached encoder feature map as (C, h, w) float32.

        The same embeddings the decoder segments from, handed out raw so a
        caller can measure distances in feature space instead of asking for a
        mask. set_image() must have been called first; the batch axis is
        dropped since there is only ever one image in it.
        """
        if level not in FEATURE_LEVELS:
            raise ValueError(f"unknown feature level: {level}")
        with self._lock:
            if self._feats is None or self._image_key is None:
                raise RuntimeError("SAM2 image embeddings are not set")
            return self._feats[FEATURE_LEVELS[level]][0]

    def predict_points(self, points):
        """Object mask (bool H×W) for a list of clicks.

        `points` is [(x, y, label), ...] in image pixels, label 1 = foreground
        ("include this"), 0 = background ("exclude this"). set_image() must have
        been called first. Returns (mask, score).

        Reproduces SAM2's interactive refinement, which is a *chain*: the click
        list is replayed one click at a time and each pass is given the previous
        pass's low-res logits as `mask_input`. Both details matter, and neither
        is optional —

          * **which mask token.** A lone foreground click is ambiguous (roof,
            house, or block?), so the three hypothesis tokens 1-3 compete and
            the best predicted IoU wins. From the second click on, the intent is
            pinned down and the user is refining, so token 0 (SAM2's dedicated
            single-mask output) is used.
          * **feeding the mask back.** Without `mask_input` a background click
            only nudges the mask and need not even exclude the pixel that was
            clicked; with it the mask actually retreats. Measured on a real
            scene: 19,642 px -> 18,585 px with the click still inside the mask,
            versus -> 4,910 px and properly excluded once fed back.

        Replaying the whole chain per call keeps this stateless, so the frontend
        can drop or re-order points and always get the same mask for the same
        click list. The decoder pass is small, and click counts are single
        digits in practice.
        """
        if not points:
            raise ValueError("at least one point is required")
        with self._lock:
            if self._feats is None or self._image_key is None:
                raise RuntimeError("SAM2 image embeddings are not set")
            height, width = self._orig_hw
            image_embed, hrf0, hrf1 = self._feats
            prev, score = None, 0.0
            for count in range(1, len(points) + 1):
                chain = points[:count]
                # Clicks in the encoder's input frame (transform_coords).
                coords = np.array(
                    [[[x / width * self._res, y / height * self._res]
                      for x, y, _ in chain]], dtype=np.float32)
                labels = np.array([[int(lab) for _, _, lab in chain]], dtype=np.int64)
                low_res, iou = self._dec.run(None, {
                    "image_embed": image_embed,
                    "high_res_feat0": hrf0,
                    "high_res_feat1": hrf1,
                    "point_coords": coords,
                    "point_labels": labels,
                    "mask_input": (prev if prev is not None else
                                   np.zeros((1, 1, self._mask_res, self._mask_res),
                                            dtype=np.float32)),
                    "has_mask_input": np.array(
                        [0.0 if prev is None else 1.0], dtype=np.float32),
                })
                token = 1 + int(iou[0, 1:].argmax()) if count == 1 else 0
                prev = low_res[:, token:token + 1]
                score = float(iou[0, token])
            # Upscale the final low-res logits to the original size and threshold
            # at 0, exactly like SAM2's postprocess_masks (hole/sprinkle areas 0).
            up = resize_bilinear(prev[0, 0], (height, width))
            return up > 0.0, score


def mask_to_polygons(mask, min_area=16, epsilon_frac=0.004):
    """External contours of a boolean mask as simplified [[x, y], ...] polygons.

    Coordinates are in image pixels. Tiny specks below `min_area` are dropped;
    each contour is simplified with Douglas-Peucker at `epsilon_frac` of its
    perimeter so the vertex list stays small while tracking the edge.
    """
    m = (np.asarray(mask, dtype=np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        eps = epsilon_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons
