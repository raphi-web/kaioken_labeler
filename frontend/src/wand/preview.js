import {
  WAND_EDGE_ALPHA, WAND_EDGE_DARK, WAND_EDGE_LIGHT, WAND_PREVIEW_ALPHA,
} from './constants'

// Paint the pending selection onto its own canvas. Pure pixel work: it reads
// no React state and holds none, which is why it lives outside the hook -- the
// two nested loops below are the hot part of every slider step and are easier
// to reason about with nothing else in the frame.
//
// `mask` is one byte per pixel (1 = selected) or null to clear.
export function drawWandPreview(canvas, mask, image, color) {
  if (!canvas || !image) return
  const ctx = canvas.getContext('2d')
  const { width, height } = image
  ctx.clearRect(0, 0, width, height)
  if (!mask) return
  const [r, g, b] = color
  const imgData = ctx.createImageData(width, height)
  const px = imgData.data

  // Fill first, edge second, so the outline sits on top of the fill rather
  // than being blended into it.
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    const o = i * 4
    px[o] = r
    px[o + 1] = g
    px[o + 2] = b
    px[o + 3] = WAND_PREVIEW_ALPHA
  }

  // Two-tone boundary: the outermost selected pixels light, the ring of
  // unselected pixels just outside them dark.
  //
  // One tone is not enough -- a light outline disappears on Snow, a dark one
  // disappears in shadow -- and the fill alone is nearly invisible whenever
  // the class colour resembles the imagery underneath, which is the whole
  // reason this exists.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const inside = mask[i] === 1
      // 4-connected, matching how the selection itself is grown.
      const up = y > 0 && mask[i - width] === 1
      const down = y < height - 1 && mask[i + width] === 1
      const left = x > 0 && mask[i - 1] === 1
      const right = x < width - 1 && mask[i + 1] === 1
      const touches = up || down || left || right
      const o = i * 4
      if (inside) {
        // An edge pixel is one missing a neighbour -- including at the image
        // border, so a selection running off the edge still reads as bounded.
        const open = !up || !down || !left || !right
        if (!open) continue
        px[o] = px[o + 1] = px[o + 2] = WAND_EDGE_LIGHT
        px[o + 3] = WAND_EDGE_ALPHA
      } else if (touches) {
        px[o] = px[o + 1] = px[o + 2] = WAND_EDGE_DARK
        px[o + 3] = WAND_EDGE_ALPHA
      }
    }
  }
  ctx.putImageData(imgData, 0, 0)
}
