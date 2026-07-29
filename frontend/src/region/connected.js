import { UNLABELED } from '../constants'

// The connected component of one label, as a mask.
//
// This is what makes the Region tool different from "select everything of this
// class": a click picks out the object under the cursor, not every pixel that
// happens to share its id. On this dataset that distinction is the whole point
// -- Forest in 00.tiff is 63 disconnected pieces, and deleting one speck should
// not take the other 62.
//
// Runs on the frontend's own label buffer, so there is no round trip: unlike
// the wand, which needs raster bands the frontend never sees, everything this
// needs is already here.

/**
 * The 4-connected region of pixels sharing labels[start]'s value.
 *
 * Returns a Uint8Array mask (1 = in the region), or null when the clicked
 * pixel carries no label -- there is no object there to select.
 *
 * 4-connected, not 8, matching the wand's fill and the backend's
 * _WAND_CONNECTIVITY. With 8, two regions touching at a single diagonal pixel
 * are one region, and a click deletes across a join too small to see at the
 * zoom you clicked at.
 */
export function connectedRegion(labels, width, height, start) {
  const target = labels[start]
  if (target === undefined || target === UNLABELED) return null

  const mask = new Uint8Array(labels.length)
  // A typed-array ring rather than an Array used as a queue: the worst case is
  // every pixel of a 512x512 tile, and shift() on a 262k-element Array is not
  // the thing to do 262k times.
  const queue = new Int32Array(labels.length)
  let head = 0
  let tail = 0

  queue[tail++] = start
  mask[start] = 1

  while (head < tail) {
    const i = queue[head++]
    const x = i % width
    const y = (i / width) | 0

    if (x > 0 && !mask[i - 1] && labels[i - 1] === target) {
      mask[i - 1] = 1
      queue[tail++] = i - 1
    }
    if (x < width - 1 && !mask[i + 1] && labels[i + 1] === target) {
      mask[i + 1] = 1
      queue[tail++] = i + 1
    }
    if (y > 0 && !mask[i - width] && labels[i - width] === target) {
      mask[i - width] = 1
      queue[tail++] = i - width
    }
    if (y < height - 1 && !mask[i + width] && labels[i + width] === target) {
      mask[i + width] = 1
      queue[tail++] = i + width
    }
  }
  return mask
}

/** How many pixels a mask holds. */
export function countMask(mask) {
  let n = 0
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++
  return n
}
