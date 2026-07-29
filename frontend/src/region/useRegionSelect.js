import { useCallback, useEffect, useRef, useState } from 'react'
import { drawWandPreview } from '../wand/preview'
import { connectedRegion, countMask } from './connected'

// Pick whole labeled objects and delete them.
//
// Deliberately shaped like useWand from the outside -- click to add, Backspace
// to drop the last, Enter to commit, Escape to clear -- because those are the
// keys the two other click-driven tools already use, and a third dialect would
// be a third thing to remember. Inside it is much smaller: the selection is
// computed locally from the label buffer, so there is no request to sequence,
// nothing to be stale, and no busy state.
//
// Marked for removal in the app's danger red rather than the class colour: the
// point of the highlight is that these pixels are about to stop being that
// class, and showing them in their own colour would say the opposite.
const REMOVE_COLOR = [229, 72, 77]

export function useRegionSelect({ active, image, labels, labelsVersion, onDelete }) {
  // One entry per click: the pixel that was clicked and the region it found.
  const [regions, setRegions] = useState([])
  const regionsRef = useRef(regions)
  const canvasRef = useRef(null)

  const latest = useRef(null)
  latest.current = { image, labels, onDelete }

  // The union of every selected region, rebuilt from the list so that dropping
  // one cannot leave its pixels behind in a shared accumulator.
  const paint = useCallback((list) => {
    const { image: img } = latest.current
    if (!img) return
    let union = null
    if (list.length) {
      union = new Uint8Array(img.width * img.height)
      for (const r of list) {
        for (let i = 0; i < union.length; i++) if (r.mask[i]) union[i] = 1
      }
    }
    drawWandPreview(canvasRef.current, union, img, REMOVE_COLOR)
    return union
  }, [])

  const apply = useCallback((list) => {
    regionsRef.current = list
    setRegions(list)
    paint(list)
  }, [paint])

  /** Add the object under (x, y). No-op on unlabeled ground or a repeat click. */
  const addAt = useCallback((pos) => {
    const { image: img, labels: buffer } = latest.current
    if (!img || !buffer) return
    const x = Math.min(img.width - 1, Math.max(0, Math.floor(pos.x)))
    const y = Math.min(img.height - 1, Math.max(0, Math.floor(pos.y)))
    const start = y * img.width + x
    // A second click inside a region already selected would add a duplicate
    // that Backspace then has to remove twice.
    if (regionsRef.current.some((r) => r.mask[start])) return
    const mask = connectedRegion(buffer, img.width, img.height, start)
    if (!mask) return
    apply([...regionsRef.current, { x, y, mask, count: countMask(mask) }])
  }, [apply])

  const undo = useCallback(() => {
    if (regionsRef.current.length) apply(regionsRef.current.slice(0, -1))
  }, [apply])

  const cancel = useCallback(() => {
    if (regionsRef.current.length) apply([])
  }, [apply])

  /** Delete every selected region as ONE undoable step. */
  const commit = useCallback(() => {
    const list = regionsRef.current
    if (!list.length) return
    const union = paint(list)
    apply([])
    latest.current.onDelete(union)
  }, [apply, paint])

  // Leaving the tool drops the selection rather than leaving an invisible one
  // for a stray Enter to act on later.
  useEffect(() => {
    if (!active) cancel()
  }, [active, cancel])

  // The labels can change underneath a pending selection -- an undo, a stroke
  // with another tool, a different image. Anything already selected may no
  // longer be there, so the safe reading is that the selection is stale.
  //
  // Keyed on the version counter and the image NAME, not on the identity of
  // the buffer or the image object: those are recreated by renders that
  // changed nothing about the labels, and watching them threw the selection
  // away on the very render that had just added to it.
  useEffect(() => {
    cancel()
  }, [labelsVersion, image?.name, cancel])

  return {
    canvasRef,
    regions,
    count: regions.reduce((sum, r) => sum + r.count, 0),
    addAt,
    undo,
    cancel,
    commit,
  }
}
