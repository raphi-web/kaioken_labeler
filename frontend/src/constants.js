export const UNLABELED = 255

// Fallback palette for the no-project (standalone image) case; an open
// project's own classes (from get_project) take precedence everywhere else.
// Mirrors project.CLASS_PRESETS['sentinel2'], which is what a new project is
// created with -- a standalone image opened for a quick look should not show a
// different label space from the one it would get as a project.
export const CLASSES = [
  { id: 0, name: 'Urban', color: '#e05a4f' },
  { id: 1, name: 'Agriculture-Green', color: '#3ddc6b' },
  { id: 2, name: 'Raw-Earth/Agriculture', color: '#c98f3c' },
  { id: 3, name: 'Meadow', color: '#d4e34a' },
  { id: 4, name: 'Forest', color: '#14562b' },
  { id: 5, name: 'Shrub', color: '#9ac48d' },
  { id: 6, name: 'Water', color: '#2f7ce0' },
  { id: 7, name: 'Snow', color: '#eef2f7' },
  { id: 8, name: 'Wetland', color: '#9b5fd0' },
  { id: 9, name: 'Synthetic Nodata', color: '#ff2fd0' },
]

export function hexToRgb(hex) {
  const h = hex.replace('#', '')
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16))
}

// Alpha of a painted label pixel when the overlay slider sits at 100%, out of
// 255. Fully opaque, so the mask actually covers the imagery and the slider's
// readout means what it says -- at 180 the labels stayed ~29% see-through even
// with the slider maxed. Drag the slider down to check the image underneath.
export const LABEL_ALPHA = 255

// Brush radius bounds, in image pixels. The slider, the numeric box and the
// scroll-wheel step all clamp through clampBrush, so the three controls cannot
// disagree about the range.
export const BRUSH_MIN = 1
export const BRUSH_MAX = 40

export function clampBrush(value) {
  // Empty first: Number('') is 0, not NaN, so without this an emptied input
  // would snap the brush to BRUSH_MIN mid-edit instead of leaving it alone.
  if (typeof value === 'string' && value.trim() === '') return null
  const n = Math.round(Number(value))
  if (!Number.isFinite(n)) return null
  return Math.min(BRUSH_MAX, Math.max(BRUSH_MIN, n))
}

// Keys that select a class, in class-list order: 1-9, then 0, then '-'. Enough
// for the 11 classes this fork labels; beyond that the extra classes are
// click-only, which is the right failure mode -- no key is silently reassigned
// to a class other than the one printed next to it.
export const CLASS_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-']
export const ERASER_KEY = 'e'

export function classKeyLabel(index) {
  return CLASS_KEYS[index] ?? ''
}

export function classIdForKey(key, classes) {
  const index = CLASS_KEYS.indexOf(key)
  return index >= 0 && index < classes.length ? classes[index].id : null
}
