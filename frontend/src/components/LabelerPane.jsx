import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LABEL_ALPHA, UNLABELED } from '../constants'
import Viewport from './Viewport'

// Chromium (and Qt WebEngine) ignores custom CSS cursors larger than 128px, so
// the native brush cursor is only used up to this on-screen diameter; beyond it
// the DOM circle takes over.
const MAX_CURSOR_PX = 128
// Alpha of the pending wand selection, well below the committed label alpha
// (255) so a preview never reads as something already painted, and light
// enough that the imagery stays legible underneath while you judge the fit.
const WAND_PREVIEW_ALPHA = 102 // 40% of 255
// The two tones of the selection outline. Near-white inside, near-black just
// outside: one tone always loses against some class colour or some terrain,
// the pair reads on both.
//
// Same alpha as the fill, so the edge is distinguished by its tone rather than
// by being heavier -- the near-white/near-black pair carries far more contrast
// against imagery than the class colour does, which is what makes the boundary
// findable without the outline dominating the view.
const WAND_EDGE_LIGHT = 255
const WAND_EDGE_DARK = 20
const WAND_EDGE_ALPHA = 102 // 40% of 255, matching the fill
// Past this share of the image the selection is called out as a runaway.
const WAND_WARN_FRACTION = 0.4

// Wand tolerance, as an integer slider. The backend wants an RMS per-band
// difference in normalized [0,1] units, which is an awkward 0.13-shaped number
// to put on a slider, so the UI carries 1..100 and divides.
//
// Measured over 45 labeled regions across five images: the tolerance that
// maximises IoU sits at p10 0.084, median 0.130, p90 0.280. 26 (= 0.130) is
// therefore the default, and the useful band 17..56 lands mid-travel.
//
// Re-measured since over 388 regions across 20 images, sweeping tolerance and
// sample size jointly and scoring the best *fixed* setting rather than the
// median of the per-region optima: the peak is 0.14 at mean IoU 0.349 against
// 0.347 for 0.13, and the curve is flat from 0.11 to 0.15. That is noise, so
// 26 stands. Sample size is likewise inert here (0.343 to 0.349 across 1..25);
// it is set from click-to-click agreement instead, for the reasons below.
//
// What actually pins the default is not the peak but the cliff to the right of
// it. Median over-selection runs 6% at 0.13, 25% at 0.15, 118% at 0.17 and
// 244% at 0.19, while the IoU falls away slowly -- so the two directions cost
// different things. A wand click is additive, so selecting too little costs
// another click, and selecting too much costs an undo and hand correction.
// The default belongs on the low side of the peak, which is where it is.
//
// Note this is one number serving nine classes whose own optima span 0.07 to
// 0.19 -- which is what WAND_TOL_SEEDS below does something about.
const WAND_TOL_MIN = 1
const WAND_TOL_MAX = 100
const WAND_TOL_SCALE = 200 // slider value / SCALE = the backend's tolerance
const WAND_TOL_DEFAULT = 26 // = 0.130, for a class with no measured optimum

// Starting tolerance per class, in slider units. One global number is a
// compromise between materials that want very different thresholds, and from
// the same 388-region sweep the per-class optima are consistent enough to seed
// from (p25/median/p75 of each class's per-region optimum in brackets):
//
//   Urban        0.19  (.17/.24/.32)   +0.069 IoU over the global 0.13
//   Raw-Earth    0.14  (.10/.15/.22)   +0.008   -- the most common class, and
//   Road         0.12  (.11/.13/.21)   +0.002      the reason 0.13 is the
//   Forest       0.11  (.10/.13/.18)   +0.023      global compromise
//   Wetland      0.11  (.07/.10/.16)   +0.002
//   Shrub        0.10  (.10/.11/.13)   +0.108
//   Meadow       0.09  (.09/.11/.14)   +0.130
//   Agri-Green   0.09  (.09/.10/.14)   +0.067
//   Water        0.07  (.07/.13/.27)   +0.019
//
// Worth +12% mean IoU over one global tolerance (0.353 -> 0.397); the ceiling
// if you retuned per region is 0.585, which no default can reach.
//
// Keyed by class NAME, not id: these are properties of the material, and ids
// are per project -- seeding by id would hand Urban's threshold to whatever a
// different project happens to call class 0. A class with no entry (Snow here)
// starts at WAND_TOL_DEFAULT. Thin samples: Shrub n=17, Water n=18, Wetland
// n=4, so those three are the ones to distrust.
const WAND_TOL_SEEDS = {
  urban: 38,
  rawearthagriculture: 28,
  road: 24,
  forest: 22,
  wetland: 22,
  shrub: 20,
  meadow: 18,
  agriculturegreen: 18,
  water: 14,
}

function tolSeedKey(name) {
  return String(name ?? '').toLowerCase().replace(/[^a-z0-9]/g, '')
}

// With the SAM-features switch on, the wand measures a cosine distance
// between mean-centred SAM2 feature vectors (see Api._sam_features), which has
// nothing to do with an RMS band difference -- so the slider stays 1..100 and
// maps onto its own range.
//
// Not a bare divide, because the useful distances do not start at zero. On a
// 512px Sentinel-2 tile at the fine level, three clicks each selected under
// 0.15% of the image below 0.7, 0.6-2.3% at 0.8, and 7-30% by 1.0: a region
// at 0.7-0.95, a speck below, the scene above. Mapping the slider onto
// [0.4, 1.2] puts that band across the middle of the travel and still leaves
// the deep level's lower band (0.5-0.8) inside it. A plain /100 left the whole
// working range pinned to the right-hand fifth of the slider.
const SAM_TOL_FLOOR = 0.4
const SAM_TOL_CEIL = 1.2
const SAM_TOL_DEFAULT = 50 // = 0.8, mid-band for the fine level

function samTolerance(slider) {
  return SAM_TOL_FLOOR + (slider / WAND_TOL_MAX) * (SAM_TOL_CEIL - SAM_TOL_FLOOR)
}

// Which encoder feature map to match on (see sam_service.FEATURE_LEVELS). The
// grid sizes are for the encoder's fixed 1024² input frame, so on a 512px tile
// "fine" is one cell per 2x2 pixels and "deep" one per 8x8.
//
// "fine" is the default: on Sentinel-2 it is the only level that produces
// selections worth committing, established by labeling with it rather than by
// measurement.
//
// It is worth recording that the obvious measurement says the opposite, and
// why it is the wrong measurement. Region growing needs a click's neighbours
// to resemble it, and by that test fine looks hopeless -- the median distance
// within 10 px of a click is 1.10 against a whole-image median of 1.02, so the
// ground beside a click is as unlike it as the far side of the scene, while at
// deep the same numbers are 0.41 against 0.98. But growing smoothly is not the
// same as landing on the thing you meant: one deep cell covers 8x8 px, which
// at 10 m per pixel is 80 m of ground, so its selections are the right shape
// at the wrong scale -- and Sentinel-2 classes are often a few pixels across.
// Fine grows grudgingly and needs most of the tolerance range, but its
// boundaries are where the material actually changes.
//
// The likely reason is that SAM2 learned its semantics on photographs, where a
// deep feature covers one object; a 10 m/px tile has nothing at that scale, so
// only the low-level texture features carry usable signal. If so, the ordering
// should flip on higher-resolution imagery -- which is exactly why all three
// levels stay selectable instead of the loser being deleted.
const SAM_LEVELS = [
  { id: 'fine', label: 'fine',
    title: 'high_res_feat0 — 32 channels at 256². Texture-level features and '
      + 'the sharpest boundaries. The one that works on Sentinel-2; needs a '
      + 'high tolerance (around 0.8) before it selects a region at all' },
  { id: 'mid', label: 'mid',
    title: 'high_res_feat1 — 64 channels at 128², between the two' },
  { id: 'deep', label: 'deep',
    title: 'image_embed — 256 channels at 64², what the SAM2 decoder segments '
      + 'from. Grows the most smoothly with the tolerance, but one cell is '
      + '8x8 px — too coarse for Sentinel-2 boundaries. Worth retrying on '
      + 'higher-resolution imagery' },
]

// Optional budget on how many pixels the wand may select. The slider runs
// left (tight) to right (loose) and its maximum means "no limit", so sliding
// right always widens the selection and the top of the range is simply the
// unbounded behaviour -- no separate toggle to forget. Off by default:
// connectivity already bounds most fills, and a cap nobody asked for would
// silently truncate a legitimately large region.
//
// Log-scaled, because useful budgets span three orders of magnitude: a road
// edge is a few hundred pixels and a field is tens of thousands, and a linear
// slider would spend nearly all its travel in the range you never want.
// How many pixels around each click feed that sample's reference colour. A
// one-pixel sample is a sample of size one, and on textured ground the clicked
// pixel is easily an outlier its own region fails to match. Nearest-first, so
// N always means the N pixels closest to the click; 25 is the full 5x5 window,
// which is the backend's ceiling.
//
// Default 9 (a 3x3). Measured over 28 regions, clicking six different pixels
// inside each at a fixed tolerance: going from 1 to 25 px raises click-to-click
// agreement 0.266 -> 0.288, cuts the spread in selected size from 25.6x to
// 17.3x, and lifts accuracy 0.218 -> 0.233. The gains flatten after ~9, and a
// wider window is likelier to straddle a boundary -- at 10 m pixels, 5x5 is
// 50 m of ground -- so 9 takes most of the benefit at half the blur.
//
// Note this only shows up at a FIXED tolerance. Sweeping tolerance per click
// reverses the ranking (1 px scores best), because a sweep can retune around a
// noisy reference after the fact -- which is not something anyone does while
// labeling.
const WAND_SAMPLE_MIN = 1
const WAND_SAMPLE_MAX = 25

const WAND_BUDGET_MIN = 1
const WAND_BUDGET_MAX = 100 // at the max the budget is lifted entirely

function wandBudget(slider) {
  if (slider >= WAND_BUDGET_MAX) return null // unlimited
  const raw = 10 ** (2 + (3 * (slider - 1)) / (WAND_BUDGET_MAX - 2))
  // Two significant figures: the exact value of a log slider is noise, and a
  // readout that jitters between 3,162 and 3,271 is harder to aim with.
  const mag = 10 ** (Math.floor(Math.log10(raw)) - 1)
  return Math.round(raw / mag) * mag
}

const CURSOR_PAD = 2 // room around the circle for its 1.5px stroke
const CROSS_ARM = 8 // half-length of each center crosshair arm, in px

// SAM2 click markers, matching the app's --accent / --danger (see App.css).
// Deliberately independent of CLASS_COLORS: these mark prompt points, not
// labeled pixels, and must stay readable whichever class is being painted.
const SAM_INCLUDE_COLOR = [242, 111, 181] // pink
const SAM_EXCLUDE_COLOR = [229, 72, 77]   // red

// Stroke the center crosshair ("+") marking the exact painted pixel. Uses the
// current strokeStyle/lineWidth so it matches whatever it's drawn over.
function strokeCrosshair(ctx, cx, cy) {
  ctx.beginPath()
  ctx.moveTo(cx - CROSS_ARM, cy)
  ctx.lineTo(cx + CROSS_ARM, cy)
  ctx.moveTo(cx, cy - CROSS_ARM)
  ctx.lineTo(cx, cy + CROSS_ARM)
  ctx.stroke()
}

// Turn an offscreen canvas into a CSS cursor value centered on the pointer.
function canvasCursor(canvas) {
  const hot = Math.round(canvas.width / 2)
  return `url(${canvas.toDataURL()}) ${hot} ${hot}, crosshair`
}

// Paint the brush preview into an offscreen canvas and return it as a cursor
// value: `url(<png>) <hotspot> <hotspot>, crosshair`, centered on the pointer.
// Returns null when the image would exceed the browser's cursor size cap.
function makeBrushCursor(diameter, color) {
  const size = Math.ceil(Math.max(diameter, CROSS_ARM * 2)) + CURSOR_PAD * 2
  if (size > MAX_CURSOR_PX) return null
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const c = size / 2
  ctx.beginPath()
  ctx.arc(c, c, diameter / 2, 0, Math.PI * 2)
  ctx.fillStyle = `rgba(${color.join(',')}, 0.2)`
  ctx.fill()
  ctx.lineWidth = 1.5
  ctx.strokeStyle = `rgba(${color.join(',')}, 0.9)`
  ctx.stroke()
  strokeCrosshair(ctx, c, c)
  return canvasCursor(canvas)
}

// Crosshair-only cursor for oversized brushes, where the big circle is drawn by
// the DOM `.brush-cursor` div. Rendered with the exact same canvas code as the
// native brush cursor's crosshair, so the two are pixel-identical.
function makeCrosshairCursor(color) {
  const size = CROSS_ARM * 2 + CURSOR_PAD * 2
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const c = size / 2
  ctx.lineWidth = 1.5
  ctx.strokeStyle = `rgba(${color.join(',')}, 0.9)`
  strokeCrosshair(ctx, c, c)
  return canvasCursor(canvas)
}

// Right pane: base RGB image with a brush-paintable label canvas on top.
// Labels live in a Uint8Array (values 0/1/UNLABELED); the canvas is a
// colorized rendering of that array, redrawn per stroke segment.
// Left button paints (brush mode) or places polygon vertices (polygon mode);
// middle-drag pans (handled by the surrounding Viewport).
// Every stroke/fill reports a Map<pixelIndex, previousValue> via onDiff so
// the app can undo it.
export default function LabelerPane({
  image, labels, labelsVersion, tool, view, setView, onStrokeEnd, onDiff,
  onSamSnap, onWandSelect, classColors, classes, fitRef, samAvailable,
}) {
  const canvasRef = useRef(null)
  const [canvasReady, setCanvasReady] = useState(false)
  const cursorRef = useRef(null)
  const drawing = useRef(false)
  const lastPos = useRef(null)
  const strokeDiff = useRef(null)
  const [opacity, setOpacity] = useState(1)
  const [samBusy, setSamBusy] = useState(false) // a SAM2 snap request is in flight
  const [verts, setVerts] = useState([]) // polygon vertices in image coords
  const vertsRef = useRef(verts)
  vertsRef.current = verts
  // Pending SAM2 selection: the clicks so far ({x, y, label}, label 1 =
  // include / 0 = exclude) and the mask they currently produce, previewed as
  // polygons. Nothing touches the labels until the user commits, which is what
  // makes an exclude click able to *shrink* the selection — the mask is
  // re-predicted from the whole click list rather than painted per click.
  const [samPoints, setSamPoints] = useState([])
  const [samPolys, setSamPolys] = useState([])
  const samPointsRef = useRef(samPoints)
  samPointsRef.current = samPoints
  const samPolysRef = useRef(samPolys)
  samPolysRef.current = samPolys
  const samSeq = useRef(0) // drops out-of-order responses from stale clicks
  const settings = useRef(tool)
  settings.current = tool
  const closeRef = useRef(() => {}) // fresh closePolygon for the window key listener
  const samCommitRef = useRef(() => {}) // ditto for the SAM2 selection
  const samUndoRef = useRef(() => {})
  const samCancelRef = useRef(() => {})

  // Magic wand: a pending selection previewed on its own canvas and committed
  // with Enter, like the polygon and SAM2 tools. The seed click is kept so
  // dragging the tolerance re-runs the fill from the same point, instead of
  // making you re-click to try a different threshold.
  // See WAND_TOL_* above for how the defaults were measured.
  //
  // The tolerance travels with the class: what counts as "the same material"
  // is a property of the material, and the per-class optima span 0.07 to 0.19.
  // This holds only the values you have changed; anything untouched falls
  // through to that class's measured seed. So the slider starts somewhere
  // sensible for each class and then remembers what you actually preferred.
  const [wandTolerances, setWandTolerances] = useState({})
  // Measure similarity on SAM2 encoder features instead of the image bands.
  // A switch on the wand rather than a second wand in the toolbar: measured
  // over 60 labeled regions the two score the same at their own best
  // tolerances (mean IoU 0.593 spectral, 0.568 embedding, 27/60 regions won),
  // so it is not a better tool -- but they fail on *different* regions
  // (correlation 0.538; taking the better of the two per region would score
  // 0.665) and the split runs by class. That is worth a checkbox and is not
  // worth a second set of everything.
  const [wandSam, setWandSam] = useState(false)
  const wandSamRef = useRef(wandSam)
  wandSamRef.current = wandSam
  // Its own tolerance, read through `tolerance` below: the two are not in the
  // same units, so carrying one slider's value across the switch would
  // silently mean something else.
  const [wandSamTolerance, setWandSamTolerance] = useState(SAM_TOL_DEFAULT)
  const [wandLevel, setWandLevel] = useState('fine')
  const wandLevelRef = useRef(wandLevel)
  wandLevelRef.current = wandLevel
  const [wandBudgetStep, setWandBudgetStep] = useState(WAND_BUDGET_MAX) // max = off
  const [wandSampleSize, setWandSampleSize] = useState(9)
  // On by default: this dataset arrives ~75% labeled, and an unprotected fill
  // covers 35% already-labeled pixels across several classes. Overwriting
  // finished work silently is the worse failure, so the safe mode is the one
  // you have to switch off.
  const [wandProtect, setWandProtect] = useState(true)
  // Off by default: connectivity is the safer reading of "this region", and a
  // global click at a loose tolerance can take most of the scene.
  const [wandGlobal, setWandGlobal] = useState(false)
  const [wandBusy, setWandBusy] = useState(false)
  const wandSeq = useRef(0) // drops out-of-order responses, like samSeq
  const [wandCount, setWandCount] = useState(null) // selected px; null = nothing pending
  const [wandCapped, setWandCapped] = useState(null) // {available} when the budget bit
  const [wandProtected, setWandProtected] = useState(0) // px withheld by protect
  // Every click adds a sample. Each matches on its own colour and grows from
  // its own seed with its own budget, and the results union -- so a click is
  // purely additive: it can only ever extend the selection, never redraw it.
  // [{x, y, label}] in image coords, label 1 = include / 0 = exclude, matching
  // the convention SAM2's clicks already use in this pane.
  const [wandSamples, setWandSamples] = useState([])
  const wandSamplesRef = useRef(wandSamples)
  wandSamplesRef.current = wandSamples
  const wandMaskRef = useRef(null) // Uint8Array(W*H), 1 = selected
  const wandCanvasRef = useRef(null)
  const wandCommitRef = useRef(() => {})
  const wandCancelRef = useRef(() => {})
  const wandUndoRef = useRef(() => {})
  const wandRerunRef = useRef(() => {}) // re-runs the current samples, for the class switch

  // Which tolerance every handler below reads and writes. Declared here rather
  // than beside polygonMode/samMode so the wand's own functions can close over
  // it. The embedding source keeps one tolerance for every class: its own
  // per-class figures have not been measured, and inventing seeds for it would
  // dress a guess up as the spectral wand's evidence.
  const classSeed = useMemo(() => {
    const byId = {}
    for (const c of classes ?? []) {
      const seed = WAND_TOL_SEEDS[tolSeedKey(c.name)]
      if (seed !== undefined) byId[c.id] = seed
    }
    return byId
  }, [classes])
  const tolerance = wandSam
    ? wandSamTolerance
    : (wandTolerances[tool.classId] ?? classSeed[tool.classId] ?? WAND_TOL_DEFAULT)

  // Callback ref: the canvas only mounts once the Viewport has a `view`
  // (see Viewport), which lands after image/labels are already set. Track the
  // node's presence so the redraw effect can paint the persisted mask the
  // moment the canvas appears, rather than waiting for the first stroke.
  const setCanvas = useCallback((node) => {
    canvasRef.current = node
    setCanvasReady(!!node)
  }, [])

  useEffect(() => {
    if (image && canvasReady) redraw()
  }, [image, labels, labelsVersion, canvasReady, opacity, classColors])

  // Leaving polygon mode discards an unfinished polygon.
  useEffect(() => {
    if (tool.mode !== 'polygon') setVerts([])
    // Leaving SAM2 mode drops an uncommitted selection rather than leaving an
    // invisible one to be committed later by a stray Enter.
    if (tool.mode !== 'sam') samCancelRef.current()
    if (tool.mode !== 'wand') wandCancelRef.current()
  }, [tool.mode])

  // Changing class now changes the tolerance with it, so a selection already on
  // screen has to be recomputed -- otherwise the preview is the one the
  // previous class's threshold produced while the slider reads the new class's
  // number, and Enter would commit a mask that matches neither. Skipped when
  // nothing is pending, and when the embedding source is on (it keeps a single
  // tolerance, so a class switch changes nothing about the selection).
  useEffect(() => {
    if (!wandSamRef.current && wandSamplesRef.current.length) {
      wandRerunRef.current()
    }
  }, [tool.classId])

  // Enter commits the pending shape (polygon fill / SAM2 selection), Escape
  // cancels it; in SAM2 mode Backspace drops just the last click.
  useEffect(() => {
    function onKey(e) {
      const mode = settings.current.mode
      if (mode === 'polygon') {
        if (e.key === 'Enter') closeRef.current()
        else if (e.key === 'Escape') setVerts([])
      } else if (mode === 'wand') {
        if (e.key === 'Enter') wandCommitRef.current()
        else if (e.key === 'Escape') wandCancelRef.current()
        else if (e.key === 'Backspace') {
          e.preventDefault() // otherwise the webview treats it as "go back"
          wandUndoRef.current()
        }
      } else if (mode === 'sam') {
        if (e.key === 'Enter') samCommitRef.current()
        else if (e.key === 'Escape') samCancelRef.current()
        else if (e.key === 'Backspace') {
          e.preventDefault() // otherwise the webview treats it as "go back"
          samUndoRef.current()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function redraw() {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const { width, height } = image
    const imgData = ctx.createImageData(width, height)
    for (let i = 0; i < labels.length; i++) {
      const v = labels[i]
      if (!(v in classColors)) continue
      const [r, g, b] = classColors[v]
      const o = i * 4
      imgData.data[o] = r
      imgData.data[o + 1] = g
      imgData.data[o + 2] = b
      // Bake the overlay opacity into the pixels instead of using CSS opacity
      // on the canvas element: a sub-1 CSS opacity promotes the canvas to its
      // own compositing layer, which snaps to device pixels independently of
      // the base <img> and makes the labels jitter against it while zooming.
      imgData.data[o + 3] = Math.round(LABEL_ALPHA * opacity)
    }
    ctx.clearRect(0, 0, width, height)
    ctx.putImageData(imgData, 0, 0)
  }

  function canvasPos(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * image.width,
      y: ((e.clientY - rect.top) / rect.height) * image.height,
    }
  }

  function paintPixel(diff, idx, value) {
    if (labels[idx] === value) return
    if (!diff.has(idx)) diff.set(idx, labels[idx])
    labels[idx] = value
  }

  function stamp(x, y) {
    const { classId, brushSize, eraser } = settings.current
    const value = eraser ? UNLABELED : classId
    const r = brushSize
    const { width, height } = image
    const x0 = Math.max(0, Math.floor(x - r))
    const x1 = Math.min(width - 1, Math.ceil(x + r))
    const y0 = Math.max(0, Math.floor(y - r))
    const y1 = Math.min(height - 1, Math.ceil(y + r))
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        if ((px - x) ** 2 + (py - y) ** 2 <= r * r) {
          paintPixel(strokeDiff.current, py * width + px, value)
        }
      }
    }
  }

  function strokeTo(pos) {
    const last = lastPos.current ?? pos
    const dist = Math.hypot(pos.x - last.x, pos.y - last.y)
    const steps = Math.max(1, Math.ceil(dist / (settings.current.brushSize / 2)))
    for (let i = 1; i <= steps; i++) {
      stamp(last.x + ((pos.x - last.x) * i) / steps, last.y + ((pos.y - last.y) * i) / steps)
    }
    lastPos.current = pos
    redraw()
  }

  // Even-odd scanline fill of one closed polygon (image coords) into `diff`.
  function scanFillPolygon(poly, value, diff) {
    const { width, height } = image
    const ys = poly.map((p) => p.y)
    const yStart = Math.max(0, Math.floor(Math.min(...ys)))
    const yEnd = Math.min(height - 1, Math.ceil(Math.max(...ys)))
    for (let py = yStart; py <= yEnd; py++) {
      const yc = py + 0.5
      const xs = []
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i]
        const b = poly[(i + 1) % poly.length]
        if (a.y <= yc !== b.y <= yc) {
          xs.push(a.x + ((yc - a.y) / (b.y - a.y)) * (b.x - a.x))
        }
      }
      xs.sort((p, q) => p - q)
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const xa = Math.max(0, Math.round(xs[k]))
        const xb = Math.min(width - 1, Math.round(xs[k + 1]) - 1)
        for (let px = xa; px <= xb; px++) {
          paintPixel(diff, py * width + px, value)
        }
      }
    }
  }

  // Fill one or more closed polygons as a single undoable edit, using the
  // current class (or the eraser). Shared by the polygon tool and SAM2 snap.
  function commitPolygons(polys) {
    const { classId, eraser } = settings.current
    const value = eraser ? UNLABELED : classId
    const diff = new Map()
    for (const poly of polys) {
      if (poly.length >= 3) scanFillPolygon(poly, value, diff)
    }
    if (diff.size) {
      onDiff(diff)
      redraw()
      onStrokeEnd()
    }
  }

  function closePolygon() {
    const poly = vertsRef.current
    setVerts([])
    commitPolygons([poly])
  }
  closeRef.current = closePolygon

  // Re-run SAM2 over the whole pending click list and show the mask it returns.
  // Every click re-predicts from scratch (the backend replays the click chain),
  // so removing or adding a point always yields the mask that click set implies.
  async function samPredict(points) {
    if (!points.length) {
      setSamPolys([])
      return
    }
    const seq = ++samSeq.current
    setSamBusy(true)
    try {
      const polygons = await onSamSnap(points) // [[x, y], ...][] image coords, or null
      if (seq !== samSeq.current) return // a newer click already superseded this
      if (polygons?.length) {
        setSamPolys(polygons.map((poly) => poly.map(([x, y]) => ({ x, y }))))
      }
      // On a failed/empty prediction keep the previous preview rather than
      // blanking it, so a stray click does not throw away a good selection.
    } finally {
      if (seq === samSeq.current) setSamBusy(false)
    }
  }

  // Add an include (label 1) or exclude (label 0) click to the pending selection.
  function samAddPoint(pos, label) {
    const next = [...samPointsRef.current, { ...pos, label }]
    samPointsRef.current = next
    setSamPoints(next)
    samPredict(next)
  }

  // Drop the most recent click (Backspace) — the mask reverts to what the
  // remaining clicks imply, so a mis-click costs one keystroke, not the selection.
  function samUndoPoint() {
    const next = samPointsRef.current.slice(0, -1)
    samPointsRef.current = next
    setSamPoints(next)
    if (!next.length) setSamPolys([])
    samPredict(next)
  }

  function samCancel() {
    samSeq.current++ // abandon any in-flight prediction
    samPointsRef.current = []
    setSamPoints([])
    setSamPolys([])
    setSamBusy(false)
  }
  samUndoRef.current = samUndoPoint
  samCancelRef.current = samCancel

  // ---------- magic wand ----------

  function drawWandPreview(mask) {
    const canvas = wandCanvasRef.current
    if (!canvas || !image) return
    const ctx = canvas.getContext('2d')
    const { width, height } = image
    ctx.clearRect(0, 0, width, height)
    if (!mask) return
    const [r, g, b] = tool.eraser ? [255, 255, 255] : (classColors[tool.classId] ?? [255, 255, 255])
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

  // Re-run over the whole sample list. Called on every click, sample removal
  // and tolerance change, so the preview always reflects the current samples
  // rather than accumulating the results of earlier fills.
  //
  // The selection is computed in the backend across every band (see
  // Api.wand_select), so this is a round trip. Dragging the slider fires one
  // per step; a sequence number keeps the newest answer and discards any
  // earlier one that lands late, so the preview cannot end up showing a
  // tolerance the slider has already moved past.
  async function wandRun(samples, tolerance, budgetStep, sampleSize, protect,
                         everywhere) {
    if (!samples.length) {
      wandSeq.current++
      wandMaskRef.current = null
      setWandCount(null)
      setWandCapped(null)
      setWandProtected(0)
      setWandBusy(false)
      drawWandPreview(null)
      return
    }
    const seq = ++wandSeq.current
    setWandBusy(true)
    try {
      // The budget is per sample; the backend applies it to each one
      // independently. With a single shared pot a second click halves what the
      // first one had, and you watch a good selection shrink for adding
      // information to it.
      const positives = samples.filter((p) => p.label !== 0)
      const negatives = samples.filter((p) => p.label === 0)
      if (!positives.length) {
        // Negatives alone describe nothing to select. Keep them pending rather
        // than discarding the clicks -- the next include click uses them.
        wandMaskRef.current = null
        setWandCount(null)
        setWandCapped(null)
        setWandProtected(0)
        drawWandPreview(null)
        return
      }
      // Read from the ref, not from the render scope: an in-flight run must
      // ask for the source the toggle is on *now*.
      const embedding = wandSamRef.current
      const result = await onWandSelect(positives, {
        source: embedding ? 'sam' : 'image',
        level: wandLevelRef.current,
        tolerance: embedding ? samTolerance(tolerance) : tolerance / WAND_TOL_SCALE,
        max_pixels: wandBudget(budgetStep),
        sample_size: sampleSize,
        // Never while erasing: the eraser only ever acts on labeled pixels, so
        // protecting them would make it a no-op -- the one combination where
        // the toggle inverts its own purpose.
        protect: protect && !settings.current.eraser,
        global: everywhere,
        negatives: negatives.map((p) => [p.x, p.y]),
      })
      if (seq !== wandSeq.current) return // superseded while in flight
      if (!result) return // backend refused; keep the previous preview
      wandMaskRef.current = result.mask
      setWandCount(result.count)
      setWandCapped(result.capped ? { available: result.available } : null)
      setWandProtected(result.protected || 0)
      drawWandPreview(result.mask)
    } finally {
      if (seq === wandSeq.current) setWandBusy(false)
    }
  }

  function setTolerance(value) {
    const next = Math.min(WAND_TOL_MAX, Math.max(WAND_TOL_MIN, Math.round(value)))
    if (next === tolerance) return
    if (wandSam) setWandSamTolerance(next)
    else setWandTolerances((prev) => ({ ...prev, [tool.classId]: next }))
    wandRun(wandSamplesRef.current, next, wandBudgetStep, wandSampleSize, wandProtect, wandGlobal)
  }

  // Re-runs the clicks already placed against the other source, rather than
  // clearing them: seeing the same click measured both ways is the point of
  // having the switch, and the backend caches both fields so flipping back is
  // free. The tolerance swaps with it, since the two are in different units.
  function setSamSource(on) {
    wandSamRef.current = on
    setWandSam(on)
    wandRun(wandSamplesRef.current,
            on ? wandSamTolerance
               : (wandTolerances[tool.classId] ?? classSeed[tool.classId]
                  ?? WAND_TOL_DEFAULT),
            wandBudgetStep, wandSampleSize, wandProtect, wandGlobal)
  }

  // Switching feature level re-runs the existing samples rather than clearing
  // them: comparing fine/mid/deep on the same click is the whole point of the
  // selector, and a cached field makes going back free.
  function setLevel(next) {
    if (next === wandLevelRef.current) return
    wandLevelRef.current = next
    setWandLevel(next)
    wandRun(wandSamplesRef.current, tolerance, wandBudgetStep, wandSampleSize,
            wandProtect, wandGlobal)
  }

  // Add a sample. It matches on its own colour rather than shifting a shared
  // average, so an existing selection cannot lose pixels because of it.
  function wandAddSample(pos, label = 1) {
    const next = [...wandSamplesRef.current, { ...pos, label }]
    wandSamplesRef.current = next
    setWandSamples(next)
    wandRun(next, tolerance, wandBudgetStep, wandSampleSize, wandProtect, wandGlobal)
  }

  // Drop the most recent sample (Backspace), as SAM2 does for its clicks: a
  // mis-click costs one keystroke rather than the whole selection.
  function wandUndoSample() {
    const next = wandSamplesRef.current.slice(0, -1)
    wandSamplesRef.current = next
    setWandSamples(next)
    wandRun(next, tolerance, wandBudgetStep, wandSampleSize, wandProtect, wandGlobal)
  }
  wandUndoRef.current = wandUndoSample

  function setBudgetStep(value) {
    const next = Math.min(WAND_BUDGET_MAX,
                          Math.max(WAND_BUDGET_MIN, Math.round(value)))
    if (next === wandBudgetStep) return
    setWandBudgetStep(next)
    wandRun(wandSamplesRef.current, tolerance, next, wandSampleSize, wandProtect, wandGlobal)
  }

  function setSampleSize(value) {
    const next = Math.min(WAND_SAMPLE_MAX,
                          Math.max(WAND_SAMPLE_MIN, Math.round(value)))
    if (next === wandSampleSize) return
    setWandSampleSize(next)
    wandRun(wandSamplesRef.current, tolerance, wandBudgetStep, next, wandProtect, wandGlobal)
  }

  function setProtect(on) {
    setWandProtect(on)
    wandRun(wandSamplesRef.current, tolerance, wandBudgetStep,
            wandSampleSize, on, wandGlobal)
  }

  function setGlobal(on) {
    setWandGlobal(on)
    wandRun(wandSamplesRef.current, tolerance, wandBudgetStep,
            wandSampleSize, wandProtect, on)
  }

  wandRerunRef.current = () => wandRun(wandSamplesRef.current, tolerance,
                                       wandBudgetStep, wandSampleSize,
                                       wandProtect, wandGlobal)

  function wandCancel() {
    wandSeq.current++ // abandon any in-flight selection
    setWandBusy(false)
    wandSamplesRef.current = []
    setWandSamples([])
    wandMaskRef.current = null
    setWandCount(null)
    setWandCapped(null)
    setWandProtected(0)
    drawWandPreview(null)
  }
  wandCancelRef.current = wandCancel

  function wandCommit() {
    const mask = wandMaskRef.current
    if (!mask) return
    const value = settings.current.eraser ? UNLABELED : settings.current.classId
    const diff = new Map()
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) paintPixel(diff, i, value)
    }
    wandCancel()
    if (diff.size) {
      onDiff(diff)
      redraw()
      onStrokeEnd()
    }
  }
  wandCommitRef.current = wandCommit

  // Commit the previewed mask into the labels as a normal polygon fill, so it
  // persists to masks_user and feeds training exactly like a hand-drawn one.
  function samCommit() {
    const polys = samPolysRef.current
    if (!polys.length) return
    samCancel()
    commitPolygons(polys)
  }
  samCommitRef.current = samCommit

  // Oversized-brush fallback circle (see cursorCss): only mounted when the
  // brush is too big for a native CSS cursor. The stack's CSS space is the
  // image scaled by the zoom (layout scaling, see Viewport), so image
  // coordinates map to CSS pixels via view.scale. Positioned with `transform`
  // (compositor-only, no layout reflow) via direct style updates, avoiding a
  // React re-render per pointer move.
  function moveCursor(pos) {
    const el = cursorRef.current
    if (!el) return // native-cursor mode: no DOM circle mounted
    const scale = view?.scale ?? 1
    el.style.display = 'block'
    el.style.transform = `translate3d(${pos.x * scale}px, ${pos.y * scale}px, 0)`
  }

  function hideCursor() {
    if (cursorRef.current) cursorRef.current.style.display = 'none'
  }

  function onPointerDown(e) {
    if (!image) return
    if (settings.current.mode === 'wand') {
      // Left includes, right excludes -- the same gesture SAM2 uses here, so
      // the two click-driven tools share one language. Ctrl/Alt+left also
      // excludes, for trackpads that swallow the secondary click.
      if (e.button === 2) {
        e.preventDefault()
        wandAddSample(canvasPos(e), 0)
      } else if (e.button === 0) {
        wandAddSample(canvasPos(e), e.ctrlKey || e.altKey ? 0 : 1)
      }
      return
    }
    if (settings.current.mode === 'sam') {
      // Left click includes, right click (or Ctrl/Alt+left) excludes. Right
      // click is the primary gesture because it needs no second hand; the
      // modifier is there for trackpads that swallow secondary click.
      if (e.button === 2) {
        e.preventDefault()
        samAddPoint(canvasPos(e), 0)
      } else if (e.button === 0) {
        samAddPoint(canvasPos(e), e.ctrlKey || e.altKey ? 0 : 1)
      }
      return
    }
    if (e.button !== 0) return
    if (settings.current.mode === 'polygon') {
      const pos = canvasPos(e)
      setVerts((v) => {
        const next = [...v, pos]
        vertsRef.current = next // keep the ref fresh for same-tick key events
        return next
      })
      return
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    drawing.current = true
    lastPos.current = null
    strokeDiff.current = new Map()
    strokeTo(canvasPos(e))
  }

  function onPointerMove(e) {
    const pos = canvasPos(e)
    moveCursor(pos)
    if (drawing.current) strokeTo(pos)
  }

  function onPointerUp() {
    if (!drawing.current) return
    drawing.current = false
    lastPos.current = null
    if (strokeDiff.current?.size) onDiff(strokeDiff.current)
    strokeDiff.current = null
    onStrokeEnd()
  }

  const brushColor = tool.eraser ? [255, 255, 255] : classColors[tool.classId]
  const polygonMode = tool.mode === 'polygon'
  const samMode = tool.mode === 'sam'
  const wandMode = tool.mode === 'wand'
  // Share of the image the pending selection covers, and whether that is far
  // enough to be worth calling out before it is committed.
  const wandFraction = (wandCount !== null && image)
    ? wandCount / (image.width * image.height) : 0
  const wandPct = (wandFraction * 100).toFixed(1)
  const wandRunaway = wandFraction > WAND_WARN_FRACTION
  const budget = wandBudget(wandBudgetStep)

  // Brush cursor: draw it into a native CSS cursor (zero pointer lag) while it
  // fits the browser's 128px cap; beyond that, fall back to the DOM circle.
  const scale = view?.scale ?? 1
  const oversized = Math.ceil(tool.brushSize * 2 * scale) + CURSOR_PAD * 2 > MAX_CURSOR_PX
  const cursorCss = useMemo(() => {
    if (samMode) return samBusy ? 'wait' : 'crosshair'
    if (wandMode) return wandBusy ? 'wait' : 'crosshair'
    if (polygonMode) return 'crosshair'
    const color = tool.eraser ? [255, 255, 255] : classColors[tool.classId]
    return makeBrushCursor(tool.brushSize * 2 * scale, color) ?? makeCrosshairCursor(color)
  }, [tool.brushSize, scale, tool.eraser, tool.classId, polygonMode, samMode, wandMode, samBusy, wandBusy, classColors])

  return (
    <div className="pane">
      <div className="pane-title">
        <span className="pane-name">Interactive Labeler</span>
        {/* No class/eraser chip here: the toolbar already shows what is
            selected, so repeating it beside the title is noise. */}
        {polygonMode && (
          <span className="hint">click: add point · Enter/double-click: fill · Esc: cancel</span>
        )}
        {samMode && (
          <span className="hint">
            {samBusy ? 'SAM2 working…'
              : samPoints.length
                ? 'right-click: exclude · left-click: include · Backspace: undo point · Enter: fill · Esc: cancel'
                : 'click an object · then right-click parts to exclude them'}
          </span>
        )}
        {wandMode && (
          <span className={`hint${wandRunaway ? ' warn' : ''}`}>
            {wandCount === null
              ? (wandSam
                ? 'click a region · matched on SAM2 features · first click on an '
                  + 'image encodes it (a moment)'
                : 'click a region · click again to extend · right-click to exclude')
              : `${wandSamples.filter((p) => p.label !== 0).length} sample`
                + `${wandSamples.filter((p) => p.label !== 0).length === 1 ? '' : 's'}`
                + `${wandSamples.some((p) => p.label === 0)
                  ? ` −${wandSamples.filter((p) => p.label === 0).length}` : ''} · `
                + `${wandCount.toLocaleString()} px (${wandPct}%)`
                + (wandCapped
                  ? ` — capped, ${wandCapped.available.toLocaleString()} match`
                  : wandRunaway ? ' — most of the image' : '')
                + (wandProtected
                  ? ` · ${wandProtected.toLocaleString()} protected · `
                  : ' · ')
                + 'Backspace: drop last · Enter: fill · Esc: cancel'}
          </span>
        )}
        <span className="spacer" />
        <label className="opacity-control">
          overlay {Math.round(opacity * 100)}%
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            onWheel={(e) => {
              e.preventDefault()
              const next = opacity + (e.deltaY < 0 ? 0.05 : -0.05)
              setOpacity(Math.min(1, Math.max(0, Number(next.toFixed(2)))))
            }}
          />
        </label>
      </div>
      {/* The wand carries more settings than fit beside a title. Its own
          row, mounted only in wand mode, keeps them together and keeps the
          title bar readable -- the same reason the class picker left the
          toolbar. */}
      {wandMode && (
        <div className="wand-controls">
          {(
          <label
            className="opacity-control"
            title={wandSam
              ? 'Cosine distance between SAM2 feature vectors: how differently '
                + 'the encoder describes a pixel from the one you clicked. '
                + "Measured against this image's own mean feature, so the "
                + 'number is relative to the variety in the tile in front of '
                + 'you — the same setting can bite differently on a uniform '
                + 'tile and a busy one.'
              : 'Spectral distance from the sampled colour, across every '
                + 'band of the image rather than the three shown on screen. '
                + 'Measured as an RMS per-band difference, so the number means '
                + 'the same thing whatever the band count.\n\n'
                + 'Kept per class: each class starts at the tolerance measured '
                + 'best for that material (they range from 14 for Water to 38 '
                + 'for Urban) and then remembers whatever you set it to.'}
          >
            {/* The cosine value alongside the slider position, since the band
                that works is narrow and worth being able to name. */}
            tolerance {tolerance}{wandSam ? ` (${samTolerance(tolerance).toFixed(2)})` : ''}
            <input
              type="range"
              min={WAND_TOL_MIN}
              max={WAND_TOL_MAX}
              step="1"
              value={tolerance}
              onChange={(e) => setTolerance(Number(e.target.value))}
              onWheel={(e) => {
                e.preventDefault()
                setTolerance(tolerance + (e.deltaY < 0 ? 1 : -1))
              }}
            />
          </label>
          )}
          {(
          <label
            className="opacity-control"
            title={'How many ' + (wandSam ? 'feature cells' : 'pixels')
              + ' around each click are averaged into that '
              + "sample's reference, nearest first. Raising it steadies "
              + 'the reference on noisy ground, at the cost of blurring across '
              + 'a boundary when the window straddles one.'
              + (wandSam
                ? ' Cells, not pixels: at the deep level one already covers '
                  + 'several pixels of the image.'
                : '')}
          >
            sample {wandSampleSize} {wandSam ? 'cells' : 'px'}
            <input
              type="range"
              min={WAND_SAMPLE_MIN}
              max={WAND_SAMPLE_MAX}
              step="1"
              value={wandSampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              onWheel={(e) => {
                e.preventDefault()
                setSampleSize(wandSampleSize + (e.deltaY < 0 ? 1 : -1))
              }}
            />
          </label>
          )}
          {(
          <label
            className="opacity-control"
            title={'Cap how many pixels the wand may select, per sample. Past '
              + 'the cap the selection grows outward from your clicks and '
              + 'stops. The allowance is per click, so adding a sample never '
              + 'shrinks what the previous ones selected. Use it when a region '
              + 'runs spectrally continuous into its surroundings and '
              + 'connectivity alone will not stop the fill. At the far right '
              + 'there is no cap.'}
          >
            max {budget === null
              ? '\u221e'
              : `${budget.toLocaleString()} px${wandSamples.length > 1
                  ? ' / sample' : ''}`}
            <input
              type="range"
              min={WAND_BUDGET_MIN}
              max={WAND_BUDGET_MAX}
              step="1"
              value={wandBudgetStep}
              onChange={(e) => setBudgetStep(Number(e.target.value))}
              onWheel={(e) => {
                e.preventDefault()
                setBudgetStep(wandBudgetStep + (e.deltaY < 0 ? 1 : -1))
              }}
            />
          </label>
          )}
          <span className="spacer" />
          <label
            className="wand-toggle"
            title={samAvailable
              ? 'Match on SAM2 encoder features instead of the image bands: '
                + 'texture and learned appearance rather than reflectance. '
                + 'Scores about the same overall (mean IoU 0.568 against 0.593 '
                + 'over 60 labeled regions) but succeeds and fails on '
                + 'different regions, so it is worth trying where the spectral '
                + 'fill will not converge. Costs one encoder pass per image, '
                + 'and gives up the non-visible bands the wand normally uses.'
              : 'SAM2 model files not found in sam2/onnx/'}
          >
            <input
              type="checkbox"
              checked={wandSam}
              disabled={!samAvailable}
              onChange={(e) => setSamSource(e.target.checked)}
            />
            SAM features
          </label>
          {/* Which encoder feature map the distance is measured on. Switching
              re-runs the clicks already placed, so the three are directly
              comparable on one region -- which is the point of exposing them
              at all rather than picking one. */}
          {wandSam && (
            <label className="wand-levels">
              features
              <span className="segmented">
                {SAM_LEVELS.map((l) => (
                  <button
                    key={l.id}
                    type="button"
                    className={wandLevel === l.id ? 'active' : ''}
                    title={l.title}
                    onClick={() => setLevel(l.id)}
                  >
                    {l.label}
                  </button>
                ))}
              </span>
            </label>
          )}
          <label
            className="wand-toggle"
            title={'Fill only where nothing is labeled yet, so a selection can '
              + 'never overwrite finished work. Switch it off to deliberately '
              + 're-classify. Ignored while the eraser is active, which only '
              + 'ever acts on labeled pixels.'}
          >
            <input
              type="checkbox"
              checked={wandProtect}
              onChange={(e) => setProtect(e.target.checked)}
            />
            protect labels
          </label>
          <label
            className="wand-toggle"
            title={'Take every matching pixel in the scene, not just the '
              + 'connected piece under the cursor. These classes come in dozens '
              + 'of separate fragments, so one click can reach them all. The '
              + 'pixel budget still applies, spending on the nearest first.'}
          >
            <input
              type="checkbox"
              checked={wandGlobal}
              onChange={(e) => setGlobal(e.target.checked)}
            />
            whole image
          </label>
        </div>
      )}
      <Viewport image={image} view={view} setView={setView} fitRef={fitRef}>
        <img src={`data:image/png;base64,${image?.png}`} alt="" draggable={false} />
        <canvas
          ref={setCanvas}
          width={image?.width}
          height={image?.height}
          style={{ cursor: cursorCss }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={hideCursor}
          onWheel={hideCursor}
          onDoubleClick={polygonMode ? closePolygon : undefined}
          // In SAM2 mode right-click is the "exclude" gesture, so the webview's
          // context menu must not open on it.
          onContextMenu={samMode || wandMode ? (e) => e.preventDefault() : undefined}
        />
        {wandMode && (
          <canvas
            ref={wandCanvasRef}
            className="wand-preview"
            width={image?.width}
            height={image?.height}
          />
        )}
        {wandMode && wandSamples.length > 0 && image && (
          // Markers for the sampled pixels. Without them the samples are
          // invisible once the fill covers them, and there is no way to see
          // what Backspace is about to remove.
          <svg
            className="poly-preview"
            viewBox={`0 0 ${image.width} ${image.height}`}
          >
            {wandSamples.map((p, i) => {
              const r = view ? 5 / view.scale : 5
              const last = i === wandSamples.length - 1
              return (
                <circle
                  key={i}
                  cx={p.x}
                  cy={p.y}
                  r={r}
                  fill={`rgba(${(p.label === 0 ? SAM_EXCLUDE_COLOR : SAM_INCLUDE_COLOR)
                    .join(',')}, ${last ? 0.9 : 0.5})`}
                  stroke="#fff"
                  strokeWidth={last ? 2 : 1}
                  vectorEffect="non-scaling-stroke"
                />
              )
            })}
          </svg>
        )}
        {polygonMode && verts.length > 0 && image && (
          <svg
            className="poly-preview"
            viewBox={`0 0 ${image.width} ${image.height}`}
          >
            <polyline
              points={verts.map((p) => `${p.x},${p.y}`).join(' ')}
              fill={`rgba(${brushColor.join(',')}, 0.15)`}
              stroke={`rgba(${brushColor.join(',')}, 0.9)`}
              vectorEffect="non-scaling-stroke"
            />
            {verts.map((p, i) => (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={view ? 3 / view.scale : 3}
                fill={`rgb(${brushColor.join(',')})`}
              />
            ))}
          </svg>
        )}
        {samMode && (samPolys.length > 0 || samPoints.length > 0) && image && (
          <svg className="poly-preview" viewBox={`0 0 ${image.width} ${image.height}`}>
            {samPolys.map((poly, i) => (
              <polygon
                key={i}
                points={poly.map((p) => `${p.x},${p.y}`).join(' ')}
                fill={`rgba(${brushColor.join(',')}, 0.35)`}
                stroke={`rgba(${brushColor.join(',')}, 0.9)`}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {/* Include = pink, exclude = red (the app's --accent / --danger).
                Translucent fill with the same colour at full opacity for the
                stroke, so a marker stays legible over both the image and the
                selection fill. The exclude marker keeps its bar: pink and red
                are close enough that colour alone is a weak distinction. */}
            {samPoints.map((p, i) => {
              const r = view ? 6 / view.scale : 6
              const color = p.label ? SAM_INCLUDE_COLOR : SAM_EXCLUDE_COLOR
              return (
                <g key={i}>
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={r}
                    fill={`rgba(${color.join(',')}, 0.7)`}
                    stroke={`rgb(${color.join(',')})`}
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                  />
                  {!p.label && (
                    <line
                      x1={p.x - r * 0.6}
                      y1={p.y}
                      x2={p.x + r * 0.6}
                      y2={p.y}
                      stroke={`rgb(${color.join(',')})`}
                      strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                </g>
              )
            })}
          </svg>
        )}
        {!polygonMode && !samMode && !wandMode && oversized && (
          <div
            ref={cursorRef}
            className="brush-cursor"
            style={{
              width: tool.brushSize * 2 * scale,
              height: tool.brushSize * 2 * scale,
              marginLeft: -tool.brushSize * scale,
              marginTop: -tool.brushSize * scale,
              borderWidth: 1.5,
              borderColor: `rgba(${brushColor.join(',')}, 0.9)`,
              background: `rgba(${brushColor.join(',')}, 0.2)`,
            }}
          />
        )}
      </Viewport>
    </div>
  )
}
