import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  EMBED_TOL_DEFAULT, MS_TOL_DEFAULT, SAM_TOL_DEFAULT, WAND_BUDGET_MAX,
  WAND_BUDGET_MIN, WAND_SAMPLE_DEFAULT, WAND_SAMPLE_MAX, WAND_SAMPLE_MIN,
  WAND_TOL_DEFAULT, WAND_TOL_MAX, WAND_TOL_MIN, WAND_TOL_SEEDS, tolSeedKey,
  toleranceFor, wandBudget,
} from './constants'
import { drawWandPreview } from './preview'

// Everything the wand is, minus the markup: the pending clicks, the settings,
// the round trip to the backend and the preview canvas.
//
// The settings are ONE object edited through patch(), rather than a state
// variable per knob. The previous shape threaded six positional arguments into
// a re-run function from eleven call sites, ten of which differed by exactly
// one argument -- so every new knob meant editing eleven lines, and two of
// them (the SAM source and the feature level) ended up smuggled through refs
// purely to avoid an eighth positional argument. patch({ protect: true }) is
// the whole of that now, and re-running is the hook's business rather than
// each caller's.
//
// The settings live in a ref as well as in state: the ref is the truth that
// callbacks read, state is only what renders. That is what lets every callback
// below be stable (useCallback with no deps) while still seeing fresh values,
// which in turn lets the window-level key handler call them directly.

const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, Math.round(value)))

const INITIAL = {
  // Per class, and only the values you have changed -- anything untouched
  // falls through to that class's measured seed. What counts as "the same
  // material" is a property of the material, and the per-class optima span
  // 0.07 to 0.19.
  tolerances: {},
  // What the wand measures distance in: 'image' (spectral), 'sam' (SAM2
  // encoder features), 'embed' (auto-encoder land-cover embeddings) or
  // 'meanshift' (spectral on a mean-shift-filtered PCA image). One string
  // rather than a bool per source, so a fifth source is an array entry and
  // not another flag nobody can keep exclusive.
  source: 'image',
  // The embedding sources keep one tolerance for every class: their own
  // per-class figures have not been measured, and inventing seeds for them
  // would dress a guess up as the spectral wand's evidence. One field each,
  // so toggling back and forth restores what each was last set to.
  samTolerance: SAM_TOL_DEFAULT,
  embedTolerance: EMBED_TOL_DEFAULT,
  msTolerance: MS_TOL_DEFAULT,
  level: 'fine',
  budgetStep: WAND_BUDGET_MAX, // max = no budget
  sampleSize: WAND_SAMPLE_DEFAULT,
  // On by default: this dataset arrives ~75% labeled, and an unprotected fill
  // covers 35% already-labeled pixels across several classes. Overwriting
  // finished work silently is the worse failure, so the safe mode is the one
  // you have to switch off.
  protect: true,
  // Off by default: connectivity is the safer reading of "this region", and a
  // global click at a loose tolerance can take most of the scene.
  everywhere: false,
}

// The slider position for the active class, or for a non-spectral source,
// each of which is one value for all classes.
function sliderFor(settings, classId, seeds) {
  if (settings.source === 'sam') return settings.samTolerance
  if (settings.source === 'embed') return settings.embedTolerance
  if (settings.source === 'meanshift') return settings.msTolerance
  return settings.tolerances[classId] ?? seeds[classId] ?? WAND_TOL_DEFAULT
}

// Which settings field a slider move writes to.
function toleranceField(source) {
  if (source === 'sam') return 'samTolerance'
  if (source === 'embed') return 'embedTolerance'
  if (source === 'meanshift') return 'msTolerance'
  return 'tolerances'
}

export function useWand({ active, image, tool, classes, classColors, onSelect, onCommit }) {
  const [samples, setSamples] = useState([])
  const [settings, setSettings] = useState(INITIAL)
  const [result, setResult] = useState({ count: null, capped: null, protected: 0 })
  const [busy, setBusy] = useState(false)

  const samplesRef = useRef(samples)
  const settingsRef = useRef(settings)
  const maskRef = useRef(null)      // Uint8Array(W*H), 1 = selected
  const canvasRef = useRef(null)
  const seq = useRef(0)             // drops out-of-order responses from stale clicks

  // One ref for everything that comes from the render scope, so the callbacks
  // below can stay stable without a dependency array per prop.
  const latest = useRef(null)
  latest.current = { image, tool, classColors, onSelect, onCommit }

  const seeds = useMemo(() => {
    const byId = {}
    for (const c of classes ?? []) {
      const seed = WAND_TOL_SEEDS[tolSeedKey(c.name)]
      if (seed !== undefined) byId[c.id] = seed
    }
    return byId
  }, [classes])
  const seedsRef = useRef(seeds)
  seedsRef.current = seeds

  const draw = useCallback((mask) => {
    const { image: img, tool: t, classColors: colors } = latest.current
    const color = t.eraser ? [255, 255, 255] : (colors[t.classId] ?? [255, 255, 255])
    drawWandPreview(canvasRef.current, mask, img, color)
  }, [])

  const clear = useCallback(() => {
    maskRef.current = null
    setResult({ count: null, capped: null, protected: 0 })
    draw(null)
  }, [draw])

  // Re-run over the whole sample list. Called on every click, sample removal
  // and settings change, so the preview always reflects the current samples
  // rather than accumulating the results of earlier fills.
  //
  // The selection is computed in the backend across every band (see
  // Api.wand_select), so this is a round trip. Dragging a slider fires one per
  // step; a sequence number keeps the newest answer and discards any earlier
  // one that lands late, so the preview cannot end up showing a value the
  // slider has already moved past.
  const run = useCallback(async (nextSettings, nextSamples) => {
    const { tool: t, onSelect: select } = latest.current
    if (!nextSamples.length) {
      seq.current++
      setBusy(false)
      clear()
      return
    }
    const mine = ++seq.current
    setBusy(true)
    try {
      const positives = nextSamples.filter((p) => p.label !== 0)
      const negatives = nextSamples.filter((p) => p.label === 0)
      if (!positives.length) {
        // Negatives alone describe nothing to select. Keep them pending rather
        // than discarding the clicks -- the next include click uses them.
        clear()
        return
      }
      const slider = sliderFor(nextSettings, t.classId, seedsRef.current)
      const selection = await select(positives, {
        source: nextSettings.source,
        level: nextSettings.level,
        tolerance: toleranceFor(nextSettings.source, slider),
        // The budget is per sample; the backend applies it to each one
        // independently. With a single shared pot a second click halves what
        // the first one had, and you watch a good selection shrink for adding
        // information to it.
        max_pixels: wandBudget(nextSettings.budgetStep),
        sample_size: nextSettings.sampleSize,
        // Never while erasing: the eraser only ever acts on labeled pixels, so
        // protecting them would make it a no-op -- the one combination where
        // the toggle inverts its own purpose.
        protect: nextSettings.protect && !t.eraser,
        global: nextSettings.everywhere,
        negatives: negatives.map((p) => [p.x, p.y]),
      })
      if (mine !== seq.current) return // superseded while in flight
      if (!selection) return // backend refused; keep the previous preview
      maskRef.current = selection.mask
      setResult({
        count: selection.count,
        capped: selection.capped ? { available: selection.available } : null,
        protected: selection.protected || 0,
      })
      draw(selection.mask)
    } finally {
      if (mine === seq.current) setBusy(false)
    }
  }, [clear, draw])

  // The one way settings change: merge, publish, re-run. Every knob below is a
  // one-line call to this.
  const patch = useCallback((partial) => {
    const next = { ...settingsRef.current, ...partial }
    settingsRef.current = next
    setSettings(next)
    run(next, samplesRef.current)
  }, [run])

  const setSamples_ = useCallback((next) => {
    samplesRef.current = next
    setSamples(next)
    run(settingsRef.current, next)
  }, [run])

  // Add a sample. It matches on its own reference rather than shifting a
  // shared average, so an existing selection cannot lose pixels because of it.
  const addSample = useCallback((pos, label = 1) => {
    setSamples_([...samplesRef.current, { ...pos, label }])
  }, [setSamples_])

  // Drop the most recent sample (Backspace), as SAM2 does for its clicks: a
  // mis-click costs one keystroke rather than the whole selection.
  const undoSample = useCallback(() => {
    setSamples_(samplesRef.current.slice(0, -1))
  }, [setSamples_])

  const cancel = useCallback(() => {
    seq.current++ // abandon any in-flight selection
    setBusy(false)
    samplesRef.current = []
    setSamples([])
    clear()
  }, [clear])

  const commit = useCallback(() => {
    const mask = maskRef.current
    if (!mask) return
    cancel()
    latest.current.onCommit(mask)
  }, [cancel])

  const setTolerance = useCallback((value) => {
    const s = settingsRef.current
    const { tool: t } = latest.current
    const next = clamp(value, WAND_TOL_MIN, WAND_TOL_MAX)
    if (next === sliderFor(s, t.classId, seedsRef.current)) return
    const field = toleranceField(s.source)
    patch(field === 'tolerances'
      ? { tolerances: { ...s.tolerances, [t.classId]: next } }
      : { [field]: next })
  }, [patch])

  const setBudgetStep = useCallback((value) => {
    const next = clamp(value, WAND_BUDGET_MIN, WAND_BUDGET_MAX)
    if (next !== settingsRef.current.budgetStep) patch({ budgetStep: next })
  }, [patch])

  const setSampleSize = useCallback((value) => {
    const next = clamp(value, WAND_SAMPLE_MIN, WAND_SAMPLE_MAX)
    if (next !== settingsRef.current.sampleSize) patch({ sampleSize: next })
  }, [patch])

  // Switching source or level re-runs the clicks already placed rather than
  // clearing them: seeing the same click measured all three ways, or at all
  // three SAM levels, is the point of having the switch, and the backend
  // caches every field so flipping back is free. The tolerance swaps with the
  // source by itself now, since sliderFor reads which one is active.
  const setSource = useCallback((source) => {
    if (source !== settingsRef.current.source) patch({ source })
  }, [patch])
  const setLevel = useCallback((level) => {
    if (level !== settingsRef.current.level) patch({ level })
  }, [patch])
  const setProtect = useCallback((on) => patch({ protect: !!on }), [patch])
  const setGlobal = useCallback((on) => patch({ everywhere: !!on }), [patch])

  // Leaving the tool drops an uncommitted selection rather than leaving an
  // invisible one to be committed later by a stray Enter.
  useEffect(() => {
    if (!active) cancel()
  }, [active, cancel])

  // The tolerance travels with the class, so a selection already on screen has
  // to be recomputed when the class changes -- otherwise the preview is the one
  // the previous class's threshold produced while the slider reads the new
  // class's number, and Enter would commit a mask that matches neither.
  // Skipped for the embedding sources: each keeps a single tolerance, so a
  // class switch changes nothing about the selection.
  useEffect(() => {
    if (settingsRef.current.source === 'image' && samplesRef.current.length) {
      run(settingsRef.current, samplesRef.current)
    }
  }, [tool.classId, run])

  const slider = sliderFor(settings, tool.classId, seeds)
  const fraction = (result.count !== null && image)
    ? result.count / (image.width * image.height) : 0

  return {
    canvasRef,
    samples,
    settings,
    busy,
    // Readout
    count: result.count,
    capped: result.capped,
    protectedCount: result.protected,
    fraction,
    tolerance: slider,                       // slider position, 1..100
    toleranceValue: toleranceFor(settings.source, slider), // what the source sends
    budget: wandBudget(settings.budgetStep),
    // Actions
    addSample,
    undoSample,
    cancel,
    commit,
    setTolerance,
    setBudgetStep,
    setSampleSize,
    setSource,
    setLevel,
    setProtect,
    setGlobal,
  }
}
