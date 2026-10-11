import {
  WAND_BUDGET_MAX, WAND_BUDGET_MIN, WAND_SAMPLE_MAX, WAND_SAMPLE_MIN,
  WAND_TOL_MAX, WAND_TOL_MIN,
} from './constants'

// The wand carries more settings than fit beside a pane title, so they get
// their own row -- the same reason the class picker left the toolbar. Split
// out of LabelerPane because it is markup for one thing and reads better on
// its own than as the longest branch of a component that also owns the brush,
// the polygon tool and the canvas.
//
// Every control does the same thing: hand one value to the wand and let it
// re-run itself. None of them know that a selection is recomputed as a result.

// What distance the wand measures. Both sources read the raster's own bands
// and need no model file, so neither can be unavailable and neither carries a
// gate. The numbers quoted are the bench_wand.py sources run over this
// project's labels (see implementation_log.txt).
const SOURCES = [
  {
    id: 'image', label: 'spectral',
    title: 'RMS per-band difference across every band of the image, not '
      + 'just the three on screen. The default: its per-class tolerances '
      + 'are measured, and water vs terrain shadow separate cleanly in '
      + 'NIR/SWIR.',
  },
  {
    id: 'meanshift', label: 'mean-shift',
    title: 'The spectral measure on a mean-shift-filtered image: PCA to '
      + '3 components, then every pixel walks to the mode of its '
      + 'spatio-colour neighbourhood (OpenCV pyrMeanShiftFiltering). Each '
      + 'material collapses onto one near-constant colour and the boundary '
      + 'between materials sharpens, so a tolerance that drifts on raw '
      + 'bands becomes a cliff here. Costs one filter pass per image '
      + '(~0.25 s), and works on any band count.',
  },
]

export default function WandControls({ wand }) {
  const { settings } = wand
  const msSource = settings.source === 'meanshift'

  const toleranceTitle = msSource
    ? 'RMS colour distance on the mean-shift-filtered image, in the same '
      + '[0,1] units as spectral -- but each material has already collapsed '
      + 'onto one near-constant colour, so the number that separates a '
      + 'region from its surroundings is small and sharp. One value for '
      + 'every class.'
    : 'Spectral distance from the sampled colour, across every '
      + 'band of the image rather than the three shown on screen. '
      + 'Measured as an RMS per-band difference, so the number means '
      + 'the same thing whatever the band count.\n\n'
      + 'Kept per class: each class starts at the tolerance measured '
      + 'best for that material (they range from 14 for Water to 38 '
      + 'for Urban) and then remembers whatever you set it to.'

  return (
    <div className="wand-controls">
      <label
        className="opacity-control"
        title={toleranceTitle}
      >
        {/* The mapped tolerance alongside the slider position, since mean-shift's
            useful band is narrow and worth being able to name. */}
        tolerance {wand.tolerance}
        {msSource ? ` (${wand.toleranceValue.toFixed(3)})` : ''}
        <input
          type="range"
          min={WAND_TOL_MIN}
          max={WAND_TOL_MAX}
          step="1"
          value={wand.tolerance}
          onChange={(e) => wand.setTolerance(Number(e.target.value))}
          onWheel={(e) => {
            e.preventDefault()
            wand.setTolerance(wand.tolerance + (e.deltaY < 0 ? 1 : -1))
          }}
        />
      </label>

      <label
        className="opacity-control"
        title={'How many pixels around each click are averaged into that '
          + "sample's reference, nearest first. Raising it steadies "
          + 'the reference on noisy ground, at the cost of blurring across '
          + 'a boundary when the window straddles one.'}
      >
        sample {settings.sampleSize} px
        <input
          type="range"
          min={WAND_SAMPLE_MIN}
          max={WAND_SAMPLE_MAX}
          step="1"
          value={settings.sampleSize}
          onChange={(e) => wand.setSampleSize(Number(e.target.value))}
          onWheel={(e) => {
            e.preventDefault()
            wand.setSampleSize(settings.sampleSize + (e.deltaY < 0 ? 1 : -1))
          }}
        />
      </label>

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
        max {wand.budget === null
          ? '∞'
          : `${wand.budget.toLocaleString()} px${wand.samples.length > 1
              ? ' / sample' : ''}`}
        <input
          type="range"
          min={WAND_BUDGET_MIN}
          max={WAND_BUDGET_MAX}
          step="1"
          value={settings.budgetStep}
          onChange={(e) => wand.setBudgetStep(Number(e.target.value))}
          onWheel={(e) => {
            e.preventDefault()
            wand.setBudgetStep(settings.budgetStep + (e.deltaY < 0 ? 1 : -1))
          }}
        />
      </label>

      <span className="spacer" />

      {/* What the wand measures distance in. Switching re-runs the clicks
          already placed rather than clearing them, so the same click can be
          seen under both measures -- the backend caches every field, so
          flipping back and forth is free after the first look. */}
      <label className="wand-levels">
        match on
        <span className="segmented">
          {SOURCES.map((s) => (
            <button
              key={s.id}
              type="button"
              className={settings.source === s.id ? 'active' : ''}
              title={s.title}
              onClick={() => wand.setSource(s.id)}
            >
              {s.label}
            </button>
          ))}
        </span>
      </label>

      <label
        className="wand-toggle"
        title={'Fill only where nothing is labeled yet, so a selection can '
          + 'never overwrite finished work. Switch it off to deliberately '
          + 're-classify. Ignored while the eraser is active, which only '
          + 'ever acts on labeled pixels.'}
      >
        <input
          type="checkbox"
          checked={settings.protect}
          onChange={(e) => wand.setProtect(e.target.checked)}
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
          checked={settings.everywhere}
          onChange={(e) => wand.setGlobal(e.target.checked)}
        />
        whole image
      </label>
    </div>
  )
}
