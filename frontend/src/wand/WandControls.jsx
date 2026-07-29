import {
  SAM_LEVELS, WAND_BUDGET_MAX, WAND_BUDGET_MIN, WAND_SAMPLE_MAX,
  WAND_SAMPLE_MIN, WAND_TOL_MAX, WAND_TOL_MIN,
} from './constants'

// The wand carries more settings than fit beside a pane title, so they get
// their own row -- the same reason the class picker left the toolbar. Split
// out of LabelerPane because it is markup for one thing and reads better on
// its own than as the longest branch of a component that also owns the brush,
// the polygon tool and the canvas.
//
// Every control does the same thing: hand one value to the wand and let it
// re-run itself. None of them know that a selection is recomputed as a result.
export default function WandControls({ wand, samAvailable }) {
  const { settings } = wand

  return (
    <div className="wand-controls">
      <label
        className="opacity-control"
        title={settings.sam
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
        tolerance {wand.tolerance}
        {settings.sam ? ` (${wand.toleranceValue.toFixed(2)})` : ''}
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
        title={'How many ' + (settings.sam ? 'feature cells' : 'pixels')
          + ' around each click are averaged into that '
          + "sample's reference, nearest first. Raising it steadies "
          + 'the reference on noisy ground, at the cost of blurring across '
          + 'a boundary when the window straddles one.'
          + (settings.sam
            ? ' Cells, not pixels: at the deep level one already covers '
              + 'several pixels of the image.'
            : '')}
      >
        sample {settings.sampleSize} {settings.sam ? 'cells' : 'px'}
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
          checked={settings.sam}
          disabled={!samAvailable}
          onChange={(e) => wand.setSamSource(e.target.checked)}
        />
        SAM features
      </label>

      {/* Which encoder feature map the distance is measured on. Switching
          re-runs the clicks already placed, so the three are directly
          comparable on one region -- which is the point of exposing them
          at all rather than picking one. */}
      {settings.sam && (
        <label className="wand-levels">
          features
          <span className="segmented">
            {SAM_LEVELS.map((l) => (
              <button
                key={l.id}
                type="button"
                className={settings.level === l.id ? 'active' : ''}
                title={l.title}
                onClick={() => wand.setLevel(l.id)}
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
