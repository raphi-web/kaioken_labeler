// The wand's tunables, and the measurements that set them.
//
// Separated from the hook because these are the numbers someone will come back
// to argue with, and they should be readable without the state machine around
// them. Every default here was measured against this project's own images and
// labels; spec.txt carries the long form.

// Alpha of the pending selection, well below the committed label alpha (255)
// so a preview never reads as something already painted, and light enough that
// the imagery stays legible underneath while you judge the fit.
export const WAND_PREVIEW_ALPHA = 102 // 40% of 255
// The two tones of the selection outline. Near-white inside, near-black just
// outside: one tone always loses against some class colour or some terrain,
// the pair reads on both.
//
// Same alpha as the fill, so the edge is distinguished by its tone rather than
// by being heavier -- the near-white/near-black pair carries far more contrast
// against imagery than the class colour does, which is what makes the boundary
// findable without the outline dominating the view.
export const WAND_EDGE_LIGHT = 255
export const WAND_EDGE_DARK = 20
export const WAND_EDGE_ALPHA = 102 // 40% of 255, matching the fill
// Past this share of the image the selection is called out as a runaway.
export const WAND_WARN_FRACTION = 0.4

// Tolerance, as an integer slider. The backend wants an RMS per-band
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
export const WAND_TOL_MIN = 1
export const WAND_TOL_MAX = 100
export const WAND_TOL_SCALE = 200 // slider value / SCALE = the backend's tolerance
export const WAND_TOL_DEFAULT = 26 // = 0.130, for a class with no measured optimum

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
export const WAND_TOL_SEEDS = {
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

export function tolSeedKey(name) {
  return String(name ?? '').toLowerCase().replace(/[^a-z0-9]/g, '')
}

// With the SAM-features switch on, the wand measures a cosine distance between
// mean-centred SAM2 feature vectors (see Api._sam_features), which has nothing
// to do with an RMS band difference -- so the slider stays 1..100 and maps onto
// its own range.
//
// Not a bare divide, because the useful distances do not start at zero. On a
// 512px Sentinel-2 tile at the fine level, three clicks each selected under
// 0.15% of the image below 0.7, 0.6-2.3% at 0.8, and 7-30% by 1.0: a region
// at 0.7-0.95, a speck below, the scene above. Mapping the slider onto
// [0.4, 1.2] puts that band across the middle of the travel and still leaves
// the deep level's lower band (0.5-0.8) inside it. A plain /100 left the whole
// working range pinned to the right-hand fifth of the slider.
export const SAM_TOL_FLOOR = 0.4
export const SAM_TOL_CEIL = 1.2
export const SAM_TOL_DEFAULT = 50 // = 0.8, mid-band for the fine level

export function samTolerance(slider) {
  return SAM_TOL_FLOOR + (slider / WAND_TOL_MAX) * (SAM_TOL_CEIL - SAM_TOL_FLOOR)
}

// With the embedding source on, the wand measures a cosine distance between
// mean-centred 8-dim land-cover embeddings from auto_encoder/embedding.onnx
// (see backend/wand.py embed_features and auto_encoder/ONNX.md). Same
// construction as SAM's band: the slider spans the measured useful range.
//
// Measured by bench_wand.py sources over the project's own labels (see
// implementation_log.txt): the embed arm peaks around 0.08 mean IoU-wise with
// the within-10%-of-peak band spanning roughly 0.04..0.10, and IoU collapses
// past ~0.12 as fills merge whole class mosaics. The floor sits just under
// the band so the left of the slider still narrows, and the ceiling past the
// collapse so the right end is visibly "too much" rather than silently pinned.
export const EMBED_TOL_FLOOR = 0.02
export const EMBED_TOL_CEIL = 0.16
export const EMBED_TOL_DEFAULT = 50 // = 0.09, mid-band

export function embedTolerance(slider) {
  return EMBED_TOL_FLOOR + (slider / WAND_TOL_MAX) * (EMBED_TOL_CEIL - EMBED_TOL_FLOOR)
}

// The tolerance a slider position means, in the units of `source`. The three
// distance measures (RMS band difference, SAM cosine, embed cosine) share
// nothing but the 1..100 slider, so each maps through its own curve.
export function toleranceFor(source, slider) {
  if (source === 'sam') return samTolerance(slider)
  if (source === 'embed') return embedTolerance(slider)
  return slider / WAND_TOL_SCALE
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
export const SAM_LEVELS = [
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
export const WAND_BUDGET_MIN = 1
export const WAND_BUDGET_MAX = 100 // at the max the budget is lifted entirely

export function wandBudget(slider) {
  if (slider >= WAND_BUDGET_MAX) return null // unlimited
  const raw = 10 ** (2 + (3 * (slider - 1)) / (WAND_BUDGET_MAX - 2))
  // Two significant figures: the exact value of a log slider is noise, and a
  // readout that jitters between 3,162 and 3,271 is harder to aim with.
  const mag = 10 ** (Math.floor(Math.log10(raw)) - 1)
  return Math.round(raw / mag) * mag
}

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
export const WAND_SAMPLE_MIN = 1
export const WAND_SAMPLE_MAX = 25
export const WAND_SAMPLE_DEFAULT = 9
