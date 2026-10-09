import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { WAND_TOL_SCALE, embedTolerance, samTolerance } from './constants'
import { useWand } from './useWand'

// The wand's state machine, with the backend replaced by a spy. What is worth
// pinning here is not "does a click select pixels" -- smoke.html does that
// against the real thing -- but the rules the hook enforces on top: which
// tolerance a click uses, what survives a settings change, and which answers
// get thrown away.
//
// Nothing attaches a canvas, so drawWandPreview early-returns. That is fine:
// the painting is smoke.html's job, and the hook's job is what runs before it.

const CLASSES = [
  { id: 0, name: 'Urban', color: '#e05a4f' },               // seed 38
  { id: 1, name: 'Agriculture-Green', color: '#3ddc6b' },   // seed 18
  { id: 6, name: 'Water', color: '#2f7ce0' },               // seed 14
  { id: 7, name: 'Snow', color: '#eef2f7' },                // no seed -> 26
]

const SELECTION = { mask: new Uint8Array(16), count: 4, available: 4,
                    protected: 0, capped: false }

function setup({ onSelect, classId = 0, eraser = false } = {}) {
  const select = onSelect ?? vi.fn(async () => SELECTION)
  const onCommit = vi.fn()
  const view = renderHook(
    ({ tool }) => useWand({
      active: true,
      image: { width: 4, height: 4 },
      tool,
      classes: CLASSES,
      classColors: { 0: [1, 2, 3], 1: [4, 5, 6], 7: [7, 8, 9], 8: [0, 0, 0] },
      onSelect: select,
      onCommit,
    }),
    { initialProps: { tool: { classId, eraser, mode: 'wand' } } },
  )
  return { ...view, select, onCommit }
}

const lastOptions = (select) => select.mock.calls.at(-1)[1]

async function click(result, x = 1, y = 1) {
  await act(async () => { result.current.addSample({ x, y }, 1) })
}

describe('tolerance', () => {
  it('uses the seed measured for the active class', async () => {
    const { result, select } = setup({ classId: 0 })
    await click(result)
    expect(lastOptions(select).tolerance).toBe(38 / WAND_TOL_SCALE)
  })

  it('falls back to the default for a class with no measured seed', async () => {
    const { result, select } = setup({ classId: 8 })  // Snow
    await click(result)
    expect(lastOptions(select).tolerance).toBe(26 / WAND_TOL_SCALE)
  })

  it('is remembered per class', async () => {
    const { result, select, rerender } = setup({ classId: 0 })
    await click(result)
    await act(async () => { result.current.setTolerance(70) })
    expect(lastOptions(select).tolerance).toBe(70 / WAND_TOL_SCALE)

    // Another class keeps its own seed (Agriculture-Green: 18) ...
    rerender({ tool: { classId: 1, eraser: false, mode: 'wand' } })
    await waitFor(() => expect(lastOptions(select).tolerance).toBe(18 / WAND_TOL_SCALE))

    // ... and coming back restores what was set, not the seed.
    rerender({ tool: { classId: 0, eraser: false, mode: 'wand' } })
    await waitFor(() => expect(lastOptions(select).tolerance).toBe(70 / WAND_TOL_SCALE))
  })

  it('changing class re-runs the pending selection', async () => {
    const { result, select, rerender } = setup({ classId: 0 })
    await click(result)
    const before = select.mock.calls.length
    rerender({ tool: { classId: 1, eraser: false, mode: 'wand' } })
    await waitFor(() => expect(select.mock.calls.length).toBeGreaterThan(before))
  })

  it('does not re-run on a class change with nothing pending', async () => {
    const { select, rerender } = setup({ classId: 0 })
    rerender({ tool: { classId: 1, eraser: false, mode: 'wand' } })
    await new Promise((r) => setTimeout(r, 20))
    expect(select).not.toHaveBeenCalled()
  })
})

describe('the source switch', () => {
  it('swaps the source and the tolerance units, and back again', async () => {
    const { result, select } = setup({ classId: 0 })
    await click(result)
    expect(lastOptions(select).source).toBe('image')

    await act(async () => { result.current.setSource('sam') })
    expect(lastOptions(select).source).toBe('sam')
    expect(lastOptions(select).tolerance).toBeCloseTo(samTolerance(50), 10)
    expect(lastOptions(select).level).toBe('fine')

    await act(async () => { result.current.setSource('image') })
    expect(lastOptions(select).source).toBe('image')
    expect(lastOptions(select).tolerance).toBe(38 / WAND_TOL_SCALE)
  })

  it('keeps one tolerance for every class', async () => {
    const { result, select, rerender } = setup({ classId: 0 })
    await click(result)
    await act(async () => { result.current.setSource('sam') })
    await act(async () => { result.current.setTolerance(80) })
    const wanted = samTolerance(80)

    rerender({ tool: { classId: 7, eraser: false, mode: 'wand' } })
    await act(async () => { result.current.setLevel('deep') })
    expect(lastOptions(select).tolerance).toBeCloseTo(wanted, 10)
    expect(lastOptions(select).level).toBe('deep')
  })

  it('re-runs on a level change but not on a no-op one', async () => {
    const { result, select } = setup()
    await click(result)
    await act(async () => { result.current.setSource('sam') })
    const before = select.mock.calls.length
    await act(async () => { result.current.setLevel('fine') })  // already fine
    expect(select.mock.calls.length).toBe(before)
    await act(async () => { result.current.setLevel('mid') })
    expect(select.mock.calls.length).toBe(before + 1)
  })

  it('a no-op source change does not re-run', async () => {
    const { result, select } = setup()
    await click(result)
    const before = select.mock.calls.length
    await act(async () => { result.current.setSource('image') })  // already image
    expect(select.mock.calls.length).toBe(before)
  })
})

describe('the embed source', () => {
  it('sends the embed source with its own tolerance mapping', async () => {
    const { result, select } = setup({ classId: 0 })
    await click(result)
    await act(async () => { result.current.setSource('embed') })
    expect(lastOptions(select).source).toBe('embed')
    expect(lastOptions(select).tolerance).toBeCloseTo(embedTolerance(50), 10)
    // The level is a SAM-only knob; it still travels (the backend ignores it)
    // so the wire shape stays one object.
    expect(lastOptions(select).level).toBe('fine')
  })

  it('keeps its own tolerance, separate from SAM and spectral', async () => {
    const { result, select } = setup({ classId: 0 })
    await click(result)
    await act(async () => { result.current.setSource('embed') })
    await act(async () => { result.current.setTolerance(70) })
    expect(lastOptions(select).tolerance).toBeCloseTo(embedTolerance(70), 10)

    await act(async () => { result.current.setSource('sam') })
    expect(lastOptions(select).tolerance).toBeCloseTo(samTolerance(50), 10)

    await act(async () => { result.current.setSource('embed') })
    expect(lastOptions(select).tolerance).toBeCloseTo(embedTolerance(70), 10)

    await act(async () => { result.current.setSource('image') })
    expect(lastOptions(select).tolerance).toBe(38 / WAND_TOL_SCALE)
  })

  it('keeps one tolerance for every class, like SAM', async () => {
    const { result, select, rerender } = setup({ classId: 0 })
    await click(result)
    await act(async () => { result.current.setSource('embed') })
    await act(async () => { result.current.setTolerance(60) })
    const wanted = embedTolerance(60)
    rerender({ tool: { classId: 7, eraser: false, mode: 'wand' } })
    await new Promise((r) => setTimeout(r, 20))
    // A class switch must not re-run (one tolerance across classes) nor move
    // the number.
    expect(lastOptions(select).tolerance).toBeCloseTo(wanted, 10)
    expect(lastOptions(select).source).toBe('embed')
  })

  it('carries every other knob unchanged', async () => {
    const { result, select } = setup()
    await click(result)
    await act(async () => { result.current.setSource('embed') })
    await act(async () => { result.current.setGlobal(true) })
    await act(async () => { result.current.setProtect(false) })
    const o = lastOptions(select)
    expect(o.source).toBe('embed')
    expect(o.global).toBe(true)
    expect(o.protect).toBe(false)
    expect(o.sample_size).toBe(9)
  })

  it('switching source re-runs the pending samples', async () => {
    const { result, select } = setup()
    await click(result, 1, 1)
    const before = select.mock.calls.length
    await act(async () => { result.current.setSource('embed') })
    expect(select.mock.calls.length).toBe(before + 1)
    expect(select.mock.calls.at(-1)[0]).toEqual([{ x: 1, y: 1, label: 1 }])
    await act(async () => { result.current.setSource('sam') })
    expect(lastOptions(select).source).toBe('sam')
  })
})

describe('what reaches the backend', () => {
  it('sends the shipped defaults', async () => {
    const { result, select } = setup()
    await click(result)
    expect(lastOptions(select)).toMatchObject({
      source: 'image', level: 'fine', sample_size: 9,
      max_pixels: null, protect: true, global: false, negatives: [],
    })
  })

  it('never protects while the eraser is active', async () => {
    // The one combination where the toggle inverts its own purpose: the eraser
    // only ever acts on labeled pixels, so protecting them makes it a no-op.
    const { result, select } = setup({ eraser: true })
    await click(result)
    expect(result.current.settings.protect).toBe(true)
    expect(lastOptions(select).protect).toBe(false)
  })

  it('splits positive and negative clicks', async () => {
    const { result, select } = setup()
    await click(result, 1, 1)
    await act(async () => { result.current.addSample({ x: 2, y: 2 }, 0) })
    expect(select.mock.calls.at(-1)[0]).toEqual([{ x: 1, y: 1, label: 1 }])
    expect(lastOptions(select).negatives).toEqual([[2, 2]])
  })

  it('does not call the backend for negatives alone', async () => {
    const { result, select } = setup()
    await act(async () => { result.current.addSample({ x: 2, y: 2 }, 0) })
    expect(select).not.toHaveBeenCalled()
    expect(result.current.count).toBe(null)
  })
})

describe('the pending selection', () => {
  it('reports the count and the share of the image', async () => {
    const { result } = setup()
    await click(result)
    expect(result.current.count).toBe(4)
    expect(result.current.fraction).toBe(4 / 16)
  })

  it('undoing a sample drops the last one and re-runs', async () => {
    const { result, select } = setup()
    await click(result, 1, 1)
    await click(result, 2, 2)
    expect(result.current.samples).toHaveLength(2)
    await act(async () => { result.current.undoSample() })
    expect(result.current.samples).toEqual([{ x: 1, y: 1, label: 1 }])
    expect(select.mock.calls.at(-1)[0]).toEqual([{ x: 1, y: 1, label: 1 }])
  })

  it('cancel clears the samples and the readout', async () => {
    const { result } = setup()
    await click(result)
    await act(async () => { result.current.cancel() })
    expect(result.current.samples).toEqual([])
    expect(result.current.count).toBe(null)
    expect(result.current.busy).toBe(false)
  })

  it('commit hands the mask over and then clears', async () => {
    const { result, onCommit } = setup()
    await click(result)
    await act(async () => { result.current.commit() })
    expect(onCommit).toHaveBeenCalledWith(SELECTION.mask)
    expect(result.current.samples).toEqual([])
    expect(result.current.count).toBe(null)
  })

  it('commit with nothing pending does nothing', async () => {
    const { result, onCommit } = setup()
    await act(async () => { result.current.commit() })
    expect(onCommit).not.toHaveBeenCalled()
  })

  it('is dropped when the tool goes inactive', async () => {
    const onCommit = vi.fn()
    const select = vi.fn(async () => SELECTION)
    const { result, rerender } = renderHook(
      ({ active }) => useWand({
        active,
        image: { width: 4, height: 4 },
        tool: { classId: 0, eraser: false, mode: 'wand' },
        classes: CLASSES,
        classColors: { 0: [1, 2, 3] },
        onSelect: select,
        onCommit,
      }),
      { initialProps: { active: true } },
    )
    await act(async () => { result.current.addSample({ x: 1, y: 1 }, 1) })
    expect(result.current.count).toBe(4)
    rerender({ active: false })
    await waitFor(() => expect(result.current.count).toBe(null))
    expect(result.current.samples).toEqual([])
  })
})

describe('out-of-order responses', () => {
  it('keeps the newest answer and discards one that lands late', async () => {
    // Dragging a slider fires one request per step. Without the sequence guard
    // a slow early response can land last and leave the preview showing a
    // value the slider has already moved past.
    const pending = []
    const select = vi.fn(() => new Promise((resolve) => { pending.push(resolve) }))
    const { result } = setup({ select, onSelect: select })

    await act(async () => { result.current.addSample({ x: 1, y: 1 }, 1) })
    await act(async () => { result.current.setTolerance(70) })
    expect(pending).toHaveLength(2)

    // Resolve the NEWER one first, then let the stale one land.
    await act(async () => { pending[1]({ ...SELECTION, count: 99 }) })
    await act(async () => { pending[0]({ ...SELECTION, count: 11 }) })

    expect(result.current.count).toBe(99)
    expect(result.current.busy).toBe(false)
  })

  it('a response that lands after cancel is ignored', async () => {
    const pending = []
    const select = vi.fn(() => new Promise((resolve) => { pending.push(resolve) }))
    const { result } = setup({ onSelect: select })

    await act(async () => { result.current.addSample({ x: 1, y: 1 }, 1) })
    await act(async () => { result.current.cancel() })
    await act(async () => { pending[0](SELECTION) })

    expect(result.current.count).toBe(null)
    expect(result.current.samples).toEqual([])
  })
})
