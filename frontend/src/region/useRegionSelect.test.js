import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { UNLABELED } from '../constants'
import { useRegionSelect } from './useRegionSelect'

// The selection state on top of connectedRegion: what a click adds, what
// Backspace takes away, and exactly which pixels a commit hands over.
//
// No canvas is attached, so the preview early-returns. That is the same split
// as the wand: what gets painted is smoke.html's business, in a real browser.

const W = 6
const H = 4

//  0 0 . 1 1 .
//  0 0 . 1 1 .
//  . . . . . .
//  2 . . . . .
function labels() {
  const buf = new Uint8Array(W * H).fill(UNLABELED)
  const put = (x, y, v) => { buf[y * W + x] = v }
  put(0, 0, 0); put(1, 0, 0); put(0, 1, 0); put(1, 1, 0)
  put(3, 0, 1); put(4, 0, 1); put(3, 1, 1); put(4, 1, 1)
  put(0, 3, 2)
  return buf
}

const IMAGE = { name: 't.tif', width: W, height: H }

function setup(buffer = labels()) {
  const onDelete = vi.fn()
  const view = renderHook(() => useRegionSelect({
    active: true, image: IMAGE, labels: buffer, labelsVersion: 1, onDelete,
  }))
  return { ...view, onDelete, buffer }
}

const click = (result, x, y) => act(() => { result.current.addAt({ x, y }) })

describe('selecting', () => {
  it('takes the object under the click, not the whole class', async () => {
    const { result } = setup()
    await click(result, 0, 0)
    expect(result.current.regions).toHaveLength(1)
    expect(result.current.count).toBe(4)   // the left block only
  })

  it('adds a second object without disturbing the first', async () => {
    const { result } = setup()
    await click(result, 0, 0)
    await click(result, 3, 0)
    expect(result.current.regions).toHaveLength(2)
    expect(result.current.count).toBe(8)
  })

  it('ignores a click on unlabeled ground', async () => {
    const { result } = setup()
    await click(result, 2, 2)
    expect(result.current.regions).toHaveLength(0)
    expect(result.current.count).toBe(0)
  })

  it('ignores a second click inside a region already selected', async () => {
    // Otherwise Backspace has to be pressed twice to undo one visible action.
    const { result } = setup()
    await click(result, 0, 0)
    await click(result, 1, 1)
    expect(result.current.regions).toHaveLength(1)
  })

  it('clamps a click outside the image rather than reading past the buffer', async () => {
    const { result } = setup()
    await click(result, -5, -5)          // clamps to (0, 0), which is labeled
    expect(result.current.count).toBe(4)
  })
})

describe('undoing and cancelling', () => {
  it('Backspace drops the most recent region only', async () => {
    const { result } = setup()
    await click(result, 0, 0)
    await click(result, 0, 3)            // the lone class-2 pixel
    expect(result.current.count).toBe(5)
    await act(() => { result.current.undo() })
    expect(result.current.regions).toHaveLength(1)
    expect(result.current.count).toBe(4)
  })

  it('cancel clears everything', async () => {
    const { result } = setup()
    await click(result, 0, 0)
    await act(() => { result.current.cancel() })
    expect(result.current.regions).toHaveLength(0)
  })

  it('a changed label buffer drops the selection, which may no longer exist', async () => {
    const onDelete = vi.fn()
    const { result, rerender } = renderHook(
      ({ version }) => useRegionSelect({
        active: true, image: IMAGE, labels: labels(), labelsVersion: version, onDelete,
      }),
      { initialProps: { version: 1 } },
    )
    await act(() => { result.current.addAt({ x: 0, y: 0 }) })
    expect(result.current.regions).toHaveLength(1)
    rerender({ version: 2 })             // an undo, or another tool's stroke
    expect(result.current.regions).toHaveLength(0)
  })

  it('going inactive drops the selection', async () => {
    const onDelete = vi.fn()
    const buffer = labels()
    const { result, rerender } = renderHook(
      ({ active }) => useRegionSelect({
        active, image: IMAGE, labels: buffer, labelsVersion: 1, onDelete,
      }),
      { initialProps: { active: true } },
    )
    await act(() => { result.current.addAt({ x: 0, y: 0 }) })
    rerender({ active: false })
    expect(result.current.regions).toHaveLength(0)
  })
})

describe('committing', () => {
  it('hands over exactly the selected pixels, then clears', async () => {
    const { result, onDelete } = setup()
    await click(result, 0, 0)
    await click(result, 0, 3)
    await act(() => { result.current.commit() })

    expect(onDelete).toHaveBeenCalledTimes(1)
    const mask = onDelete.mock.calls[0][0]
    const on = []
    for (let i = 0; i < mask.length; i++) if (mask[i]) on.push(i)
    expect(on).toEqual([0, 1, W, W + 1, 3 * W])   // the block plus the lone pixel
    expect(result.current.regions).toHaveLength(0)
  })

  it('commits both regions as one call, so one undo brings them all back', async () => {
    const { result, onDelete } = setup()
    await click(result, 0, 0)
    await click(result, 3, 0)
    await act(() => { result.current.commit() })
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('does nothing with an empty selection', async () => {
    const { result, onDelete } = setup()
    await act(() => { result.current.commit() })
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('a dropped region is not deleted', async () => {
    // The union is rebuilt from the list, so undoing a click has to actually
    // remove its pixels rather than leave them in an accumulator.
    const { result, onDelete } = setup()
    await click(result, 0, 0)
    await click(result, 3, 0)
    await act(() => { result.current.undo() })
    await act(() => { result.current.commit() })
    const mask = onDelete.mock.calls[0][0]
    expect(mask[3]).toBe(0)      // (3, 0) belonged to the dropped region
    expect(mask[0]).toBe(1)
  })
})
