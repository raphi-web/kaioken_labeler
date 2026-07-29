import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fromBase64, toBase64 } from './bridge'

// Undo/redo and the label buffer, driven through the app rather than in
// isolation, because the stacks are refs inside App and the contract they
// implement -- a stroke is one Map<pixelIndex, previousValue> -- is what every
// tool commits through.
//
// The bridge is mocked; the canvas work it triggers is inert under happy-dom
// (no 2d context) and is covered by smoke.html instead.

const W = 8
const H = 8
const api = vi.fn()
vi.mock('./bridge', async (original) => ({
  ...(await original()),
  api: (...args) => api(...args),
}))

const { default: App } = await import('./App')

// Ten pixels of class 3, so there is something to clear and restore.
const PAINTED = 10
function seeded() {
  const buf = new Uint8Array(W * H).fill(255)
  for (let i = 0; i < PAINTED; i++) buf[i] = 3
  return buf
}
let pushed = []   // every label buffer the app has sent back

function respond(method, args) {
  switch (method) {
    case 'get_image':
      return { name: 't.tif', png: '', width: W, height: H,
               labels: toBase64(seeded()), labeled_pixels: PAINTED }
    case 'sam_available': return { available: false }
    case 'set_labels': {
      const buffer = fromBase64(args[0])
      pushed.push(buffer)
      return { labeled_pixels: buffer.filter((v) => v !== 255).length }
    }
    default: return { ok: true }
  }
}

beforeEach(() => {
  pushed = []
  api.mockReset()
  api.mockImplementation(async (method, ...args) => respond(method, args))
})

/** Mount the app and wait for the image to land. */
async function mount() {
  render(<App />)
  await waitFor(() => expect(api).toHaveBeenCalledWith('get_image'))
  await screen.findByText(/Labeled/)
}

const undoButton = () => screen.getByRole('button', { name: 'Undo' })
const redoButton = () => screen.getByRole('button', { name: 'Redo' })
const labeledCount = () => document.querySelector('.stat-value').textContent
const clearButton = () => screen.getByRole('button', { name: 'Clear Labels' })

/** The label buffer as the app last sent it. */
const latest = () => pushed.at(-1)

// An edit that needs no canvas coordinates. Painting goes through
// LabelerPane, which turns a pointer position into a pixel index -- and
// happy-dom has no layout, so every rect is zero and the position comes out
// NaN. Clear Labels commits through exactly the same contract (one
// Map<pixelIndex, previousValue>, recordUndo, pushLabels), which is what these
// tests are about. Painting itself is covered in smoke.html.
async function edit() {
  fireEvent.click(clearButton())
  await waitFor(() => expect(undoButton().disabled).toBe(false))
}

describe('undo and redo', () => {
  it('start disabled, and a keystroke with nothing to undo does nothing', async () => {
    await mount()
    expect(undoButton().disabled).toBe(true)
    expect(redoButton().disabled).toBe(true)
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await new Promise((r) => setTimeout(r, 20))
    expect(pushed).toHaveLength(0)
  })

  it('an edit enables undo, and undoing it restores every pixel', async () => {
    await mount()
    expect(labeledCount()).toBe(`${PAINTED.toLocaleString()} px`)
    await edit()
    expect(latest().every((v) => v === 255)).toBe(true)
    expect(labeledCount()).toBe('0 px')

    fireEvent.click(undoButton())
    await waitFor(() => expect(redoButton().disabled).toBe(false))
    expect([...latest()]).toEqual([...seeded()])
    expect(labeledCount()).toBe(`${PAINTED.toLocaleString()} px`)
    expect(undoButton().disabled).toBe(true)

    fireEvent.click(redoButton())
    await waitFor(() => expect(undoButton().disabled).toBe(false))
    expect(latest().every((v) => v === 255)).toBe(true)
  })

  it('a new edit forks history, so redo cannot resurrect an older one', async () => {
    // Otherwise redo pastes back a state that predates what is on screen.
    await mount()
    await edit()
    fireEvent.click(undoButton())
    await waitFor(() => expect(redoButton().disabled).toBe(false))

    await edit()
    expect(redoButton().disabled).toBe(true)

    // And the stack itself must be empty, not merely reported as empty: the
    // keyboard path does not consult the button's disabled state, so a stale
    // entry left behind would still be reachable with Ctrl+Y.
    const before = pushed.length
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    await new Promise((r) => setTimeout(r, 30))
    expect(pushed).toHaveLength(before)
  })

  it('Ctrl+Z undoes; Ctrl+Shift+Z and Ctrl+Y both redo', async () => {
    await mount()
    await edit()

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await waitFor(() => expect(redoButton().disabled).toBe(false))

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true, shiftKey: true })
    await waitFor(() => expect(undoButton().disabled).toBe(false))

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await waitFor(() => expect(redoButton().disabled).toBe(false))
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    await waitFor(() => expect(undoButton().disabled).toBe(false))
  })

  it('every step is pushed to the backend', async () => {
    // The buffer only reaches disk through set_labels, so an undo that edits
    // the local array without pushing would be lost on the next image switch.
    await mount()
    await edit()
    expect(pushed).toHaveLength(1)
    fireEvent.click(undoButton())
    await waitFor(() => expect(pushed).toHaveLength(2))
    fireEvent.click(redoButton())
    await waitFor(() => expect(pushed).toHaveLength(3))
  })
})

describe('class selection', () => {
  it('number keys pick a class and E toggles the eraser', async () => {
    await mount()
    const active = () => document.querySelector('.class-item.active')?.textContent
    expect(active()).toContain('Urban')
    fireEvent.keyDown(window, { key: '2' })
    await waitFor(() => expect(active()).toContain('Road'))
    fireEvent.keyDown(window, { key: 'e' })
    await waitFor(() => expect(active()).toContain('Eraser'))
  })

  it('ignores class keys while a modifier is held, so Ctrl+Y is not class 9', async () => {
    await mount()
    const active = () => document.querySelector('.class-item.active')?.textContent
    fireEvent.keyDown(window, { key: '3', ctrlKey: true })
    await new Promise((r) => setTimeout(r, 20))
    expect(active()).toContain('Urban')
  })
})
