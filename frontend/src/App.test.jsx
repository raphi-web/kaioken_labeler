import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fromBase64, toBase64 } from './bridge'

// Undo/redo and the label buffer, driven through the app rather than in
// isolation, because the stacks are refs inside App and the contract they
// implement -- a stroke is one Map<pixelIndex, previousValue> -- is what every
// tool commits through.
//
// The cap-at-50 rule (undoStack.shift() when full) is a performance invariant
// rather than a correctness one; exercising it requires 51 distinct strokes
// which happy-dom cannot drive without pointer coordinates, so the rule lives
// here as documentation only.
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
    case 'get_project':
      // The frontend treats the absence of get_project as the standalone-image
      // case, so the existing test corpus expects null. The new add_image tests
      // override this with a real project; that override is per-test via
      // api.mockImplementationOnce below.
      return null
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
const paletteRow = (name) => [...document.querySelectorAll('.class-item')]
  .find((r) => r.textContent.includes(name))

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
    await waitFor(() => expect(active()).toContain('Agriculture-Green'))
    fireEvent.keyDown(window, { key: 'e' })
    await waitFor(() => expect(active()).toContain('Eraser'))
  })

  it('locking a class protects it from clear labels', async () => {
    await mount()
    // The seeded pixels are class 3 == Meadow since the Road removal shifted
    // the ids down; locking that class must spare them from Clear Labels.
    fireEvent.click(screen.getByRole('button', { name: 'Lock Meadow' }))
    await waitFor(() => expect(
      screen.getByRole('button', { name: 'Unlock Meadow' })).toBeTruthy())

    fireEvent.click(clearButton())
    await new Promise((r) => setTimeout(r, 20))
    expect(pushed).toHaveLength(0)
    expect(undoButton().disabled).toBe(true)
    expect(labeledCount()).toBe(`${PAINTED.toLocaleString()} px`)
  })

  it('hiding a class changes no pixel and no count', async () => {
    // The whole contract of the toggle: it is a view setting. Nothing about
    // the labels, what is saved, or what the palette counts may move.
    await mount()
    const countBefore = labeledCount()
    fireEvent.click(screen.getByRole('button', { name: 'Hide Urban' }))
    await waitFor(() => expect(
      screen.getByRole('button', { name: 'Show Urban' })).toBeTruthy())
    expect(labeledCount()).toBe(countBefore)
    expect(pushed).toHaveLength(0)      // nothing was written back
    // The seeded pixels are class 3, whichever name the palette gives it
    // (Meadow since Road left the taxonomy and the ids shifted down).
    const painted = paletteRow('Meadow')
    expect(painted.querySelector('.class-count').textContent)
      .toBe(PAINTED.toLocaleString())
  })

  it('picking a hidden class shows it again, from the palette and the keyboard', async () => {
    // You cannot paint what you cannot see, so selecting has to bring it back.
    await mount()
    const eye = (name) => screen.queryByRole('button', { name })

    fireEvent.click(screen.getByRole('button', { name: 'Hide Urban' }))
    await waitFor(() => expect(eye('Show Urban')).toBeTruthy())
    fireEvent.click(paletteRow('Urban').querySelector('.class-select'))
    await waitFor(() => expect(eye('Hide Urban')).toBeTruthy())

    // ... and via the number key, which goes through the same path.
    fireEvent.click(screen.getByRole('button', { name: 'Hide Agriculture-Green' }))
    await waitFor(() => expect(eye('Show Agriculture-Green')).toBeTruthy())
    fireEvent.keyDown(window, { key: '2' })
    await waitFor(() => expect(eye('Hide Agriculture-Green')).toBeTruthy())
  })

  it('ignores class keys while a modifier is held, so Ctrl+Y is not class 9', async () => {
    await mount()
    const active = () => document.querySelector('.class-item.active')?.textContent
    fireEvent.keyDown(window, { key: '3', ctrlKey: true })
    await new Promise((r) => setTimeout(r, 20))
    expect(active()).toContain('Urban')
  })
})

const PROJECT = () => ({
  root: '/p', name: 'demo',
  // Real enough classes to drive the toolbar's class palette and the brush
  // cursor's color lookup. The unique foci are the toolbar wiring (Add Image…
  // exists, Project dropdown enabled) and the dialog round-trip, not the
  // palette's internal rendering.
  classes: [
    { id: 0, name: 'Urban', color: '#e05a4f' },
    { id: 1, name: 'Agriculture-Green', color: '#3ddc6b' },
    { id: 2, name: 'Raw-Earth/Agriculture', color: '#c98f3c' },
    { id: 3, name: 'Meadow', color: '#d4e34a' },
  ],
  data_profile: { input_channels: 4, display_bands: [0, 1, 2],
                  band_names: ['a', 'b', 'c', 'd'] },
  paths: { images_folder: '.', masks_user: 'masks_user', masks_ai: 'masks_ai' },
  images: ['t.tif'], active: 't.tif' })

/** Mount the app with a real-looking project so the Project dropdown is enabled. */
async function mountWithProject() {
  api.mockImplementationOnce(async (method) => {
    if (method === 'get_project') return PROJECT()
    return respond(method, [])
  })
  return mount()
}

describe('add image to project', () => {
  it('offers an Add Image entry in the Project dropdown', async () => {
    await mountWithProject()
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    expect(screen.getByRole('button', { name: 'Add Image…' })).toBeTruthy()
  })

  it('asks the backend to add the chosen file', async () => {
    await mountWithProject()
    api.mockImplementationOnce(async (method) => {
      if (method === 'pick_file') return { path: 'fresh.tif' }
      return { ok: true }
    })
    api.mockImplementationOnce(async (method) => {
      if (method === 'add_image') return { ok: true, name: 'fresh.tif',
                                            width: W, height: H, bands: 4,
                                            project: { images: ['t.tif', 'fresh.tif'] } }
      return { ok: true }
    })
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Image…' }))
    await waitFor(() => expect(api).toHaveBeenCalledWith('add_image', 'fresh.tif', false))
  })

  it('shows a confirmation modal when the dimensions differ', async () => {
    await mountWithProject()
    api.mockImplementationOnce(async (method) => {
      if (method === 'pick_file') return { path: 'fresh.tif' }
      return { ok: true }
    })
    api.mockImplementationOnce(async (method, source) => {
      if (method === 'add_image') return {
        ok: false, requires_confirm: true, source,
        existing: { width: W, height: H, name: 't.tif' },
        candidate: { width: W + 8, height: H + 8, name: 'fresh.tif', bands: 4 },
      }
      return { ok: true }
    })
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Image…' }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog.textContent).toContain(`${W}×${H}`)
    expect(dialog.textContent).toContain(`${W + 8}×${H + 8}`)
  })

  it('re-calls add_image with force=true on confirm', async () => {
    await mountWithProject()
    api.mockImplementationOnce(async (method) => {
      if (method === 'pick_file') return { path: 'fresh.tif' }
      return { ok: true }
    })
    api.mockImplementationOnce(async (method, source) => {
      if (method === 'add_image') return {
        ok: false, requires_confirm: true, source,
        existing: { width: W, height: H, name: 't.tif' },
        candidate: { width: W + 8, height: H + 8, name: 'fresh.tif', bands: 4 },
      }
      return { ok: true }
    })
    api.mockImplementationOnce(async (method, source, force) => {
      if (method === 'add_image' && force === true) return {
        ok: true, name: 'fresh.tif', width: W + 8, height: H + 8, bands: 4,
        project: { images: ['t.tif', 'fresh.tif'] },
      }
      return { ok: true }
    })
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Image…' }))
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: 'Add anyway' }))
    await waitFor(() => expect(api).toHaveBeenCalledWith('add_image', 'fresh.tif', true))
  })

  it('just closes the dialog on Cancel', async () => {
    await mountWithProject()
    api.mockImplementationOnce(async (method) => {
      if (method === 'pick_file') return { path: 'fresh.tif' }
      return { ok: true }
    })
    api.mockImplementationOnce(async (method, source) => {
      if (method === 'add_image') return {
        ok: false, requires_confirm: true, source,
        existing: { width: W, height: H, name: 't.tif' },
        candidate: { width: W + 8, height: H + 8, name: 'fresh.tif', bands: 4 },
      }
      return { ok: true }
    })
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Image…' }))
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    const addCalls = api.mock.calls.filter(([m]) => m === 'add_image')
    expect(addCalls).toHaveLength(1)
  })

  it('surfaces an error from the backend as a toast', async () => {
    await mountWithProject()
    api.mockImplementationOnce(async (method) => {
      if (method === 'pick_file') return { path: 'fresh.tif' }
      return { ok: true }
    })
    api.mockImplementationOnce(async (method) => {
      if (method === 'add_image') return { ok: false, error: 'no good' }
      return { ok: true }
    })
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Image…' }))
    await waitFor(() => expect(screen.getByText('no good')).toBeTruthy())
  })
})
