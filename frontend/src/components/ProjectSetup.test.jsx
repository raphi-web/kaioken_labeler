import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The dialog asks the backend what a deletion would cost, so the bridge is
// mocked rather than stubbed on window.
const api = vi.fn()
vi.mock('../bridge', () => ({ api: (...args) => api(...args) }))

const { default: ProjectSetup } = await import('./ProjectSetup')

// Removing a class rewrites every saved mask and cannot be undone. It is the
// only irreversible thing in the app, which is why it gets its own file: the
// warning, the count and the confirm are the guard rails, and a regression in
// any of them destroys labels silently.

const CLASSES = [
  { id: 0, name: 'Urban', color: '#e05a4f' },
  { id: 1, name: 'Road', color: '#4a4a55' },
  { id: 4, name: 'Meadow', color: '#d4e34a' },
]

const DEFAULTS = {
  project_name: 'demo',
  input_channels: 4,
  input_patch_size: 96,
  band_names: ['a', 'b', 'c', 'd'],
  display_bands: [0, 1, 2],
  images_folder: '.',
  masks_user: 'masks_user',
  bands_detected: true,
  classes: CLASSES,
}

function open(props = {}) {
  const onSubmit = vi.fn()
  render(<ProjectSetup mode="edit" root="/p" defaults={DEFAULTS}
                       onSubmit={onSubmit} onCancel={() => {}} {...props} />)
  return { onSubmit }
}

const rows = () => document.querySelectorAll('.class-row')
const removeRow = (i) => fireEvent.click(document.querySelectorAll('.class-row-remove')[i])
const save = () => screen.getByRole('button', { name: /Save Settings/ })

beforeEach(() => {
  api.mockReset()
  api.mockResolvedValue({ ok: true, usage: {} })
})

describe('the class list', () => {
  it('renders a row per class, with its shortcut key', () => {
    open()
    expect(rows()).toHaveLength(3)
    const keys = [...document.querySelectorAll('.class-row-key')].map((e) => e.textContent)
    expect(keys).toEqual(['1', '2', '3'])
  })

  it('adds and removes rows', () => {
    open()
    fireEvent.click(screen.getByRole('button', { name: /Add class/ }))
    expect(rows()).toHaveLength(4)
    removeRow(3)
    expect(rows()).toHaveLength(3)
  })

  it('will not let the last class be removed', () => {
    open()
    removeRow(0)
    removeRow(0)
    expect(rows()).toHaveLength(1)
    expect(document.querySelector('.class-row-remove').disabled).toBe(true)
  })

  it('blocks saving while a class has no name', () => {
    open()
    const name = document.querySelector('.class-row input[type=text]')
    fireEvent.change(name, { target: { value: '  ' } })
    expect(save().disabled).toBe(true)
    fireEvent.change(name, { target: { value: 'Rooftop' } })
    expect(save().disabled).toBe(false)
  })
})

describe('removing a class that has labels', () => {
  it('asks the backend what the deletion would destroy', async () => {
    open()
    removeRow(2)  // Meadow, id 4
    await waitFor(() => expect(api).toHaveBeenCalledWith('class_usage', [4]))
  })

  it('reports the count and demands a confirmation before saving', async () => {
    api.mockResolvedValue({ ok: true, usage: { 4: { pixels: 12345, images: 7 } } })
    const { onSubmit } = open()
    removeRow(2)

    const warning = await screen.findByText(/unlabels/)
    expect(warning.textContent).toContain('Meadow')
    // toLocaleString, so the separator is whatever the runtime's locale uses.
    expect(warning.textContent).toContain(`${(12345).toLocaleString()} px in 7 images`)
    expect(save().disabled).toBe(true)

    fireEvent.click(screen.getByRole('checkbox', { name: /unlabel those pixels/i }))
    expect(save().disabled).toBe(false)
    fireEvent.click(save())
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('re-arms the confirmation when the removal set changes', async () => {
    api.mockResolvedValue({ ok: true, usage: { 1: { pixels: 5, images: 1 },
                                               4: { pixels: 9, images: 1 } } })
    open()
    removeRow(2)
    await screen.findByText(/unlabels/)
    fireEvent.click(screen.getByRole('checkbox', { name: /unlabel those pixels/i }))
    expect(save().disabled).toBe(false)

    removeRow(1)  // a second removal: the earlier tick must not carry over
    await waitFor(() => expect(save().disabled).toBe(true))
  })

  it('needs no confirmation for a class nobody painted', async () => {
    api.mockResolvedValue({ ok: true, usage: { 4: { pixels: 0, images: 0 } } })
    open()
    removeRow(2)
    expect(await screen.findByText(/no labeled pixels/)).toBeTruthy()
    expect(save().disabled).toBe(false)
  })
})

describe('what the dialog submits', () => {
  it('keeps ids on surviving rows and omits them on new ones', async () => {
    const { onSubmit } = open()
    fireEvent.change(document.querySelectorAll('.class-row input[type=text]')[0],
                     { target: { value: 'Built-up' } })
    removeRow(1)  // drop Road
    fireEvent.click(screen.getByRole('button', { name: /Add class/ }))
    await waitFor(() => expect(save().disabled).toBe(false))
    fireEvent.click(save())

    const { classes } = onSubmit.mock.calls[0][0]
    expect(classes.map((c) => [c.id, c.name])).toEqual([
      [0, 'Built-up'],        // renamed, id kept -- its pixels stay its pixels
      [4, 'Meadow'],
      [undefined, 'Class 3'], // new: the backend assigns from its own counter
    ])
  })

  it('offers presets only when creating, not when editing', () => {
    open()
    expect(document.querySelector('.class-preset')).toBe(null)

    const presets = { binary: [{ id: 0, name: 'Target', color: '#ff5028' },
                               { id: 1, name: 'Background', color: '#3c8cff' }] }
    render(<ProjectSetup mode="create" root="/p"
                         defaults={{ ...DEFAULTS, class_presets: presets,
                                     class_preset: 'binary' }}
                         onSubmit={() => {}} onCancel={() => {}} />)
    const dialogs = document.querySelectorAll('.modal')
    expect(dialogs[1].querySelector('.class-preset select')).toBeTruthy()
  })

  it('replaces the whole list when a preset is chosen', () => {
    const presets = { binary: [{ id: 0, name: 'Target', color: '#ff5028' },
                               { id: 1, name: 'Background', color: '#3c8cff' }] }
    render(<ProjectSetup mode="create" root="/p"
                         defaults={{ ...DEFAULTS, class_presets: presets,
                                     class_preset: 'sentinel2' }}
                         onSubmit={() => {}} onCancel={() => {}} />)
    fireEvent.change(document.querySelector('.class-preset select'),
                     { target: { value: 'binary' } })
    const names = [...document.querySelectorAll('.class-row input[type=text]')]
      .map((i) => i.value)
    expect(names).toEqual(['Target', 'Background'])
  })
})
