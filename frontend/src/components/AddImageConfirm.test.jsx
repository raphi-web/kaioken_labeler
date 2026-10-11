import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AddImageConfirm from './AddImageConfirm'

// The size-mismatch dialog has one job: show both grids in a way that makes
// the difference obvious, and only commit when the user explicitly says yes.
// The dismiss/cancel paths all clear the dialog without calling onConfirm.

const EXISTING = { width: 256, height: 256, name: 'existing.tif' }
const CANDIDATE = { width: 320, height: 320, name: 'fresh.tif', bands: 4 }

afterEach(cleanup)

describe('AddImageConfirm', () => {
  it('shows both grids and the candidate name', () => {
    render(<AddImageConfirm existing={EXISTING} candidate={CANDIDATE}
                            onConfirm={() => {}} onCancel={() => {}} />)
    const dialog = screen.getByRole('dialog')
    expect(dialog.textContent).toContain('256×256')
    expect(dialog.textContent).toContain('320×320')
    expect(dialog.textContent).toContain('fresh.tif')
  })

  it('Cancel does not call onConfirm', () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()
    render(<AddImageConfirm existing={EXISTING} candidate={CANDIDATE}
                            onConfirm={onConfirm} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('Add anyway does not call onCancel', () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()
    render(<AddImageConfirm existing={EXISTING} candidate={CANDIDATE}
                            onConfirm={onConfirm} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: 'Add anyway' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('Escape closes the dialog without confirming', () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()
    render(<AddImageConfirm existing={EXISTING} candidate={CANDIDATE}
                            onConfirm={onConfirm} onCancel={onCancel} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onConfirm).not.toHaveBeenCalled()
  })
})
