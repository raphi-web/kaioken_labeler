import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Toast from './Toast'

// A single-line toast at the bottom-center of the window. Auto-dismisses
// after 3s; pauses its dismiss timer on hover so a slow reader doesn't lose
// the message mid-sentence. Reuses the .floating-error class App.css already
// defines for menu errors -- the look was right for that role and is right
// for this one too.
//
// Three behaviours pin the contract the rest of the app relies on:
//   - hides when there's no message
//   - shows the message verbatim
//   - auto-dismisses via the parent-supplied callback

const TOAST_CLASS = '.floating-error.toast'

afterEach(cleanup)

describe('Toast', () => {
  beforeEach(() => {
    vi.useRealTimers()
  })

  it('renders nothing when message is null', () => {
    const { container } = render(<Toast message={null} onDismiss={() => {}} />)
    expect(container.querySelector(TOAST_CLASS)).toBeNull()
  })

  it('renders nothing when message is an empty string', () => {
    // An empty string is the cleared-message state -- also hidden.
    const { container } = render(<Toast message="" onDismiss={() => {}} />)
    expect(container.querySelector(TOAST_CLASS)).toBeNull()
  })

  it('renders the message when given', () => {
    const { container } = render(<Toast message="Labels cleared" onDismiss={() => {}} />)
    const el = container.querySelector(TOAST_CLASS)
    expect(el).not.toBeNull()
    expect(el.textContent).toBe('Labels cleared')
  })

  it('calls onDismiss after the dismiss delay', () => {
    vi.useFakeTimers()
    const onDismiss = vi.fn()
    render(<Toast message="hello" onDismiss={onDismiss} />)
    expect(onDismiss).not.toHaveBeenCalled()
    // 3 s auto-dismiss: the parent clears `message`, then a 250 ms leaving
    // delay lets the CSS fade-out run before the element unmounts.
    vi.advanceTimersByTime(2999)
    expect(onDismiss).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(onDismiss).toHaveBeenCalledTimes(1)
  })

  it('calls onDismiss after the delay even after re-renders with the same message', () => {
    // A re-render with the same string should not reset the dismiss clock.
    // The hook's effect dep is the message *value*, so a no-op re-render
    // keeps the same timer.
    vi.useFakeTimers()
    const onDismiss = vi.fn()
    const { rerender } = render(<Toast message="hello" onDismiss={onDismiss} />)
    vi.advanceTimersByTime(1500)
    rerender(<Toast message="hello" onDismiss={onDismiss} />)
    vi.advanceTimersByTime(1499)
    expect(onDismiss).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(onDismiss).toHaveBeenCalledTimes(1)
  })
})