import { useEffect, useRef, useState } from 'react'

// A single-line toast at the bottom-center of the window. Auto-dismisses
// after 3 s; pauses its dismiss timer on hover so a slow reader doesn't
// lose the message mid-sentence. Reuses the .floating-error class App.css
// already defines for menu errors -- the look was right for that role and
// is right for this one too.
//
// The "leaving" state keeps the element mounted for LEAVING_MS after
// onDismiss so the CSS fade-out has time to run. Without it the parent
// would clear `message` to '' and the element would unmount instantly,
// cutting the animation short.
const DISMISS_MS = 3000
const LEAVING_MS = 250

export default function Toast({ message, onDismiss }) {
  const [leaving, setLeaving] = useState(false)
  const [paused, setPaused] = useState(false)
  // Refs (not state) so changes don't re-render: the timer is fire-and-forget,
  // and the pause flag is already covered by `paused`.
  const dismissRef = useRef(null)
  const leavingRef = useRef(null)
  // Mirror onDismiss into a ref so the timer callback always calls the
  // latest version. Otherwise the effect would have to list onDismiss in
  // its deps, and a new function reference on every parent render would
  // restart the 3 s dismiss clock -- the test "calls onDismiss after the
  // delay even after re-renders with the same message" pins exactly that.
  const onDismissRef = useRef(onDismiss)
  onDismissRef.current = onDismiss

  function clearDismissTimer() {
    if (dismissRef.current) {
      clearTimeout(dismissRef.current)
      dismissRef.current = null
    }
  }

  function clearLeavingTimer() {
    if (leavingRef.current) {
      clearTimeout(leavingRef.current)
      leavingRef.current = null
    }
  }

  // (Re)start whenever the message changes or we unpause. A no-op re-render
  // (same message) is intentionally ignored so the dismiss clock survives
  // parent re-renders that didn't change the toast.
  useEffect(() => {
    if (!message) {
      clearDismissTimer()
      return clearLeavingTimer
    }
    if (!paused) {
      clearDismissTimer()
      dismissRef.current = setTimeout(() => {
        dismissRef.current = null
        setLeaving(false)
        onDismissRef.current()
        leavingRef.current = setTimeout(() => {
          leavingRef.current = null
          setLeaving(true)
        }, LEAVING_MS)
      }, DISMISS_MS)
    }
    return clearDismissTimer
  }, [message, paused])

  // Cleanup on unmount.
  useEffect(() => () => {
    clearDismissTimer()
    clearLeavingTimer()
  }, [])

  if (!message) return null

  // The CSS class controls position, fade-in and fade-out. The leaving
  // modifier is what lets the unmount happen after the animation finishes;
  // it must outlast onDismiss by LEAVING_MS so React keeps us in the DOM.
  const className = leaving ? 'floating-error toast toast--leaving' : 'floating-error toast'

  return (
    <div
      className={className}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      role="status"
      aria-live="polite"
    >
      {message}
    </div>
  )
}