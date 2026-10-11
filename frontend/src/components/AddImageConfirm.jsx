import { useEffect, useRef } from 'react'

// Confirm-or-cancel dialog for a size mismatch on add_image.
//
// The dimensions of the existing project's first image and the candidate
// filename are shown side by side so the user can eyeball the difference
// before accepting. We deliberately do not show band counts: the backend
// already rejects band-count mismatches outright, so anything that reaches
// this dialog has matching bands and the only thing to decide is the grid.
//
// A Confirm dialog is the only place that calls onConfirm with the necessary
// payload; the button label is "Add anyway" because the warning is the only
// reason this branch exists, and the affirmative verb should match the
// action that is about to happen.
export default function AddImageConfirm({ existing, candidate, onConfirm, onCancel }) {
  const ref = useRef(null)

  // Match the project-setup modal's keyboard contract: Escape backs out,
  // Enter on the default button (Cancel) does not commit. The Add anyway
  // button is the only one that should commit, and the user has to reach
  // for it with the mouse -- deliberate, since this is the "I really
  // mean it" path.
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  useEffect(() => {
    // Initial focus on the Cancel button: the danger is the secondary path,
    // so the safe option is the one the keyboard drops you on.
    ref.current?.querySelector('button[data-default]')?.focus()
  }, [])

  return (
    <div className="modal-backdrop" role="dialog" aria-label="Add image with different size" ref={ref}>
      <div className="modal">
        <h2>Different image size</h2>
        <div className="modal-sub">
          {candidate.name} is {candidate.width}×{candidate.height} px, but this
          project's existing image ({existing.name}) is {existing.width}×{existing.height} px.
          Mixed-size images are unusual and can confuse the training pipeline.
        </div>
        <div className="modal-actions">
          <button data-default type="button" onClick={onCancel}>Cancel</button>
          <button type="button" className="primary" onClick={onConfirm}>Add anyway</button>
        </div>
      </div>
    </div>
  )
}
