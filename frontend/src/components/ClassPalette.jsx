
import { useMemo, useRef, useState } from 'react'
import { ERASER_KEY, UNLABELED, classKeyLabel } from '../constants'

// Drag-to-resize bounds. The floor keeps the key/swatch/count columns readable;
// the ceiling stops the palette from eating the labeling pane.
const MIN_WIDTH = 120
const MAX_WIDTH = 420
const DEFAULT_WIDTH = 168

// Painted-pixel count per class, so a class that is present but scarce (or that
// you meant to paint and haven't) is visible without generating a report. One
// pass over the label buffer; recomputed only when the labels actually change.
function useClassCounts(labels, labelsVersion) {
  return useMemo(() => {
    const counts = new Map()
    if (!labels) return counts
    for (let i = 0; i < labels.length; i++) {
      const v = labels[i]
      if (v !== UNLABELED) counts.set(v, (counts.get(v) ?? 0) + 1)
    }
    return counts
    // labelsVersion is the change signal: labels is a mutable buffer painted
    // in place, so its identity does not change when its contents do.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labels, labelsVersion])
}

export default function ClassPalette({
  classes, tool, setTool, onSelectClass, hiddenClasses, onToggleVisible,
  lockedClasses, onToggleLocked, labels, labelsVersion,
}) {
  const counts = useClassCounts(labels, labelsVersion)
  const [width, setWidth] = useState(DEFAULT_WIDTH)
  const [dragging, setDragging] = useState(false)
  // Start point of the drag, so the width follows the cursor exactly instead of
  // jumping by the offset between the pointer and the border it grabbed.
  const dragRef = useRef(null)

  function handleResizeDown(e) {
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    dragRef.current = { x: e.clientX, width }
    setDragging(true)
  }

  function handleResizeMove(e) {
    const start = dragRef.current
    if (!start) return
    const next = start.width + (e.clientX - start.x)
    setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, next)))
  }

  function handleResizeUp(e) {
    if (!dragRef.current) return
    dragRef.current = null
    setDragging(false)
    e.currentTarget.releasePointerCapture(e.pointerId)
  }

  return (
    <div className="class-palette" style={{ width }}>
      <div className="class-palette-title">Class</div>
      <div className="class-palette-list">
        {classes.map((c, i) => {
          const active = !tool.eraser && tool.classId === c.id
          const count = counts.get(c.id) ?? 0
          const hidden = !!hiddenClasses?.has(c.id)
          const locked = !!lockedClasses?.has(c.id)
          // A row rather than a button, because it holds two of them: a button
          // inside a button is invalid, and a click on the eye would select the
          // class on its way up.
          return (
            <div
              key={c.id}
              className={`class-item${active ? ' active' : ''}${hidden ? ' hidden' : ''}${locked ? ' locked' : ''}`}
            >
              <button
                className="class-select"
                onClick={() => onSelectClass(c.id)}
                title={`${c.name} — ${count.toLocaleString()} px painted`}
              >
                <span className="class-key">{classKeyLabel(i)}</span>
                <span className="class-swatch" style={{ background: c.color }} />
                <span className="class-name">{c.name}</span>
                <span className={`class-count${count ? '' : ' empty'}`}>
                  {count ? count.toLocaleString() : '0'}
                </span>
              </button>
              {/* Visibility is a view setting: the labels stay, they are just
                  not drawn, so the imagery under them can be checked. */}
              <button
                className={`class-eye${hidden ? ' off' : ''}`}
                aria-label={`${hidden ? 'Show' : 'Hide'} ${c.name}`}
                aria-pressed={hidden}
                title={hidden
                  ? `${c.name} is hidden in the labeler — click to show it`
                  : `Hide ${c.name} in the labeler (its labels are kept)`}
                onClick={() => onToggleVisible(c.id)}
              />
              <button
                className={`class-lock${locked ? ' off' : ''}`}
                aria-label={`${locked ? 'Unlock' : 'Lock'} ${c.name}`}
                aria-pressed={locked}
                title={locked
                  ? `${c.name} is locked — click to allow editing`
                  : `Lock ${c.name} to prevent editing or overwriting`}
                onClick={() => onToggleLocked(c.id)}
              />
            </div>
          )
        })}
        <button
          className={`class-item eraser${tool.eraser ? ' active' : ''}`}
          onClick={() => setTool({ ...tool, eraser: !tool.eraser })}
          title="Erase back to unlabeled"
        >
          <span className="class-key">{ERASER_KEY.toUpperCase()}</span>
          <span className="class-swatch hollow" />
          <span className="class-name">Eraser</span>
        </button>
      </div>
      <div
        className={`class-palette-resizer${dragging ? ' dragging' : ''}`}
        onPointerDown={handleResizeDown}
        onPointerMove={handleResizeMove}
        onPointerUp={handleResizeUp}
        onPointerCancel={handleResizeUp}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
        title="Drag to resize — double-click to reset"
      />
    </div>
  )
}
