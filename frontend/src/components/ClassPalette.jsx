import { useMemo } from 'react'
import { ERASER_KEY, UNLABELED, classKeyLabel } from '../constants'

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

export default function ClassPalette({ classes, tool, setTool, labels, labelsVersion }) {
  const counts = useClassCounts(labels, labelsVersion)

  return (
    <div className="class-palette">
      <div className="class-palette-title">Class</div>
      <div className="class-palette-list">
        {classes.map((c, i) => {
          const active = !tool.eraser && tool.classId === c.id
          const count = counts.get(c.id) ?? 0
          return (
            <button
              key={c.id}
              className={`class-item${active ? ' active' : ''}`}
              onClick={() => setTool({ ...tool, classId: c.id, eraser: false })}
              title={`${c.name} — ${count.toLocaleString()} px painted`}
            >
              <span className="class-key">{classKeyLabel(i)}</span>
              <span className="class-swatch" style={{ background: c.color }} />
              <span className="class-name">{c.name}</span>
              <span className={`class-count${count ? '' : ' empty'}`}>
                {count ? count.toLocaleString() : '0'}
              </span>
            </button>
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
    </div>
  )
}
