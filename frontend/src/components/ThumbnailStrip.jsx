// Bottom strip of project image thumbnails; click one to make it the active
// image, right-click one for its context menu. Thumbnails stream in while the
// backend generates them.
//
// Each thumbnail carries a coverage bar: the share of its pixels that are
// labeled. A plain "has labels" dot would be useless on a dataset that arrives
// partly labeled everywhere — every thumbnail would light up. Coverage instead
// points straight at the images with work left in them.

import { forwardRef, useImperativeHandle, useRef } from 'react'

// Below this, a thumbnail is called out as needing attention rather than just
// being drawn short.
const LOW_COVERAGE = 0.5

function fmtPct(fraction) {
  const pct = fraction * 100
  // Keep a decimal only where rounding would read as a milestone it hasn't
  // reached: 99.6% must not display as "100%".
  if (pct > 99 && pct < 100) return '99.9%'
  return `${Math.round(pct)}%`
}

function Thumbnail({ name, thumb, active, stats, onSelect, onContextMenu, slotRef }) {
  const coverage = stats?.total ? stats.labeled / stats.total : null
  const low = coverage !== null && coverage < LOW_COVERAGE
  const title = coverage === null
    ? name
    : `${name} — ${fmtPct(coverage)} labeled `
      + `(${stats.labeled.toLocaleString()} / ${stats.total.toLocaleString()} px)`

  // The context handler sits on the wrapper rather than the button so it keeps
  // working regardless of the button's own state.
  return (
    <div className="thumb-slot" ref={slotRef} onContextMenu={onContextMenu}>
      <button
        className={`thumb${active ? ' active' : ''}`}
        onClick={onSelect}
        title={title}
      >
        {thumb
          ? <img src={`data:image/png;base64,${thumb}`} alt={name} draggable={false} />
          : <span className="thumb-placeholder" />}
        {coverage !== null && (
          <span className={`thumb-coverage${low ? ' low' : ''}`}>
            <span style={{ width: `${Math.round(coverage * 100)}%` }} />
          </span>
        )}
        <span className="thumb-name">
          {name}
          {coverage !== null && (
            <span className={`thumb-pct${low ? ' low' : ''}`}>{fmtPct(coverage)}</span>
          )}
        </span>
      </button>
    </div>
  )
}

// Exposes centerOn(name) via ref so a caller outside the strip can scroll a
// thumbnail into view without the strip re-centering on every ordinary click,
// which would feel jumpy.
const ThumbnailStrip = forwardRef(function ThumbnailStrip({
  images, thumbs, activeName, progress, labelStats, onSelect, onContextMenu,
}, ref) {
  const nodes = useRef(new Map())

  useImperativeHandle(ref, () => ({
    centerOn(name) {
      nodes.current.get(name)?.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' })
    },
  }), [])

  if (!images?.length) return null

  // Pooled over pixels, not averaged over images, so one tiny image cannot
  // swing the headline the way a per-image mean would.
  const counts = labelStats?.counts ?? {}
  let labeled = 0
  let total = 0
  for (const name of images) {
    const s = counts[name]
    if (s) { labeled += s.labeled; total += s.total }
  }
  const overall = total ? labeled / total : null
  const scanning = labelStats?.running

  function handleWheel(e) {
    if (e.deltaY === 0) return
    e.currentTarget.scrollLeft += e.deltaY
    e.preventDefault()
  }

  return (
    <div className="thumb-strip-wrap">
      <div className="strip-summary">
        <span className="group-label">Images</span>
        <span className="strip-count">{images.length}</span>
        {overall !== null && (
          <span
            className="strip-coverage"
            title={`${labeled.toLocaleString()} of ${total.toLocaleString()} pixels labeled`}
          >
            {fmtPct(overall)} labeled
          </span>
        )}
        {scanning && <span className="hint">scanning {labelStats.done}/{labelStats.total}…</span>}
        {progress?.running && (
          <span className="hint">thumbnails {progress.done}/{progress.total}…</span>
        )}
      </div>
      <div className="thumb-strip" onWheel={handleWheel}>
        {images.map((name) => (
          <Thumbnail
            key={name}
            name={name}
            thumb={thumbs?.[name]}
            active={name === activeName}
            stats={counts[name]}
            onSelect={() => onSelect(name)}
            onContextMenu={(e) => {
              e.preventDefault()
              onContextMenu(name, e.clientX, e.clientY)
            }}
            slotRef={(el) => {
              if (el) nodes.current.set(name, el)
              else nodes.current.delete(name)
            }}
          />
        ))}
      </div>
    </div>
  )
})

export default ThumbnailStrip
