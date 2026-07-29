import { classKeyLabel } from '../constants'

// The project's label space, as an editable list. Split out of ProjectSetup
// because it is the one part of that dialog with rules of its own: ids are
// what every saved mask stores, so a row is not just a row.
//
// It owns no state. The form lives in ProjectSetup and the deletion warning is
// computed there, because saving is what acts on it -- this renders and
// reports, which keeps the destructive path in one place rather than two.
export default function ClassListEditor({
  classes, presets, preset, editing, clearing, removedNames, usage,
  removedCount, confirmRemoval, setConfirmRemoval,
  onField, onAdd, onRemove, onPreset,
}) {
  const needsConfirm = clearing.length > 0
  const totalCleared = clearing.reduce((sum, u) => sum + u.pixels, 0)

  return (
    <div className="class-editor">
      {!editing && presets && (
        <div className="class-preset">
          start from
          <select value={preset ?? ''} onChange={(e) => onPreset(e.target.value)}>
            {Object.keys(presets).map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <span className="hint">then edit the list below</span>
        </div>
      )}

      {classes.map((c, i) => (
        <div key={c.id ?? `new-${i}`} className="class-row">
          {/* The shortcut key, so it is visible that the list has grown past
              the ten-and-one keys CLASS_KEYS provides and the rest are
              click-only. */}
          <span className="class-row-key">{classKeyLabel(i) || '·'}</span>
          <input
            type="color"
            value={c.color || '#888888'}
            onChange={(e) => onField(i, 'color', e.target.value)}
          />
          <input
            type="text"
            value={c.name}
            placeholder="Class name"
            onChange={(e) => onField(i, 'name', e.target.value)}
          />
          <button
            type="button"
            className="class-row-remove"
            disabled={classes.length < 2}
            title={classes.length < 2
              ? 'A project needs at least one class'
              : `Remove ${c.name}`}
            onClick={() => onRemove(i)}
          >
            ×
          </button>
        </div>
      ))}

      <button type="button" className="class-add" onClick={onAdd}>
        + Add class
      </button>

      {editing && removedCount > 0 && (
        <div className={needsConfirm ? 'modal-error' : 'hint'}>
          {usage === null
            ? 'checking what those classes have labeled…'
            : needsConfirm
              ? (
                <>
                  <div>
                    Removing {clearing.map((u) => (
                      `${removedNames[u.id] ?? `class ${u.id}`} `
                      + `(${u.pixels.toLocaleString()} px in ${u.images} `
                      + `image${u.images === 1 ? '' : 's'})`)).join(', ')}
                    {' '}unlabels {totalCleared.toLocaleString()} px in the
                    saved masks. This cannot be undone. AI masks are left
                    alone.
                  </div>
                  <label className="class-confirm">
                    <input
                      type="checkbox"
                      checked={confirmRemoval}
                      onChange={(e) => setConfirmRemoval(e.target.checked)}
                    />
                    Yes, unlabel those pixels
                  </label>
                </>
              )
              : 'those classes have no labeled pixels'}
        </div>
      )}
    </div>
  )
}
