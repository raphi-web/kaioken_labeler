import { useEffect, useMemo, useState } from 'react'
import { api } from '../bridge'
import { classKeyLabel } from '../constants'

const RGB_CHANNELS = [{ key: 'r', label: 'R' }, { key: 'g', label: 'G' }, { key: 'b', label: 'B' }]

// Colours handed to classes added by hand, in order. Only a starting point --
// every row has a colour picker -- but a new class arriving in a colour that
// is already on screen is worse than useless, so these avoid the presets.
const NEW_CLASS_COLORS = ['#f2c94c', '#56ccf2', '#bb6bd9', '#eb5757', '#6fcf97',
                          '#f2994a', '#2d9cdb', '#828282']

function nextClassColor(classes) {
  const used = new Set(classes.map((c) => (c.color || '').toLowerCase()))
  return NEW_CLASS_COLORS.find((c) => !used.has(c)) ?? '#888888'
}

// A valid [R, G, B] band-index triple: default to the first three bands, and
// clamp every entry into the available band range.
function normalizeDisplayBands(bands, channels) {
  const out = Array.isArray(bands) ? bands.slice(0, 3) : []
  while (out.length < 3) out.push(Math.min(out.length, channels - 1))
  return out.map((b) => Math.min(Math.max(Number(b) || 0, 0), channels - 1))
}

// Number of tiles along one axis, mirroring data.tile_grid: fixed-size steps
// with the last one snapped back to the edge. Only an estimate for the dialog —
// the backend does the real thing.
function tileCount(total, tile, stride) {
  if (!total || !tile || stride < 1) return 0
  if (tile >= total) return 1
  const n = Math.floor((total - tile) / stride) + 1
  return (n - 1) * stride === total - tile ? n : n + 1
}

// Centered modal for a project's data profile. In 'create' mode (a folder
// without a project_config.json) the user reviews the proposed defaults before
// anything is written; in 'geotiff' mode it additionally collects the tile
// geometry used to cut one raster into a project; in 'edit' mode it edits the
// open project's settings. Band count is fixed once the project exists and
// becomes read-only in edit mode.
export default function ProjectSetup({ mode = 'create', root, source, defaults, error, busy, progress, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => ({
    ...defaults,
    display_bands: normalizeDisplayBands(defaults.display_bands, defaults.input_channels),
    classes: (defaults.classes ?? []).map((c) => ({ ...c })),
  }))
  const editing = mode === 'edit'
  const tiling = mode === 'geotiff'

  // Class ids the open project already has. A row that keeps its id keeps its
  // painted pixels; a row that disappears from the list is a deletion, and the
  // masks storing that id have to be dealt with before it can be saved.
  const originalIds = useMemo(
    () => (defaults.classes ?? []).map((c) => c.id).filter((id) => id !== undefined),
    [defaults.classes])
  const removedIds = originalIds.filter(
    (id) => !form.classes.some((c) => c.id === id))
  const removedKey = removedIds.join(',')
  const [usage, setUsage] = useState(null)
  const [confirmRemoval, setConfirmRemoval] = useState(false)

  // What deleting these classes would actually cost, counted from the saved
  // masks and this session's unsaved edits. Fetched when the set of removed
  // rows changes -- one scan of the project, so not on every keystroke.
  useEffect(() => {
    if (!editing || !removedKey) {
      setUsage(null)
      return undefined
    }
    let cancelled = false
    api('class_usage', removedKey.split(',').map(Number)).then((res) => {
      if (!cancelled && res?.ok) setUsage(res.usage)
    })
    return () => { cancelled = true }
  }, [editing, removedKey])

  // Only a deletion that destroys labels needs confirming; dropping a class
  // nobody ever painted is just an edit.
  const clearing = removedIds
    .map((id) => ({ id, ...(usage?.[String(id)] ?? { pixels: 0, images: 0 }) }))
    .filter((u) => u.pixels > 0)
  const removedNames = Object.fromEntries(
    (defaults.classes ?? []).map((c) => [c.id, c.name]))
  const needsConfirm = clearing.length > 0
  const totalCleared = clearing.reduce((sum, u) => sum + u.pixels, 0)
  // Tile validity: the overlap must leave a forward step. There is no minimum
  // tile size here — the full app needs tiles at least as large as the model's
  // patch, but any tile you can see is a tile you can label.
  const tileW = Number(form.tile_width) || 0
  const tileH = Number(form.tile_height) || 0
  const overlap = Number(form.overlap) || 0
  const tileTooSmall = tiling && Math.min(tileW, tileH) < 1
  const overlapInvalid = tiling && (overlap < 0 || overlap >= Math.min(tileW, tileH))
  const outputMissing = tiling && !String(form.output_root || '').trim()
  const estimate = tiling && !tileTooSmall && !overlapInvalid
    ? tileCount(defaults.source_width, Math.min(tileW, defaults.source_width), tileW - overlap) *
      tileCount(defaults.source_height, Math.min(tileH, defaults.source_height), tileH - overlap)
    : 0

  useEffect(() => {
    function onKey(e) {
      // Not while working: the Cancel button is disabled for the same reason,
      // and tiling keeps running on the backend regardless of the dialog.
      if (e.key === 'Escape' && !busy) onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel, busy])

  function set(key, value) {
    setForm((f) => ({ ...f, [key]: value }))
  }

  // Changing the band count keeps existing names and pads with generic ones,
  // and clamps the displayed-band selection into the new range.
  function setChannels(n) {
    n = Math.max(1, Math.min(64, Math.round(n) || 1))
    setForm((f) => ({
      ...f,
      input_channels: n,
      band_names: Array.from({ length: n }, (_, i) => f.band_names[i] ?? `Band ${i + 1}`),
      display_bands: f.display_bands.map((b) => Math.min(b, n - 1)),
    }))
  }

  function setBandName(i, name) {
    setForm((f) => {
      const band_names = [...f.band_names]
      band_names[i] = name
      return { ...f, band_names }
    })
  }

  function setDisplayBand(channel, index) {
    setForm((f) => {
      const display_bands = [...f.display_bands]
      display_bands[channel] = index
      return { ...f, display_bands }
    })
  }

  // Rows are edited by position, not by id: a class added in this dialog has
  // no id yet (the backend assigns one on save, from a counter that never
  // reuses), so index is the only handle that exists for every row.
  function setClassField(index, key, value) {
    setForm((f) => ({
      ...f,
      classes: f.classes.map((c, i) => (i === index ? { ...c, [key]: value } : c)),
    }))
  }

  function addClass() {
    setForm((f) => ({
      ...f,
      classes: [...f.classes,
                { name: `Class ${f.classes.length + 1}`,
                  color: nextClassColor(f.classes) }],
    }))
  }

  function removeClass(index) {
    setForm((f) => ({ ...f, classes: f.classes.filter((_, i) => i !== index) }))
    setConfirmRemoval(false) // a changed removal set has not been confirmed
  }

  // Presets are a starting point for a new project, not a mode: the list stays
  // editable afterwards. Not offered when editing -- swapping the palette of a
  // project that already has masks is a deletion of everything in it, which
  // should look like the row-by-row deletion it is.
  function applyPreset(name) {
    const preset = defaults.class_presets?.[name]
    if (!preset) return
    setForm((f) => ({ ...f, class_preset: name,
                      classes: preset.map((c) => ({ ...c })) }))
  }

  // Re-probe the band count when the images folder changes (create only: in
  // edit mode the band count is fixed with the model, and when tiling the
  // folder does not exist yet — the count comes from the source raster).
  // Native folder chooser for the tiling output. The picker returns an existing
  // folder, which the backend requires to be empty — so the typed default
  // (a fresh, non-existent sibling) stays available as the other option.
  async function browseOutput() {
    const res = await api('pick_folder')
    if (res?.path) set('output_root', res.path)
  }

  async function probeFolder() {
    if (editing || tiling) return
    const res = await api('probe_bands', root, form.images_folder || '.')
    if (res.bands) setChannels(res.bands)
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>{editing ? 'Project Settings' : tiling ? 'New Project from GeoTIFF' : 'New Project'}</h2>
        <div className="modal-sub">
          {tiling
            ? `${source} — ${defaults.source_width}x${defaults.source_height} px, ${defaults.input_channels} bands`
            : root}
        </div>
        {tiling && !defaults.source_georeferenced && (
          <div className="modal-error">
            This GeoTIFF has no CRS; tiles will carry none either.
          </div>
        )}

        <div className="form-row">
          <label>Project name</label>
          <input
            type="text"
            value={form.project_name}
            onChange={(e) => set('project_name', e.target.value)}
          />
        </div>

        {tiling && (
          <>
            <div className="form-row">
              <label>Output folder</label>
              <div className="path-picker">
                <input
                  type="text"
                  value={form.output_root}
                  onChange={(e) => set('output_root', e.target.value)}
                  title="The tiles are written here and this becomes the project folder"
                />
                <button type="button" disabled={busy} onClick={browseOutput}>Browse…</button>
              </div>
            </div>
            <div className="form-row">
              <label>Tile size</label>
              <div className="tile-size">
                <input
                  type="number"
                  min="32"
                  step="32"
                  value={form.tile_width}
                  onChange={(e) => set('tile_width', Number(e.target.value))}
                />
                <span className="tile-x">×</span>
                <input
                  type="number"
                  min="32"
                  step="32"
                  value={form.tile_height}
                  onChange={(e) => set('tile_height', Number(e.target.value))}
                />
              </div>
              <span className={tileTooSmall ? 'modal-error' : 'hint'}>
                {tileTooSmall
                  ? 'px — must be at least 1 px' 
                  : 'px, width × height'}
              </span>
            </div>
            <div className="form-row">
              <label>Overlap</label>
              <input
                type="number"
                min="0"
                step="16"
                value={form.overlap}
                onChange={(e) => set('overlap', Number(e.target.value))}
              />
              <span className={overlapInvalid ? 'modal-error' : 'hint'}>
                {overlapInvalid
                  ? 'px — must be smaller than the tile'
                  : `px between neighbouring tiles${estimate ? ` — about ${estimate} tiles` : ''}`}
              </span>
            </div>
          </>
        )}

        <div className="form-row">
          <label>Images folder</label>
          <input
            type="text"
            value={form.images_folder}
            onChange={(e) => set('images_folder', e.target.value)}
            onBlur={probeFolder}
            title="Relative to the project folder ('.' = the folder itself)"
          />
        </div>

        <div className="form-row">
          <label>Number of bands</label>
          <input
            type="number"
            min="1"
            max="64"
            value={form.input_channels}
            disabled={editing || tiling}
            onChange={(e) => setChannels(Number(e.target.value))}
          />
          <span className="hint">
            {editing
              ? 'fixed for this project'
              : tiling
                ? 'from the source GeoTIFF'
                : defaults.bands_detected ? 'detected from the images' : 'no readable image found'}
          </span>
        </div>

        <div className="form-row align-top">
          <label>Band names</label>
          <div className="band-grid">
            {form.band_names.map((name, i) => (
              <input
                key={i}
                type="text"
                value={name}
                placeholder={`Band ${i + 1}`}
                onChange={(e) => setBandName(i, e.target.value)}
              />
            ))}
          </div>
        </div>

        <div className="form-row">
          <label title="Which bands are mapped to the red, green and blue display channels">
            Displayed bands
          </label>
          <div className="rgb-bands">
            {RGB_CHANNELS.map((ch, c) => (
              <div key={ch.key} className="rgb-band">
                <span className={`rgb-dot rgb-${ch.key}`}>{ch.label}</span>
                <select
                  value={form.display_bands[c]}
                  onChange={(e) => setDisplayBand(c, Number(e.target.value))}
                >
                  {form.band_names.map((name, i) => (
                    <option key={i} value={i}>{name}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>

        <div className="form-row align-top">
          <label title="The label space: one id per class, stored in every mask">
            Classes
          </label>
          <div className="class-editor">
            {!editing && defaults.class_presets && (
              <div className="class-preset">
                start from
                <select
                  value={form.class_preset ?? ''}
                  onChange={(e) => applyPreset(e.target.value)}
                >
                  {Object.keys(defaults.class_presets).map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
                <span className="hint">then edit the list below</span>
              </div>
            )}
            {form.classes.map((c, i) => (
              <div key={c.id ?? `new-${i}`} className="class-row">
                {/* The shortcut key, so it is visible that the list has grown
                    past the ten-and-one keys CLASS_KEYS provides and the rest
                    are click-only. */}
                <span className="class-row-key">{classKeyLabel(i) || '·'}</span>
                <input
                  type="color"
                  value={c.color || '#888888'}
                  onChange={(e) => setClassField(i, 'color', e.target.value)}
                />
                <input
                  type="text"
                  value={c.name}
                  placeholder="Class name"
                  onChange={(e) => setClassField(i, 'name', e.target.value)}
                />
                <button
                  type="button"
                  className="class-row-remove"
                  disabled={form.classes.length < 2}
                  title={form.classes.length < 2
                    ? 'A project needs at least one class'
                    : `Remove ${c.name}`}
                  onClick={() => removeClass(i)}
                >
                  ×
                </button>
              </div>
            ))}
            <button type="button" className="class-add" onClick={addClass}>
              + Add class
            </button>
            {editing && removedIds.length > 0 && (
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
        </div>

        <div className="form-row">
          <label>Masks folder</label>
          <input
            type="text"
            value={form.masks_user}
            onChange={(e) => set('masks_user', e.target.value)}
          />
        </div>

        {error && <div className="modal-error">{error}</div>}

        {progress && (
          <div className="tiling-progress">
            <div className="tiling-bar">
              <div
                className={progress.total ? 'tiling-fill' : 'tiling-fill indeterminate'}
                style={progress.total
                  ? { width: `${Math.round((progress.done / progress.total) * 100)}%` }
                  : undefined}
              />
            </div>
            <span className="hint">
              {progress.total
                ? `tiling ${progress.done} / ${progress.total}`
                : 'preparing…'}
            </span>
          </div>
        )}

        <div className="modal-actions">
          <button onClick={onCancel} disabled={busy}>Cancel</button>
          <button
            className="primary"
            disabled={busy || !form.project_name.trim()
              || tileTooSmall || overlapInvalid || outputMissing
              || !form.classes.length
              || form.classes.some((c) => !String(c.name).trim())
              || (needsConfirm && !confirmRemoval)}
            onClick={() => onSubmit(form)}
          >
            {editing
              ? (busy ? 'Saving…' : 'Save Settings')
              : (busy ? (tiling ? 'Tiling…' : 'Creating…') : 'Create Project')}
          </button>
        </div>
      </div>
    </div>
  )
}
