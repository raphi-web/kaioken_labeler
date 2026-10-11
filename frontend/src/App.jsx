import { useEffect, useMemo, useRef, useState } from 'react'
import { api, fromBase64, toBase64 } from './bridge'
import { CLASSES, ERASER_KEY, UNLABELED, classIdForKey } from './constants'
import { classColorsMap } from './colors'
import ClassPalette from './components/ClassPalette'
import ContextMenu from './components/ContextMenu'
import LabelerPane from './components/LabelerPane'
import AddImageConfirm from './components/AddImageConfirm'
import ProjectSetup from './components/ProjectSetup'
import ThumbnailStrip from './components/ThumbnailStrip'
import Toast from './components/Toast'
import Toolbar from './components/Toolbar'
import './App.css'

const UNDO_LIMIT = 50

export default function App() {
  const [image, setImage] = useState(null)
  const [tool, setTool] = useState({ classId: 0, brushSize: 8, eraser: false, mode: 'brush' })
  const [labeledPixels, setLabeledPixels] = useState(0)
  const [labelsVersion, setLabelsVersion] = useState(0) // bumped on undo
  const [undoDepth, setUndoDepth] = useState(0)
  const [redoDepth, setRedoDepth] = useState(0)
  const [view, setView] = useState(null) // zoom/pan: {scale, x, y}
  const [project, setProject] = useState(null) // {root, name, images, active, classes}
  const [setup, setSetup] = useState(null) // {root, defaults} while the creation dialog is open
  const [setupError, setSetupError] = useState('')
  const [setupBusy, setSetupBusy] = useState(false)
  const [tilingProgress, setTilingProgress] = useState(null) // {done, total} while tiling
  const [thumbs, setThumbs] = useState({}) // image name -> base64 PNG | null
  const [thumbProgress, setThumbProgress] = useState(null)
  const [labelStats, setLabelStats] = useState(null) // {counts, running, done, total}
  const [menu, setMenu] = useState(null) // {name, x, y} while a thumbnail menu is open
  const [menuError, setMenuError] = useState('')
  const [samAvailable, setSamAvailable] = useState(false) // Efficient-SAM2 files present
  // Transient status messages ("Labels cleared", "Project opened", etc.) shown
  // as a fading toast at the bottom-center of the window. Single-message at a
  // time: a new showToast replaces the previous, which keeps the UI honest
  // and avoids any queue management.
  const [toast, setToast] = useState('')
  const showToast = (msg) => setToast(msg)
  const labelsRef = useRef(null)
  const undoStack = useRef([]) // Map<pixelIndex, previousValue> per stroke/fill
  const redoStack = useRef([]) // inverse diffs of undone steps
  const autosaveTimer = useRef(null)
  const thumbStripRef = useRef(null)
  // The active image name, readable from callbacks that outlive the render
  // they were created in. The Ctrl+Z listener is registered once with []
  // deps, so anything it reaches through a state variable would be frozen at
  // the first render -- when `image` is still null.
  const imageNameRef = useRef(null)
  const fitRef = useRef(null) // Viewport.fit(), for the toolbar button

  // The open project's own class palette (colors editable in Settings), or the
  // standalone-image default when no project is open. Declared up here because
  // the class-shortcut effect below lists it as a dependency.
  // Classes hidden from the overlay. A view setting, not part of the project:
  // the labels stay on disk, keep their counts and are still seen by every
  // tool -- they are simply not drawn, so you can look at the imagery under
  // them. Session-only, and deliberately not written to project_config.json,
  // which is shared with the training app.
  const [hiddenClasses, setHiddenClasses] = useState(() => new Set())
  // Locked classes are protected from all label writes for the session. Like
  // visibility, this is deliberately session-only and does not alter labels.
  const [lockedClasses, setLockedClasses] = useState(() => new Set())

  function toggleClassVisible(id) {
    setHiddenClasses((prev) => {
      const next = new Set(prev)
      if (!next.delete(id)) next.add(id)
      return next
    })
  }

  function toggleClassLocked(id) {
    setLockedClasses((prev) => {
      const next = new Set(prev)
      if (!next.delete(id)) next.add(id)
      return next
    })
  }

  // The add-image flow: open a native file picker, ask the backend to copy
  // the chosen file into the project, and either apply a fresh project list
  // back (so the new image shows up in the strip) or -- on a size or
  // bandwidth mismatch -- show a confirmation dialog first.
  //
  // The dialog callback is the one place that re-calls the backend with
  // force=true; the rest of the wiring is a single async function so the
  // success and error paths look the same regardless of whether the user
  // confirmed the dialog or skipped it.
  const [addImageConfirm, setAddImageConfirm] = useState(null)
  // Latest values the deferred callback reads, so the picker closure we
  // hand off can find the source path the user picked three frames later.
  const pendingSourceRef = useRef(null)

  async function performAddImage(source, force = false) {
    const res = await api('add_image', source, force)
    if (res?.requires_confirm) {
      pendingSourceRef.current = source
      setAddImageConfirm({ source, existing: res.existing,
                           candidate: res.candidate })
      return
    }
    if (!res?.ok) {
      showToast(res?.error || 'Could not add image')
      return
    }
    // Stand-alone images (no project yet) only get here when run through
    // setup; the toolbar entry is disabled when hasProject is false, so the
    // happy path is always "set project, refresh thumbnails".
    if (res.project) setProject(res.project)
    // New image is added but not loaded; keep the user on what they were
    // painting. The thumbnail + label scan threads pick the new file up on
    // their next tick.
    showToast(`Added ${res.name} (${res.width}×${res.height} px)`)
  }

  async function handleAddImage() {
    if (!project) return
    const res = await api('pick_file')
    if (!res?.path) return // user cancelled the picker
    performAddImage(res.path)
  }

  // Every path that picks a class goes through here -- the palette and the
  // number keys -- so neither can leave you painting into something invisible.
  function selectClass(id) {
    setHiddenClasses((prev) => {
      if (!prev.has(id)) return prev
      const next = new Set(prev)
      next.delete(id)
      return next
    })
    setTool((t) => ({ ...t, classId: id, eraser: false }))
  }

  const classes = project?.classes ?? CLASSES
  // Memoized on `classes` so a wheel tick or an unrelated state change does
  // not rebuild the label overlay's color table. The redraw effect in
  // LabelerPane lists `classColors` in its deps, and rebuilding it per
  // render tears down and rebuilds the overlay pixel-by-pixel.
  const classColors = useMemo(() => classColorsMap(classes), [classes])

  // Install a backend image payload (base image + its persisted labels) and
  // reset all per-image UI state.
  function applyImagePayload(img) {
    if (!img) {
      setImage(null)
      imageNameRef.current = null
      labelsRef.current = null
      setLabeledPixels(0)
      return
    }
    labelsRef.current = img.labels
      ? fromBase64(img.labels)
      : new Uint8Array(img.width * img.height).fill(UNLABELED)
    setImage({ png: img.png, width: img.width, height: img.height, name: img.name })
    imageNameRef.current = img.name
    setLabeledPixels(img.labeled_pixels ?? 0)
    undoStack.current = []
    redoStack.current = []
    setUndoDepth(0)
    setRedoDepth(0)
    setLabelsVersion((v) => v + 1)
    setView(null) // let the viewport re-fit to the new image size
  }

  useEffect(() => {
    api('get_project').then((p) => { if (p) setProject(p) })
    api('get_image').then(applyImagePayload)
    api('sam_available').then((r) => setSamAvailable(!!r.available))
  }, [])

  // Poll thumbnail generation while it runs; thumbnails appear as they finish.
  useEffect(() => {
    if (!project) return
    let cancelled = false
    async function poll() {
      for (;;) {
        const t = await api('get_thumbnails')
        if (cancelled) return
        setThumbs(t.thumbs)
        setThumbProgress({ running: t.running, done: t.done, total: t.total })
        if (!t.running) return
        await new Promise((r) => setTimeout(r, 800))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [project])

  // Poll the per-image label scan the same way, so the coverage bars fill in
  // as it walks the folder instead of appearing all at once at the end.
  useEffect(() => {
    if (!project) return
    let cancelled = false
    async function poll() {
      for (;;) {
        const s = await api('get_label_stats')
        if (cancelled) return
        setLabelStats(s)
        if (!s.running) return
        await new Promise((r) => setTimeout(r, 400))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [project])

  async function pushLabels() {
    const res = await api('set_labels', toBase64(labelsRef.current))
    setLabeledPixels(res.labeled_pixels)
    // Keep this image's coverage bar in step with the stroke that just landed.
    // The backend already recomputed it; mirroring it here avoids a round trip
    // per stroke purely to refresh one number.
    const name = imageNameRef.current
    if (name) {
      setLabelStats((prev) => {
        const previous = prev?.counts?.[name]
        if (!previous) return prev
        return {
          ...prev,
          counts: { ...prev.counts,
                    [name]: { ...previous, labeled: res.labeled_pixels } },
        }
      })
    }
    scheduleAutosave()
  }

  // Debounced write-to-disk of the current image's mask, so user edits persist
  // automatically without a full GeoTIFF write on every stroke. Backend also
  // autosaves on image switch, so a pending timer can never lose work.
  function scheduleAutosave() {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current)
    autosaveTimer.current = setTimeout(() => {
      autosaveTimer.current = null
      api('autosave_user_mask')
    }, 1000)
  }

  function recordUndo(diff) {
    undoStack.current.push(diff)
    if (undoStack.current.length > UNDO_LIMIT) undoStack.current.shift()
    // A new edit forks history: anything previously undone is unreachable now,
    // and keeping it would let redo paste back a stroke that predates what is
    // currently on screen.
    redoStack.current = []
    setUndoDepth(undoStack.current.length)
    setRedoDepth(0)
  }

  // Apply a diff (Map<pixelIndex, valueToWrite>) and return its inverse, ready
  // to push onto the opposite stack. Undo and redo are one operation run in
  // opposite directions, so they share this.
  async function applyDiff(diff) {
    const inverse = new Map()
    const labels = labelsRef.current
    for (const [i, v] of diff) {
      inverse.set(i, labels[i])
      labels[i] = v
    }
    setLabelsVersion((v) => v + 1)
    await pushLabels()
    return inverse
  }

  async function handleUndo() {
    const diff = undoStack.current.pop()
    if (!diff) return
    redoStack.current.push(await applyDiff(diff))
    setUndoDepth(undoStack.current.length)
    setRedoDepth(redoStack.current.length)
  }

  async function handleRedo() {
    const diff = redoStack.current.pop()
    if (!diff) return
    undoStack.current.push(await applyDiff(diff))
    setUndoDepth(undoStack.current.length)
    setRedoDepth(redoStack.current.length)
  }

  // The key listener is registered once, so it reaches the handlers through
  // refs rather than closing over the render that installed it.
  const undoRef = useRef(handleUndo)
  const redoRef = useRef(handleRedo)
  undoRef.current = handleUndo
  redoRef.current = handleRedo

  useEffect(() => {
    function onKey(e) {
      if (!(e.ctrlKey || e.metaKey)) return
      const key = e.key.toLowerCase()
      // Ctrl+Shift+Z and Ctrl+Y both redo: the first is the editor convention,
      // the second the Windows one, and neither is worth making anyone learn.
      if (key === 'z' && e.shiftKey) {
        e.preventDefault()
        redoRef.current()
      } else if (key === 'z') {
        e.preventDefault()
        undoRef.current()
      } else if (key === 'y') {
        e.preventDefault()
        redoRef.current()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Class shortcuts: 1-9/0/- pick a class in palette order, E toggles the
  // eraser. Held separately from the undo handler because it depends on the
  // live class list and tool, and must stay out of the way while a modal or a
  // text field has the user's typing.
  // The class-key listener is registered per class list, but selectClass is a
  // fresh closure each render; a ref keeps the listener on the current one.
  const selectRef = useRef(selectClass)
  selectRef.current = selectClass

  useEffect(() => {
    function onKey(e) {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      if (setup || menu) return
      const tag = e.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target?.isContentEditable) return
      if (e.key.toLowerCase() === ERASER_KEY) {
        e.preventDefault()
        setTool((t) => ({ ...t, eraser: !t.eraser }))
        return
      }
      const id = classIdForKey(e.key, classes)
      if (id !== null) {
        e.preventDefault()
        selectRef.current(id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, setup, menu])

  // SAM2 snap: segment from the pending include/exclude clicks and return
  // polygon vertices (image coords) to preview; null on error/no object.
  async function handleSamSnap(points) {
    const res = await api('sam_snap', points.map((p) => [p.x, p.y, p.label]))
    if (!res.ok) {
      if (res.error) showToast(res.error)
      return null
    }
    return res.polygons
  }

  // Magic wand: the selection is computed in the backend over every band, not
  // in the browser over the 3-band composite — water and shadow look alike in
  // true colour and separate in NIR/SWIR. The mask comes back bit-packed;
  // unpack it into the byte-per-pixel form the preview and commit both want.
  async function handleWandSelect(samples, options) {
    const res = await api('wand_select', samples.map((p) => [p.x, p.y]), options)
    if (!res.ok) {
      if (res.error) showToast(res.error)
      return null
    }
    const packed = fromBase64(res.mask)
    const mask = new Uint8Array(res.width * res.height)
    for (let i = 0; i < mask.length; i++) {
      mask[i] = (packed[i >> 3] >> (7 - (i & 7))) & 1
    }
    return { mask, count: res.count, valid: res.valid,
             available: res.available, capped: res.capped }
  }

  // Reset every labeled pixel of the current image to unlabeled, undoably.
  async function handleClearLabels() {
    const labels = labelsRef.current
    if (!labels) return
    const diff = new Map()
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== UNLABELED && !lockedClasses.has(labels[i])) {
        diff.set(i, labels[i])
        labels[i] = UNLABELED
      }
    }
    if (!diff.size) return
    recordUndo(diff)
    setLabelsVersion((v) => v + 1)
    showToast(`Labels cleared (${diff.size.toLocaleString()} px)`)
    await pushLabels()
  }

  async function handleExport(method) {
    const res = await api(method)
    if (res.ok) showToast(`Saved: ${res.path}`)
    else if (res.error) showToast(res.error)
  }

  // Permanently remove an image (and its mask/thumbnail) from disk. When the
  // deleted image was active, the backend hands back whatever replaces it
  // (another image, or null once the project is empty).
  async function handleDeleteImage(name) {
    setMenuError('')
    const res = await api('delete_image', name)
    if (!res.ok) {
      if (res.error) setMenuError(res.error)
      return
    }
    setProject(res.project)
    setThumbs((t) => {
      if (!(name in t)) return t
      const next = { ...t }
      delete next[name]
      return next
    })
    if ('image' in res) {
      applyImagePayload(res.image)
      if (res.image_error) showToast(res.image_error)
    }
  }

  // Reset every labeled pixel of one image to UNLABELED. Per-image counterpart
  // to the toolbar's Clear Labels: same action, different image. If the
  // cleared image is active, the backend has already reset the in-memory
  // buffer; we re-fetch the payload so the local copy mirrors it, and a
  // subsequent stroke does not bring the cleared pixels back.
  async function handleClearImageLabels(name) {
    setMenuError('')
    const res = await api('clear_image_labels', name)
    if (!res.ok) {
      if (res.error) showToast(res.error)
      return
    }
    if (res.active && image?.name === name) {
      const refreshed = await api('get_image')
      applyImagePayload(refreshed)
    }
    showToast(`Cleared ${res.pixels.toLocaleString()} px from ${name}`)
  }

  function installProject(res) {
    setProject(res.project)
    setThumbs({})
    applyImagePayload(res.image)
    if (res.image_error) showToast(res.image_error)
    else if (!res.project.images.length) showToast('Project opened — no images found in folder')
    else showToast(res.created ? `Project created: ${res.project.name}` : `Project opened: ${res.project.name}`)
  }

  async function handleNewProject() {
    const res = await api('new_project')
    if (!res.ok) {
      if (res.error) showToast(res.error)
      return
    }
    if (res.needs_setup) {
      // No config in that folder yet: collect the data profile first.
      setSetupError('')
      setSetup({ mode: 'create', root: res.root, defaults: res.defaults })
      return
    }
    installProject(res)
  }

  // Build a project by tiling one big GeoTIFF: same setup dialog, plus the
  // tile geometry. Nothing is written until the dialog is confirmed.
  async function handleNewFromGeotiff() {
    const res = await api('new_project_from_geotiff')
    if (!res.ok) {
      if (res.error) showToast(res.error)
      return
    }
    setSetupError('')
    setSetup({ mode: 'geotiff', source: res.source, defaults: res.defaults })
  }

  // Open the settings dialog prefilled with the current project's config.
  function handleOpenSettings() {
    if (!project) return
    const dp = project.data_profile
    setSetupError('')
    setSetup({
      mode: 'edit',
      root: project.root,
      defaults: {
        project_name: project.name,
        input_channels: dp.input_channels,
        band_names: dp.band_names ?? [],
        display_bands: dp.display_bands,
        images_folder: project.paths.images_folder,
        masks_user: project.paths.masks_user,
        bands_detected: true,
        classes: project.classes,
      },
    })
  }

  async function handleSubmitSetup(settings) {
    setSetupBusy(true)
    let res
    if (setup.mode === 'edit') {
      res = await api('update_settings', settings)
    } else if (setup.mode === 'geotiff') {
      // Tiling runs on a backend worker; poll it so the dialog can show a bar.
      const started = await api('create_project_from_geotiff', setup.source, settings)
      if (!started.ok) {
        setSetupBusy(false)
        setSetupError(started.error || 'Could not create the project')
        return
      }
      setTilingProgress({ done: 0, total: 0 })
      for (;;) {
        const p = await api('tiling_progress')
        setTilingProgress({ done: p.done, total: p.total })
        if (!p.running) {
          res = p.error ? { ok: false, error: p.error } : p.result
          break
        }
        await new Promise((r) => setTimeout(r, 300))
      }
      setTilingProgress(null)
    } else {
      res = await api('create_project', setup.root, settings)
    }
    setSetupBusy(false)
    if (!res.ok) {
      setSetupError(res.error || (setup.mode === 'edit' ? 'Could not save settings' : 'Could not create the project'))
      return
    }
    setSetup(null)
    if (setup.mode === 'edit') {
      if (res.reopened) {
        // A folder change reopens the project: reload images/labels/thumbnails.
        installProject(res)
      } else {
        setProject(res.project)
        if (res.cleared?.pixels) {
          // Removing a class rewrote the masks, including the buffer this pane
          // is holding. Take the whole payload back, labels included -- keeping
          // the local copy would push the deleted class's pixels straight back
          // on the next stroke.
          applyImagePayload(res.image)
          showToast(`Settings saved — unlabeled ${res.cleared.pixels.toLocaleString()} px `
            + `across ${res.cleared.images} image${res.cleared.images === 1 ? '' : 's'}`)
        } else {
          // Only the display bands changed: swap the base RGB, keep labels/view.
          if (res.image) setImage((img) => (img ? { ...img, png: res.image.png } : img))
          showToast('Settings saved')
        }
      }
    } else {
      installProject(res)
    }
  }

  async function handleSelectImage(name) {
    if (name === image?.name) return
    const res = await api('load_image', name)
    if (!res.ok) {
      if (res.error) showToast(res.error)
      return
    }
    applyImagePayload(res.image)
    setProject((p) => (p ? { ...p, active: name } : p))
  }

  return (
    <div className="app">
      <Toolbar
        tool={tool}
        setTool={setTool}
        labeledPixels={labeledPixels}
        undoDepth={undoDepth}
        redoDepth={redoDepth}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onFit={() => fitRef.current?.fit()}
        hasImage={!!image}
        onClearLabels={handleClearLabels}
        hasProject={!!project}
        onNewProject={handleNewProject}
        onNewFromGeotiff={handleNewFromGeotiff}
        onAddImage={handleAddImage}
        onSaveProject={() => handleExport('save_project')}
        onSaveUserMask={async () => { await pushLabels(); await handleExport('save_user_mask') }}
        onOpenSettings={handleOpenSettings}
        samAvailable={samAvailable}
      />
      <div className="panes">
        <ClassPalette
          classes={classes}
          tool={tool}
          setTool={setTool}
          onSelectClass={selectClass}
          hiddenClasses={hiddenClasses}
          onToggleVisible={toggleClassVisible}
          lockedClasses={lockedClasses}
          onToggleLocked={toggleClassLocked}
          labels={labelsRef.current}
          labelsVersion={labelsVersion}
        />
        <LabelerPane
          image={image}
          labels={labelsRef.current}
          labelsVersion={labelsVersion}
          tool={tool}
          view={view}
          setView={setView}
          onStrokeEnd={pushLabels}
          onDiff={recordUndo}
          onSamSnap={handleSamSnap}
          onWandSelect={handleWandSelect}
          classColors={classColors}
          classes={classes}
          lockedClasses={lockedClasses}
          hiddenClasses={hiddenClasses}
          fitRef={fitRef}
        />
      </div>
      <ThumbnailStrip
        ref={thumbStripRef}
        images={project?.images}
        thumbs={thumbs}
        activeName={image?.name}
        progress={thumbProgress}
        labelStats={labelStats}
        onSelect={handleSelectImage}
        onContextMenu={(name, x, y) => {
          setMenuError('')
          setMenu({ name, x, y })
        }}
      />
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <span className="context-title">{menu.name}</span>
          <button
            className="danger"
            onClick={() => {
              if (window.confirm(`Clear all labels on "${menu.name}"? This writes a blank mask and cannot be undone.`)) {
                handleClearImageLabels(menu.name)
              }
            }}
          >
            Clear all labels
          </button>
          <button
            className="danger"
            onClick={() => {
              if (window.confirm(`Delete "${menu.name}"? This removes the image and its mask from disk.`)) {
                handleDeleteImage(menu.name)
              }
            }}
          >
            Delete image
          </button>
        </ContextMenu>
      )}
      {menuError && <div className="floating-error">{menuError}</div>}
      <Toast message={toast} onDismiss={() => setToast('')} />
      {addImageConfirm && (
        <AddImageConfirm
          existing={addImageConfirm.existing}
          candidate={addImageConfirm.candidate}
          onConfirm={() => {
            const source = pendingSourceRef.current
            setAddImageConfirm(null)
            pendingSourceRef.current = null
            if (source) performAddImage(source, true)
          }}
          onCancel={() => {
            setAddImageConfirm(null)
            pendingSourceRef.current = null
          }}
        />
      )}
      {setup && (
        <ProjectSetup
          mode={setup.mode}
          root={setup.root}
          source={setup.source}
          defaults={setup.defaults}
          error={setupError}
          busy={setupBusy}
          progress={tilingProgress}
          onSubmit={handleSubmitSetup}
          onCancel={() => setSetup(null)}
        />
      )}
    </div>
  )
}
