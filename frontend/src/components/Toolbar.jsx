import { BRUSH_MAX, BRUSH_MIN, clampBrush } from '../constants'
import Dropdown from './Dropdown'

export default function Toolbar({
  tool, setTool, labeledPixels, undoDepth, redoDepth, hasImage,
  onUndo, onRedo, onFit, onClearLabels,
  hasProject, onNewProject, onNewFromGeotiff, onAddImage,
  onSaveProject, onSaveUserMask, onOpenSettings, samAvailable,
  showWorldcover, setShowWorldcover, hasWorldcover,
}) {
  // Shared by the slider, the numeric box and the wheel, so all three clamp
  // identically. clampBrush returns null for a blank or non-numeric entry,
  // which leaves the current size in place instead of forcing it to a bound.
  function setBrush(value) {
    const size = clampBrush(value)
    if (size !== null) setTool({ ...tool, brushSize: size })
  }

  function handleBrushWheel(e) {
    e.preventDefault()
    setBrush(tool.brushSize + (e.deltaY < 0 ? 1 : -1))
  }

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <span className="group-label">File</span>
        <Dropdown label="Project">
          <button onClick={onNewProject}>New / Open</button>
          <button
            onClick={onNewFromGeotiff}
            title="Cut one large GeoTIFF into tiles and use them as a new project"
          >
            From GeoTIFF…
          </button>
          <button
            disabled={!hasProject}
            onClick={onAddImage}
            title="Copy a GeoTIFF into this project's images folder"
          >
            Add Image…
          </button>
          <button disabled={!hasProject} onClick={onSaveProject}>Save Project</button>
          <button disabled={!hasProject} onClick={onSaveUserMask}>Save Masks</button>
          <button disabled={!hasProject} onClick={onOpenSettings}>Settings…</button>
        </Dropdown>
      </div>

      <div className="toolbar-group">
        <span className="group-label">Tool</span>
        <div className="segmented">
          <button
            className={tool.mode === 'brush' ? 'active' : ''}
            onClick={() => setTool({ ...tool, mode: 'brush' })}
          >
            Brush
          </button>
          <button
            className={tool.mode === 'polygon' ? 'active' : ''}
            onClick={() => setTool({ ...tool, mode: 'polygon' })}
          >
            Polygon
          </button>
          <button
            className={tool.mode === 'wand' ? 'active' : ''}
            title="Click a region to select connected pixels of a similar colour, then Enter to fill"
            onClick={() => setTool({ ...tool, mode: 'wand' })}
          >
            Wand
          </button>
          <button
            className={tool.mode === 'region' ? 'active' : ''}
            title="Click a labeled object to select it, then Enter to delete it — the connected region under the cursor, not every pixel of its class"
            onClick={() => setTool({ ...tool, mode: 'region' })}
          >
            Region
          </button>
          <button
            className={tool.mode === 'sam' ? 'active' : ''}
            disabled={!samAvailable}
            title={samAvailable
              ? 'Click an object to snap a polygon to its edges (Efficient-SAM2)'
              : 'SAM2 model files not found in sam2/onnx/'}
            onClick={() => setTool({ ...tool, mode: 'sam' })}
          >
            SAM2
          </button>
        </div>
        <button disabled={!undoDepth} onClick={onUndo} title="Ctrl+Z">Undo</button>
        <button
          disabled={!redoDepth}
          onClick={onRedo}
          title="Ctrl+Shift+Z or Ctrl+Y"
        >
          Redo
        </button>
        <button
          className="danger"
          disabled={!labeledPixels}
          onClick={onClearLabels}
          title="Remove all labels from this image (undoable with Ctrl+Z)"
        >
          Clear Labels
        </button>
      </div>

      <div className="toolbar-group">
        <span className="group-label">View</span>
        <button
          disabled={!hasImage}
          onClick={onFit}
          title="Fit the image to the pane (resets zoom and pan)"
        >
          Fit
        </button>
      </div>

      <div className="toolbar-group">
        <span className="group-label">Brush</span>
        <input
          type="range"
          min={BRUSH_MIN}
          max={BRUSH_MAX}
          value={tool.brushSize}
          onChange={(e) => setBrush(e.target.value)}
          onWheel={handleBrushWheel}
        />
        {/* Typing an exact width, for when dragging the slider is fiddly. It
            clamps to the slider's range so the two can never show different
            things; a blank or non-numeric entry leaves the size alone rather
            than snapping to 0. */}
        <input
          type="number"
          className="brush-size"
          min={BRUSH_MIN}
          max={BRUSH_MAX}
          step="1"
          value={tool.brushSize}
          onChange={(e) => setBrush(e.target.value)}
          onWheel={handleBrushWheel}
          title={`Brush width in pixels (${BRUSH_MIN}-${BRUSH_MAX})`}
        />
        <span className="hint">px</span>
        {/* The WorldCover overlay sits to the right of the brush size, as a
            checkbox in its own right: it is a view toggle, but it modifies
            what the brush paints over, so it belongs with the brush. */}
        <label
          className="worldcover-toggle"
          title={hasWorldcover
            ? 'Overlay the ESA WorldCover 2021 land cover map (worldcover/ folder) over the image and the labels'
            : 'No worldcover/ overlay found for this image'}
        >
          <input
            type="checkbox"
            checked={showWorldcover}
            disabled={!hasWorldcover}
            onChange={(e) => setShowWorldcover(e.target.checked)}
          />
          Show Worldcover
        </label>
      </div>

      {/* Labels autosave a second after the last stroke, so this doubles as the
          "your work is on disk" readout -- there is no explicit save to forget. */}
      <div className="toolbar-group status">
        <div className="stat">
          <span className="stat-label">Labeled</span>
          <span className="stat-value">{labeledPixels.toLocaleString()} px</span>
        </div>
      </div>
    </div>
  )
}
