import { cleanup } from '@testing-library/react'
import { afterEach, beforeAll } from 'vitest'

// Unmount between tests. Without it a component from an earlier test is still
// in the document and every getBy* query starts matching two things.
afterEach(cleanup)

// happy-dom (and jsdom) return null from getContext('2d'), and this app draws
// on mount -- the brush cursor is rendered into an offscreen canvas before
// anything else happens -- so without a stand-in nothing here can even mount.
//
// This is a shim, not an implementation: every drawing call is a no-op and
// getImageData hands back transparent black. That is deliberate. Tests in this
// directory are about state and wiring; whether the right pixels land on the
// canvas is checked in frontend/smoke.html, which runs the real thing in a
// real browser. An assertion about canvas CONTENT here would be measuring the
// shim, so treat one as a bug in the test.
const noop = () => {}

function stubContext(canvas) {
  const pixels = (w, h) => ({ data: new Uint8ClampedArray(Math.max(0, w * h * 4)),
                              width: w, height: h, colorSpace: 'srgb' })
  const base = {
    canvas,
    createImageData: (w, h) => pixels(w, h),
    getImageData: (x, y, w, h) => pixels(w, h),
    measureText: () => ({ width: 0 }),
  }
  // Anything else the app calls is a no-op; anything it reads back is a
  // function, which is harmless for the style properties nobody asserts on.
  return new Proxy(base, {
    get: (target, prop) => (prop in target ? target[prop] : noop),
    set: (target, prop, value) => { target[prop] = value; return true },
  })
}

beforeAll(() => {
  const canvas = window.HTMLCanvasElement.prototype
  canvas.getContext = function getContext() { return stubContext(this) }
  canvas.toDataURL = () => 'data:image/png;base64,'
  // LabelerPane captures the pointer while stroking; happy-dom has no such thing.
  const element = window.Element.prototype
  element.setPointerCapture ??= noop
  element.releasePointerCapture ??= noop
  element.hasPointerCapture ??= () => false
})
