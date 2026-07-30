import { describe, expect, it } from 'vitest'
import { classColorsMap } from './colors'

// The helper is what App.jsx's classColors depends on. The interesting thing
// to pin is *not* "this function works" -- it is "App.jsx does not rebuild the
// result on every render", which is the regression the viewport-jitter spec
// thought it had already fixed. The memoization is in App.jsx; this file pins
// the pure helper's contract so the memoized consumer has a tested input.

describe('classColorsMap', () => {
  it('returns an object keyed by class id', () => {
    const classes = [
      { id: 0, name: 'Urban', color: '#ff5028' },
      { id: 1, name: 'Water', color: '#2f7ce0' },
    ]
    const got = classColorsMap(classes)
    expect(Object.keys(got).sort()).toEqual(['0', '1'])
  })

  it('decodes hex colours into [r, g, b] triples', () => {
    const classes = [
      { id: 0, name: 'Urban', color: '#ff5028' },  // 255, 80, 40
      { id: 1, name: 'Water', color: '#2f7ce0' },  // 47, 124, 224
      { id: 2, name: 'Forest', color: '#14562b' }, // 20, 86, 43
    ]
    const got = classColorsMap(classes)
    expect(got[0]).toEqual([255, 80, 40])
    expect(got[1]).toEqual([47, 124, 224])
    expect(got[2]).toEqual([20, 86, 43])
  })

  it('passes each value through unchanged rather than mutating the input', () => {
    // The object is consumed by React effects, so it must be a fresh object --
    // otherwise memoization by reference is impossible.
    const classes = [{ id: 0, name: 'A', color: '#000000' }]
    const a = classColorsMap(classes)
    const b = classColorsMap(classes)
    expect(a).not.toBe(b)
    expect(a[0]).not.toBe(b[0])
    expect(a[0]).toEqual(b[0])
  })

  it('handles an empty list', () => {
    expect(classColorsMap([])).toEqual({})
  })
})