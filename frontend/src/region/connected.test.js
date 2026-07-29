import { describe, expect, it } from 'vitest'
import { UNLABELED } from '../constants'
import { connectedRegion, countMask } from './connected'

// A tiny grid literal, so the shape being tested is visible in the test.
// '.' is unlabeled; every other character is a class id by its digit.
function grid(rows) {
  const height = rows.length
  const width = rows[0].length
  const labels = new Uint8Array(width * height)
  rows.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      labels[y * width + x] = ch === '.' ? UNLABELED : Number(ch)
    })
  })
  return { labels, width, height }
}

const at = ({ width }, x, y) => y * width + x

/** The selected pixels as "x,y" strings, sorted -- readable on failure. */
function selected(mask, width) {
  const out = []
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) out.push(`${i % width},${(i / width) | 0}`)
  }
  return out.sort()
}

describe('connectedRegion', () => {
  it('takes the object under the click, not every pixel of its class', () => {
    // Two separate 1-blobs. Clicking the left one must not take the right one.
    const g = grid([
      '11..11',
      '11..11',
      '......',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 0, 0))
    expect(selected(mask, g.width)).toEqual(['0,0', '0,1', '1,0', '1,1'])
    expect(countMask(mask)).toBe(4)
  })

  it('is 4-connected, so a diagonal touch does not join two regions', () => {
    // With 8-connectivity these two blocks would be one region.
    const g = grid([
      '11..',
      '11..',
      '..11',
      '..11',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 0, 0))
    expect(countMask(mask)).toBe(4)
    expect(mask[at(g, 2, 2)]).toBe(0)
  })

  it('stops at a different class', () => {
    const g = grid([
      '1122',
      '1122',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 0, 0))
    expect(countMask(mask)).toBe(4)
    expect(mask[at(g, 2, 0)]).toBe(0)
  })

  it('stops at unlabeled ground', () => {
    const g = grid([
      '1.1',
      '1.1',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 0, 0))
    expect(selected(mask, g.width)).toEqual(['0,0', '0,1'])
  })

  it('follows a shape around a corner', () => {
    const g = grid([
      '1..',
      '1..',
      '111',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 2, 2))
    expect(countMask(mask)).toBe(5)
  })

  it('handles a region running off every edge', () => {
    const g = grid([
      '111',
      '111',
      '111',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 1, 1))
    expect(countMask(mask)).toBe(9)
  })

  it('does not wrap around a row boundary', () => {
    // (2,0) and (0,1) are adjacent in the flat buffer but not on the grid.
    const g = grid([
      '..1',
      '1..',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 2, 0))
    expect(selected(mask, g.width)).toEqual(['2,0'])
  })

  it('selects a lone pixel', () => {
    const g = grid([
      '.1.',
      '...',
    ])
    const mask = connectedRegion(g.labels, g.width, g.height, at(g, 1, 0))
    expect(countMask(mask)).toBe(1)
  })

  it('returns null on unlabeled ground, because there is no object there', () => {
    const g = grid([
      '.1',
      '..',
    ])
    expect(connectedRegion(g.labels, g.width, g.height, at(g, 0, 0))).toBe(null)
  })

  it('handles a whole-image region without blowing the queue', () => {
    const width = 512
    const height = 512
    const labels = new Uint8Array(width * height).fill(3)
    const mask = connectedRegion(labels, width, height, 0)
    expect(countMask(mask)).toBe(width * height)
  })
})
