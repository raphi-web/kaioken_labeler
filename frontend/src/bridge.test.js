import { describe, expect, it } from 'vitest'
import { fromBase64, toBase64 } from './bridge'

// The label buffer crosses the bridge as base64 on every stroke end, so these
// two are the narrowest place a whole image's labels can be corrupted.

describe('base64 round trip', () => {
  it('preserves every byte value', () => {
    const all = new Uint8Array(256)
    for (let i = 0; i < 256; i++) all[i] = i
    expect([...fromBase64(toBase64(all))]).toEqual([...all])
  })

  it('survives a buffer larger than the chunk size', () => {
    // toBase64 encodes in 0x8000 slices because String.fromCharCode.apply
    // blows the argument limit on a whole label buffer. A 512x512 image is
    // 262,144 bytes -- eight chunks -- so anything smaller than this would
    // never exercise the seam.
    const big = new Uint8Array(512 * 512)
    for (let i = 0; i < big.length; i++) big[i] = (i * 7 + (i >> 8)) & 0xff
    const back = fromBase64(toBase64(big))
    expect(back.length).toBe(big.length)
    // Compare as arrays only on failure; a 262k element diff is unreadable.
    let firstBad = -1
    for (let i = 0; i < big.length; i++) {
      if (back[i] !== big[i]) { firstBad = i; break }
    }
    expect(firstBad).toBe(-1)
  })

  it('handles the empty buffer', () => {
    expect(fromBase64(toBase64(new Uint8Array(0))).length).toBe(0)
  })

  it('returns a Uint8Array, not a string or a plain array', () => {
    expect(fromBase64(toBase64(new Uint8Array([1, 2, 3])))).toBeInstanceOf(Uint8Array)
  })
})
