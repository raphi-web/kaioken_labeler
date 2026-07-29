import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { UNLABELED } from '../constants'
import ClassPalette from './ClassPalette'

// The palette's two jobs: pick a class, and toggle whether its labels are
// drawn. Hiding is a view setting, so the interesting assertions are about
// what it does NOT change -- the counts stay, the labels stay, and picking a
// hidden class brings it back rather than leaving you painting into nothing.

const CLASSES = [
  { id: 0, name: 'Urban', color: '#e05a4f' },
  { id: 1, name: 'Road', color: '#4a4a55' },
  { id: 7, name: 'Water', color: '#2f7ce0' },
]

// Four Urban pixels, one Water.
function labels() {
  const buf = new Uint8Array(16).fill(UNLABELED)
  buf[0] = 0; buf[1] = 0; buf[2] = 0; buf[3] = 0
  buf[9] = 7
  return buf
}

function show({ hidden = [], tool = { classId: 0, eraser: false } } = {}) {
  const onSelectClass = vi.fn()
  const onToggleVisible = vi.fn()
  render(<ClassPalette
    classes={CLASSES}
    tool={tool}
    setTool={vi.fn()}
    onSelectClass={onSelectClass}
    hiddenClasses={new Set(hidden)}
    onToggleVisible={onToggleVisible}
    labels={labels()}
    labelsVersion={1}
  />)
  return { onSelectClass, onToggleVisible }
}

const eyes = () => document.querySelectorAll('.class-eye')
const rows = () => document.querySelectorAll('.class-item')

describe('the class rows', () => {
  it('gives every class a toggle, and the eraser none', () => {
    show()
    expect(eyes()).toHaveLength(CLASSES.length)
    // The eraser is a row too, so there is one more row than there are toggles.
    expect(rows()).toHaveLength(CLASSES.length + 1)
    expect(document.querySelector('.class-item.eraser .class-eye')).toBe(null)
  })

  it('reports which class was picked', () => {
    const { onSelectClass } = show()
    fireEvent.click(screen.getByTitle(/^Road/))
    expect(onSelectClass).toHaveBeenCalledWith(1)
  })

  it('reports which class was toggled, without picking it', () => {
    const { onToggleVisible, onSelectClass } = show()
    fireEvent.click(screen.getByRole('button', { name: 'Hide Water' }))
    expect(onToggleVisible).toHaveBeenCalledWith(7)
    // The eye sits inside the row; a click on it must not also select.
    expect(onSelectClass).not.toHaveBeenCalled()
  })
})

describe('the hidden state', () => {
  it('marks the row and flips the icon', () => {
    show({ hidden: [1] })
    const road = [...rows()].find((r) => r.textContent.includes('Road'))
    expect(road.className).toContain('hidden')
    expect(road.querySelector('.class-eye').className).toContain('off')
    // and only that row
    const urban = [...rows()].find((r) => r.textContent.includes('Urban'))
    expect(urban.className).not.toContain('hidden')
  })

  it('says what a click will do, in both directions', () => {
    show({ hidden: [1] })
    expect(screen.getByRole('button', { name: 'Show Road' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Hide Urban' })).toBeTruthy()
  })

  it('keeps the painted-pixel count, because those pixels are still there', () => {
    show({ hidden: [0] })
    const urban = [...rows()].find((r) => r.textContent.includes('Urban'))
    expect(urban.querySelector('.class-count').textContent).toBe('4')
    expect(urban.querySelector('.class-count').className).not.toContain('empty')
  })

  it('still marks a class with no pixels as empty', () => {
    show()
    const road = [...rows()].find((r) => r.textContent.includes('Road'))
    expect(road.querySelector('.class-count').className).toContain('empty')
  })
})
