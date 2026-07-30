import { hexToRgb } from './constants'

// Build the {classId: [r, g, b]} table the label overlay paints from. Pure
// function: returns a fresh object on every call, so the caller can memoize
// it on the class list and rely on referential equality across renders.
//
// Lifted out of App.jsx so the label redraw effect's "classColors changed"
// signal (and therefore its dependency array) cannot drift if the helper
// changes shape.
export function classColorsMap(classes) {
  return Object.fromEntries(classes.map((c) => [c.id, hexToRgb(c.color)]))
}