import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the built index.html works from file:// inside pywebview
export default defineConfig({
  base: './',
  plugins: [react()],
  test: {
    // happy-dom rather than jsdom: lighter, and nothing here needs the parts
    // jsdom does better. Note NEITHER implements a canvas 2d context, which is
    // why anything that paints -- the wand preview, the label canvas, the brush
    // -- is covered by frontend/smoke.html in a real browser instead.
    environment: 'happy-dom',
    setupFiles: ['./src/test-setup.js'],
    // No globals: an imported `expect` is one less thing to explain, and it
    // matches how everything else in this repo imports what it uses.
    globals: false,
    include: ['src/**/*.test.{js,jsx}'],
  },
})
