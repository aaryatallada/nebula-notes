import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite dev server on 5173 by default
export default defineConfig({
  plugins: [react()]
})
