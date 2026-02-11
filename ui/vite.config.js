import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react({ include: /\.(jsx|js|tsx|ts)$/ }), // allow JSX in .js files
  ],
  root: '.',
  publicDir: 'public',
  server: {
    port: 3000,
    proxy: {
      '/upload': 'http://localhost:3002',
      '/update-camera': 'http://localhost:3002',
      '/camera-params': 'http://localhost:3002',
      '/sim-stream.jpg': 'http://localhost:3002',
      '/start-default-sim': 'http://localhost:3002',
      '/start-sim-stream': 'http://localhost:3002',
      '/visual_guides': 'http://localhost:3002',
      '/uploads': 'http://localhost:3002',
    },
  },
});
