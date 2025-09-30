import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        format: 'es'
      }
    }
  },
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {
      allow: ['..']
    }
  },
  worker: {
    format: 'es'
  },
  assetsInclude: ['**/*.onnx', '**/*.json'],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  }
});