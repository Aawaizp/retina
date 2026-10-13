import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static site: relative paths so it works on Cloudflare Pages, GitHub Pages, a plain folder - or served
// by the local app (`retina app`). In development, API calls go to the local app on port 8000.
const app = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  base: "./",
  optimizeDeps: { exclude: ["@duckdb/duckdb-wasm", "onnxruntime-web"] },
  build: { chunkSizeWarningLimit: 1200 },
  server: { proxy: { "/health": app, "/stores": app, "/runs": app } },
});
