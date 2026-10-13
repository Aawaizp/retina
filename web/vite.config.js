import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static site: relative paths so it works on Cloudflare Pages, GitHub Pages or a plain folder.
export default defineConfig({
  plugins: [react()],
  base: "./",
  optimizeDeps: { exclude: ["@duckdb/duckdb-wasm", "onnxruntime-web"] },
  build: { chunkSizeWarningLimit: 1200 },
});
