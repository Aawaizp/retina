# web/ - the RETINA website (v2, Phase 6)

A static site: React + Vite + Tailwind + ECharts. No server: the dashboards query Parquet files with
**DuckDB-WASM** and the what-if page runs the **student model (ONNX)** with onnxruntime-web, both in
the visitor's browser. Their engines load from the jsDelivr CDN at exact versions (too big for free
static hosting: 25 MB per file) and are cached by the browser after the first visit.

```powershell
retina web-data              # (in the repo root) build web/public/data + web/public/model
cd web
npm install                  # once
npm run dev                  # http://localhost:5173
npm run build                # static files in web/dist (any static host)
```

Pages: **Home** (what RETINA is, student panel vs engine crowd vs real statistics), **Dashboards**
(per store format: revenue and trips per day, weekday x hour traffic, departments, baskets, best
sellers), **What-if** (one shopper at one shelf: change the shopper, the trip or a price and see the
student's choice probabilities and a price curve).

Data rules: `web/public/data` holds only aggregated, SIMULATED tables built WITHOUT the private
Instacart per-shopper profiles (`RETINA_NO_PROFILES=1`, like a fresh clone) - checked by
tests/test_web_data.py. `src/lib/student.js` `features()` is a copy of `retina/student/features.py`:
change both together (the browser and Python gave identical probabilities, 78/5/2/5/11%, on the
same shelf).
