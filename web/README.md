# RETINA website

The React/Vite site has two pages. The landing page runs a slow, repeating 3D shopper demo in
the browser. It is illustrative, needs no backend data run, and writes no CSVs. **Data studio**
opens a live store through the local app (`retina app`). Its 3D floor follows NPC actions from
the running simulator; the store clock advances automatically. Choose the first store time
each day and a daily NPC limit before opening the store.

The local **Add a store** form creates a US/USD store from user-supplied operating details and a
first simulated date. A checked product CSV and typical weekday hourly visitor counts are optional;
missing inputs use labelled generic estimates. The Python API creates an isolated profile under
`data/custom_stores/`. The single NPC system uses local student models for goals and shelf
choices, with store rules for walking, stock and payment. It makes no LLM call for each action.
`npc_actions.csv` records aisle and shelf decisions as they happen; `sales.csv` and `visits.csv`
are appended only when the NPC pays.
During a run, the local list student can add a browse stop at a visible aisle, the shelf student
chooses there, and the checkout student can end a trip early. Browsing conversion is an estimate;
the input data does not observe physical aisle browsing.
Store data supplied by the user is labelled `USER_PROVIDED`; defaults are `ESTIMATED`; all sales
remain `SIMULATED`.

After a run, the business export has a row for every carried product on every completed day,
including days with zero sales. Raw receipts, inventory and unprefixed sales are at the simulated
NPC scale. `sample_weight` describes how many expected shoppers one NPC represents;
`estimated_store_*` totals expand the sample. Those estimates are blank on days when the NPC
arrival limit may have clipped traffic. `VALIDATION.json` and `VALIDATION.md` check table links,
accounting totals, SKU-day coverage and inventory balances. They make no realism claim without
observed store data.

Data Studio can compare a completed run with full-store daily totals. Upload UTF-8
`daily_store_kpis.csv` (`date,transactions[,net_sales]`) or `daily_product_sales.csv`
(`date,product_id,units_sold[,net_sales]`), up to 10 MB per file. Use `YYYY-MM-DD` dates and
product IDs from the run. Comparisons show exact-key coverage and errors against the expanded
estimates. The observed CSVs are stored outside the downloadable synthetic-data ZIP. For larger
reference files, run `retina validate --run <run-name> --reference <folder>` from the command line.

```powershell
cd web
npm install
npm run dev          # frontend development at http://localhost:5173
npm run build        # build web/dist for `retina app`
cd ..
retina app           # local app at http://localhost:8000
```
