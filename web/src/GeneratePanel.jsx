import { useEffect, useRef, useState } from "react";
import { url } from "./lib/json.js";

// Generate data with the local RETINA app (`retina app` on your PC): the student model runs every NPC in
// RETINA Supercenter, each finished day is shown walking in the 3D store, and the CSV files come back here.
const AMOUNTS = [
  ["1 week", 7], ["2 weeks", 14], ["1 month", 30], ["3 months", 91], ["6 months", 182], ["1 year", 365],
];
// Strict: the student model handles the NPCs - every shelf choice of every shopper.
const MIND = "student";
// ESTIMATED run time, MEASURED on the owner's PC (15 Oct 2026, RETINA Supercenter, student):
// 7 days ~4 min, 30 days ~5.5 min, 91 days ~48 min. Days get slower as the shoppers' memory grows,
// so beyond 91 days it grows with days squared.
const MEASURED = [[0, 180], [7, 240], [30, 330], [91, 2870]];
function tenthScale(days) {
  if (days >= 91) return 2870 * (days / 91) ** 2;
  const k = MEASURED.findIndex(([d]) => d >= days);
  const [d0, s0] = MEASURED[k - 1], [d1, s1] = MEASURED[k];
  return s0 + ((s1 - s0) * (days - d0)) / (d1 - d0);
}
// Since 18 Oct 1 NPC = 1 real shopper (10x more): 2 days took 290 s (~70 s start + ~110 s a day, about
// 12.7x the per-day time above). Longer runs: the curve above times 12.7 (ESTIMATED, not yet measured).
const runSeconds = (days) => 70 + 12.7 * (tenthScale(days) - 180);
const BUSINESS = {
  "sales_transactions.csv": "one row per product sold: price, discount, cost, profit",
  "baskets.csv": "one row per receipt: customer, times, items, total",
  "daily_product_sales.csv": "product x day: units, sales, profit, promotion, lost sales (forecasting)",
  "inventory_daily.csv": "stock movements: opening, delivered, wasted, sold, closing, out of stock",
  "daily_store_kpis.csv": "one row per day: customers, sales, margin, basket, stock-outs, waste",
  "products.csv": "product master: brand, category, size, price, cost",
  "customers.csv": "customers: household, income band, price sensitivity, visits, spend",
  "promotions.csv": "promotion calendar: product, week, discount",
  "DATA_DICTIONARY.md": "every column explained",
  "DATASET_CARD.md": "what this dataset is and how it was made",
};
const FILES = {
  "receipts.csv": "one row per product bought",
  "trips.csv": "one row per shopping trip",
  "stockouts.csv": "empty shelves met by shoppers",
  "hourly.csv": "trips per hour",
  "decisions.csv": "every shelf choice: options and who decided",
  "report.txt": "what the thinking shoppers did vs real statistics",
  "report.json": "the same report as data",
  "not_sold_here.csv": "wanted but not sold in this store",
};

const api = (path, options) =>
  fetch(url(path), options).then(async (r) => {
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail;          // a text, or a list of checks (422)
      throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || `${r.status}`);
    }
    return r.json();
  });
const size = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
const time = (s) => (s < 90 ? `${Math.max(5, Math.round(s / 5) * 5)} seconds`
  : s < 5400 ? `${Math.round(s / 60)} minutes` : `${Math.round(s / 3600)} hours`);

export default function GeneratePanel({ onJob }) {
  const [app, setApp] = useState(null); // null = checking, false = not running, object = ok
  const store = "retina";
  const [days, setDays] = useState(30);
  const [watch, setWatch] = useState(true);    // the next day is made only after the 3D store played this one
  const [job, setJob] = useState(null);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const refreshRuns = () => api("runs").then(setRuns).catch(() => {});

  useEffect(() => {
    api("health")
      .then((h) => {
        setApp(h);
        refreshRuns();
        api("runs/current").then((s) => s.running && follow());
      })
      .catch(() => setApp(false));
    return () => clearInterval(timer.current);
  }, []);

  function follow() {
    clearInterval(timer.current);
    const tick = () =>
      api("runs/current").then((s) => {
        setJob(s);
        onJob?.(s);
        if (!s.running) {
          clearInterval(timer.current);
          refreshRuns();
        }
      });
    tick();
    timer.current = setInterval(tick, 1500);
  }

  async function start() {
    setError(null);
    try {
      await api("runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ store, days, mind: MIND, reset: true, view: true, wait_for_view: watch }),
      });
      follow();
    } catch (e) {
      setError(e.message);
    }
  }

  const estimate = runSeconds(Math.max(1, days));
  const running = job?.running;
  const done = job && !job.running;
  const result = done ? runs.find((r) => r.name === job.run) : null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">Generate your own data</h2>
        <p className="text-sm muted">
          Choose how much data you want. The student model runs every shopper in RETINA Supercenter; each finished
          day plays above in the 3D store. You get business-ready sales data: transactions, daily sales per product,
          inventory, store KPIs, products, customers and promotions - as CSV files with a data dictionary.
          Every row is <span className="label">SIMULATED</span>.
        </p>
      </div>

      {app === null && <p className="card muted">Looking for the local RETINA app...</p>}
      {app === false && (
        <div className="card space-y-2">
          <h2 className="font-semibold">Start the local app to generate data</h2>
          <p className="text-sm muted">
            Generating runs the Python simulation, so it happens on your own PC (free, no account, no key).
            In the RETINA folder:
          </p>
          <pre className="overflow-x-auto rounded-xl bg-slate-100 p-3 text-sm dark:bg-slate-800">retina app</pre>
          <p className="text-sm muted">Then open http://localhost:8000 - this page will have a Generate button.</p>
        </div>
      )}

      {app && (
        <div className="card space-y-5">
          <div className="space-y-2 text-sm">
            <span className="muted">How much data</span>
            <div className="flex flex-wrap gap-2">
              {AMOUNTS.map(([label, d]) => (
                <button key={d} disabled={running} onClick={() => setDays(d)}
                  className={`btn border ${days === d ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-700/30 dark:text-brand-100" : "border-slate-300 dark:border-slate-700"}`}>
                  {label}
                </button>
              ))}
              <label className="flex items-center gap-2">
                <input type="number" min={1} max={365} value={days} disabled={running}
                  onChange={(e) => setDays(Math.min(365, Math.max(1, Number(e.target.value) || 1)))}
                  className="input w-24" aria-label="days" />
                <span className="muted">days (1-365)</span>
              </label>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button onClick={start} disabled={running}
              className="btn bg-brand-500 px-5 py-2 text-white hover:bg-brand-600 disabled:opacity-50">
              {running ? "Generating..." : `Generate ${days} day${days > 1 ? "s" : ""}`}
            </button>
            <span className="text-sm muted">
              {watch ? `about 2 minutes per day while you watch, ${time(Math.max(estimate, 70 + days * 120))} in all`
                : `about ${time(estimate)} on a normal PC`} (ESTIMATED)
            </span>
          </div>
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" checked={watch} disabled={running} onChange={(e) => setWatch(e.target.checked)} className="mt-1" />
            <span>
              <b>Make the data as I watch (live)</b> - the NPCs shop NOW, on one store clock that runs at the speed
              you watch: every sales row is written the moment its shopper pays. Untick for big runs: the same NPCs
              and rules at full computer speed, nothing to watch.
            </span>
          </label>
          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
      )}

      {job && (
        <div className="card space-y-3">
          <div className="flex items-baseline justify-between">
            <h2 className="font-semibold">
              {running ? "Generating" : job.exit_code === 0 ? "Finished" : "Stopped"} - {job.request?.store}, {job.days_wanted} days
            </h2>
            <span className="text-sm muted">day {job.days_done} of {job.days_wanted}</span>
          </div>
          <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-800">
            <div className="h-2 rounded-full bg-brand-500 transition-all"
              style={{ width: `${Math.min(100, (100 * job.days_done) / Math.max(job.days_wanted, 1))}%` }} />
          </div>
          <pre className="max-h-40 overflow-auto rounded-xl bg-slate-100 p-3 text-xs dark:bg-slate-800">
            {(job.output || []).slice(-8).join("\n")}
          </pre>
          {done && job.exit_code !== 0 && <p className="text-sm text-red-600">The run stopped with an error - see the lines above.</p>}
          {running && job.request?.wait_for_view && (
            <button onClick={() => api("runs/current/nowait", { method: "POST" }).catch(() => {})}
              className="btn border border-slate-700 text-sm">Full speed - stop running the store at watching speed</button>
          )}
        </div>
      )}

      {result && <Downloads run={result} highlight />}

      {app && runs.filter((r) => r.files.length && r.name !== result?.name && !(running && r.name === job?.run)).length > 0 && (
        <div className="space-y-3">
          <h2 className="font-semibold">Earlier generated data on this PC</h2>
          {runs.filter((r) => r.files.length && r.name !== result?.name && !(running && r.name === job?.run)).map((r) => <Downloads key={r.name} run={r} />)}
        </div>
      )}
    </div>
  );
}

function Downloads({ run, highlight }) {
  const total = [...Object.values(run.sizes || {}), ...Object.values(run.business_sizes || {})].reduce((a, b) => a + b, 0);
  const business = run.business || [];
  return (
    <div className={`card space-y-3 ${highlight ? "border-brand-500" : ""}`}>
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="font-semibold">{run.name}</h3>
        <span className="text-sm muted">{run.days_done} days · {size(total)}{run.updated ? ` · ${run.updated.replace("T", " ")}` : ""}</span>
        <a href={url(`runs/${run.name}/zip`)} className="btn ml-auto bg-brand-500 text-white hover:bg-brand-600">Download all (ZIP)</a>
      </div>
      {business.length > 0 && (
        <div className="divide-y divide-slate-200 dark:divide-slate-800">
          {business.map((f) => (
            <FileRow key={f} href={url(`runs/${run.name}/business/${f}`)} name={f} text={BUSINESS[f]} bytes={run.business_sizes?.[f]} />
          ))}
        </div>
      )}
      <details className="text-sm" open={!business.length}>
        <summary className="cursor-pointer muted">{business.length ? "Extra: simulation logs" : "Files"}</summary>
        <div className="divide-y divide-slate-200 dark:divide-slate-800">
          {run.files.map((f) => (
            <FileRow key={f} href={url(`runs/${run.name}/files/${f}`)} name={f} text={FILES[f]} bytes={run.sizes?.[f]} />
          ))}
        </div>
      </details>
    </div>
  );
}

function FileRow({ href, name, text, bytes }) {
  return (
    <div className="flex flex-wrap items-center gap-2 py-2 text-sm">
      <a href={href} className="font-medium text-brand-600 hover:underline">{name}</a>
      <span className="muted">{text || ""}</span>
      <span className="ml-auto muted">{size(bytes || 0)}</span>
    </div>
  );
}
