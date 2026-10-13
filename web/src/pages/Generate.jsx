import { useEffect, useRef, useState } from "react";
import { url } from "../lib/json.js";

// Generate data with the local RETINA app (`retina app` on your PC): the student model decides for every
// shopper, Python runs the store, and the CSV files come back here to download.
const AMOUNTS = [
  ["1 week", 7], ["2 weeks", 14], ["1 month", 30], ["3 months", 91], ["6 months", 182], ["1 year", 365],
];
const MINDS = {
  student: "Thinking shoppers - every NPC decides with the student model",
  fast: "Engine only - fastest, statistical shoppers",
};
// ESTIMATED seconds per day for the supermarket (7-day runs on the owner's PC, 14 Oct 2026)
const SECONDS_PER_DAY = { student: 2.3, fast: 0.7 };
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
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status}`);
    return r.json();
  });
const size = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
const time = (s) => (s < 90 ? `${Math.max(5, Math.round(s / 5) * 5)} seconds` : `${Math.round(s / 60)} minutes`);

export default function Generate() {
  const [app, setApp] = useState(null); // null = checking, false = not running, object = ok
  const [stores, setStores] = useState([]);
  const [store, setStore] = useState("supermarket");
  const [days, setDays] = useState(30);
  const [mind, setMind] = useState("student");
  const [job, setJob] = useState(null);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const refreshRuns = () => api("runs").then(setRuns).catch(() => {});

  useEffect(() => {
    api("health")
      .then((h) => {
        setApp(h);
        api("stores").then(setStores);
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
        body: JSON.stringify({ store, days, mind, reset: true }),
      });
      follow();
    } catch (e) {
      setError(e.message);
    }
  }

  const traffic = (stores.find((s) => s.name === store)?.shoppers_per_week || 13583) / 13583;
  const estimate = days * SECONDS_PER_DAY[mind] * traffic;
  const running = job?.running;
  const done = job && !job.running;
  const result = done ? runs.find((r) => r.name === job.run) : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Generate data</h1>
        <p className="text-sm muted">
          Choose a store and how much data you want. The RETINA simulation runs on your PC: the student model decides
          for every shopper, the store engine handles traffic, prices, stock and checkout. Then download the CSV files.
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
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block space-y-1 text-sm">
              <span className="muted">Store</span>
              <select className="input" value={store} onChange={(e) => setStore(e.target.value)} disabled={running}>
                {stores.map((s) => <option key={s.name} value={s.name}>{s.title}</option>)}
              </select>
            </label>
            <label className="block space-y-1 text-sm">
              <span className="muted">Shoppers</span>
              <select className="input" value={mind} onChange={(e) => setMind(e.target.value)} disabled={running}>
                {Object.entries(MINDS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
          </div>

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
            <span className="text-sm muted">about {time(estimate)} on a normal PC (ESTIMATED)</span>
          </div>
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
        </div>
      )}

      {result && <Downloads run={result} highlight />}

      {app && runs.filter((r) => r.files.length && r.name !== result?.name).length > 0 && (
        <div className="space-y-3">
          <h2 className="font-semibold">Earlier generated data on this PC</h2>
          {runs.filter((r) => r.files.length && r.name !== result?.name).map((r) => <Downloads key={r.name} run={r} />)}
        </div>
      )}
    </div>
  );
}

function Downloads({ run, highlight }) {
  const total = Object.values(run.sizes || {}).reduce((a, b) => a + b, 0);
  return (
    <div className={`card space-y-3 ${highlight ? "border-brand-500" : ""}`}>
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="font-semibold">{run.name}</h3>
        <span className="text-sm muted">{run.days_done} days · {size(total)}{run.updated ? ` · ${run.updated.replace("T", " ")}` : ""}</span>
        <a href={url(`runs/${run.name}/zip`)} className="btn ml-auto bg-brand-500 text-white hover:bg-brand-600">Download all (ZIP)</a>
      </div>
      <div className="divide-y divide-slate-200 dark:divide-slate-800">
        {run.files.map((f) => (
          <div key={f} className="flex flex-wrap items-center gap-2 py-2 text-sm">
            <a href={url(`runs/${run.name}/files/${f}`)} className="font-medium text-brand-600 hover:underline">{f}</a>
            <span className="muted">{FILES[f] || ""}</span>
            <span className="ml-auto muted">{size(run.sizes?.[f] || 0)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
