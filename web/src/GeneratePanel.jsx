import { useEffect, useRef, useState } from "react";
import { url } from "./lib/json.js";
import StoreSetupPanel from "./StoreSetupPanel.jsx";
import ValidationPanel from "./ValidationPanel.jsx";

const api = (path, options) => fetch(url(path), options).then(async (response) => {
  if (!response.ok) {
    const detail = (await response.json().catch(() => ({}))).detail;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || `Request failed (${response.status})`);
  }
  return response.json();
});

const durations = [["1 day", 1], ["1 week", 7], ["30 days", 30]];

export default function GeneratePanel({ onJob, onStarted }) {
  const [app, setApp] = useState(null);
  const [stores, setStores] = useState([]);
  const [store, setStore] = useState("retina");
  const [days, setDays] = useState(1);
  const [startTime, setStartTime] = useState("09:00");
  const [clockRate, setClockRate] = useState(5);
  const [job, setJob] = useState(null);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState("");
  const timer = useRef(null);

  const refreshRuns = () => api("runs").then(setRuns).catch(() => {});
  const refreshStores = () => api("stores").then(setStores).catch(() => {});

  useEffect(() => {
    api("health").then((health) => {
      setApp(health);
      refreshRuns();
      refreshStores();
      api("runs/current").then((current) => { if (current.running) follow(); }).catch(() => {});
    }).catch(() => setApp(false));
    return () => clearInterval(timer.current);
  }, []);

  useEffect(() => {
    const chosen = stores.find((entry) => entry.name === store);
    if (!chosen) return;
    const [open, close] = chosen.open_hours || [0, 24];
    const minute = Number(startTime.slice(0, 2)) * 60 + Number(startTime.slice(3, 5));
    if (minute < open * 60 || minute >= close * 60) setStartTime(`${String(open).padStart(2, "0")}:00`);
  }, [stores, store, startTime]);

  function follow() {
    clearInterval(timer.current);
    const tick = () => api("runs/current").then((current) => {
      setJob(current);
      onJob?.(current);
      if (!current.running) { clearInterval(timer.current); refreshRuns(); }
    }).catch((e) => setError(e.message));
    tick();
    timer.current = setInterval(tick, 1500);
  }

  async function start() {
    setError("");
    try {
      await api("runs", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ store, days, mind: "student", reset: true, view: true,
          wait_for_view: true, start_time: startTime, scale: .1, clock_rate: clockRate }) });
      onStarted?.();
      follow();
    } catch (e) { setError(e.message); }
  }

  async function removeRun(name) {
    if (!window.confirm(`Remove ${name} and its generated data from this computer? This cannot be undone.`)) return;
    setError("");
    try {
      await api(`runs/${encodeURIComponent(name)}`, { method: "DELETE" });
      await refreshRuns();
      if (job?.run === name) setJob(null);
    } catch (e) { setError(e.message); }
  }

  const chosen = stores.find((entry) => entry.name === store);
  const custom = stores.filter((entry) => entry.custom);
  const modelReady = app?.student_models?.list && app?.student_models?.shelf;
  const opening = chosen?.open_hours?.[0] ?? 0;
  const closing = chosen?.open_hours?.[1] ?? 24;
  const startMinute = Number(startTime.slice(0, 2)) * 60 + Number(startTime.slice(3, 5));
  const validTime = startMinute >= opening * 60 && startMinute < closing * 60;
  const running = !!job?.running;
  const result = job && !running && job.exit_code === 0 ? runs.find((run) => run.name === job.run) : null;
  const past = runs.filter((run) => run.files.length && run.name !== result?.name && run.name !== (running ? job?.run : ""));

  return <div className="space-y-5">
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div><p className="eyebrow">DATA STUDIO</p><h1 className="mt-2 text-3xl font-extrabold tracking-tight text-slate-950 sm:text-4xl">Create your store dataset</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">Choose a store and how long to simulate. Watch shoppers make decisions live; download the finished data as one package.</p></div>
      <span className="label">US stores · USD · synthetic</span>
    </div>

    {app === null && <div className="card text-sm text-slate-600">Connecting to the local RETINA app…</div>}
    {app === false && <div className="card"><h2 className="font-semibold">Start RETINA on this computer</h2><p className="mt-2 text-sm text-slate-600">Run <code className="rounded bg-slate-100 px-1.5 py-1">retina app</code> in the project folder, then refresh this page.</p></div>}

    {app && <div className="grid gap-5 lg:grid-cols-[minmax(0,1.4fr)_minmax(280px,.6fr)]">
      <section className="card p-5 sm:p-7" aria-label="Dataset setup">
        <div className="mb-6 flex items-center gap-3"><span className="grid h-9 w-9 place-items-center rounded-full bg-brand-500 text-sm font-bold text-white">1</span><div><h2 className="font-bold text-slate-950">Choose a store</h2><p className="text-xs text-slate-500">Store details shape arrivals, products, hours, and checkout.</p></div></div>
        <select className="input" aria-label="Store" value={store} disabled={running} onChange={(e) => {
          const next = stores.find((entry) => entry.name === e.target.value);
          setStore(e.target.value); setStartTime(`${String(next?.open_hours?.[0] ?? 9).padStart(2, "0")}:00`);
        }}>
          <option value="retina">Sample store · RETINA Supercenter</option>
          {custom.map((entry) => <option key={entry.id} value={entry.name}>{entry.title}</option>)}
        </select>
        <p className="mt-2 text-xs leading-5 text-slate-500">{chosen?.custom ? "Using your store setup and uploaded products. Any missing details are estimated and labelled in the dataset." : "A ready-to-try sample store. Add your own store for a store-specific dataset."}{chosen?.start_date ? ` First simulated date: ${chosen.start_date}.` : ""}</p>

        <div className="my-6 border-t border-slate-100" />
        <div className="mb-4 flex items-center gap-3"><span className="grid h-9 w-9 place-items-center rounded-full bg-brand-500 text-sm font-bold text-white">2</span><div><h2 className="font-bold text-slate-950">Choose a duration</h2><p className="text-xs text-slate-500">Start with one day to see how it works.</p></div></div>
        <div className="flex flex-wrap gap-2">
          {durations.map(([label, value]) => <button key={value} type="button" disabled={running} onClick={() => setDays(value)}
            className={`btn border px-4 py-2.5 ${days === value ? "border-brand-600 bg-brand-50 text-brand-700" : "border-slate-200 bg-white text-slate-600 hover:border-brand-500"}`}>{label}</button>)}
          <label className="flex items-center gap-2 text-sm text-slate-600"><input type="number" min="1" max="365" disabled={running} value={days} onChange={(e) => setDays(Math.min(365, Math.max(1, Number(e.target.value) || 1)))} className="input w-20 text-center" aria-label="Custom number of days" /> days</label>
        </div>

        <details className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm">
          <summary className="cursor-pointer font-semibold text-slate-700">Time and speed settings</summary>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <label className="font-medium text-slate-700">Start each day at<input type="time" value={startTime} min={`${String(opening).padStart(2, "0")}:00`} max={`${String(Math.min(closing - 1, 23)).padStart(2, "0")}:59`} disabled={running} onChange={(e) => setStartTime(e.target.value)} className="input mt-1" /></label>
            <label className="font-medium text-slate-700">Store clock<select className="input mt-1" value={clockRate} disabled={running} onChange={(e) => setClockRate(Number(e.target.value))}><option value={1}>Slow · watch each walk</option><option value={5}>Normal · balanced</option><option value={20}>Fast · finish data sooner</option></select></label>
          </div>
          <p className="mt-3 text-xs text-slate-500">Open hours: {String(opening).padStart(2, "0")}:00–{String(closing).padStart(2, "0")}:00. Earlier arrivals are excluded. Fast clock compresses a short aisle walk into a fraction of a second.</p>
        </details>

        <div className="mt-6 border-t border-slate-100 pt-5">
          <button type="button" onClick={start} disabled={running || !modelReady || !validTime} className="btn w-full bg-brand-500 px-6 py-3.5 text-white hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto">{running ? "Store is running…" : "Open live store →"}</button>
          {!modelReady && <p className="mt-2 text-sm font-medium text-amber-700">The local shopper models must be installed before a run can start.</p>}
          {error && <p role="alert" className="mt-2 text-sm text-red-700">{error}</p>}
        </div>
      </section>

      <aside className="space-y-4">
        <div className="rounded-2xl bg-[#103d35] p-6 text-white shadow-sm"><p className="text-xs font-bold uppercase tracking-[.17em] text-emerald-200">What happens next</p>
          <ol className="mt-5 space-y-5 text-sm"><li><b className="mr-2 text-emerald-200">01</b>Shoppers arrive and walk the store.</li><li><b className="mr-2 text-emerald-200">02</b>Local student models choose shopping goals and score products at the shelf.</li><li><b className="mr-2 text-emerald-200">03</b>Paid items appear in the live table.</li><li><b className="mr-2 text-emerald-200">04</b>Download one ZIP with ready-to-use tables.</li></ol>
          <p className="mt-5 border-t border-white/20 pt-4 text-xs leading-5 text-emerald-100">Chance still varies arrivals and samples choices from the models’ probabilities. A saved seed makes a run repeatable; purchases are written only when NPCs check out.</p>
          <p className="mt-3 text-xs leading-5 text-emerald-100">One simulated shopper represents about 10 expected visitors for aggregate estimates. Individual purchases stay individual records.</p></div>
        <StoreSetupPanel disabled={running} onCreated={(created) => { refreshStores(); setStore(created.id); }} />
      </aside>
    </div>}

    {job && <section className="card flex flex-wrap items-center gap-4 p-5" aria-live="polite">
      <div className="min-w-[210px] flex-1"><h2 className="font-bold text-slate-900">{running ? "Your store is live" : job.exit_code === 0 ? "Dataset ready" : "Run stopped"}</h2><p className="mt-1 text-sm text-slate-500">{job.days_done} of {job.days_wanted} days completed</p>{job.seed != null && <details className="mt-1 text-xs text-slate-500"><summary className="cursor-pointer">Repeat this run</summary><p className="mt-1">Random seed: {job.seed}. It repeats arrival timing and model sampling with the same store, code, and models.</p></details>}</div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 sm:order-3"><div className="h-full bg-brand-500 transition-all" style={{ width: `${Math.min(100, 100 * job.days_done / Math.max(1, job.days_wanted))}%` }} /></div>
      {result && <a className="btn bg-brand-500 px-5 py-2.5 text-white hover:bg-brand-600" href={url(`runs/${result.name}/zip`)}>Download dataset ZIP ↓</a>}
      {result && <button type="button" className="btn text-slate-500 hover:text-red-700" onClick={() => removeRun(result.name)}>Remove run</button>}
      {!running && job.exit_code !== 0 && <details className="w-full text-xs text-red-700"><summary className="cursor-pointer">View error details</summary><pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap">{(job.output || []).slice(-8).join("\n")}</pre></details>}
    </section>}

    {result && <ValidationPanel key={`${result.name}-${job.started}`} run={result.name} />}

    {past.length > 0 && <details className="card text-sm"><summary className="cursor-pointer font-semibold text-slate-700">Saved datasets on this computer ({past.length})</summary>
      <div className="mt-4 divide-y divide-slate-100">{past.map((run) => <div key={run.name} className="flex flex-wrap items-center gap-3 py-3"><span className="min-w-0 flex-1 truncate font-medium text-slate-900">{run.name} <small className="ml-1 font-normal text-slate-500">{run.days_done} days</small></span><a className="btn border border-slate-200 text-brand-700" href={url(`runs/${run.name}/zip`)}>Download ZIP</a>{run.business?.includes("VALIDATION.md") && <a className="btn border border-slate-200 text-brand-700" href={url(`runs/${run.name}/business/VALIDATION.md`)}>Validation</a>}<button className="btn text-slate-500 hover:text-red-700" type="button" onClick={() => removeRun(run.name)}>Remove</button></div>)}</div>
    </details>}
  </div>;
}
