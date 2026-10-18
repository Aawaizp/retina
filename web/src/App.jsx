import { useEffect, useRef, useState } from "react";
import GeneratePanel from "./GeneratePanel.jsx";
import { clock, money, prepare, TRIP } from "./lib/day.js";
import { getJSON, url } from "./lib/json.js";
import { SHAPES, STATE_COLORS, STATE_NAMES, Store3D } from "./lib/store3d.js";

// ONE page: RETINA Supercenter in 3D. NPC shoppers (run by the student model) walk the store and every
// product they put in the basket becomes a sales row. On the owner's PC (retina app) it plays the run
// being generated, day by day; on the public website it plays days recorded by `retina web-data`.
const SPEEDS = [["2 min / sec", 2], ["5 min / sec", 5], ["15 min / sec", 15], ["1 hour / sec", 60]];
const EMPTY = { trips: 0, units: 0, rev: 0, oos: 0, sub: 0, lines: 0, done: 0, mins: 0 };

export default function App() {
  const box = useRef(null);
  const store = useRef(null);
  const S = useRef({ day: null, now: 0, playing: true, speed: 15, followed: null, source: null, queue: [], waiting: false });
  const [day, setDay] = useState(null);
  const [view, setView] = useState({ now: 0, inStore: 0, queue: 0, k: EMPTY, rows: [], feed: [] });
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(15);
  const [followed, setFollowed] = useState(null);
  const [source, setSource] = useState(null);       // {kind: "demo"|"run", label}
  const [status, setStatus] = useState("Loading the store...");
  const [job, setJob] = useState(null);

  // ---- load a day (demo file or the run being generated)
  async function load(path, label) {
    const D = await getJSON(path);
    const prepared = prepare(D);
    if (!store.current) store.current = new Store3D(box.current, D.layout, { onPick: (tr) => { S.current.followed = tr; setFollowed(tr); } });
    store.current.clearNpcs();
    Object.assign(S.current, { day: prepared, now: prepared.tMin, followed: null, waiting: false });
    setFollowed(null);
    setDay(prepared);
    setStatus(label);
  }

  // demo mode: the recorded days, played one after another
  useEffect(() => {
    getJSON("data/manifest.json")
      .then((m) => {
        S.current.source = { kind: "demo", days: m.days, i: 0 };
        setSource({ kind: "demo" });
        return load(`data/${m.days[0]}`, "playing");
      })
      .catch((e) => setStatus(`No recorded day found (${e.message}). Build it with: retina web-data`));
    return () => store.current?.dispose();
  }, []);

  // run mode: follow the days the local app is generating
  function onJob(s) {
    setJob(s);
    const src = S.current.source;
    if (!s.run || (src?.kind === "run" && src.started === s.started && src.finished && !s.running)) return;
    if (src?.kind !== "run" || src.run !== s.run || src.started !== s.started) {   // a new run (same folder too)
      S.current.source = { kind: "run", run: s.run, started: s.started, shown: -1 };
      setSource({ kind: "run", run: s.run });
    }
    S.current.source.running = s.running;
    if (S.current.waiting || S.current.source.shown < 0) nextRunDay();
  }

  async function nextRunDay() {
    const src = S.current.source;
    if (src?.kind !== "run" || src.loading) return;
    src.loading = true;
    try {
      const { days } = await fetch(url(`runs/${src.run}/view`)).then((r) => r.json());
      const next = days.find((d) => d > src.shown) ?? null;
      if (next === null) {
        S.current.waiting = true;
        setStatus(src.running ? "waiting for the simulator's next day..." : "all generated days played");
        if (!src.running) src.finished = true;
        return;
      }
      src.shown = next;
      const D = await fetch(url(`runs/${src.run}/view/${next}`)).then((r) => r.json());
      const prepared = prepare(D);
      if (!store.current) store.current = new Store3D(box.current, D.layout, { onPick: (tr) => { S.current.followed = tr; setFollowed(tr); } });
      store.current.clearNpcs();
      Object.assign(S.current, { day: prepared, now: prepared.tMin, followed: null, waiting: false });
      setFollowed(null);
      setDay(prepared);
      setStatus(`generating: day ${next + 1}`);
    } finally {
      src.loading = false;
    }
  }

  // ---- the animation loop
  useEffect(() => {
    let raf, last = performance.now(), lastUi = 0;
    const tick = (t) => {
      const st = S.current, dt = Math.min((t - last) / 1000, 0.25);
      last = t;
      if (st.day && store.current) {
        if (st.playing && !st.waiting) {
          st.now += dt * st.speed;
          if (st.now > st.day.tMax) {
            st.now = st.day.tMax;
            const src = st.source;
            if (src?.kind === "run") {                // this day was watched: the simulator may make the next
              st.waiting = true;
              fetch(url("runs/current/watched"), { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ day: src.shown }) }).catch(() => {}).finally(nextRunDay);
            }
            else if (src?.kind === "demo") {
              src.i = (src.i + 1) % src.days.length;
              load(`data/${src.days[src.i]}`, "playing");
            }
          }
        }
        const r = store.current.update(st.day, st.now, st.followed);
        if (t - lastUi > 120) {                         // the side panels: a few times a second
          lastUi = t;
          const d = st.day, i = d.idxAt(st.now), k = i > 0 ? d.cum[i - 1] : EMPTY;
          const rows = [], feed = [];
          for (let j = i - 1; j >= 0 && (rows.length < 12 || feed.length < 8); j--) {
            const e = d.all[j];
            if (e.event === "PURCHASE" && !e.merge && rows.length < 12) rows.push({ ...e, qty: d.lineQty(e, st.now) });
            if (!["VIEW_PRODUCT", "MOVE_TO_DEPARTMENT", "LEAVE"].includes(e.event) && feed.length < 8) feed.push(e);
          }
          setView({ now: st.now, ...r, k, rows, feed });
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const toggle = () => { S.current.playing = !S.current.playing; setPlaying(S.current.playing); };
  const changeSpeed = (v) => { S.current.speed = v; setSpeed(v); };
  const seek = (v) => { S.current.now = v; };
  const D = day?.D;
  const k = view.k;

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-950/90 px-4 py-3">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-1">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-500 font-bold text-white">R</span>
          <h1 className="text-lg font-bold">RETINA Supercenter <span className="font-normal muted">· shoppers walk, sales data is created</span></h1>
          <span className="label">SIMULATED</span>
          <span className="ml-auto text-sm muted">
            {D ? `${D.date} · ${source?.kind === "run" ? `your run ${source.run}` : "recorded day"} · ${status}` : status}
          </span>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-[minmax(0,1fr)] gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        <section className="min-w-0 space-y-3">
          <div className="card relative overflow-hidden p-0">
            <div ref={box} className="h-[58vh] min-h-[360px] w-full cursor-grab" />
            <div className="pointer-events-none absolute left-3 top-3 rounded-xl bg-slate-950/80 px-3 py-2 text-3xl font-bold tabular-nums text-white">
              {clock(view.now)}
            </div>
            <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-x-3 gap-y-1 rounded-xl bg-slate-950/80 px-3 py-2 text-xs text-slate-200">
              {Object.values(SHAPES).map((s) => <span key={s}>{s}</span>)}
              {Object.entries(STATE_COLORS).map(([k2, c]) => (
                <span key={k2} className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />{STATE_NAMES[k2]}</span>
              ))}
              <span>green bar on top = basket filling</span>
            </div>
          </div>
          <div className="card flex flex-wrap items-center gap-3 py-3">
            <button onClick={toggle} className="btn bg-brand-500 text-white hover:bg-brand-600">{playing ? "Pause" : "Play"}</button>
            <select value={speed} onChange={(e) => changeSpeed(+e.target.value)} className="input w-auto py-1.5 text-sm">
              {SPEEDS.map(([l, v]) => <option key={v} value={v}>{l}</option>)}
            </select>
            {day && (
              <input type="range" min={day.tMin} max={day.tMax} step={0.5} value={view.now}
                onChange={(e) => seek(+e.target.value)} className="min-w-[160px] flex-1 accent-brand-500" aria-label="time of day" />
            )}
            <span className="text-xs muted">Drag to turn the store · scroll to zoom · click a shopper to follow them</span>
          </div>
          {followed && (
            <div className="card space-y-2">
              <div className="flex items-baseline gap-2">
                <h2 className="font-semibold">Shopper {followed.shopper}</h2>
                <span className="text-sm muted">{TRIP[followed.mission] || "trip"} · {followed.minutes} min in store{followed.total ? ` · paid ${money(followed.total)}` : ""}</span>
                <button onClick={() => { S.current.followed = null; setFollowed(null); }} className="btn ml-auto border border-slate-700 text-xs">Stop following</button>
              </div>
              <ul className="max-h-48 space-y-1 overflow-auto text-sm">
                {followed.events.filter((e) => !["VIEW_PRODUCT", "MOVE_TO_DEPARTMENT"].includes(e.event)).map((e, i) => (
                  <li key={i} className={e.t > view.now ? "opacity-35" : ""}><span className="mr-2 tabular-nums muted">{clock(e.t)}</span>{day.describe(e)}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <aside className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="shoppers in the store" value={view.inStore} />
            <Stat label="waiting at checkout" value={view.queue} />
            <Stat label="trips today" value={k.trips.toLocaleString()} />
            <Stat label="sales at checkout" value={`$${Math.round(k.rev).toLocaleString()}`} />
            <Stat label="sales rows created" value={k.lines.toLocaleString()} />
            <Stat label="units sold" value={k.units.toLocaleString()} />
            <Stat label="empty shelves found" value={k.oos} />
            <Stat label="swaps / skips" value={k.sub} />
          </div>
          <div className="card space-y-2 p-4">
            <h2 className="text-xs font-semibold uppercase tracking-wide muted">Sales data being created (sales_transactions)</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-left muted"><tr><th className="pr-2">time</th><th className="pr-2">customer</th><th className="pr-2">product</th><th className="pr-2 text-right">qty</th><th className="text-right">price</th></tr></thead>
                <tbody>
                  {view.rows.map((e, i) => (
                    <tr key={`${e.shopper}-${e.t}-${i}`} className="border-t border-slate-800">
                      <td className="py-1 pr-2 tabular-nums muted">{clock(e.t)}</td>
                      <td className="pr-2 tabular-nums">C{String(e.shopper).padStart(5, "0")}</td>
                      <td className="max-w-[150px] truncate pr-2" title={day.pname(e.product)}>{day.pname(e.product)}</td>
                      <td className="pr-2 text-right">{e.qty}</td>
                      <td className="text-right tabular-nums">{money(e.price)}{e.promo ? " ★" : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs muted">★ = on promotion. Every row comes from a shopper's choice in the store.</p>
          </div>
          <div className="card space-y-2 p-4">
            <h2 className="text-xs font-semibold uppercase tracking-wide muted">What just happened</h2>
            <ul className="space-y-1 text-xs">
              {view.feed.map((e, i) => <li key={i}><span className="mr-2 tabular-nums muted">{clock(e.t)}</span>{day.describe(e)}</li>)}
            </ul>
          </div>
        </aside>
      </main>

      <section className="mx-auto max-w-7xl px-4 pb-10">
        <GeneratePanel onJob={onJob} />
      </section>

      <footer className="border-t border-slate-800 py-6 text-center text-xs muted">
        Every number on this site is <span className="label">SIMULATED</span> - not real sales. Shoppers, routes, choices, prices and
        receipts come from the RETINA simulation; the floor plan and walking speed are ESTIMATED. Product data: Open Food
        Facts / Open Prices (ODbL). Code MIT.
      </footer>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="card p-3">
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      <div className="text-xs muted">{label}</div>
    </div>
  );
}
