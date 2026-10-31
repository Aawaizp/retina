import { useEffect, useRef, useState } from "react";
import GeneratePanel from "./GeneratePanel.jsx";
import { clock, money, TRIP } from "./lib/day.js";
import { url } from "./lib/json.js";
import { LiveDay } from "./lib/live.js";
import { SHAPES, STATE_COLORS, STATE_NAMES, Store3D } from "./lib/store3d.js";

// The local store follows NPC actions as the backend makes them on one shared clock.
const EMPTY = { trips: 0, units: 0, rev: 0, oos: 0, sub: 0, lines: 0, done: 0, mins: 0, student: 0, studentRoutes: 0, studentBrowses: 0, studentCheckouts: 0 };
const decisionLabel = (mind) => ({ student: "Student model", autopilot: "Habit", llm: "LLM" })[mind] || "Store rule";
const storeClock = (minute) => minute >= 1440 ? "24:00" : clock(minute);

function StoreClock({ minute, closed = false, compact = false }) {
  const time = Math.max(0, Number(minute) || 0);
  const hours = (time % 720) / 720 * 360;
  const minutes = (time % 60) / 60 * 360;
  const progress = Math.min(100, time / 1440 * 100);
  return (
    <div className={`store-clock ${compact ? "store-clock-compact" : ""}`}
      style={{ "--store-hour": `${hours}deg`, "--store-minute": `${minutes}deg`, "--store-progress": `${progress}%` }}
      aria-label={`Store time ${storeClock(time)}`}>
      <span className="store-clock-dial" aria-hidden="true">
        <span className="store-clock-face">
          <i className="store-clock-mark store-clock-mark-top" /><i className="store-clock-mark store-clock-mark-right" />
          <i className="store-clock-mark store-clock-mark-bottom" /><i className="store-clock-mark store-clock-mark-left" />
          <i className="store-clock-hand store-clock-hour" /><i className="store-clock-hand store-clock-minute" />
          <i className="store-clock-pin" />
        </span>
      </span>
      <span className="store-clock-readout"><small>{closed ? "STORE CLOSED" : "STORE TIME"}</small><strong>{storeClock(time)}</strong></span>
    </div>
  );
}

export default function Studio() {
  const box = useRef(null);
  const store = useRef(null);
  const S = useRef({ day: null, now: 0, followed: null, source: null, clockOrigin: null,
                     clockStart: 0, clockRate: 5, ended: false, endAt: 1440 });
  const [day, setDay] = useState(null);
  const [view, setView] = useState({ now: 0, inStore: 0, queue: 0, k: EMPTY, rows: [], feed: [] });
  const [followed, setFollowed] = useState(null);
  const [status, setStatus] = useState("Ready to open a live store");
  const [job, setJob] = useState(null);
  const [control, setControl] = useState({ rate: 5, paused: false });
  const [theater, setTheater] = useState(false);
  const [walkMode, setWalkMode] = useState(false);

  useEffect(() => {
    if (!theater) return;
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = old; };
  }, [theater]);

  useEffect(() => {
    return () => { store.current?.dispose(); clearInterval(S.current.poll); };
  }, []);

  // The only viewing mode: the page reads decisions while the store clock runs.
  function startLive(s) {
    clearInterval(S.current.poll);
    store.current?.clearNpcs();
    Object.assign(S.current, { day: null, now: 0, followed: null, clockOrigin: null, ended: false });
    setDay(null);
    setFollowed(null);
    setWalkMode(false);
    setView({ now: 0, inStore: 0, queue: 0, k: EMPTY, rows: [], feed: [] });
    S.current.source = { kind: "live", run: s.run, started: s.started, offset: 0, running: true, layout: null };
    setControl({ rate: s.request?.clock_rate || 5, paused: false });
    setStatus("opening the store...");
    const poll = async () => {
      const src = S.current.source;
      if (src?.kind !== "live" || src.busy) return;
      src.busy = true;
      try {
        if (!src.layout) {
          const r = await fetch(url(`runs/${src.run}/layout`));
          if (!r.ok) return;
          src.layout = (await r.json()).layout;
          if (!src.layout) return;                         // the store is still opening
          store.current?.dispose();
          store.current = new Store3D(box.current, src.layout, { autoFocus: true,
            onPick: (tr) => { S.current.followed = tr; setFollowed(tr); } });
          setWalkMode(false);
        }
        const { offset, events } = await fetch(url(`runs/${src.run}/stream?offset=${src.offset}`)).then((r) => r.json());
        src.offset = offset;
        for (const e of events) {
          if (e.k === "day") {
            const d = new LiveDay(src.layout, e.date, e.t);
            store.current.clearNpcs();
            Object.assign(S.current, { day: d, now: e.t, followed: null, ended: false,
              clockOrigin: e.wall_started_at ? e.wall_started_at * 1000 : Date.now(),
              clockStart: e.t, clockRate: e.clock_rate || 5, endAt: 1440 });
            setFollowed(null);
            setDay(d);
            src.dayNo = e.day + 1;
          } else if (e.k === "day_end") {
            Object.assign(S.current, { ended: true, endAt: e.t, now: e.t,
              clockStart: e.t, clockOrigin: Date.now(), clockRate: 0 });
          } else S.current.day?.add(e);
        }
        const clockState = await fetch(url(`runs/${src.run}/clock`)).then((r) => r.json());
        if (!S.current.ended && clockState.day === (src.dayNo ?? 1) - 1 && clockState.minute != null) {
          Object.assign(S.current, {
            clockOrigin: clockState.wall_at * 1000, clockStart: clockState.minute,
            clockRate: clockState.paused ? 0 : clockState.rate,
          });
          setControl((old) => old.rate === clockState.rate && old.paused === clockState.paused
            ? old : { rate: clockState.rate, paused: clockState.paused });
        }
        if (events.length) setStatus(`LIVE: day ${src.dayNo} - the NPCs are shopping now`);
        else if (!src.running) {
          setStatus("live run finished");
          clearInterval(S.current.poll);
        }
      } catch (err) {
        /* the app may be busy: try again */
      } finally {
        src.busy = false;
      }
    };
    poll();
    S.current.poll = setInterval(poll, 500);
  }

  async function setClockControl(next) {
    setControl(next);
    Object.assign(S.current, { clockStart: S.current.now, clockOrigin: Date.now(),
      clockRate: next.paused ? 0 : next.rate });
    try {
      const r = await fetch(url("runs/current/control"), {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next),
      });
      if (!r.ok) throw new Error("Clock control could not be changed");
    } catch (err) {
      setStatus(err.message);
    }
  }

  function onJob(s) {
    setJob(s);
    const src = S.current.source;
    if (s.run && s.request?.wait_for_view && s.realtime) {
      if (src?.kind !== "live" || src.started !== s.started) {
        startLive(s);
        if (s.running) setTheater(true);
      }
      S.current.source.running = s.running;
    }
  }

  function walkInside() {
    store.current?.setWalkMode(true);
    setWalkMode(Boolean(store.current?.walkMode));
  }

  function fitStore() {
    store.current?.fitStore();
    setWalkMode(false);
  }

  // ---- the animation loop
  useEffect(() => {
    let raf, lastUi = 0;
    const tick = (t) => {
      const st = S.current;
      if (st.day && store.current) {
        if (st.clockOrigin) {
          const elapsed = Math.max(0, (Date.now() - st.clockOrigin) / 1000);
          st.now = Math.min(st.ended ? st.endAt : 1440,
            st.clockStart + Math.min(elapsed, 2.5) * st.clockRate);
        }
        const r = store.current.update(st.day, st.now, st.followed, st.clockRate);
        if (t - lastUi > 120) {                         // the side panels: a few times a second
          lastUi = t;
          const d = st.day, i = d.idxAt(st.now), k = i > 0 ? d.cum[i - 1] : EMPTY;
          setView({ now: st.now, ...r, k, ...d.recent(i, st.now, 20, 5) });
          setWalkMode((old) => old === Boolean(store.current?.walkMode) ? old : Boolean(store.current?.walkMode));
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const D = day?.D;
  const k = view.k;

  return (
    <div className="min-h-screen">
      <section id="generate" className={theater ? "hidden" : "mx-auto max-w-7xl px-4 pb-4 pt-10"}>
        <GeneratePanel onJob={onJob} onStarted={() => setTheater(true)} />
      </section>

      <section id="store-floor" className={theater ? "live-stage" : "mx-auto max-w-7xl px-4 pb-12 pt-8"}>
        {!theater && <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">LIVE STORE</p>
            <h2 className="mt-1 text-2xl font-bold tracking-tight">Watch the store work</h2>
            <p className="mt-1 text-sm muted">Walk the aisles with shoppers. Their paid items appear in the table below.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {D && <button className="btn bg-brand-500 text-white hover:bg-brand-600" onClick={() => setTheater(true)}>Open full screen ↗</button>}
            <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
              {D ? `${D.date} · ${status}` : status}
            </span>
          </div>
        </div>}
        <div className={theater ? "absolute inset-0" : "grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-[minmax(0,1fr)_380px]"}>
        <section className={theater ? "absolute inset-0" : "min-w-0 space-y-3"}>
          <div className={theater ? "absolute inset-0 overflow-hidden bg-[#10242b]" : "relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-900 shadow-sm"}>
            <div ref={box} className={theater ? "absolute inset-0 h-full w-full cursor-grab" : "h-[58vh] min-h-[360px] w-full cursor-grab"} />
            {!day && (
              <div className="absolute inset-0 grid place-items-center bg-[#10232c] px-8 text-center text-white">
                <div className="max-w-sm">
                  <div className="mx-auto mb-6 grid h-16 w-16 place-items-center rounded-2xl border border-white/20 bg-white/5 text-3xl text-[#9bd8c7]">↗</div>
                  <h3 className="text-2xl font-semibold tracking-tight">Your store will open here</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-300">Choose a store and a duration above. Watch shoppers arrive, choose products, and check out in real time.</p>
                  {!theater && <button onClick={() => document.getElementById("generate")?.scrollIntoView({ behavior: "smooth" })}
                    className="mt-5 inline-flex rounded-lg bg-white px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-slate-100">Set up a run</button>}
                </div>
              </div>
            )}
            {day && !theater && <div className="store-clock-embedded"><StoreClock minute={view.now} closed={S.current.ended} compact /></div>}
          </div>
          {!theater && day && <div className="card flex flex-wrap items-center gap-3 py-3">
            <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700"><i className="h-2 w-2 rounded-full bg-emerald-500" />LIVE STORE</span>
            <span className="text-xs muted">{view.drawn ?? 0} shown · {view.inStore} shopping</span>
            {job?.running && <div className="flex w-full flex-wrap items-center gap-2 border-t border-slate-100 pt-3">
              <button className="btn border border-slate-300 bg-white text-slate-700" onClick={() => setClockControl({ ...control, paused: !control.paused })}>
                {control.paused ? "Resume" : "Pause"}
              </button>
              <label className="text-xs font-semibold text-slate-600">Clock speed</label>
              <select className="input w-auto" value={control.rate} onChange={(e) => setClockControl({ ...control, rate: Number(e.target.value) })}>
                <option value={1}>Slow · watch walking</option><option value={5}>Normal · balanced</option><option value={20}>Fast · finish data sooner</option>
              </select>
              <span className="ml-auto text-xs muted">1 shopper = about 10 expected visitors in aggregate estimates</span>
            </div>}
            <div className="store-view-controls">
              <button className={`btn store-view-button ${walkMode ? "store-view-button-active" : ""}`} onClick={walkInside} aria-pressed={walkMode}>Walk inside</button>
              <button className={`btn store-view-button ${!walkMode ? "store-view-button-active" : ""}`} onClick={fitStore} aria-pressed={!walkMode}>Fit store</button>
              <span className="store-view-hint">{walkMode ? "WASD / arrows move · Shift faster · drag to look · Esc exits" : "Drag to pan · right-drag to turn · wheel to zoom · click a shopper to follow"}</span>
            </div>
            <details className="w-full border-t border-slate-100 pt-3 text-xs text-slate-600">
              <summary className="cursor-pointer font-semibold text-slate-700">Shopper legend</summary>
              <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
                {Object.values(SHAPES).map((s) => <span key={s}>{s}</span>)}
                {Object.entries(STATE_COLORS).map(([k2, c]) => (
                  <span key={k2} className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />{STATE_NAMES[k2]}</span>
                ))}
                <span>Shopping bag fills as products are picked</span>
              </div>
            </details>
          </div>}
          {!theater && followed && (
            <div className="card space-y-2">
              <div className="flex items-baseline gap-2">
                <h2 className="font-semibold">Shopper {followed.shopper}</h2>
                <span className="text-sm muted">{TRIP[followed.mission] || "trip"} · {followed.minutes} min in store{followed.total ? ` · paid ${money(followed.total)}` : ""}</span>
                <button onClick={() => { S.current.followed = null; setFollowed(null); }} className="btn ml-auto border border-slate-300 text-xs">Stop following</button>
              </div>
              {followed.requirements && <div className="rounded-xl bg-slate-50 px-3 py-2 text-sm text-slate-700">
                <span className="font-semibold">NPC brain: {followed.listBrain === "student" ? "local student model" : "shopping rules"}</span>
                <div className="mt-1">Requirements · household of {followed.requirements.household_size} · target {followed.requirements.target_items} items
                {followed.requirements.budget != null && ` · ${money(followed.requirements.budget)} budget`}
                </div>
                {followed.requirements.essentials.length > 0 && <div className="mt-1 text-slate-600">
                  Needs now: {followed.requirements.essentials.join(", ")}
                  {followed.essentialItems != null && ` · bought ${followed.essentialItemsBought}/${followed.essentialItems}`}
                </div>}
              </div>}
              <ul className="max-h-48 space-y-1 overflow-auto text-sm">
                {followed.events.filter((e) => !["VIEW_PRODUCT", "MOVE_TO_DEPARTMENT"].includes(e.event)).map((e, i) => (
                  <li key={i} className={e.t > view.now ? "opacity-35" : ""}><span className="mr-2 tabular-nums muted">{clock(e.t)}</span>{day.describe(e)}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {!theater && (day ? <aside className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="In store" value={view.inStore} />
            <Stat label="At checkout" value={view.queue} />
            <Stat label="Sales paid" value={`$${Math.round(k.rev).toLocaleString()}`} />
            <Stat label="Sales rows" value={k.lines.toLocaleString()} />
          </div>
          <details className="card text-sm">
            <summary className="cursor-pointer font-semibold">More run metrics</summary>
            <div className="mt-4 grid grid-cols-2 gap-3 text-slate-600">
              <span>Trips<br /><b className="text-slate-900">{k.trips.toLocaleString()}</b></span>
              <span>Units sold<br /><b className="text-slate-900">{k.units.toLocaleString()}</b></span>
              <span>Empty shelves<br /><b className="text-slate-900">{k.oos}</b></span>
              <span>Swaps / skips<br /><b className="text-slate-900">{k.sub}</b></span>
              {job?.request?.mind === "student" && <>
                <span>Student shelf choices<br /><b className="text-slate-900">{k.student.toLocaleString()}</b></span>
                <span>Student aisle choices<br /><b className="text-slate-900">{k.studentRoutes.toLocaleString()}</b></span>
                <span>Browse stops<br /><b className="text-slate-900">{k.studentBrowses.toLocaleString()}</b></span>
                <span>Early checkouts<br /><b className="text-slate-900">{k.studentCheckouts.toLocaleString()}</b></span>
              </>}
            </div>
          </details>
          <div className="card space-y-2 p-4">
            <h2 className="text-xs font-semibold uppercase tracking-wide muted">Store activity</h2>
            <ul className="space-y-2 text-xs leading-5">
              {view.feed.map((e, i) => <li key={i}><span className="mr-2 tabular-nums muted">{clock(e.t)}</span>{day.describe(e)}</li>)}
            </ul>
          </div>
        </aside> : <aside className="card flex flex-col justify-between p-7">
          <div>
            <p className="eyebrow">From action to data</p>
            <h3 className="mt-2 text-xl font-semibold tracking-tight">What you will see</h3>
            <ol className="mt-6 space-y-5 text-sm text-slate-600">
              <li><span className="mr-3 font-bold text-brand-600">01</span>Shoppers enter on the store clock.</li>
              <li><span className="mr-3 font-bold text-brand-600">02</span>They walk aisles and choose products.</li>
              <li><span className="mr-3 font-bold text-brand-600">03</span>Checkout creates rows in the purchase table.</li>
            </ol>
          </div>
          <p className="mt-8 border-t border-slate-100 pt-5 text-xs leading-5 muted">All sales shown here are synthetic. Store details supplied by you are labelled separately in the dataset.</p>
        </aside>)}
        </div>
        {!theater && <section className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm" aria-label="Live purchases">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-4">
            <div><p className="eyebrow">CHECKOUT LEDGER</p><h3 className="mt-1 text-lg font-bold text-slate-900">Purchases as they happen</h3></div>
            <span className="text-xs text-slate-500">Latest 20 paid product lines · full dataset in ZIP</span>
          </div>
          <div className="overflow-x-auto"><table className="w-full min-w-[680px] text-sm">
            <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500"><tr><th className="px-5 py-3">Time</th><th className="px-3 py-3">Shopper</th><th className="px-3 py-3">Product</th><th className="px-3 py-3 text-right">Qty</th><th className="px-3 py-3">Decision</th><th className="px-5 py-3 text-right">Sale</th></tr></thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {day && view.rows.length ? view.rows.map((row, i) => <tr key={`${row.shopper}-${row.t}-${i}`} className={i === 0 ? "bg-emerald-50/60" : ""}>
                <td className="px-5 py-3 tabular-nums text-slate-500">{clock(row.t)}</td><td className="px-3 py-3 font-medium tabular-nums">#{row.shopper}</td><td className="px-3 py-3 font-medium text-slate-900">{day.pname(row.product)}{row.promo ? <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-bold text-amber-800">DEAL</span> : null}</td><td className="px-3 py-3 text-right tabular-nums">{row.qty}</td><td className="px-3 py-3 text-slate-500">{decisionLabel(row.mind)}</td><td className="px-5 py-3 text-right font-semibold tabular-nums">{money(row.price * row.qty)}</td>
              </tr>) : <tr><td colSpan="6" className="px-5 py-10 text-center text-slate-500">{day ? "Shoppers are walking the store. The first paid purchase will appear here." : "Start a run to see every purchase reach checkout."}</td></tr>}
            </tbody>
          </table></div>
        </section>}
        {theater && <>
          <div className="live-stage-shade" aria-hidden="true" />
          <header className="live-stage-top">
            <div className="flex min-w-0 items-center gap-3"><span className="grid h-9 w-9 flex-none place-items-center rounded-xl bg-emerald-200 text-lg font-black text-emerald-950">R</span><div className="min-w-0"><p className="text-sm font-black tracking-[.15em] text-white">RETINA <span className="text-emerald-300">LIVE</span></p><p className="truncate text-xs text-slate-300">{D?.date || "Opening the store"} · {job?.days_done ?? 0} of {job?.days_wanted ?? 1} days complete</p></div></div>
            <div className="live-stage-clock"><StoreClock minute={view.now || S.current.clockStart || 0} closed={S.current.ended} /></div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              {job?.running && <><button className="live-stage-control" onClick={() => setClockControl({ ...control, paused: !control.paused })}>{control.paused ? "Resume" : "Pause"}</button><select aria-label="Clock speed" className="live-stage-control" value={control.rate} onChange={(e) => setClockControl({ ...control, rate: Number(e.target.value) })}><option value={1}>Slow · watch walks</option><option value={5}>Normal</option><option value={20}>Fast · finish sooner</option></select></>}
              {day && <><button className={`live-stage-control ${walkMode ? "live-stage-control-active" : ""}`} onClick={walkInside} aria-pressed={walkMode}>Walk inside</button><button className="live-stage-control" onClick={fitStore}>Fit store</button></>}
              {job && !job.running && job.exit_code === 0 && <><a className="live-stage-download" href={url(`runs/${job.run}/zip`)}>Download ZIP ↓</a><a className="live-stage-control" href={url(`runs/${job.run}/business/VALIDATION.md`)}>Validation report</a></>}
              <button className="live-stage-control" onClick={() => setTheater(false)}>Back to setup ✕</button>
            </div>
          </header>
          <div className="live-stage-left">
            <div className="live-stage-badge"><span className="live-pulse" /> {job?.running ? "SHOPPERS IN STORE" : "STORE RUN COMPLETE"}</div>
            <div className="live-stage-statrow"><span><b>{view.inStore}</b> inside</span><span><b>{view.queue}</b> checkout</span><span><b>{k.lines}</b> paid lines</span></div>
            {job?.request?.mind === "student" && <div className="mt-2 rounded-lg border border-emerald-200/30 bg-emerald-950/55 px-3 py-2 text-xs text-emerald-100">Local student brain · {k.student} shelf choices · {k.studentRoutes} aisle choices</div>}
            {followed && <div className="live-stage-follow"><b>Shopper #{followed.shopper}</b><p>{TRIP[followed.mission] || "Shopping"} · {followed.total ? `paid ${money(followed.total)}` : "choosing products"}</p><button onClick={() => { S.current.followed = null; setFollowed(null); }}>Stop following</button></div>}
          </div>
          <aside className="live-stage-ledger" aria-label="Live checkout purchases">
            <div className="flex items-center justify-between gap-2 border-b border-white/15 px-4 py-3"><div><p className="text-[10px] font-bold tracking-[.14em] text-emerald-200">CHECKOUT COUNTER</p><h2 className="mt-1 text-base font-bold text-white">Just paid</h2></div><span className="text-xs text-slate-300">{money(k.rev)}</span></div>
            <div className="max-h-[30vh] overflow-auto"><table className="w-full table-fixed text-xs"><thead className="text-left text-slate-400"><tr><th className="w-14 px-4 py-2">Time</th><th className="px-2 py-2">Product</th><th className="w-16 px-3 py-2 text-right">Amount</th></tr></thead><tbody>
              {day && view.rows.length ? view.rows.slice(0, 8).map((row, i) => <tr key={`${row.shopper}-${row.t}-${i}`} className="border-t border-white/10 text-slate-100"><td className="px-4 py-2 tabular-nums text-slate-300">{clock(row.t)}</td><td className="px-2 py-2" title={`Shopper #${row.shopper}: ${day.pname(row.product)}`}><div className="truncate">{day.pname(row.product)}{row.qty > 1 ? ` ×${row.qty}` : ""}</div><div className="mt-0.5 text-[10px] text-emerald-200">{decisionLabel(row.mind)}</div></td><td className="px-3 py-2 text-right font-semibold tabular-nums">{money(row.price * row.qty)}</td></tr>) : <tr><td colSpan="3" className="px-4 py-5 text-slate-300">Paid products appear here when shoppers finish at checkout.</td></tr>}
            </tbody></table></div>
          </aside>
          <div className="live-stage-hint">{walkMode ? "WASD / arrows move · Shift faster · drag to look · Esc exits" : "Drag to pan · right-drag to turn · wheel to zoom · click a shopper to follow"}{control.rate === 20 && job?.running && <span> · Slow clock shows longer walks</span>}</div>
        </>}
      </section>

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
