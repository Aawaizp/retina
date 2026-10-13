import { useEffect, useMemo, useState } from "react";
import Chart from "../components/Chart.jsx";
import { getJSON } from "../lib/json.js";
import { chooseMany, loadStudent } from "../lib/student.js";

const SHOPPERS = {
  "Steady regular": { sensitivity: 1.0, household: 2, kids: 0, income: 2, budget: 110, rebuy: 0.6 },
  "Deal hunter": { sensitivity: 2.2, household: 2, kids: 0, income: 1, budget: 70, rebuy: 0.3 },
  "Busy family": { sensitivity: 1.4, household: 4, kids: 2, income: 2, budget: 160, rebuy: 0.5 },
  "Convenience first": { sensitivity: 0.6, household: 1, kids: 0, income: 3, budget: 80, rebuy: 0.4 },
};
const ROUTES = { need: "they ran out", repeat: "a regular buy", new: "something new", partner: "goes with the list" };
const MISSIONS = { top_up: "quick top-up", routine: "routine trip", stock_up: "big stock-up" };

// The option the model sees for one product at a given price
function option(p, i, factor, usual, times) {
  const price = Math.round(p.regular * factor * 100) / 100;
  const deal = factor < 0.995;
  return {
    product: p.id, price, regular: p.regular, deal, off: deal ? (1 - factor) * 100 : 0,
    times: i === usual ? times : 0, usual: i === usual, top: true, pop: p.pop, last_price: null, sold_out: 0,
  };
}

export default function WhatIf() {
  const [shelves, setShelves] = useState([]);
  const [shelf, setShelf] = useState(0);
  const [who, setWho] = useState("Steady regular");
  const [npc, setNpc] = useState(SHOPPERS["Steady regular"]);
  const [route, setRoute] = useState("repeat");
  const [mission, setMission] = useState("routine");
  const [usual, setUsual] = useState(0);
  const [times, setTimes] = useState(3);
  const [factors, setFactors] = useState([1, 1, 1, 1, 1]);
  const [focus, setFocus] = useState(0);
  const [probs, setProbs] = useState(null);
  const [curve, setCurve] = useState(null);
  const [status, setStatus] = useState("Loading the student model...");
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    getJSON("data/whatif.json").then((w) => setShelves(w.shelves)).catch((e) => setStatus(`Data not found: ${e.message}`));
    loadStudent().then(({ meta }) => { setMeta(meta); setStatus(""); }).catch((e) => setStatus(`Model not loaded: ${e.message}`));
  }, []);

  const products = shelves[shelf]?.products || [];
  const engine = products[usual]?.id ?? products[0]?.id;

  useEffect(() => {
    if (!products.length || !meta) return;
    let alive = true;
    const steps = Array.from({ length: 11 }, (_, k) => 0.5 + k * 0.1);
    const decision = (fs) => ({ npc, route, mission, engine, options: products.map((p, i) => option(p, i, fs[i], usual, times)) });
    const scenarios = [decision(factors), ...steps.map((f) => decision(factors.map((x, i) => (i === focus ? f : x))))];
    chooseMany(scenarios)
      .then(([now, ...ys]) => {
        if (!alive) return;
        setProbs(now);
        setCurve(steps.map((f, k) => [Math.round(f * 100), Math.round(ys[k][focus] * 1000) / 10]));
      })
      .catch((e) => alive && setStatus(`Model error: ${e.message}`));
    return () => { alive = false; };
  }, [products, meta, npc, route, mission, usual, times, factors, focus, engine]);

  const pickShelf = (i) => { setShelf(i); setFactors([1, 1, 1, 1, 1]); setUsual(0); setFocus(0); };
  const setFactor = (i, v) => setFactors((f) => f.map((x, k) => (k === i ? v : x)));

  const curveOption = useMemo(() => curve && {
    tooltip: { trigger: "axis", valueFormatter: (v) => `${v}%` },
    grid: { left: 45, right: 15, top: 15, bottom: 45 },
    xAxis: { type: "category", data: curve.map((c) => `${c[0]}%`), name: "price vs normal", nameLocation: "middle", nameGap: 30 },
    yAxis: { type: "value", axisLabel: { formatter: "{value}%" }, max: 100 },
    series: [{ type: "line", smooth: true, data: curve.map((c) => c[1]), areaStyle: { opacity: 0.15 } }],
  }, [curve]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">What-if: one shopper at one shelf</h1>
        <p className="text-sm muted">
          The RETINA student model runs in your browser (ONNX). Change the shopper, the reason for the trip or a price,
          and see how likely each product is to be picked. The model was trained on the choices of six AI shoppers and
          includes one price correction marked <span className="label">ESTIMATED</span>.
        </p>
      </div>
      {status && <p className="card muted">{status}</p>}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card space-y-4 lg:col-span-1">
          <Field label="Shelf">
            <select className="input" value={shelf} onChange={(e) => pickShelf(Number(e.target.value))}>
              {shelves.map((s, i) => <option key={s.aisle} value={i}>{s.aisle}</option>)}
            </select>
          </Field>
          <Field label="Shopper">
            <select className="input" value={who} onChange={(e) => { setWho(e.target.value); setNpc(SHOPPERS[e.target.value]); }}>
              {Object.keys(SHOPPERS).map((k) => <option key={k}>{k}</option>)}
            </select>
          </Field>
          <Slider label={`Price sensitivity ${npc.sensitivity.toFixed(1)}`} min={0.4} max={2.7} step={0.1}
            value={npc.sensitivity} onChange={(v) => setNpc({ ...npc, sensitivity: v })} />
          <Slider label={`Re-buy habit ${Math.round(npc.rebuy * 100)}%`} min={0.1} max={0.99} step={0.01}
            value={npc.rebuy} onChange={(v) => setNpc({ ...npc, rebuy: v })} />
          <Field label="Why it is on the list">
            <select className="input" value={route} onChange={(e) => setRoute(e.target.value)}>
              {Object.entries(ROUTES).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </Field>
          <Field label="Trip">
            <select className="input" value={mission} onChange={(e) => setMission(e.target.value)}>
              {Object.entries(MISSIONS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </Field>
          <Field label="Their usual product">
            <select className="input" value={usual} onChange={(e) => setUsual(Number(e.target.value))}>
              {products.map((p, i) => <option key={p.id} value={i}>{p.name}</option>)}
            </select>
          </Field>
          <Slider label={`Bought it before: ${times} times`} min={0} max={15} step={1} value={times} onChange={setTimes} />
        </div>

        <div className="card space-y-4 lg:col-span-2">
          <h2 className="font-semibold">{shelves[shelf]?.aisle} <span className="muted text-sm">· {shelves[shelf]?.department}</span></h2>
          {products.map((p, i) => {
            const price = p.regular * factors[i];
            const pr = probs?.[i] ?? 0;
            return (
              <div key={p.id} className={`rounded-xl border p-3 ${focus === i ? "border-brand-500" : "border-slate-200 dark:border-slate-800"}`}
                onClick={() => setFocus(i)}>
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="font-medium">{p.name}</span>
                  {i === usual && <span className="label">usual</span>}
                  {factors[i] < 0.995 && <span className="label bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200">deal -{Math.round((1 - factors[i]) * 100)}%</span>}
                  <span className="ml-auto text-sm">
                    ${price.toFixed(2)} <span className="muted">(normal ${p.regular.toFixed(2)})</span>
                  </span>
                </div>
                <input type="range" className="w-full accent-brand-500" min={0.5} max={1.5} step={0.05} value={factors[i]}
                  onChange={(e) => setFactor(i, Number(e.target.value))} aria-label={`price of ${p.name}`} />
                <div className="flex items-center gap-2">
                  <div className="h-2 flex-1 rounded-full bg-slate-100 dark:bg-slate-800">
                    <div className="h-2 rounded-full bg-brand-500 transition-all" style={{ width: `${pr * 100}%` }} />
                  </div>
                  <span className="w-14 text-right text-sm font-semibold">{Math.round(pr * 100)}%</span>
                </div>
              </div>
            );
          })}
          <p className="text-xs muted">Click a product to see its price curve below. Probabilities add up to 100% across the shelf.</p>
        </div>
      </div>

      {curveOption && (
        <div className="card">
          <h2 className="mb-2 font-semibold">How the chance of picking "{products[focus]?.name}" changes with its price</h2>
          <Chart height={280} option={curveOption} />
          <p className="text-xs muted">Other products keep their current prices. Real grocery research: about -2.5 own-price elasticity.</p>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block space-y-1 text-sm">
      <span className="muted">{label}</span>
      {children}
    </label>
  );
}

function Slider({ label, ...props }) {
  return (
    <label className="block space-y-1 text-sm">
      <span className="muted">{label}</span>
      <input type="range" className="w-full accent-brand-500" {...props} value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))} />
    </label>
  );
}
