import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getJSON, STORE_NAMES } from "../lib/json.js";

const fmt = (n, d = 0) => (n == null ? "-" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d }));
const pct = (x) => (x == null ? "-" : `${Math.round(x * 100)}%`);

export default function Home() {
  const [manifest, setManifest] = useState(null);
  const [student, setStudent] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getJSON("data/manifest.json").then(setManifest).catch((e) => setError(e.message));
    getJSON("data/student.json").then(setStudent).catch(() => {});
  }, []);

  const stores = manifest ? Object.entries(manifest.stores) : [];
  const trips = stores.reduce((a, [, s]) => a + s.trips, 0);
  const groups = student?.groups || {};
  const crowd = groups["engine crowd"];
  const panel = groups["thinking panel (student)"];

  return (
    <div className="space-y-10">
      <section className="space-y-4">
        <span className="label">SIMULATED data</span>
        <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">A virtual store whose shoppers think.</h1>
        <p className="max-w-3xl text-lg muted">
          RETINA simulates a grocery store day by day. Thousands of NPC shoppers - with habits calibrated on public
          statistics - walk the aisles, react to prices, deals and empty shelves, and leave receipts. Some of them
          <i> think</i>: a small model, trained on the choices of several AI shoppers, decides at the shelf. The result
          is realistic retail data anyone can download and study.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link to="/dashboards" className="btn bg-brand-500 text-white hover:bg-brand-600">Open the dashboards</Link>
          <Link to="/what-if" className="btn border border-slate-300 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">
            Try the what-if
          </Link>
        </div>
      </section>

      {error && <p className="card text-red-600">Data not found ({error}). Build it with: retina web-data</p>}

      <section className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        <Stat label="Store formats" value={fmt(stores.length)} />
        <Stat label="Simulated days per store" value={fmt(manifest?.days)} />
        <Stat label="Shopping trips on this site" value={fmt(trips)} />
        <Stat label="Thinking choices, LLM calls" value={student ? `${fmt(student.minds?.student)}, 0` : "-"} />
      </section>

      {crowd && panel && (
        <section className="card space-y-3">
          <h2 className="text-xl font-semibold">Do the thinking shoppers act like people?</h2>
          <p className="muted text-sm">
            Same store, same {student.days} days: a panel of NPCs decided by the student model vs the statistical engine
            crowd, next to public statistics about real shoppers.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left muted">
                <tr><th className="py-2 pr-4"></th><th className="pr-4">Student panel</th><th className="pr-4">Engine crowd</th><th>Real shoppers</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                <tr><td className="py-2 pr-4">Products bought before (re-buys)</td><td>{pct(panel.repeat_share)}</td><td>{pct(crowd.repeat_share)}</td><td>~60% of single choices</td></tr>
                <tr><td className="py-2 pr-4">Units bought on a deal</td><td>{pct(panel.units_on_deal)}</td><td>{pct(crowd.units_on_deal)}</td><td>~24-31%</td></tr>
                <tr><td className="py-2 pr-4">Reaction to own price (elasticity)</td><td>{student.price?.llm_elasticity ?? "-"}</td><td>{student.price?.engine_elasticity ?? "-"}</td><td>about -2.5</td></tr>
                <tr><td className="py-2 pr-4">Spend per trip</td><td>${fmt(panel.spend_per_trip, 2)}</td><td>${fmt(crowd.spend_per_trip, 2)}</td><td>-</td></tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs muted">
            Real values: Instacart reorder rate, Nielsen 2019-2020 and Bernstein 2022 promotion shares, grocery
            research on price elasticity (all PUBLIC_STATISTIC). The student's price reaction includes one
            correction marked ESTIMATED.
          </p>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <Feature title="Statistical engine" text="Traffic by hour, baskets, promotions, stock-outs and waste - calibrated on public data and checked by a scorecard." />
        <Feature title="Thinking NPCs" text="A memory card per shopper; routine buys on autopilot; real choices by a student model trained on six AI shoppers, balanced by model family." />
        <Feature title="Honest labels" text="Every value is labelled REAL_OBSERVED, PUBLIC_STATISTIC, ESTIMATED, SIMULATED or USER_PROVIDED. Stores are generic formats, never real retailers." />
      </section>

      {stores.length > 0 && (
        <section className="card">
          <h2 className="mb-3 text-xl font-semibold">The four store formats</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {stores.map(([id, s]) => (
              <div key={id} className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
                <div className="font-medium">{STORE_NAMES[id] || id}</div>
                <div className="text-sm muted">
                  {fmt(s.trips)} trips · ${fmt(s.avg_basket, 2)} per basket · {fmt(s.minutes_in_store, 1)} min in store
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="card">
      <div className="text-sm muted">{label}</div>
      <div className="mt-1 text-xl font-semibold sm:text-2xl">{value}</div>
    </div>
  );
}

function Feature({ title, text }) {
  return (
    <div className="card">
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-1 text-sm muted">{text}</p>
    </div>
  );
}
