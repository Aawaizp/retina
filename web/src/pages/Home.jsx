import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getJSON } from "../lib/json.js";

const fmt = (n, d = 0) => (n == null ? "-" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d }));
const show = (v, format) => (v == null ? "-" : typeof v === "string" ? v : format === "pct" ? `${(v * 100).toFixed(1)}%` : fmt(v, 2));

export default function Home() {
  const [manifest, setManifest] = useState(null);
  const [card, setCard] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getJSON("data/manifest.json").then(setManifest).catch((e) => setError(e.message));
    getJSON("data/scorecard.json").then(setCard).catch(() => {});
  }, []);

  const store = manifest?.stores?.retina_supercenter;

  return (
    <div className="space-y-10">
      <section className="space-y-4">
        <span className="label">SIMULATED data</span>
        <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">Retail sales data, made by shoppers that think.</h1>
        <p className="max-w-3xl text-lg muted">
          RETINA Supercenter is a virtual store with real FMCG products - food, household, personal care and baby. Every
          shopper is run by a small AI model: it writes the shopping list from the shopper's own situation, then picks
          the product at the shelf. The store records it like a real retailer: transactions, baskets, daily KPIs,
          inventory and promotions. Synthetic, but checked against public numbers.
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
        <Stat label="Simulated days" value={fmt(store?.days)} />
        <Stat label="Shopping trips" value={fmt(store?.trips)} />
        <Stat label="Net sales" value={store ? `$${fmt(store.revenue)}` : "-"} />
        <Stat label="Choices by the student model" value={card ? `${Math.round(card.decided_by_student * 100)}%` : "-"} />
      </section>

      {card && (
        <section className="card space-y-3">
          <h2 className="text-xl font-semibold">Does it look like real shopping?</h2>
          <p className="muted text-sm">
            {fmt(card.baskets)} baskets and {fmt(card.lines)} receipt lines from {card.days} days of {card.store},
            next to public numbers about real shoppers. Shopping lists: {Math.round(card.lists_by_list_student * 100)}%
            written by the list student.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left muted">
                <tr><th className="py-2 pr-4">Measure</th><th className="pr-4 text-right">RETINA</th><th className="pr-4 text-right">Real</th><th>Source</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {card.rows.map((r) => (
                  <tr key={r.measure}>
                    <td className="py-2 pr-4">{r.measure}</td>
                    <td className="pr-4 text-right font-medium">{show(r.ours, r.format)}</td>
                    <td className="pr-4 text-right">{show(r.real, r.format)}</td>
                    <td className="muted">{r.source} <span className="label">{r.label}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="list-disc pl-5 text-xs muted">
            {card.notes.map((n) => <li key={n}>{n}</li>)}
            <li>The student models carry corrections marked ESTIMATED (price reaction, re-buy habit, list size); see the model cards.</li>
          </ul>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <Feature title="Student-run shoppers" text="A list student plans each shopping list (what and how many) from the shopper's own state; a shelf student picks the product. Both learned from several AI shoppers and run in ONNX - no LLM at run time." />
        <Feature title="Company-style tables" text="sales_transactions, baskets, daily_store_kpis, daily_product_sales, inventory_daily, products, customers and promotions - with a data dictionary." />
        <Feature title="Honest labels" text="Every value is labelled REAL_OBSERVED, PUBLIC_STATISTIC, ESTIMATED, SIMULATED or USER_PROVIDED. The store is generic, never a real retailer." />
      </section>
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
