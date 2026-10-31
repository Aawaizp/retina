import { useEffect, useState } from "react";
import { url } from "./lib/json.js";

const formats = {
  daily_store_kpis: {
    label: "Daily store totals",
    columns: "date,transactions,net_sales",
    help: "Use the same calendar dates as the simulated run.",
  },
  daily_product_sales: {
    label: "Daily product sales",
    columns: "date,product_id,units_sold,net_sales",
    help: "Product IDs must match products.csv in the dataset ZIP.",
  },
};

function percent(value) {
  return value == null ? "—" : `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
}

function amount(value, unit) {
  if (value == null) return "—";
  const number = Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
  return unit === "USD" ? `$${number}` : `${number} ${unit}`;
}

export default function ValidationPanel({ run }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [kind, setKind] = useState("daily_store_kpis");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setReport(null);
    setError("");
    fetch(url(`runs/${encodeURIComponent(run)}/validation`)).then((response) => {
      if (!response.ok) throw new Error("Validation report is not available for this run.");
      return response.json();
    }).then((value) => { if (active) { setReport(value); setLoading(false); } })
      .catch((problem) => { if (active) { setError(problem.message); setLoading(false); } });
    return () => { active = false; };
  }, [run]);

  async function compare(event) {
    event.preventDefault();
    const form = event.currentTarget;
    setError("");
    if (!file) return setError("Choose a CSV file first.");
    if (file.size > 10_000_000) return setError("The comparison CSV must be 10 MB or smaller.");
    setBusy(true);
    try {
      const csv_text = await file.text();
      if (csv_text.includes("\ufffd")) throw new Error("Save the comparison CSV as UTF-8, then upload it again.");
      const response = await fetch(url(`runs/${encodeURIComponent(run)}/validation`), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, csv_text }),
      });
      if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        throw new Error(typeof problem.detail === "string" ? problem.detail : `Comparison failed (${response.status}).`);
      }
      setReport(await response.json());
      setFile(null);
      form.reset();
    } catch (problem) {
      setError(problem.message);
    } finally {
      setBusy(false);
    }
  }

  const integrity = report?.integrity;
  const realism = report?.realism;
  const failed = integrity?.checks?.filter((check) => check.status === "fail") || [];

  return <section className="card space-y-5 p-5 sm:p-6" aria-label="Dataset validation">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><p className="eyebrow">DATA QUALITY</p><h2 className="mt-1 text-xl font-bold text-slate-950">Validation report</h2>
        <p className="mt-1 text-sm text-slate-600">Check receipt and stock accounting. Add real store totals to measure how close the simulated data is.</p></div>
      {report && <a className="btn border border-slate-200 text-brand-700" href={url(`runs/${encodeURIComponent(run)}/business/VALIDATION.md`)}>Read full report ↓</a>}
    </div>

    {loading && <p className="text-sm text-slate-500">Checking this dataset…</p>}
    {report && <>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className={`rounded-xl border p-4 ${integrity?.status === "pass" ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50"}`}>
          <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Data checks</p>
          <p className="mt-1 text-lg font-bold text-slate-900">{integrity?.status === "pass" ? "Passed" : "Needs attention"}</p>
          <p className="mt-1 text-xs text-slate-600">{failed.length ? `${failed.length} check(s) found an issue.` : "Receipts, products, daily totals and inventory were checked."}</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Real-store comparison</p>
          <p className="mt-1 text-lg font-bold text-slate-900">{realism?.status === "evaluated" ? "Compared" : realism?.status === "no_overlap" ? "No matching dates or products" : "Not yet verified"}</p>
          <p className="mt-1 text-xs text-slate-600">{realism?.note}</p>
        </div>
      </div>
      {failed.length > 0 && <ul className="list-disc space-y-1 pl-5 text-sm text-rose-700">{failed.map((check) => <li key={check.name}>{check.name}: {check.failures} issue(s)</li>)}</ul>}
      {realism?.comparisons?.length > 0 && <div className="overflow-x-auto"><table className="w-full min-w-[650px] text-sm">
        <thead className="border-b border-slate-200 text-left text-xs font-bold uppercase tracking-wide text-slate-500"><tr><th className="py-2 pr-3">Measure</th><th className="px-3 py-2 text-right">Matched rows</th><th className="px-3 py-2 text-right">Coverage</th><th className="px-3 py-2 text-right" title="Mean absolute error per matched row">MAE</th><th className="px-3 py-2 text-right">Error (WAPE)</th><th className="py-2 pl-3 text-right">Bias</th></tr></thead>
        <tbody>{realism.comparisons.map((item) => <tr key={item.name} className="border-b border-slate-100"><td className="py-2 pr-3 font-medium">{item.name}</td><td className="px-3 py-2 text-right tabular-nums">{item.matched_rows} / {item.reference_rows}</td><td className="px-3 py-2 text-right tabular-nums">{percent(item.coverage_pct)}</td><td className="px-3 py-2 text-right tabular-nums">{amount(item.mae, item.unit)}</td><td className="px-3 py-2 text-right tabular-nums">{percent(item.wape_pct)}</td><td className="py-2 pl-3 text-right tabular-nums">{percent(item.bias_pct)}</td></tr>)}</tbody>
      </table><p className="mt-2 text-xs text-slate-500">Errors apply only to matching rows. Check coverage before interpreting them.</p></div>}
      <p className="text-xs text-slate-500">Each raw receipt belongs to one simulated NPC. Sample weight: {report.sampling?.sample_weight ?? "unknown"}. Expanded totals are estimates.</p>
    </>}

    <form onSubmit={compare} className="border-t border-slate-100 pt-5">
      <h3 className="font-semibold text-slate-900">Compare with your store data</h3>
      <p className="mt-1 text-xs leading-5 text-slate-500">Upload aggregate daily totals only. Your file stays on this computer outside the downloadable synthetic-data ZIP.</p>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="text-sm font-medium text-slate-700">Data type<select className="input mt-1" value={kind} onChange={(event) => setKind(event.target.value)}>
          {Object.entries(formats).map(([value, option]) => <option key={value} value={value}>{option.label}</option>)}
        </select></label>
        <label className="min-w-[200px] flex-1 text-sm font-medium text-slate-700">CSV file<input className="input mt-1" type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] || null)} /></label>
        <button className="btn bg-brand-500 px-5 py-2.5 text-white disabled:opacity-50" disabled={busy || !file}>{busy ? "Comparing…" : "Compare data"}</button>
      </div>
      <p className="mt-2 text-xs text-slate-500">Columns: <code>{formats[kind].columns}</code> (last column optional). {formats[kind].help}</p>
      {error && <p className="mt-2 text-sm text-rose-700" role="alert">{error}</p>}
    </form>
  </section>;
}
