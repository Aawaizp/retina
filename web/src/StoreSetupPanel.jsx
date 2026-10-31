import { useState } from "react";
import { url } from "./lib/json.js";

function todayLocal() {
  const day = new Date();
  return `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, "0")}-${String(day.getDate()).padStart(2, "0")}`;
}

function hourlyCounts(text) {
  if (!text.trim()) return null;
  const counts = {};
  for (const line of text.trim().split(/\r?\n/)) {
    if (/^hour\s*,/i.test(line)) continue;
    const parts = line.split(",").map((x) => x.trim());
    if (parts.length !== 2 || !/^\d+$/.test(parts[0]) || !/^\d+$/.test(parts[1])) {
      throw new Error("Hourly traffic needs two columns: hour,shoppers (one hour per line).");
    }
    if (counts[parts[0]] !== undefined) throw new Error(`Hour ${parts[0]} appears twice in hourly traffic.`);
    counts[parts[0]] = Number(parts[1]);
  }
  return counts;
}

export default function StoreSetupPanel({ onCreated, disabled }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({ name: "", city: "", state: "", store_type: "supermarket",
    open_hour: 8, close_hour: 22, shoppers_per_week: 1000, checkout_lanes: 4,
    start_date: todayLocal(), hourly: "" });
  const set = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

  async function create(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    if (file?.size > 20_000_000) return setError("The product CSV must be 20 MB or smaller.");
    setBusy(true);
    try {
      let products_csv = "";
      if (file) {
        const bytes = await file.arrayBuffer();
        products_csv = new TextDecoder("utf-8").decode(bytes);
        if (products_csv.includes("\ufffd")) products_csv = new TextDecoder("windows-1252").decode(bytes);
      }
      const body = { ...form, open_hour: Number(form.open_hour), close_hour: Number(form.close_hour),
        shoppers_per_week: Number(form.shoppers_per_week), checkout_lanes: Number(form.checkout_lanes),
        hourly_counts: hourlyCounts(form.hourly), products_csv };
      delete body.hourly;
      const response = await fetch(url("custom-stores"), { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) });
      if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        const detail = Array.isArray(problem.detail) ? problem.detail.map((item) => item.msg).join("; ") : problem.detail;
        throw new Error(typeof detail === "string" ? detail : `Store setup failed (${response.status}).`);
      }
      const created = await response.json();
      setResult(created);
      onCreated(created);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold">Use your own store</h3>
          <p className="mt-1 text-sm muted">Enter a few facts you know. RETINA labels the rest as estimates.</p>
        </div>
        <button type="button" onClick={() => setOpen(!open)} className="btn border border-slate-300">
          {open ? "Close" : "Set up store"}
        </button>
      </div>
      {open && <form onSubmit={create} className="space-y-4 text-sm">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="sm:col-span-2">Store name<input required minLength={2} maxLength={100} className="input mt-1" value={form.name}
            onChange={(e) => set("name", e.target.value)} placeholder="Example: Main Street Market" /></label>
          <label>Store type<select className="input mt-1" value={form.store_type} onChange={(e) => set("store_type", e.target.value)}>
            <option value="supermarket">Supermarket</option><option value="supercenter">Supercenter</option>
            <option value="neighborhood">Neighborhood market</option><option value="convenience">Convenience store</option>
          </select></label>
          <label>Opens (hour, 0–23)<input required type="number" min={0} max={23} className="input mt-1" value={form.open_hour}
            onChange={(e) => set("open_hour", e.target.value)} /></label>
          <label>Closes (hour, up to 24)<input required type="number" min={1} max={24} className="input mt-1" value={form.close_hour}
            onChange={(e) => set("close_hour", e.target.value)} /></label>
          <label>Estimated visitors per week<input required type="number" min={7} max={200000} className="input mt-1" value={form.shoppers_per_week}
            onChange={(e) => set("shoppers_per_week", e.target.value)} /></label>
          <label>First simulated date<input required type="date" className="input mt-1" value={form.start_date} onChange={(e) => set("start_date", e.target.value)} />
            <span className="muted">To compare with real data, use its first date.</span></label>
        </div>
        <p className="-mt-2 text-xs text-slate-500">If you do not know weekly traffic, keep the starting estimate and refine it later.</p>
        <div className="space-y-2">
          <label className="block">Products CSV (optional; for the store’s own range and prices)
            <input type="file" accept=".csv,text/csv" className="input mt-1" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </label>
          <a className="text-brand-600 underline" href={url("store_products_template.csv")} download>Download a product template</a>
          <p className="muted">Upload your products for store-specific names and prices. Without a file, RETINA uses a generic US catalog.</p>
        </div>
        <details className="rounded-xl border border-slate-200 bg-slate-50 p-3">
          <summary className="cursor-pointer font-semibold text-slate-700">More store details (optional)</summary>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <label>City<input className="input mt-1" value={form.city} onChange={(e) => set("city", e.target.value)} /></label>
            <label>State<input className="input mt-1" value={form.state} onChange={(e) => set("state", e.target.value)} placeholder="e.g. TX" /></label>
            <label>Checkout lanes<input required type="number" min={1} max={100} className="input mt-1" value={form.checkout_lanes} onChange={(e) => set("checkout_lanes", e.target.value)} /></label>
          </div>
          <label className="mt-3 block">Typical weekday visitors by hour
            <textarea className="input mt-1 h-24 font-mono text-xs" value={form.hourly} onChange={(e) => set("hourly", e.target.value)} placeholder={"hour,shoppers\n8,20\n9,35\n10,50"} />
            <span className="muted">Only add this if you know the hourly pattern. It replaces the weekly estimate.</span>
          </label>
        </details>
        <button disabled={busy || disabled} className="btn bg-brand-500 px-5 py-2 text-white disabled:opacity-50">
          {busy ? "Checking products..." : "Save store"}
        </button>
        {error && <pre className="whitespace-pre-wrap text-red-600">{error}</pre>}
        {result && <p className="text-green-600">Store ready: {result.name}. {result.products ? `${result.products} products checked.` : "Using a generic catalog."} It is selected above and ready for a run.
          {result.warnings?.length > 0 && <span> {result.warnings.length} warning(s); see the product check report in its data folder.</span>}
        </p>}
      </form>}
    </div>
  );
}
