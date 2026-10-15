import { useEffect, useMemo, useState } from "react";
import Chart from "../components/Chart.jsx";
import { getJSON, query, STORE_NAMES } from "../lib/data.js";

const fmt = (n, d = 0) => (n == null ? "-" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d }));
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function Dashboards() {
  const [stores, setStores] = useState([]);
  const [store, setStore] = useState("retina_supercenter");
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("Starting the in-browser database...");

  useEffect(() => {
    getJSON("data/manifest.json")
      .then((m) => {
        const ids = Object.keys(m.stores);
        setStores(ids);
        if (!ids.includes(store)) setStore(ids[0]);
      })
      .catch((e) => setStatus(`Data not found (${e.message}). Build it with: retina web-data`));
  }, []);

  useEffect(() => {
    if (!stores.includes(store)) return;
    let alive = true;
    setData(null);
    setStatus("Querying Parquet files with DuckDB in your browser...");
    const f = (name) => `${store}/${name}.parquet`;
    Promise.all([
      query(`SELECT * FROM '${f("daily")}' ORDER BY date`, [f("daily")]),
      query(`SELECT weekday, hour, trips FROM '${f("hourly")}'`, [f("hourly")]),
      query(`SELECT department, SUM(revenue) AS revenue, SUM(items) AS items FROM '${f("departments")}'
             GROUP BY department ORDER BY revenue DESC`, [f("departments")]),
      query(`SELECT * FROM '${f("top_products")}' ORDER BY revenue DESC LIMIT 15`, [f("top_products")]),
      query(`SELECT measure, low, high, share FROM '${f("baskets")}'`, [f("baskets")]),
    ])
      .then(([daily, hourly, depts, top, baskets]) => alive && setData({ daily, hourly, depts, top, baskets }))
      .catch((e) => alive && setStatus(`Could not read the data: ${e.message}`));
    return () => {
      alive = false;
    };
  }, [store, stores]);

  const kpi = useMemo(() => {
    if (!data) return null;
    const sum = (k) => data.daily.reduce((a, r) => a + (r[k] || 0), 0);
    const trips = sum("trips");
    return {
      revenue: sum("revenue"), trips, items: sum("items"), profit: sum("gross_profit"),
      basket: sum("revenue") / Math.max(trips, 1), lost: sum("lost_sales"), waste: sum("waste_cost"),
      deal: data.daily.reduce((a, r) => a + (r.promo_share || 0), 0) / Math.max(data.daily.length, 1),
    };
  }, [data]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <h1 className="text-2xl font-bold">Dashboards</h1>
          <p className="text-sm muted">
            {STORE_NAMES[store] || store}, every shopper run by the student model (SIMULATED). Read straight from
            Parquet files by DuckDB-WASM - nothing is sent to a server.
          </p>
        </div>
        {stores.length > 1 && <select
          className="ml-auto rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          value={store}
          onChange={(e) => setStore(e.target.value)}
        >
          {stores.map((s) => <option key={s} value={s}>{STORE_NAMES[s] || s}</option>)}
        </select>}
      </div>

      {!data ? (
        <p className="card muted">{status}</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
            <Kpi label="Revenue" value={`$${fmt(kpi.revenue)}`} sub={`gross profit $${fmt(kpi.profit)}`} />
            <Kpi label="Trips" value={fmt(kpi.trips)} sub={`${fmt(data.daily.length)} days`} />
            <Kpi label="Average basket" value={`$${fmt(kpi.basket, 2)}`} sub={`${fmt(kpi.items)} items sold`} />
            <Kpi label="Lost to empty shelves" value={`$${fmt(kpi.lost)}`} sub={`waste $${fmt(kpi.waste)}`} />
          </div>

          <div className="card">
            <h2 className="mb-2 font-semibold">Revenue and trips per day</h2>
            <Chart height={320} option={dailyOption(data.daily)} />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h2 className="mb-2 font-semibold">When shoppers come (average trips)</h2>
              <Chart height={300} option={heatmapOption(data.hourly)} />
            </div>
            <div className="card">
              <h2 className="mb-2 font-semibold">Revenue by department</h2>
              <Chart height={300} option={deptOption(data.depts)} />
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h2 className="mb-2 font-semibold">Basket size</h2>
              <Chart height={260} option={histOption(data.baskets, "items", "items per trip")} />
            </div>
            <div className="card">
              <h2 className="mb-2 font-semibold">Spend per trip</h2>
              <Chart height={260} option={histOption(data.baskets, "spend", "$ per trip")} />
            </div>
          </div>

          <div className="card overflow-x-auto">
            <h2 className="mb-2 font-semibold">Best sellers</h2>
            <table className="w-full text-sm">
              <thead className="text-left muted">
                <tr><th className="py-2 pr-4">Product</th><th className="pr-4">Department</th><th className="pr-4 text-right">Units</th><th className="pr-4 text-right">Revenue</th><th className="text-right">On deal</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {data.top.map((r) => (
                  <tr key={r.product_id}>
                    <td className="py-2 pr-4">{r.product_name}{r.brand && !r.product_name.toLowerCase().includes(r.brand.toLowerCase()) && <span className="muted"> {r.brand}</span>}</td>
                    <td className="pr-4 muted">{r.department}</td>
                    <td className="pr-4 text-right">{fmt(r.units)}</td>
                    <td className="pr-4 text-right">${fmt(r.revenue)}</td>
                    <td className="text-right">{Math.round(r.deal_share * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function Kpi({ label, value, sub }) {
  return (
    <div className="card">
      <div className="text-sm muted">{label}</div>
      <div className="mt-1 text-xl font-semibold sm:text-2xl">{value}</div>
      <div className="text-xs muted">{sub}</div>
    </div>
  );
}

const grid = { left: 50, right: 50, top: 30, bottom: 40 };

function dailyOption(daily) {
  return {
    tooltip: { trigger: "axis" },
    legend: { data: ["Revenue", "Trips"] },
    grid,
    xAxis: { type: "category", data: daily.map((r) => r.date) },
    yAxis: [{ type: "value", name: "$" }, { type: "value", name: "trips" }],
    series: [
      { name: "Revenue", type: "line", smooth: true, showSymbol: false, areaStyle: { opacity: 0.15 }, data: daily.map((r) => Math.round(r.revenue)) },
      { name: "Trips", type: "line", yAxisIndex: 1, smooth: true, showSymbol: false, data: daily.map((r) => r.trips) },
    ],
  };
}

function heatmapOption(hourly) {
  const hours = [...new Set(hourly.map((r) => r.hour))].sort((a, b) => a - b);
  const max = Math.max(...hourly.map((r) => r.trips), 1);
  return {
    tooltip: { formatter: (p) => `${WEEKDAYS[p.value[1]]} ${hours[p.value[0]]}:00 - ${p.value[2]} trips` },
    grid: { left: 40, right: 10, top: 10, bottom: 60 },
    xAxis: { type: "category", data: hours.map((h) => `${h}h`) },
    yAxis: { type: "category", data: WEEKDAYS },
    visualMap: { min: 0, max, orient: "horizontal", left: "center", bottom: 0, calculable: true },
    series: [{ type: "heatmap", data: hourly.map((r) => [hours.indexOf(r.hour), r.weekday, r.trips]) }],
  };
}

function deptOption(depts) {
  const top = depts.slice(0, 12).reverse();
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 130, right: 20, top: 10, bottom: 30 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: top.map((r) => r.department) },
    series: [{ type: "bar", data: top.map((r) => Math.round(r.revenue)), itemStyle: { borderRadius: [0, 6, 6, 0] } }],
  };
}

function histOption(baskets, measure, unit) {
  const rows = baskets.filter((r) => r.measure === measure);
  const label = (r) => (r.high >= 1000 ? `${r.low}+` : `${r.low}-${r.high}`);
  return {
    tooltip: { trigger: "axis", valueFormatter: (v) => `${v}%` },
    grid: { left: 40, right: 10, top: 10, bottom: 40 },
    xAxis: { type: "category", data: rows.map(label), name: unit, nameLocation: "middle", nameGap: 28 },
    yAxis: { type: "value", axisLabel: { formatter: "{value}%" } },
    series: [{ type: "bar", data: rows.map((r) => Math.round(r.share * 1000) / 10), itemStyle: { borderRadius: [6, 6, 0, 0] } }],
  };
}
