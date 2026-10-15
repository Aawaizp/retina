// One simulated store day (written by the RETINA simulator: retina --view / the local app).
// This file only READS it: every route, choice, price, minute and total comes from the simulation.

export const REASON = {
  had_at_home: "still has some at home", too_expensive: "felt too expensive", aisle_removed: "no longer sold",
  other_store: "will buy it elsewhere", delay: "will buy it next trip", skip: "went without",
  out_of_stock: "shelf was empty", price_switch: "usual one got pricier", delisted_switch: "usual one no longer sold",
  over_budget: "this week's food money is spent", mind_choice: "the student chose it",
};
const WHY = { need: "ran out at home", repeat: "usual item", partner: "goes with the list", new: "new on the list", impulse: "unplanned" };
export const TRIP = { top_up: "quick top-up", routine: "routine trip", stock_up: "big stock-up" };

export const money = (v) => "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const clock = (m) => `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(Math.floor(m % 60)).padStart(2, "0")}`;
const short = (s, n = 34) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

export function prepare(D) {
  const pname = (i) => D.products[i] || "a product";
  const trips = new Map();
  for (const e of D.events) {
    if (!trips.has(e.shopper)) trips.set(e.shopper, { shopper: e.shopper, events: [] });
    const tr = trips.get(e.shopper);
    tr.events.push(e);
    if (e.event === "SHOPPER_ENTER") Object.assign(tr, { group: e.group, mission: e.mission, path: e.path, minutes: e.minutes, start: e.t });
    if (e.event === "CHECKOUT") Object.assign(tr, { lane: e.lane, served: e.t, ready: e.t - e.wait_min, total: e.total, items: e.items });
  }
  const T = [...trips.values()].filter((t) => t.path && t.path.length);
  T.forEach((t) => {
    t.end = t.path[t.path.length - 1][0];
    t.events.sort((a, b) => a.t - b.t);
  });
  const all = D.events.filter((e) => e.t !== undefined).sort((a, b) => a.t - b.t);
  const c = { trips: 0, units: 0, rev: 0, oos: 0, sub: 0, lines: 0, done: 0, mins: 0 };
  const cum = all.map((e) => {
    if (e.event === "SHOPPER_ENTER") c.trips++;
    if (e.event === "PURCHASE") { c.units += e.qty; c.lines++; }
    if (e.event === "CHECKOUT") c.rev += e.total;
    if (e.event === "STOCKOUT") c.oos++;
    if (e.event === "SUBSTITUTE" || e.event === "SKIP") c.sub++;
    if (e.event === "LEAVE") { c.done++; c.mins += trips.get(e.shopper)?.minutes || 0; }
    return { ...c };
  });
  const idxAt = (t) => {
    let lo = 0, hi = all.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (all[m].t <= t) lo = m + 1; else hi = m; }
    return lo;
  };
  const tMin = Math.floor(Math.min(...T.map((t) => t.start)) / 60) * 60;
  const tMax = Math.ceil(Math.max(...T.map((t) => t.end)) / 60) * 60;

  function describe(e) {
    const who = `Shopper ${e.shopper}`;
    switch (e.event) {
      case "SHOPPER_ENTER": return `${who} came in (${TRIP[e.mission] || "trip"})`;
      case "PURCHASE": return `${who} put ${short(pname(e.product))}${e.qty > 1 ? " ×" + e.qty : ""} in the basket – ${money(e.price)}${e.promo ? " (deal)" : ""}${WHY[e.route] ? ", " + WHY[e.route] : ""}`;
      case "STOCKOUT": return `${who} found ${short(pname(e.product))} out of stock`;
      case "SUBSTITUTE": return `${who} took ${short(pname(e.product), 26)} instead of ${short(pname(e.instead_of), 26)} (${REASON[e.reason] || e.reason})`;
      case "SKIP": return `${who} skipped ${e.product >= 0 ? short(pname(e.product)) : "an aisle"} (${REASON[e.reason] || e.reason})`;
      case "CHECKOUT": return `${who} paid ${money(e.total)} for ${e.items} item${e.items === 1 ? "" : "s"} at lane ${e.lane + 1}`;
      case "LEAVE": return `${who} left the store`;
      default: return e.event;
    }
  }
  return { D, T, all, cum, idxAt, tMin, tMax, describe, pname };
}

export function posAt(tr, t) {
  const p = tr.path;
  if (t <= p[0][0]) return { x: p[0][1], y: p[0][2] };
  for (let i = 1; i < p.length; i++)
    if (t <= p[i][0]) {
      const a = p[i - 1], b = p[i], f = (t - a[0]) / Math.max(b[0] - a[0], 1e-6);
      return { x: a[1] + (b[1] - a[1]) * f, y: a[2] + (b[2] - a[2]) * f };
    }
  const z = p[p.length - 1];
  return { x: z[1], y: z[2] };
}

// What the shopper is doing at time t: walking, just picked / found empty / swapped / skipped, waiting to pay.
export function stateOf(tr, t) {
  if (tr.ready !== undefined && t >= tr.ready && t < tr.served) return "queue";
  for (let i = tr.events.length - 1; i >= 0; i--) {
    const e = tr.events[i];
    if (e.t > t) continue;
    if (t - e.t < 1.5) {
      if (e.event === "PURCHASE") return "buy";
      if (e.event === "STOCKOUT") return "oos";
      if (e.event === "SUBSTITUTE") return "sub";
      if (e.event === "SKIP") return "skip";
    }
    break;
  }
  return "walk";
}

export const unitsBy = (tr, t) => tr.events.reduce((n, e) => n + (e.event === "PURCHASE" && e.t <= t ? e.qty : 0), 0);
