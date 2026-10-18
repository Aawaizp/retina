// A store day that grows WHILE it happens: built from the simulator's live stream (retina --live).
// Same shape as prepare() in day.js, so the 3D store and the side panels draw it the same way.
// Every move, choice and sales row comes from the simulation, in the order it happened.
import { describer } from "./day.js";

const EMPTY = { trips: 0, units: 0, rev: 0, oos: 0, sub: 0, lines: 0, done: 0, mins: 0 };

export class LiveDay {
  constructor(layout, date) {
    this.D = { layout, date };
    this.T = [];
    this.trips = new Map();
    this.all = [];
    this.cum = [];
    this.c = { ...EMPTY };
    this.names = new Map();
    this.tMin = null;
    this.tMax = 0;
    this.live = true;
    this.pname = (i) => this.names.get(i) || "a product";
    this.describe = describer(this.pname);
  }

  idxAt(t) {
    let lo = 0, hi = this.all.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.all[m].t <= t) lo = m + 1; else hi = m; }
    return lo;
  }

  push(ev) {
    const c = this.c;
    if (ev.event === "SHOPPER_ENTER") c.trips++;
    if (ev.event === "PURCHASE") c.units += ev.qty;
    if (ev.event === "STOCKOUT") c.oos++;
    if (ev.event === "SUBSTITUTE" || ev.event === "SKIP") c.sub++;
    if (ev.event === "CHECKOUT") { c.rev += ev.total; c.lines += ev.lines.length; }
    if (ev.event === "LEAVE") { c.done++; c.mins += this.trips.get(ev.shopper)?.minutes || 0; }
    this.all.push(ev);
    this.cum.push({ ...c });
  }

  // one event of the stream: {k, t, ...}
  add(e) {
    const t = e.t, L = this.D.layout;
    this.tMax = Math.max(this.tMax, t);
    if (e.k === "enter") {
      const tr = { shopper: e.shopper, mission: e.mission, group: e.group, path: e.path, start: t, end: Infinity, events: [] };
      this.trips.set(e.shopper, tr);
      this.T.push(tr);
      if (this.tMin === null) this.tMin = Math.floor(t / 60) * 60;
      this.push({ event: "SHOPPER_ENTER", shopper: e.shopper, mission: e.mission, t });
      return;
    }
    const tr = this.trips.get(e.shopper);
    if (!tr) return;                                     // came in before we started listening
    if (e.k === "shelf") {
      for (const x of e.events) {
        if (x.product >= 0 && x.name) this.names.set(x.product, x.name);
        if (x.instead_of !== undefined && x.instead_of_name) this.names.set(x.instead_of, x.instead_of_name);
        const ev = { ...x, t, shopper: e.shopper };
        tr.events.push(ev);
        this.push(ev);
      }
    } else if (e.k === "checkout") {                      // the lane, the queue, then out
      const x = L.lanes_x[e.lane];
      Object.assign(tr, { lane: e.lane, ready: t, served: e.start });
      tr.path = [...tr.path, [e.start, x, L.lane_y], [e.served, x, L.lane_y + 1.5], [e.served + 1, L.exit[0], L.exit[1]]];
      tr.end = e.served + 1;
    } else if (e.k === "pay") {                           // the sales rows are written now
      for (const x of e.lines) this.names.set(x.product, x.name);
      const items = e.lines.reduce((n, x) => n + x.qty, 0);
      Object.assign(tr, { total: e.total, items, minutes: Math.round((t + 1 - tr.start) * 10) / 10 });
      const ev = { event: "CHECKOUT", shopper: e.shopper, customer: e.customer, t, total: e.total, items, lane: e.lane, lines: e.lines };
      tr.events.push(ev);
      this.push(ev);
    } else if (e.k === "leave") {
      this.push({ event: "LEAVE", shopper: e.shopper, t });
    }
  }

  // the newest sales rows (from the receipts paid) and events, up to event index i
  recent(i, now, nRows = 12, nFeed = 8) {
    const rows = [], feed = [];
    for (let j = i - 1; j >= 0 && (rows.length < nRows || feed.length < nFeed); j--) {
      const e = this.all[j];
      if (e.event === "CHECKOUT")
        for (const x of e.lines) if (rows.length < nRows) rows.push({ t: e.t, shopper: e.shopper, product: x.product, qty: x.qty, price: x.price, promo: x.promo });
      if (!["LEAVE"].includes(e.event) && feed.length < nFeed) feed.push(e);
    }
    return { rows, feed };
  }
}
