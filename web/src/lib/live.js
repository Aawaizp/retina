// A store day that grows WHILE it happens: built from the simulator's live stream (retina --live).
// Same shape as prepare() in day.js, so the 3D store and the side panels draw it the same way.
// Every move, choice and sales row comes from the simulation, in the order it happened.
import { describer } from "./day.js";

const EMPTY = { trips: 0, units: 0, rev: 0, oos: 0, sub: 0, lines: 0, done: 0, mins: 0, student: 0, studentRoutes: 0, studentBrowses: 0, studentCheckouts: 0 };

export class LiveDay {
  constructor(layout, date, startMinute = 0) {
    this.D = { layout, date };
    this.T = [];
    this.trips = new Map();
    this.all = [];
    this.cum = [];
    this.c = { ...EMPTY };
    this.names = new Map();
    this.tMin = startMinute;
    this.tMax = startMinute;
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
    if (ev.event === "MIND_DECISION" && ev.mind === "student") c.student++;
    if ((ev.event === "MIND_MOVE" || ev.event === "MIND_BROWSE") && ev.brain === "student") c.studentRoutes++;
    if (ev.event === "MIND_BROWSE") c.studentBrowses++;
    if (ev.event === "MIND_CHECKOUT") c.studentCheckouts++;
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
      const tr = { shopper: e.shopper, mission: e.mission, group: e.group,
                   listBrain: e.list_brain, requirements: e.requirements,
                   path: e.path, start: t, end: Infinity, events: [] };
      this.trips.set(e.shopper, tr);
      this.T.push(tr);
      this.push({ event: "SHOPPER_ENTER", shopper: e.shopper, mission: e.mission, t });
      return;
    }
    const tr = this.trips.get(e.shopper);
    if (!tr) return;                                     // came in before we started listening
    if (e.k === "move") {
      const route = (e.path || []).map((point) => [...point]);
      // Older streams ended a checkout route inside the register row. Draw its
      // approach in the cross-aisle so the shopper can walk into the chosen lane.
      if (e.to === "checkout" && route.length) route[route.length - 1][2] = L.y_front;
      for (const point of route) {
        const previous = tr.path.at(-1);
        if (previous && previous[0] === point[0] && previous[1] === point[1] && previous[2] === point[2]) continue;
        tr.path.push(point);
      }
      if (e.decision?.reason === "basket_complete") {
        const ev = { event: "MIND_CHECKOUT", shopper: e.shopper, t, ...e.decision, lines: e.lines };
        tr.events.push(ev);
        this.push(ev);
      } else if (e.browse) {
        this.names.set(e.browse.product, e.browse.name);
        const ev = { event: "MIND_BROWSE", shopper: e.shopper, t, to_name: e.to_name,
                     product: e.browse.product, brain: e.brain, probability: e.browse.probability };
        tr.events.push(ev);
        this.push(ev);
      } else if (e.to !== "checkout") {
        const ev = { event: "MIND_MOVE", shopper: e.shopper, t, to: e.to, to_name: e.to_name, brain: e.brain };
        tr.events.push(ev);
        this.push(ev);
      }
    } else if (e.k === "shelf") {
      for (const x of e.decisions || []) {
        if (x.product >= 0 && x.name) this.names.set(x.product, x.name);
        if (x.product < 0 && x.wanted >= 0 && x.name) this.names.set(x.wanted, x.name);
        const ev = { event: "MIND_DECISION", shopper: e.shopper, t, ...x };
        tr.events.push(ev);
        this.push(ev);
      }
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
      const standingX = x - 1.35;
      const reach = Math.min(e.served - .15, t + Math.min(1.2, Math.max(.45, (e.start - t) * .5)));
      const pass = t + (reach - t) * .65;
      tr.path.push([pass, standingX, L.y_front], [reach, standingX, L.lane_y - 2.1]);
      if (e.start > reach) tr.path.push([e.start, standingX, L.lane_y - 2.1]);
      tr.path.push([e.served, standingX, L.lane_y - 2.1],
                   [e.served + .65, L.width - 2, L.lane_y - 2.1],
                   [e.served + 1, L.exit[0], L.exit[1]]);
      tr.end = e.served + 1;
    } else if (e.k === "pay") {                           // the sales rows are written now
      for (const x of e.lines) this.names.set(x.product, x.name);
      const items = e.lines.reduce((n, x) => n + x.qty, 0);
      Object.assign(tr, { total: e.total, items, minutes: Math.round((t + 1 - tr.start) * 10) / 10,
                          paidAt: t, paidLines: e.lines,
                          essentialItems: e.essential_items, essentialItemsBought: e.essential_items_bought });
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
        for (const x of e.lines) if (rows.length < nRows) rows.push({ t: e.t, shopper: e.shopper, product: x.product, qty: x.qty, price: x.price, promo: x.promo, mind: x.mind });
      if (!["LEAVE"].includes(e.event) && feed.length < nFeed) feed.push(e);
    }
    return { rows, feed };
  }
}
