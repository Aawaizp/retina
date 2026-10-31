import { useEffect, useMemo, useRef, useState } from "react";
import { clock, money, TRIP } from "./lib/day.js";
import { LiveDay } from "./lib/live.js";
import { Store3D } from "./lib/store3d.js";

const START = 9 * 60;
const LOOP_MINUTES = 78;

// A small illustrated store for the landing page. These shoppers never call the API or write data.
const LAYOUT = {
  width: 52, depth: 36, y_back: 5, y_front: 27, y_top: 2, y_bottom: 29,
  corridor_x: [10, 22, 34, 46], entrance: [6, 33], exit: [47, 33],
  lanes_x: [24, 30, 36], lane_y: 32, lanes_open: Array(24).fill(3),
  aisles: [
    { name: "Fresh food", department: "Fresh food", x: 10, y: 16 },
    { name: "Pantry", department: "Pantry", x: 22, y: 16 },
    { name: "Home care", department: "Home care", x: 34, y: 16 },
    { name: "Snacks", department: "Snacks", x: 46, y: 16 },
  ],
  colors: { "Fresh food": "#71c6a1", Pantry: "#e7c979", "Home care": "#91b7ed", Snacks: "#e8a883" },
  shelves: [
    [0, 7, 8, 8.5, 25], [0, 11.5, 8, 13, 25],
    [1, 19, 8, 20.5, 25], [1, 23.5, 8, 25, 25],
    [2, 31, 8, 32.5, 25], [2, 35.5, 8, 37, 25],
    [3, 43, 8, 44.5, 25], [3, 47.5, 8, 49, 25],
    [0, 3, 0.3, 14, 0.9], [1, 15, 0.3, 26, 0.9],
    [2, 27, 0.3, 38, 0.9], [3, 39, 0.3, 50, 0.9],
  ],
};

const SHOPPERS = [
  { at: 0, mission: "routine", first: [0, 1, "Bananas", 2.18], second: [1, 2, "Breakfast cereal", 4.79] },
  { at: 4, mission: "top_up", first: [0, 3, "Whole milk", 3.89], second: [3, 4, "Potato chips", 3.49] },
  { at: 10, mission: "stock_up", first: [2, 5, "Laundry detergent", 11.99], second: [1, 6, "Pasta", 2.29] },
  { at: 18, mission: "routine", first: [1, 7, "Coffee", 7.49], second: [0, 8, "Fresh apples", 4.25] },
  { at: 26, mission: "top_up", first: [3, 9, "Chocolate", 2.39], second: [0, 10, "Yogurt", 1.89] },
  { at: 36, mission: "routine", first: [2, 11, "Hand soap", 4.19], second: [3, 12, "Crackers", 3.59] },
];

function buildDemoDay() {
  const day = new LiveDay(LAYOUT, "Illustrative demo", START);
  const events = [];
  let order = 0;
  const add = (k, t, data) => events.push({ k, t, ...data, order: order++ });

  SHOPPERS.forEach((spec, i) => {
    const shopper = 101 + i;
    const t = START + spec.at;
    const [firstAisle, firstId, firstName, firstPrice] = spec.first;
    const [secondAisle, secondId, secondName, secondPrice] = spec.second;
    const x1 = LAYOUT.aisles[firstAisle].x;
    const x2 = LAYOUT.aisles[secondAisle].x;
    const lane = i % LAYOUT.lanes_x.length;
    const laneX = LAYOUT.lanes_x[lane];
    const lines = [spec.first, spec.second].map(([, product, name, price]) =>
      ({ product, name, qty: 1, price, promo: false, mind: "demo" }));
    const shelf = (minute, aisle, product, name, price) => add("shelf", minute, {
      shopper, aisle, aisle_name: LAYOUT.aisles[aisle].name,
      decisions: [{ action: "pick", product, wanted: product, name, mind: "demo", why: "illustrative choice" }],
      events: [{ event: "PURCHASE", product, name, qty: 1, price, promo: false, route: "new" }],
    });

    add("enter", t, { shopper, mission: spec.mission, group: i % 3, path: [[t, ...LAYOUT.entrance]] });
    add("move", t, { shopper, to: firstAisle, to_name: LAYOUT.aisles[firstAisle].name,
      brain: "demo", path: [[t + 1, 6, 27], [t + 3, x1, 27], [t + 7, x1, 16]] });
    shelf(t + 7, firstAisle, firstId, firstName, firstPrice);
    add("move", t + 8, { shopper, to: secondAisle, to_name: LAYOUT.aisles[secondAisle].name,
      brain: "demo", path: [[t + 10, x1, 27], [t + 12, x2, 27], [t + 16, x2, 16]] });
    shelf(t + 16, secondAisle, secondId, secondName, secondPrice);
    add("move", t + 17, { shopper, to: "checkout", to_name: "Checkout", brain: "demo",
      path: [[t + 19, x2, 27], [t + 22, laneX, 30]] });
    add("checkout", t + 22, { shopper, lane, start: t + 22, served: t + 25 });
    add("pay", t + 25, { shopper, lane, lines, total: firstPrice + secondPrice });
    add("leave", t + 26, { shopper });
  });

  events.sort((a, b) => a.t - b.t || a.order - b.order).forEach(day.add.bind(day));
  return day;
}

export default function DemoStore() {
  const box = useRef(null);
  const selectedRef = useRef(null);
  const day = useMemo(buildDemoDay, []);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(false);
  const [view, setView] = useState({ now: START, inStore: 0, paid: 0, feed: [], rows: [] });

  useEffect(() => {
    let store;
    try {
      store = new Store3D(box.current, LAYOUT, { autoFocus: true, onPick: (shopper) => {
        selectedRef.current = shopper;
        setSelected(shopper);
      } });
    } catch {
      setError(true);
      return;
    }
    let frame, previousCycle = 0, lastPanel = 0;
    const began = performance.now() - 12 * 550; // open the scene with several shoppers already in view
    const tick = (time) => {
      const elapsed = Math.max(0, (time - began) / 550); // clear but lively movement
      const cycle = Math.floor(elapsed / LOOP_MINUTES);
      if (cycle !== previousCycle) {
        store.clearNpcs();
        selectedRef.current = null;
        setSelected(null);
        previousCycle = cycle;
      }
      const now = START + elapsed % LOOP_MINUTES; // one store minute per real second: easy to watch
      const position = store.update(day, now, selectedRef.current, 1000 / 550);
      if (time - lastPanel > 180) {
        const i = day.idxAt(now);
        const paid = i ? day.cum[i - 1].done : 0;
        const { rows, feed: recentFeed } = day.recent(i, now, 5, 18);
        const feed = recentFeed
          .filter((event) => ["SHOPPER_ENTER", "PURCHASE", "CHECKOUT"].includes(event.event)).slice(0, 4);
        setView({ now, inStore: position.inStore, paid, feed, rows });
        lastPanel = time;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(frame); store.dispose(); };
  }, [day]);

  return <main className="demo-shell">
    <div ref={box} className="demo-canvas" aria-label="Animated 3D store with shoppers walking to shelves and checkout" />
    {error && <div className="absolute inset-0 z-10 grid place-items-center bg-slate-950 p-8 text-center text-white">The store view needs WebGL in your browser.</div>}
    <div className="demo-shade" aria-hidden="true" />
    <header className="demo-topbar">
      <a href="#demo" className="flex items-center gap-3 text-white" aria-label="RETINA home">
        <span className="brand-mark">R<span className="brand-dot" /></span>
        <span className="text-lg font-black tracking-[0.16em]">RETINA</span>
      </a>
      <span className="hidden text-xs font-medium tracking-wide text-white/70 sm:block">THE LIVING RETAIL DATA STUDIO</span>
      <a href="#studio" className="demo-nav-link">Generate data <span aria-hidden="true">↗</span></a>
    </header>

    <section className="demo-hero" aria-labelledby="demo-title">
      <div className="demo-eyebrow"><span className="live-pulse" /> INSIDE THE STORE · VISUAL DEMO</div>
      <h1 id="demo-title">A store you can<br /><em>actually watch.</em></h1>
      <p>Shoppers walk the aisles, choose products, and pay at checkout. Every completed trip can become a row in your synthetic retail dataset.</p>
      <div className="mt-7 flex flex-wrap gap-3">
        <a href="#studio" className="demo-primary">Want to generate data? <span>→</span></a>
        <span className="demo-secondary">Explore the store below ↓</span>
      </div>
      <div className="demo-steps" aria-label="How RETINA works">
        <span><b>01</b> Arrive</span><i /><span><b>02</b> Shop</span><i /><span><b>03</b> Checkout</span><i /><span><b>04</b> Dataset</span>
      </div>
    </section>

    <aside className="demo-storeboard" aria-label="Store activity">
      <div className="demo-storeboard-head"><span className="live-pulse" /> STORE OPEN <strong>{clock(view.now)}</strong></div>
      <div className="demo-storeboard-stats"><div><strong>{view.inStore}</strong><span>shopping now</span></div><div><strong>{view.paid}</strong><span>checked out</span></div></div>
      <div className="demo-ledger-title">PAID AT CHECKOUT <span>LIVE PREVIEW</span></div>
      <div className="demo-ledger-wrap"><table className="demo-ledger"><thead><tr><th>Time</th><th>Shopper</th><th>Product</th><th>Sale</th></tr></thead><tbody>
        {view.rows.length ? view.rows.map((row, i) => <tr key={`${row.shopper}-${row.t}-${i}`}><td>{clock(row.t)}</td><td>#{row.shopper}</td><td title={day.pname(row.product)}>{day.pname(row.product)}</td><td>{money(row.price)}</td></tr>)
          : <tr><td colSpan="4" className="demo-ledger-empty">Follow the shoppers. Paid items appear here.</td></tr>}
      </tbody></table></div>
      {selected && <div className="demo-selected"><b>Following shopper #{selected.shopper}</b><span>{TRIP[selected.mission]} · {selected.total ? `paid ${money(selected.total)}` : "choosing products"}</span></div>}
      <p className="demo-disclaimer">Illustrative store loop · No data saved on this page</p>
    </aside>

    <div className="demo-bottom-note"><span className="live-pulse" /> Drag to explore · Scroll to zoom · Select a shopper</div>
  </main>;
}
