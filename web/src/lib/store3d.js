// The 3D RETINA Supercenter (three.js). It only DRAWS a simulated day: the floor plan, routes, choices
// and receipts all come from the simulator. Shapes: cone = quick top-up, sphere = routine trip,
// cube = big stock-up. Colour = what the shopper is doing right now.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { posAt, stateOf, unitsBy } from "./day.js";

export const STATE_COLORS = {
  walk: "#60a5fa", buy: "#22c55e", oos: "#ef4444", sub: "#f59e0b", skip: "#94a3b8", queue: "#a78bfa",
};
export const STATE_NAMES = {
  walk: "walking", buy: "picked a product", oos: "empty shelf", sub: "swapped product", skip: "skipped", queue: "waiting to pay",
};
export const SHAPES = { top_up: "▲ cone = quick top-up", routine: "● sphere = routine trip", stock_up: "■ cube = big stock-up" };

function labelSprite(text, { size = 1.6, color = "#e2e8f0", bg = "rgba(15,23,42,0.75)" } = {}) {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  const font = "600 44px Inter, system-ui, sans-serif";
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 36;
  c.width = w;
  c.height = 64;
  ctx.font = font;
  ctx.fillStyle = bg;
  ctx.beginPath();
  ctx.roundRect(0, 0, w, 64, 14);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, 18, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set((size * w) / 64, size, 1);
  s.renderOrder = 10;
  return s;
}

export class Store3D {
  constructor(container, layout, { onPick } = {}) {
    this.container = container;
    this.L = layout;
    this.onPick = onPick;
    this.npcs = new Map();
    const w = container.clientWidth, h = container.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);                  // the canvas follows its box (CSS), never wider
    Object.assign(this.renderer.domElement.style, { width: "100%", height: "100%", display: "block" });
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#0b1120");
    this.scene.fog = new THREE.Fog("#0b1120", 90, 170);
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.5, 400);
    this.camera.position.set(0, 38, 46);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 2);
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 140;
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight("#cbd5e1", "#1e293b", 1.4));
    const sun = new THREE.DirectionalLight("#ffffff", 1.6);
    sun.position.set(-30, 60, 40);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -50, right: 50, top: 35, bottom: -35, near: 1, far: 150 });
    this.scene.add(sun);
    this.geo = {
      top_up: new THREE.ConeGeometry(0.75, 2.0, 24),        // shoppers are drawn larger than life, to be seen
      routine: new THREE.SphereGeometry(0.8, 28, 18),
      stock_up: new THREE.BoxGeometry(1.4, 1.4, 1.4),
      other: new THREE.OctahedronGeometry(0.9),
      basket: new THREE.BoxGeometry(0.7, 1, 0.7),
    };
    this.mat = Object.fromEntries(Object.entries(STATE_COLORS).map(([k, c]) =>
      [k, new THREE.MeshStandardMaterial({ color: c, roughness: 0.45, metalness: 0.1, emissive: c, emissiveIntensity: 0.15 })]));
    this.basketMat = new THREE.MeshStandardMaterial({ color: "#16a34a", roughness: 0.6 });
    this.buildStore();
    this.raycaster = new THREE.Raycaster();
    this.onClick = (ev) => this.pick(ev);
    this.renderer.domElement.addEventListener("click", this.onClick);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
  }

  X(x) { return x - this.L.width / 2; }
  Z(y) { return y - this.L.depth / 2; }

  buildStore() {
    const L = this.L;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(L.width + 4, L.depth + 4),
      new THREE.MeshStandardMaterial({ color: "#1e293b", roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(Math.max(L.width, L.depth) + 4, 42, "#334155", "#273449");
    grid.position.y = 0.01;
    this.scene.add(grid);
    const walk = new THREE.MeshStandardMaterial({ color: "#26344d", roughness: 1 });
    for (const y of [L.y_back, L.y_front]) {                       // cross aisles
      const m = new THREE.Mesh(new THREE.PlaneGeometry(L.width, 3), walk);
      m.rotation.x = -Math.PI / 2;
      m.position.set(0, 0.02, this.Z(y));
      this.scene.add(m);
    }
    // shelves: one block per shelf piece, coloured by department; wall shelves are taller
    const mats = {};
    for (const [a, x0, y0, x1, y1] of L.shelves) {
      const dep = L.aisles[a].department;
      const wall = x0 < 0.5 || x1 > L.width - 0.5 || y1 <= 1.0;
      const h = wall ? 2.4 : 1.8;
      mats[dep] = mats[dep] || new THREE.MeshStandardMaterial({ color: L.colors[dep] || "#94a3b8", roughness: 0.7 });
      const m = new THREE.Mesh(new THREE.BoxGeometry(Math.max(x1 - x0, 0.3), h, Math.max(y1 - y0, 0.3)), mats[dep]);
      m.position.set(this.X((x0 + x1) / 2), h / 2, this.Z((y0 + y1) / 2));
      m.castShadow = m.receiveShadow = true;
      this.scene.add(m);
    }
    // department names over the walls, walkway numbers at the front
    const boxes = {};
    for (const [a, x0, y0, x1, y1] of L.shelves) {
      const wall = y1 <= 1.0 ? "B" : x0 < 0.5 ? "L" : x1 > L.width - 0.5 ? "R" : null;
      if (!wall) continue;
      const k = wall + L.aisles[a].department;
      const b = boxes[k] || (boxes[k] = { d: L.aisles[a].department, x0, y0, x1, y1 });
      Object.assign(b, { x0: Math.min(b.x0, x0), y0: Math.min(b.y0, y0), x1: Math.max(b.x1, x1), y1: Math.max(b.y1, y1) });
    }
    for (const b of Object.values(boxes)) {
      const s = labelSprite(b.d.toUpperCase(), { size: 1.5 });
      s.position.set(this.X((b.x0 + b.x1) / 2), 3.6, this.Z((b.y0 + b.y1) / 2));
      this.scene.add(s);
    }
    const inCorridor = L.corridor_x.map(() => new Set());
    L.aisles.forEach((a) => {
      const c = L.corridor_x.findIndex((x) => Math.abs(x - a.x) < 0.01);
      if (c >= 0) inCorridor[c].add(a.department);
    });
    L.corridor_x.forEach((x, i) => {
      const s = labelSprite(`${i + 1}`, { size: 1.1, bg: "rgba(37,99,235,0.9)", color: "#fff" });
      s.position.set(this.X(x), 0.9, this.Z(L.y_bottom + 1.0));
      this.scene.add(s);
    });
    // checkout lanes, entrance, exit
    this.lanes = L.lanes_x.map((x) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.0, 2.4), new THREE.MeshStandardMaterial({ color: "#475569" }));
      m.position.set(this.X(x), 0.5, this.Z(L.lane_y));
      m.castShadow = true;
      this.scene.add(m);
      return m;
    });
    const cs = labelSprite("CHECKOUT", { size: 1.3 });
    cs.position.set(this.X(L.lanes_x[0] - 3), 1.6, this.Z(L.lane_y));
    this.scene.add(cs);
    for (const [pos, text, color] of [[L.entrance, "ENTRANCE", "#22c55e"], [L.exit, "EXIT", "#ef4444"]]) {
      const pad = new THREE.Mesh(new THREE.BoxGeometry(3, 0.08, 1.4), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4 }));
      pad.position.set(this.X(pos[0]), 0.04, this.Z(pos[1]));
      this.scene.add(pad);
      const s = labelSprite(text, { size: 1.2, color });
      s.position.set(this.X(pos[0]), 1.4, this.Z(pos[1]));
      this.scene.add(s);
    }
    this.route = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: "#facc15" }));
    this.route.visible = false;
    this.scene.add(this.route);
    this.ring = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.55, 32), new THREE.MeshBasicMaterial({ color: "#facc15", side: THREE.DoubleSide }));
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.visible = false;
    this.scene.add(this.ring);
  }

  npc(tr) {
    let n = this.npcs.get(tr.shopper);
    if (n) return n;
    const body = new THREE.Mesh(this.geo[tr.mission] || this.geo.other, this.mat.walk);
    body.castShadow = true;
    const basket = new THREE.Mesh(this.geo.basket, this.basketMat);
    basket.visible = false;
    body.add(basket);
    body.userData.trip = tr;
    this.scene.add(body);
    n = { body, basket };
    this.npcs.set(tr.shopper, n);
    return n;
  }

  // Draw the store at minute `now` of the day. Returns {inStore, queue}.
  update(day, now, followed) {
    const L = this.L;
    const open = (L.lanes_open || [])[Math.floor(now / 60) % 24] || L.lanes_x.length;
    this.lanes.forEach((m, i) => m.material.color.set(i < open ? "#94a3b8" : "#334155"));
    const waiting = {};
    for (const t of day.T) if (t.ready !== undefined && now >= t.ready && now < t.served) (waiting[t.lane] = waiting[t.lane] || []).push(t);
    const rank = new Map();
    Object.values(waiting).forEach((list) => list.sort((a, b) => a.ready - b.ready).forEach((t, k) => rank.set(t, k)));
    const live = new Set();
    let inStore = 0;
    for (const tr of day.T) {
      if (now < tr.start || now > tr.end) continue;
      inStore++;
      live.add(tr.shopper);
      const n = this.npc(tr);
      const state = stateOf(tr, now);
      let p = posAt(tr, now);
      if (rank.has(tr)) p = { x: L.lanes_x[tr.lane], y: L.lane_y - 2.0 - rank.get(tr) * 1.7 };
      const bob = state === "walk" ? Math.abs(Math.sin(now * 9 + tr.shopper)) * 0.25 : 0;
      const half = tr.mission === "top_up" ? 1.0 : tr.mission === "stock_up" ? 0.7 : 0.8;
      n.body.position.set(this.X(p.x), half + bob, this.Z(p.y));
      n.body.material = this.mat[state];
      n.body.rotation.y = tr.mission === "stock_up" ? now * 0.5 + tr.shopper : 0;
      const units = now < tr.served ? unitsBy(tr, now) : 0;
      n.basket.visible = units > 0;
      if (units > 0) {
        const hgt = Math.min(2.5, 0.2 + units * 0.1);
        n.basket.scale.set(1, hgt, 1);
        n.basket.position.set(0, half + 0.05 + hgt / 2, 0);
      }
      n.body.scale.setScalar(tr === followed ? 1.35 : 1);
    }
    for (const [id, n] of this.npcs) if (!live.has(id)) { this.scene.remove(n.body); this.npcs.delete(id); }
    if (followed && live.has(followed.shopper)) {
      const pts = followed.path.filter((p) => p[0] <= now).map((p) => new THREE.Vector3(this.X(p[1]), 0.08, this.Z(p[2])));
      const q = posAt(followed, now);
      pts.push(new THREE.Vector3(this.X(q.x), 0.08, this.Z(q.y)));
      this.route.geometry.dispose();
      this.route.geometry = new THREE.BufferGeometry().setFromPoints(pts);
      this.route.visible = pts.length > 1;
      const b = this.npcs.get(followed.shopper).body.position;
      this.ring.position.set(b.x, 0.05, b.z);
      this.ring.visible = true;
    } else {
      this.route.visible = this.ring.visible = false;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    return { inStore, queue: rank.size };
  }

  clearNpcs() {
    for (const n of this.npcs.values()) this.scene.remove(n.body);
    this.npcs.clear();
  }

  pick(ev) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(v, this.camera);
    const hit = this.raycaster.intersectObjects([...this.npcs.values()].map((n) => n.body), false)[0];
    this.onPick?.(hit ? hit.object.userData.trip : null);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  dispose() {
    this.observer.disconnect();
    this.renderer.domElement.removeEventListener("click", this.onClick);
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
