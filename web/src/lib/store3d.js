// The 3D store only draws shopper events. The landing demo supplies an illustrative loop;
// the data studio supplies live decisions from the simulator.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clock, money, posAt, stateOf, unitsBy } from "./day.js";
import { humanAssets, makeHuman, poseHuman } from "./human.js";

export const STATE_COLORS = {
  walk: "#60a5fa", buy: "#22c55e", oos: "#ef4444", sub: "#f59e0b", skip: "#94a3b8", queue: "#a78bfa",
};
export const STATE_NAMES = {
  walk: "walking", buy: "picked a product", oos: "empty shelf", sub: "swapped product", skip: "skipped", queue: "waiting to pay",
};
export const SHAPES = { person: "Human shoppers wear different clothes; click one to follow" };
const MAX_DRAWN_SHOPPERS = 14;

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
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: true, transparent: true }));
  s.scale.set((size * w) / 64, size, 1);
  s.renderOrder = 10;
  return s;
}

function receiptSprite(tr) {
  const lines = tr.paidLines || [];
  const c = document.createElement("canvas");
  c.width = 520;
  c.height = 252;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgba(248,255,250,.97)";
  ctx.beginPath();
  ctx.roundRect(4, 4, 512, 244, 20);
  ctx.fill();
  ctx.strokeStyle = "#49b78b";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.fillStyle = "#185846";
  ctx.font = "bold 31px system-ui, sans-serif";
  ctx.fillText(`SHOPPER #${tr.shopper} PAID`, 25, 45);
  ctx.fillStyle = "#27443c";
  ctx.font = "25px system-ui, sans-serif";
  lines.slice(0, 3).forEach((line, i) => {
    const name = line.name.length > 27 ? `${line.name.slice(0, 26)}…` : line.name;
    ctx.fillText(`${line.qty} × ${name}`, 25, 83 + i * 33);
  });
  if (lines.length > 3) ctx.fillText(`+ ${lines.length - 3} more products`, 25, 180);
  ctx.fillStyle = "#0b6f50";
  ctx.font = "bold 36px system-ui, sans-serif";
  ctx.fillText(money(tr.total || 0), 25, 228);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sprite.scale.set(5.6, 2.7, 1);
  sprite.renderOrder = 30;
  return sprite;
}

export class Store3D {
  constructor(container, layout, { onPick, autoFocus = false } = {}) {
    this.container = container;
    this.L = layout;
    this.onPick = onPick;
    this.autoFocus = autoFocus;
    this.manualCamera = false;
    this.walkMode = false;
    this.walkKeys = new Set();
    this.walkYaw = 0;
    this.walkPitch = -0.06;
    this.dragLook = null;
    this.draggedLook = false;
    this.slowFrameSeconds = 0;
    this.npcs = new Map();
    this.receipts = new Map();
    this.seenPaid = new Set();
    this.focusId = null;
    this.focusHoldUntil = 0;
    this.lastFrameAt = null;
    this.checkoutFocusId = null;
    this.checkoutFocusUntil = 0;
    this.lastCheckoutFocusAt = -Infinity;
    const w = Math.max(1, container.clientWidth), h = Math.max(1, container.clientHeight);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setSize(w, h, false);                  // the canvas follows its box (CSS), never wider
    Object.assign(this.renderer.domElement.style, { width: "100%", height: "100%", display: "block" });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#10242b");
    this.scene.fog = new THREE.Fog("#10242b", 125, 240);
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.5, 400);
    this.overview = this.overviewFor(w, h);
    this.scene.fog.far = Math.max(240, this.overview.length() * 2.4);
    this.camera.position.copy(this.overview);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0);
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.minDistance = 12;
    this.controls.maxDistance = Math.max(140, this.overview.length() * 1.4);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = false;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
    this.controls.addEventListener("start", () => { this.manualCamera = true; });
    this.scene.add(new THREE.HemisphereLight("#ffffff", "#63757b", 2.0));
    const sun = new THREE.DirectionalLight("#fff7e9", 2.2);
    sun.position.set(-30, 60, 40);
    sun.castShadow = true;
    sun.shadow.bias = -0.0001;
    sun.shadow.normalBias = .025;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -50, right: 50, top: 35, bottom: -35, near: 1, far: 150 });
    this.scene.add(sun);
    this.humanAssets = humanAssets(STATE_COLORS);
    this.buildStore();
    this.raycaster = new THREE.Raycaster();
    this.onClick = (ev) => this.pick(ev);
    this.renderer.domElement.addEventListener("click", this.onClick);
    this.renderer.domElement.tabIndex = 0;
    this.renderer.domElement.style.touchAction = "none";
    this.onPointerDown = (ev) => {
      if (!this.walkMode || ev.button !== 0) return;
      this.renderer.domElement.focus();
      this.dragLook = { id: ev.pointerId, x: ev.clientX, y: ev.clientY };
      this.draggedLook = false;
      this.renderer.domElement.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    };
    this.onPointerMove = (ev) => {
      if (!this.walkMode || this.dragLook?.id !== ev.pointerId) return;
      const dx = ev.clientX - this.dragLook.x;
      const dy = ev.clientY - this.dragLook.y;
      this.dragLook.x = ev.clientX;
      this.dragLook.y = ev.clientY;
      if (Math.abs(dx) + Math.abs(dy) > 0) this.draggedLook = true;
      this.walkYaw -= dx * 0.004;
      this.walkPitch = THREE.MathUtils.clamp(this.walkPitch - dy * 0.004, -1.16, 1.16);
      this.setWalkRotation();
      ev.preventDefault();
    };
    this.onPointerUp = (ev) => {
      if (this.dragLook?.id !== ev.pointerId) return;
      this.dragLook = null;
      if (this.renderer.domElement.hasPointerCapture(ev.pointerId)) this.renderer.domElement.releasePointerCapture(ev.pointerId);
    };
    this.onKeyDown = (ev) => {
      if (!this.walkMode) return;
      if (ev.key === "Escape") { this.fitStore(); ev.preventDefault(); return; }
      if (ev.target !== this.renderer.domElement && /INPUT|SELECT|TEXTAREA/.test(ev.target?.tagName || "")) return;
      const key = ev.key.toLowerCase();
      if (["w", "a", "s", "d", "arrowup", "arrowleft", "arrowdown", "arrowright", "shift"].includes(key)) {
        this.walkKeys.add(key);
        ev.preventDefault();
      }
    };
    this.onKeyUp = (ev) => this.walkKeys.delete(ev.key.toLowerCase());
    this.onWindowBlur = () => this.walkKeys.clear();
    this.renderer.domElement.addEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.addEventListener("pointermove", this.onPointerMove);
    this.renderer.domElement.addEventListener("pointerup", this.onPointerUp);
    this.renderer.domElement.addEventListener("pointercancel", this.onPointerUp);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.onWindowBlur);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
  }

  X(x) { return x - this.L.width / 2; }
  Z(y) { return y - this.L.depth / 2; }

  overviewFor(width, height) {
    const aspect = Math.max(0.5, width / Math.max(1, height));
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const floorSpan = Math.max(this.L.width / aspect, this.L.depth * 0.82);
    const distance = floorSpan / (2 * Math.tan(halfFov)) * 1.28;
    return new THREE.Vector3(0, distance * 0.82, distance * 0.57);
  }

  fitStore() {
    this.walkMode = false;
    this.walkKeys.clear();
    if (this.dragLook && this.renderer.domElement.hasPointerCapture(this.dragLook.id))
      this.renderer.domElement.releasePointerCapture(this.dragLook.id);
    this.dragLook = null;
    this.controls.enabled = true;
    this.manualCamera = true;
    this.focusId = null;
    this.camera.position.copy(this.overview);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setWalkMode(enabled) {
    if (!enabled) { this.fitStore(); return; }
    if (this.walkMode) return;
    this.walkMode = true;
    this.manualCamera = true;
    this.focusId = null;
    this.walkKeys.clear();
    this.controls.enabled = false;
    const [entryX, entryY] = this.L.entrance;
    this.camera.position.set(this.X(entryX), 1.82, this.Z(entryY));
    this.camera.rotation.order = "YXZ";
    this.walkYaw = 0;
    this.walkPitch = -0.06;
    this.setWalkRotation();
    this.renderer.domElement.focus();
  }

  setWalkRotation() {
    this.camera.rotation.set(this.walkPitch, this.walkYaw, 0, "YXZ");
  }

  walkable(x, z) {
    const sx = x + this.L.width / 2, sy = z + this.L.depth / 2;
    const radius = 0.48;
    if (sx < radius || sy < radius || sx > this.L.width - radius || sy > this.L.depth - radius) return false;
    for (const [, x0, y0, x1, y1] of this.L.shelves) {
      if (sx > x0 - radius && sx < x1 + radius && sy > y0 - radius && sy < y1 + radius) return false;
    }
    for (const laneX of this.L.lanes_x) {
      if (sx > laneX - 1.05 && sx < laneX + 1.05 && sy > this.L.lane_y - 1.95 && sy < this.L.lane_y + 1.95) return false;
    }
    return true;
  }

  moveWalkCamera(frameSeconds) {
    if (!this.walkMode || !this.walkKeys.size || frameSeconds <= 0) return;
    const keys = this.walkKeys;
    let forward = Number(keys.has("w") || keys.has("arrowup")) - Number(keys.has("s") || keys.has("arrowdown"));
    let right = Number(keys.has("d") || keys.has("arrowright")) - Number(keys.has("a") || keys.has("arrowleft"));
    const length = Math.hypot(forward, right);
    if (!length) return;
    forward /= length;
    right /= length;
    const speed = keys.has("shift") ? 11 : 6.5;
    const step = speed * Math.min(frameSeconds, .06);
    const dx = (-Math.sin(this.walkYaw) * forward + Math.cos(this.walkYaw) * right) * step;
    const dz = (-Math.cos(this.walkYaw) * forward - Math.sin(this.walkYaw) * right) * step;
    const p = this.camera.position;
    if (this.walkable(p.x + dx, p.z)) p.x += dx;
    if (this.walkable(p.x, p.z + dz)) p.z += dz;
  }

  buildStore() {
    const L = this.L;
    const mat = (color, roughness = 0.9) => new THREE.MeshStandardMaterial({ color, roughness });
    const box = (w, h, d, x, y, z, color, cast = false) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), typeof color === "string" ? mat(color) : color);
      m.position.set(x, y, z);
      m.castShadow = cast;
      m.receiveShadow = true;
      this.scene.add(m);
      return m;
    };
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(L.width + 5, L.depth + 5),
      mat("#dce8e3"));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const walk = mat("#ecf4ee");
    for (const y of [L.y_back, L.y_front]) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(L.width, 3.5), walk);
      m.rotation.x = -Math.PI / 2;
      m.position.set(0, 0.02, this.Z(y));
      this.scene.add(m);
    }
    // Architectural frame: back wall, side walls and a low front fascia keep the shop legible in 3D.
    const wall = mat("#e7f0e9");
    box(L.width + 1, 4.3, .5, 0, 2.15, this.Z(-.65), wall);
    box(.5, 3.4, L.depth + 1, this.X(-.65), 1.7, 0, wall);
    box(.5, 3.4, L.depth + 1, this.X(L.width + .65), 1.7, 0, wall);
    box(L.width + 1, .42, .55, 0, .21, this.Z(L.depth + .55), mat("#9bb9aa"));
    box(L.width + 1, .28, .58, 0, 3.55, this.Z(-.65), mat("#166d5d"));
    const brand = labelSprite("RETINA MARKET  •  FRESH EVERY DAY", { size: 2.1, bg: "rgba(14,88,72,0.96)", color: "#f1fff4" });
    brand.position.set(0, 5.15, this.Z(.1));
    this.scene.add(brand);
    const clockCanvas = document.createElement("canvas");
    clockCanvas.width = 384;
    clockCanvas.height = 384;
    this.clockCtx = clockCanvas.getContext("2d");
    this.clockTexture = new THREE.CanvasTexture(clockCanvas);
    this.clockTexture.colorSpace = THREE.SRGBColorSpace;
    this.clockSign = new THREE.Mesh(new THREE.CircleGeometry(1.31, 64),
      new THREE.MeshBasicMaterial({ map: this.clockTexture, transparent: true, side: THREE.DoubleSide }));
    this.clockSign.position.set(this.X(L.width - 4), 2.78, this.Z(.02));
    this.clockSign.renderOrder = 12;
    this.scene.add(this.clockSign);
    this.clockMinute = null;

    // Shelves have a neutral frame, department-colour headers, and visible stock facings.
    const productColors = ["#eecba6", "#e4f0d4", "#d4e5ef", "#f5d7a8", "#e5cce7", "#f0d4d0"];
    for (const [a, x0, y0, x1, y1] of L.shelves) {
      const dep = L.aisles[a].department;
      const wall = x0 < 0.5 || x1 > L.width - 0.5 || y1 <= 1.0;
      const h = wall ? 2.45 : 1.95;
      const w = Math.max(x1 - x0, .35), d = Math.max(y1 - y0, .35);
      const x = this.X((x0 + x1) / 2), z = this.Z((y0 + y1) / 2);
      box(w, h, d, x, h / 2, z, "#fafbf5", true);
      box(w + .08, .18, d + .08, x, h + .09, z, L.colors[dep] || "#80b8a1");
      for (const sy of [.47, 1.03, 1.53]) {
        if (sy >= h - .24) continue;
        box(w + .06, .07, d + .1, x, sy, z, "#b6c9bb");
      }
      // On long shelf ends, stock rows make the floor read like a working market.
      if (d > 2.5) {
        const count = Math.min(10, Math.floor(d / 1.4));
        for (let i = 0; i < count; i++) {
          const itemZ = this.Z(y0 + (i + .5) * (y1 - y0) / count);
          const c = productColors[(a * 3 + i) % productColors.length];
          box(.35, .32, .48, x - w / 2 - .01, .71, itemZ, c);
          box(.35, .32, .48, x + w / 2 + .01, .71, itemZ, productColors[(a * 3 + i + 2) % productColors.length]);
        }
      }
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
      const s = labelSprite(b.d.toUpperCase(), { size: 1.0 });
      s.position.set(this.X((b.x0 + b.x1) / 2), 3.5, this.Z((b.y0 + b.y1) / 2));
      this.scene.add(s);
    }
    L.corridor_x.forEach((x, i) => {
      const s = labelSprite(`AISLE ${i + 1}`, { size: .74, bg: "rgba(13,96,78,0.95)", color: "#fff" });
      s.position.set(this.X(x), 3.15, this.Z(L.y_front - 1));
      this.scene.add(s);
    });
    // Checkout counters with conveyor tops and lane numbers.
    this.lanes = L.lanes_x.map((x) => {
      const m = box(1.1, .87, 2.9, this.X(x), .44, this.Z(L.lane_y), mat("#356857"), true);
      box(1.25, .08, 3.1, this.X(x), .9, this.Z(L.lane_y), "#303b3c");
      box(.44, .56, .5, this.X(x + .38), 1.2, this.Z(L.lane_y - 1), "#d2ddd3");
      const laneSign = labelSprite(`${L.lanes_x.indexOf(x) + 1}`, { size: .9, bg: "rgba(18,100,78,.95)", color: "#fff" });
      laneSign.position.set(this.X(x), 2.2, this.Z(L.lane_y - 1.4));
      this.scene.add(laneSign);
      m.castShadow = true;
      return m;
    });
    const cs = labelSprite("CHECKOUT", { size: 1.35, bg: "rgba(13,96,78,.92)", color: "#fff" });
    cs.position.set(this.X(L.lanes_x[0] - 3), 2.3, this.Z(L.lane_y));
    this.scene.add(cs);
    for (const [pos, text, color] of [[L.entrance, "ENTRANCE", "#22c55e"], [L.exit, "EXIT", "#ef4444"]]) {
      box(3.2, .06, 1.6, this.X(pos[0]), .09, this.Z(pos[1]), mat(text === "ENTRANCE" ? "#b0e2bf" : "#d2e0d6"));
      const s = labelSprite(text, { size: 1.1, bg: text === "ENTRANCE" ? "rgba(15,110,78,.94)" : "rgba(63,103,96,.94)", color: "#fff" });
      s.position.set(this.X(pos[0]), 2.1, this.Z(pos[1]));
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
    n = makeHuman(tr.shopper, tr, this.humanAssets);
    n.label = labelSprite(`#${tr.shopper}`, { size: .72, bg: "rgba(6,55,48,.92)", color: "#ffffff" });
    n.label.position.y = 3.2;
    n.body.add(n.label);
    this.scene.add(n.body);
    this.npcs.set(tr.shopper, n);
    return n;
  }

  drawClock(now) {
    const minute = Math.floor(now);
    if (minute === this.clockMinute) return;
    this.clockMinute = minute;
    const ctx = this.clockCtx;
    const cx = 192, cy = 192;
    ctx.clearRect(0, 0, 384, 384);
    ctx.beginPath();
    ctx.arc(cx, cy, 185, 0, Math.PI * 2);
    ctx.fillStyle = "#e8f7ea";
    ctx.fill();
    ctx.lineWidth = 12;
    ctx.strokeStyle = "#176d55";
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 169, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#69b995";
    ctx.stroke();
    for (let i = 0; i < 60; i++) {
      const a = i * Math.PI / 30 - Math.PI / 2;
      const major = i % 5 === 0;
      const r0 = major ? 139 : 151;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * 159, cy + Math.sin(a) * 159);
      ctx.lineWidth = major ? 6 : 2;
      ctx.strokeStyle = major ? "#0e5948" : "#7aa996";
      ctx.stroke();
    }
    const hand = (angle, length, thickness, color) => {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.sin(angle) * length, cy - Math.cos(angle) * length);
      ctx.lineWidth = thickness;
      ctx.lineCap = "round";
      ctx.strokeStyle = color;
      ctx.stroke();
    };
    hand((minute / 60 % 12) * Math.PI / 6, 88, 14, "#164e43");
    hand((minute % 60) * Math.PI / 30, 125, 7, "#168d69");
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, Math.PI * 2);
    ctx.fillStyle = "#164e43";
    ctx.fill();
    ctx.fillStyle = "#164e43";
    ctx.font = "700 19px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("STORE TIME", cx, 95);
    ctx.font = "700 28px ui-monospace, monospace";
    ctx.fillText(clock(now), cx, 290);
    this.clockTexture.needsUpdate = true;
  }

  // Draw the store at minute `now` of the day. Returns {inStore, queue}.
  update(day, now, followed, rate = 5) {
    const L = this.L;
    this.drawClock(now);
    const wallNow = performance.now();
    const frameSeconds = this.lastFrameAt == null ? 0 : Math.min(.1, Math.max(0, (wallNow - this.lastFrameAt) / 1000));
    this.lastFrameAt = wallNow;
    if (!document.hidden && this.renderer.getPixelRatio() > 1 && frameSeconds > 0) {
      this.slowFrameSeconds = Math.max(0, this.slowFrameSeconds +
        (frameSeconds > .038 ? frameSeconds : -frameSeconds * .5));
      if (this.slowFrameSeconds > 2.5) this.renderer.setPixelRatio(1);
    }
    this.moveWalkCamera(frameSeconds);
    const active = [];
    const waiting = {};
    for (const tr of day.T) {
      if (now >= tr.start && now <= tr.end) active.push(tr);
      if (tr.ready !== undefined && now >= tr.ready && now < tr.served)
        (waiting[tr.lane] = waiting[tr.lane] || []).push(tr);
      if (tr.paidAt == null || tr.paidAt > now || this.seenPaid.has(tr.shopper)) continue;
      this.seenPaid.add(tr.shopper);
      if (now - tr.paidAt <= Math.max(3, rate * 5)) this.receipts.set(tr.shopper, { tr, until: wallNow + 6000 });
    }
    for (const [id, receipt] of this.receipts) if (receipt.until <= wallNow) this.receipts.delete(id);
    const open = (L.lanes_open || [])[Math.floor(now / 60) % 24] || L.lanes_x.length;
    if (open !== this.lastOpenLanes) {
      this.lanes.forEach((m, i) => m.material.color.set(i < open ? "#94a3b8" : "#334155"));
      this.lastOpenLanes = open;
    }
    const rank = new Map();
    Object.values(waiting).forEach((list) => list.sort((a, b) => a.ready - b.ready).forEach((t, k) => rank.set(t, k)));
    const shown = active.slice(0, MAX_DRAWN_SHOPPERS);
    if (followed && active.includes(followed) && !shown.includes(followed)) shown[shown.length - 1] = followed;
    const paidShowing = [...this.receipts.values()].sort((a, b) => b.tr.paidAt - a.tr.paidAt).slice(0, 2);
    if (!followed && paidShowing.length && wallNow - this.lastCheckoutFocusAt > 8000) {
      this.checkoutFocusId = paidShowing[0].tr.shopper;
      this.checkoutFocusUntil = wallNow + 3000;
      this.lastCheckoutFocusAt = wallNow;
    }
    for (const { tr } of paidShowing) if (!shown.includes(tr)) shown.push(tr);
    const live = new Set();
    for (const tr of shown) {
      live.add(tr.shopper);
      const n = this.npc(tr);
      const paid = this.receipts.has(tr.shopper) && paidShowing.some((r) => r.tr === tr);
      const state = paid ? "queue" : stateOf(tr, now);
      let p = posAt(tr, now);
      if (paid) p = { x: L.lanes_x[tr.lane] - 1.35, y: L.lane_y - 2.1 };
      const targetX = this.X(p.x), targetZ = this.Z(p.y);
      const oldX = n.body.position.x, oldZ = n.body.position.z;
      if (!n.positioned) {
        n.body.position.set(targetX, 0, targetZ);
        n.positioned = true;
      } else {
        const dx = targetX - oldX, dz = targetZ - oldZ;
        const remaining = Math.hypot(dx, dz);
        // Follow the event clock closely, while giving newly streamed route legs
        // a few real frames to cross the floor instead of teleporting.
        const alpha = 1 - Math.exp(-frameSeconds * (20 + Math.max(0, rate) * 10));
        const maxStep = Math.max(.35, frameSeconds * (32 + Math.max(0, rate) * 22));
        const step = Math.min(remaining, remaining * alpha, maxStep);
        if (remaining > 1e-6) {
          n.body.position.x += dx / remaining * step;
          n.body.position.z += dz / remaining * step;
        }
      }
      const before = posAt(tr, Math.max(tr.start, now - 0.08));
      const after = posAt(tr, now + 0.08);
      const dx = after.x - before.x, dz = after.y - before.y;
      const moved = Math.hypot(n.body.position.x - oldX, n.body.position.z - oldZ);
      const moving = moved > 0.012 && !paid;
      if (moving) {
        const heading = Math.atan2(moved > 0.015 ? n.body.position.x - oldX : dx,
          moved > 0.015 ? n.body.position.z - oldZ : dz);
        const turn = Math.atan2(Math.sin(heading - n.body.rotation.y), Math.cos(heading - n.body.rotation.y));
        n.body.rotation.y += turn * (1 - Math.exp(-frameSeconds * (12 + Math.max(0, rate) * 2)));
      }
      const units = tr.served === undefined || now < tr.served ? unitsBy(tr, now) : 0;
      n.gaitPhase = (n.gaitPhase || tr.shopper * 1.7) + (moving ? Math.min(moved, 2.5) * 2.8 : 0);
      poseHuman(n, moving, state, units, wallNow / 1000, tr.shopper, this.humanAssets, n.gaitPhase);
      if (paid && !n.receipt) {
        n.receipt = receiptSprite(tr);
        n.receipt.position.y = 4.1;
        n.body.add(n.receipt);
      }
      if (n.receipt) n.receipt.visible = paid;
      n.label.visible = !paid && this.autoFocus && (tr === followed || shown.indexOf(tr) < 3);
      n.body.scale.setScalar((.91 + (tr.shopper % 5) * .045) * (tr === followed ? 1.18 : 1));
    }
    for (const [id, n] of this.npcs) if (!live.has(id)) { this.removeNpc(n); this.npcs.delete(id); }
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
    if (this.autoFocus && !this.manualCamera && shown.length) {
      const isWalking = (tr) => {
        const a = posAt(tr, Math.max(tr.start, now - .15));
        const b = posAt(tr, now + .15);
        return Math.hypot(b.x - a.x, b.y - a.y) > .08;
      };
      const viewable = shown.filter((tr) => active.includes(tr) && !this.receipts.has(tr.shopper));
      const current = viewable.find((tr) => tr.shopper === this.focusId);
      const checkoutFocus = wallNow < this.checkoutFocusUntil
        ? paidShowing.find(({ tr }) => tr.shopper === this.checkoutFocusId)?.tr : null;
      const keepCurrent = current && wallNow < this.focusHoldUntil;
      const tr = followed && active.includes(followed) ? followed
        : checkoutFocus || (keepCurrent ? current : viewable.find(isWalking) || current || viewable[0] || shown[0]);
      if (this.focusId !== tr.shopper) this.focusHoldUntil = wallNow + 4200;
      this.focusId = tr.shopper;
      const center = this.npcs.get(tr.shopper).body.position.clone().add(new THREE.Vector3(0, 1.1, 0));
      if (checkoutFocus === tr) center.x += 4;
      const span = Math.max(L.width, L.depth);
      const desired = center.clone().add(new THREE.Vector3(0, Math.max(25, span * .47), Math.max(18, span * .35)));
      const alpha = 1 - Math.exp(-frameSeconds * 3.2);
      this.controls.target.lerp(center, alpha);
      this.camera.position.lerp(desired, alpha);
    } else if (this.autoFocus && !this.manualCamera && !shown.length) {
      const alpha = 1 - Math.exp(-frameSeconds * 1.8);
      this.controls.target.lerp(new THREE.Vector3(0, 0, 0), alpha);
      this.camera.position.lerp(this.overview, alpha);
    }
    if (!this.walkMode) {
      this.controls.update();
      // Overview panning stays over the floor plan rather than drifting outside it.
      const x = THREE.MathUtils.clamp(this.controls.target.x, -L.width / 2, L.width / 2);
      const z = THREE.MathUtils.clamp(this.controls.target.z, -L.depth / 2, L.depth / 2);
      this.camera.position.x += x - this.controls.target.x;
      this.camera.position.z += z - this.controls.target.z;
      this.controls.target.x = x;
      this.controls.target.z = z;
    }
    this.renderer.render(this.scene, this.camera);
    return { inStore: active.length, drawn: shown.length, queue: rank.size };
  }

  clearNpcs() {
    for (const n of this.npcs.values()) this.removeNpc(n);
    this.npcs.clear();
    this.receipts.clear();
    this.seenPaid.clear();
    this.focusId = null;
    this.focusHoldUntil = 0;
    this.lastFrameAt = null;
    this.checkoutFocusId = null;
    this.checkoutFocusUntil = 0;
    this.lastCheckoutFocusAt = -Infinity;
  }

  removeNpc(n) {
    this.scene.remove(n.body);
    n.label.material.map?.dispose();
    n.label.material.dispose();
    n.receipt?.material.map?.dispose();
    n.receipt?.material.dispose();
  }

  pick(ev) {
    if (this.draggedLook) { this.draggedLook = false; return; }
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(v, this.camera);
    const hit = this.raycaster.intersectObjects([...this.npcs.values()].map((n) => n.body), true)[0];
    let object = hit?.object;
    while (object && !object.userData.trip) object = object.parent;
    if (object?.userData.trip && !this.walkMode) this.manualCamera = false;
    this.onPick?.(object?.userData.trip || null);
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth), h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.overview = this.overviewFor(w, h);
    this.scene.fog.far = Math.max(240, this.overview.length() * 2.4);
    this.controls.maxDistance = Math.max(140, this.overview.length() * 1.4);
    this.renderer.setSize(w, h, false);
  }

  dispose() {
    this.observer.disconnect();
    this.clearNpcs();
    this.renderer.domElement.removeEventListener("click", this.onClick);
    this.renderer.domElement.removeEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.removeEventListener("pointermove", this.onPointerMove);
    this.renderer.domElement.removeEventListener("pointerup", this.onPointerUp);
    this.renderer.domElement.removeEventListener("pointercancel", this.onPointerUp);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.onWindowBlur);
    this.controls.dispose();
    const geometries = new Set(), materials = new Set(), textures = new Set();
    this.scene.traverse((object) => {
      if (object.geometry) geometries.add(object.geometry);
      const list = Array.isArray(object.material) ? object.material : object.material ? [object.material] : [];
      for (const m of list) {
        materials.add(m);
        if (m.map) textures.add(m.map);
      }
    });
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => m.dispose());
    textures.forEach((t) => t.dispose());
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.humanAssets.dispose();
  }
}
