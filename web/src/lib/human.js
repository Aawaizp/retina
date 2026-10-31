import * as THREE from "three";

const SHIRTS = ["#287f71", "#cb7658", "#567ab0", "#a06d94", "#d2a04c", "#6d985a", "#b76665", "#476c93"];
const SKIN = ["#f1c6a8", "#d99c77", "#ab704f", "#71452f", "#e4b48d"];
const HAIR = ["#25262d", "#49352d", "#8b5f38", "#c49b62", "#493b52"];

const material = (color, roughness = 0.85) => new THREE.MeshStandardMaterial({ color, roughness });

export function humanAssets(statusColors) {
  const geometry = {
    torso: new THREE.CylinderGeometry(0.34, 0.43, 0.88, 10),
    head: new THREE.SphereGeometry(0.28, 14, 10),
    hair: new THREE.SphereGeometry(0.289, 14, 8, 0, Math.PI * 2, 0, Math.PI * 0.52),
    neck: new THREE.CylinderGeometry(0.11, 0.12, 0.19, 8),
    arm: new THREE.CylinderGeometry(0.125, 0.105, 0.62, 8),
    leg: new THREE.CylinderGeometry(0.15, 0.12, 0.78, 8),
    hand: new THREE.SphereGeometry(0.115, 9, 7),
    eye: new THREE.SphereGeometry(0.025, 7, 5),
    shoe: new THREE.BoxGeometry(0.25, 0.16, 0.4),
    bag: new THREE.BoxGeometry(0.54, 0.45, 0.35),
    item: new THREE.BoxGeometry(0.2, 0.28, 0.17),
    collar: new THREE.TorusGeometry(0.24, .065, 6, 12),
    hairBack: new THREE.SphereGeometry(.31, 12, 9),
    ring: new THREE.TorusGeometry(0.57, 0.055, 6, 20),
  };
  const materials = {
    shirts: SHIRTS.map((c) => material(c)), skins: SKIN.map((c) => material(c)),
    hair: HAIR.map((c) => material(c)), pants: [material("#263b51"), material("#41475c"), material("#3d4d48")],
    shoes: material("#273c3c"), eyes: material("#27242b"), bag: material("#a58355"), item: material("#86bba8"),
    collars: [material("#f1eee4"), material("#294957"), material("#faf2db")],
    status: Object.fromEntries(Object.entries(statusColors).map(([k, c]) =>
      [k, new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.82 })])),
  };
  return { geometry, materials, dispose() {
    Object.values(geometry).forEach((g) => g.dispose());
    Object.values(materials).flatMap((value) => Array.isArray(value) ? value : typeof value === "object" && !value.dispose ? Object.values(value) : [value])
      .forEach((m) => m.dispose());
  } };
}

function part(parent, geometry, material, x, y, z) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}

export function makeHuman(shopper, trip, assets) {
  const { geometry: g, materials: m } = assets;
  const root = new THREE.Group();
  root.userData.trip = trip;
  const shirt = m.shirts[shopper % m.shirts.length];
  const skin = m.skins[Math.floor(shopper / 2) % m.skins.length];
  const hairColor = m.hair[Math.floor(shopper / 3) % m.hair.length];
  const pants = m.pants[shopper % m.pants.length];

  part(root, g.torso, shirt, 0, 1.5, 0);
  part(root, g.neck, skin, 0, 2.04, 0);
  part(root, g.head, skin, 0, 2.25, 0);
  part(root, g.hair, hairColor, 0, 2.29, 0);
  if (shopper % 3 === 0) {
    const back = part(root, g.hairBack, hairColor, 0, 2.15, -.12);
    back.scale.set(1, 1.13, .72);
  }
  const collar = part(root, g.collar, m.collars[shopper % m.collars.length], 0, 1.94, 0);
  collar.rotation.x = Math.PI / 2;
  part(root, g.eye, m.eyes, -0.095, 2.28, 0.255);
  part(root, g.eye, m.eyes, 0.095, 2.28, 0.255);

  const arms = [], legs = [];
  for (const side of [-1, 1]) {
    const arm = new THREE.Group();
    arm.position.set(side * 0.46, 1.81, 0);
    root.add(arm);
    part(arm, g.arm, shirt, 0, -0.34, 0);
    part(arm, g.hand, skin, 0, -0.69, 0);
    arms.push(arm);

    const leg = new THREE.Group();
    leg.position.set(side * 0.21, 1.04, 0);
    root.add(leg);
    part(leg, g.leg, pants, 0, -0.43, 0);
    part(leg, g.shoe, m.shoes, 0, -0.86, 0.105);
    legs.push(leg);
  }

  const bag = new THREE.Group();
  bag.position.set(0.68, 0.78, 0.11);
  root.add(bag);
  const bagBody = part(bag, g.bag, m.bag, 0, 0, 0);
  bagBody.scale.set(1, 1, 1 + (shopper % 3) * .12);
  const items = part(bag, g.item, m.item, 0, 0.31, 0);
  items.visible = false;

  const ring = new THREE.Mesh(g.ring, m.status.walk);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.06;
  root.add(ring);
  root.scale.setScalar(.91 + (shopper % 5) * .045);
  return { body: root, arms, legs, bag, items, ring };
}

export function poseHuman(human, moving, state, units, now, shopper, assets, gaitPhase = now * 8.2) {
  // The stride follows actual distance across the floor, so a paused store
  // clock cannot leave shoppers walking in place and fast clocks stay legible.
  const step = moving ? Math.sin(gaitPhase) * .43 : 0;
  const idle = Math.sin(now * .95 + shopper * .8);
  const choosing = !moving && ["buy", "sub", "oos", "skip"].includes(state);
  human.legs[0].rotation.x = moving ? step : idle * .018;
  human.legs[1].rotation.x = moving ? -step : -idle * .018;
  human.arms[0].rotation.x = moving ? -step * .72 : choosing ? -.48 + idle * .09 : idle * .06;
  human.arms[1].rotation.x = moving ? step * .72 : choosing ? -.22 - idle * .08 : -idle * .06;
  human.body.position.y = moving ? Math.abs(step) * .11 : Math.sin(now * 1.4 + shopper) * .012;
  human.body.rotation.z = moving ? Math.sin(gaitPhase * .32) * .02 : idle * .008;
  human.ring.material = assets.materials.status[state] || assets.materials.status.walk;
  human.items.visible = units > 0;
  human.items.scale.y = Math.min(1.6, 0.7 + units * 0.08);
  human.bag.rotation.z = moving ? step * 0.20 : 0;
}
