// The RETINA student model in the browser: the same ONNX file Python uses, run by onnxruntime-web.
// features() is a line-by-line copy of retina/student/features.py - keep the two in step.
import * as ort from "onnxruntime-web/wasm";
import { getJSON, url } from "./json.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

const ROUTES = ["need", "repeat", "new", "partner"];
const MISSIONS = ["top_up", "routine", "stock_up"];

// npc = {sensitivity, household, kids, income (0-4), budget ($/week), rebuy (0-1)}
// options = [{product, price, regular, deal, off, times, usual, top, pop, last_price, sold_out}]
export function features(npc, route, mission, options, engineProduct, featureNames) {
  const prices = options.map((o) => Math.max(o.price, 0.01));
  const logs = prices.map(Math.log);
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length;
  const rows = options.map((o, i) => {
    const change = Math.log(Math.max(o.price, 0.01) / Math.max(o.regular, 0.01));
    const vsLast = o.last_price ? Math.log(Math.max(o.price, 0.01) / o.last_price) : 0;
    const bought = o.times > 0 ? 1 : 0;
    const usual = o.usual ? 1 : 0;
    const sens = npc.sensitivity;
    return [
      change, logs[i] - mean, o.deal ? 1 : 0, (o.off || 0) / 100, bought, Math.log1p(o.times), usual,
      o.product === engineProduct ? 1 : 0, o.top ? 1 : 0, Math.log(Math.max(o.pop || 0, 1e-6)), vsLast,
      (o.sold_out || 0) > 0 ? 1 : 0,
      sens, npc.household, npc.kids, npc.income, Math.log(Math.max(npc.budget, 1)), npc.rebuy,
      ...ROUTES.map((r) => (route === r ? 1 : 0)),
      ...MISSIONS.map((m) => (mission === m ? 1 : 0)),
      (o.deal ? 1 : 0) * sens, change * sens, usual * npc.rebuy,
    ];
  });
  if (featureNames && rows[0].length !== featureNames.length) {
    throw new Error(`feature count ${rows[0].length} != model's ${featureNames.length} - update features()`);
  }
  return new Float32Array(rows.flat());
}

let modelPromise;
export function loadStudent() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const meta = await getJSON("model/student.json");
      const session = await ort.InferenceSession.create(url("model/student.onnx"), {
        executionProviders: ["wasm"],
      });
      return { meta, session };
    })();
  }
  return modelPromise;
}

// Probabilities = softmax(scores / temperature) within each decision, exactly as in Python.
// Several decisions go in ONE run (stacked rows): the session must never run twice at the same time.
let queue = Promise.resolve();
export function chooseMany(decisions) {
  const job = queue.then(async () => {
    const { meta, session } = await loadStudent();
    const n = meta.features.length;
    const parts = decisions.map((d) => features(d.npc, d.route, d.mission, d.options, d.engine, meta.features));
    const x = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
    let at = 0;
    for (const p of parts) {
      x.set(p, at);
      at += p.length;
    }
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", x, [x.length / n, n]) });
    const scores = Array.from(out[session.outputNames[0]].data);
    let row = 0;
    return decisions.map((d) => {
      const z = scores.slice(row, row + d.options.length).map((v) => v / meta.temperature);
      row += d.options.length;
      const max = Math.max(...z);
      const exp = z.map((v) => Math.exp(v - max));
      const sum = exp.reduce((a, b) => a + b, 0);
      return exp.map((e) => e / sum);
    });
  });
  queue = job.catch(() => {});
  return job;
}

export async function choose(npc, route, mission, options, engine) {
  return (await chooseMany([{ npc, route, mission, options, engine }]))[0];
}
