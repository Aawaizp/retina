// The Parquet tables are read by DuckDB-WASM: a full SQL database running inside the visitor's
// browser (no server). Its engine (~35 MB) comes from the jsDelivr CDN at the exact installed
// version - too big for free static hosting (25 MB per file) and cached by the browser after once.
import * as duckdb from "@duckdb/duckdb-wasm";
import { url } from "./json.js";

export { getJSON, STORE_NAMES, url } from "./json.js";

let dbPromise;
function getDB() {
  if (!dbPromise) {
    dbPromise = (async () => {
      const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
      const workerUrl = URL.createObjectURL(
        new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" }),
      );
      const worker = new Worker(workerUrl);
      const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
      await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
      URL.revokeObjectURL(workerUrl);
      return db;
    })();
  }
  return dbPromise;
}

const registered = new Set();

// query("SELECT ... FROM 'generic_supermarket/daily.parquet'", ["generic_supermarket/daily.parquet"])
export async function query(sql, files) {
  const db = await getDB();
  for (const f of files) {
    if (!registered.has(f)) {
      await db.registerFileURL(f, url(`data/${f}`), duckdb.DuckDBDataProtocol.HTTP, false);
      registered.add(f);
    }
  }
  const conn = await db.connect();
  try {
    const table = await conn.query(sql);
    return table.toArray().map((row) => {
      const o = row.toJSON();
      for (const k of Object.keys(o)) if (typeof o[k] === "bigint") o[k] = Number(o[k]);
      return o;
    });
  } finally {
    await conn.close();
  }
}
