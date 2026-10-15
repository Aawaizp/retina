// Small helpers every page uses (no heavy libraries here, so the home page stays fast).
export const STORE_NAMES = {
  retina_supercenter: "RETINA Supercenter",
  generic_supermarket: "Supermarket",
  supercenter: "Supercenter",
  neighborhood_market: "Neighborhood market",
  convenience: "Convenience store",
};

export const url = (path) => new URL(path, document.baseURI).href;

const cache = new Map();
export function getJSON(path) {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(url(path)).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path);
}
