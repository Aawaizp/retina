"""The `retina` command: pick a store, a number of days and a mind, then run the store day by day.

  retina --store supermarket --days 10 --mind fast    10 more days (opens a new store the first time)
  retina --store convenience --status                 where that store is now
  retina --store supercenter --reset --days 7         start that store again from day 1
  retina --stores                                     list the store formats

Each store keeps its own saved state and CSV files in data/live/<store id>/; running again carries on
where it stopped (3 x 10 days = exactly one 30-day run). Every row is SIMULATED.

Minds (how NPCs choose at the shelf):
  fast  the statistical engine only (v1) - data/live/<store>/
  mix   a panel of 50 thinking NPCs (autopilot for routine buys, the LLM teacher for the rest);
        everyone else is the engine - data/live/<store>_mix/  (memory.sqlite, decisions.csv, report.txt)
  llm   every NPC asks the LLM for every choice - small runs only - data/live/<store>_llm/
  A thinking run prints an ESTIMATE of LLM requests and tokens first, and stops if that is above
  today's free limits (add --yes to run anyway). --provider fake = fixed rules, no LLM (dry runs, tests).
  retina --store supermarket --mind mix --report     the panel-vs-crowd report of that run

Other tools still run as modules, e.g.  python -m retina.engine.studio  or  python -m retina.engine.export
"""

import argparse
import json
import os
import sys
from pathlib import Path

from retina import __version__

ROOT = Path(__file__).resolve().parent.parent
STORES_DIR = ROOT / "stores"
ALIASES = {"supermarket": "generic_supermarket", "neighborhood": "neighborhood_market"}   # short names
MINDS = ("fast", "llm", "mix")


def store_ids():
    return sorted(p.stem for p in STORES_DIR.glob("*.json"))


def resolve_store(name):
    """Short name or full id -> store id (file name in stores/)."""
    store = ALIASES.get(name, name)
    if store not in store_ids():
        short = {v: k for k, v in ALIASES.items()}
        names = ", ".join(short.get(s, s) for s in store_ids())
        sys.exit(f"No store '{name}'. Choose one of: {names}")
    return store


def list_stores():
    short = {v: k for k, v in ALIASES.items()}
    print("Store formats (GENERIC formats, not real retailers; all data SIMULATED):\n")
    for s in store_ids():
        p = json.loads((STORES_DIR / f"{s}.json").read_text(encoding="utf-8"))
        per_week = p.get("traffic", {}).get("real_shoppers_per_week") or 0
        print(f"  {short.get(s, s):<14} {p['name']:<30} {per_week:>7,} shoppers/week")


def build_parser():
    ap = argparse.ArgumentParser(prog="retina", description="RETINA: run a simulated store day by day.")
    ap.add_argument("--store", default=os.environ.get("RETINA_STORE", "generic_supermarket"),
                    help="supermarket (default), supercenter, neighborhood, convenience (or a full id)")
    ap.add_argument("--days", type=int, default=1, help="how many days to simulate now (default 1)")
    ap.add_argument("--mind", choices=MINDS, default="fast",
                    help="fast = engine only (default); mix = panel of thinking NPCs; llm = every NPC (small runs)")
    ap.add_argument("--panel", type=int, default=50, help="new mix run only: how many NPCs think (default 50)")
    ap.add_argument("--provider", default="auto",
                    help="LLM teacher: auto (Groq -> Gemini -> Ollama), groq, gemini, ollama, cerebras, "
                         "or fake (fixed rules, no LLM)")
    ap.add_argument("--model", help="model name for that provider")
    ap.add_argument("--yes", action="store_true", help="run even if the estimate is above today's free LLM limits")
    ap.add_argument("--report", action="store_true", help="thinking runs: print the panel-vs-crowd report and stop")
    ap.add_argument("--stores", action="store_true", help="list the store formats and stop")
    ap.add_argument("--status", action="store_true", help="show where the store is and stop")
    ap.add_argument("--reset", action="store_true", help="close the saved store and open a new one")
    ap.add_argument("--watch", action="store_true", help="print each hour slowly, as it happens")
    ap.add_argument("--pace", type=float, default=0.4, help="seconds per hour with --watch")
    ap.add_argument("--view", action="store_true", help="write view.html: the NPCs walking, day by day")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait between days (to watch --view)")
    ap.add_argument("--scale", type=float, help="new store only: NPCs per real shopper (0.1 = 1 in 10)")
    ap.add_argument("--per-week", type=int, help="new store only: real shoppers a week")
    ap.add_argument("--version", action="version", version=f"retina {__version__}")
    return ap


def live_args(args):
    """The same run, as arguments for retina.engine.live."""
    out = ["--days", str(args.days), "--pace", str(args.pace), "--wait", str(args.wait), "--mind", args.mind,
           "--panel", str(args.panel), "--provider", args.provider]
    if args.model:
        out += ["--model", args.model]
    for flag in ("status", "reset", "watch", "view", "yes", "report"):
        if getattr(args, flag):
            out.append(f"--{flag}")
    if args.scale:
        out += ["--scale", str(args.scale)]
    if args.per_week:
        out += ["--per-week", str(args.per_week)]
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stores:
        list_stores()
        return
    store = resolve_store(args.store)
    if args.days < 1:
        sys.exit("--days must be 1 or more.")
    if args.panel < 1:
        sys.exit("--panel must be 1 or more.")
    os.environ["RETINA_STORE"] = store                 # must be set BEFORE the engine is imported
    from retina.engine import live
    try:
        live.main(live_args(args))
    except KeyboardInterrupt:
        print("\nStopped. Everything up to the last finished day is saved - run again to continue.")


if __name__ == "__main__":
    main()
