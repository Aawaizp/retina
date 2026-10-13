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

Your own products (USD prices; values labelled USER_PROVIDED):
  retina products --template                  example CSV + the list of 124 aisles
  retina products --check my.csv              check every row, change nothing
  retina --products my.csv --days 10          a store that sells ONLY your products
  retina --products my.csv --products-mode add    the US catalog plus your products
  retina export --products my.csv --days 30 --csv   a dataset (Parquet/CSV + data card) of that store

The student model (copies the LLM teacher, so thinking NPCs need almost no LLM calls):
  retina collect --decisions 3000     labels from Groq openai/gpt-oss-120b only (stops at 80% of its free day)
  retina collect --repeat 100         teacher self-agreement (the ceiling)  |  retina collect --status
  retina train                        compare models, export models/student (ONNX + model card)
  retina --mind mix --student only    thinking NPCs without any LLM

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
    ap.add_argument("--student", choices=("auto", "off", "only"), default="auto",
                    help="mix runs: auto = the trained student decides when sure (default); off; only = no LLM")
    ap.add_argument("--products", metavar="CSV", help="your own products (see: retina products --template)")
    ap.add_argument("--products-mode", choices=("only", "add"), default="only",
                    help="only = the store sells just your products (default); add = the US catalog plus yours")
    ap.add_argument("--skip-bad-rows", action="store_true", help="leave rows with errors out instead of stopping")
    ap.add_argument("--version", action="version", version=f"retina {__version__}")
    return ap


def products_main(argv):
    """retina products --template [--out FOLDER] | --check CSV [--skip-bad-rows]"""
    ap = argparse.ArgumentParser(prog="retina products", description="Your own products: template and check.")
    ap.add_argument("--template", action="store_true", help="write an example file + the list of 124 aisles")
    ap.add_argument("--out", default="retina_products", help="folder for --template (default retina_products)")
    ap.add_argument("--check", metavar="CSV", help="check a product file row by row (changes nothing)")
    ap.add_argument("--skip-bad-rows", action="store_true", help="count rows with errors as skipped")
    args = ap.parse_args(argv)
    if args.template:
        from retina.products.convert import write_template
        example, aisles = write_template(args.out)
        print(f"Wrote {example} (example rows - replace them with your products) and {aisles} "
              f"(the 124 aisles: write one in the 'category' or 'aisle' column).")
    elif args.check:
        from retina.products.check import check_file, report_text
        result = check_file(args.check, args.skip_bad_rows)
        print(report_text(result))
        sys.exit(0 if result.ok else 1)
    else:
        ap.print_help()


def use_products(path, mode, skip_bad_rows):
    """Check + convert the upload, then point the engine at it (before the engine is imported)."""
    from retina.products.check import check_file, report_text
    from retina.products.convert import save
    result = check_file(path, skip_bad_rows)
    if not result.ok:
        print(report_text(result))
        sys.exit(1)
    catalog, tag, summary = save(result, path, mode)
    covered = summary["demand_covered"]
    print(f"Products: {summary['uploaded_products']:,} from {summary['file']} in {summary['aisles']} aisles "
          f"({summary['warnings']} warning(s), details in data/uploads/{tag}/check_report.txt)"
          + (f"\n  these aisles hold ~{covered:.0%} of what real shoppers buy; wants elsewhere are recorded "
             f"as 'not sold here'" if covered is not None and mode == "only" else ""))
    os.environ.update(RETINA_CATALOG=str(catalog), RETINA_CATALOG_KIND=mode, RETINA_CATALOG_TAG=tag)


def live_args(args):
    """The same run, as arguments for retina.engine.live."""
    out = ["--days", str(args.days), "--pace", str(args.pace), "--wait", str(args.wait), "--mind", args.mind,
           "--panel", str(args.panel), "--provider", args.provider, "--student", args.student]
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


def export_main(argv):
    """retina export [--store S] [--products CSV [--products-mode only|add]] [--days N] [--csv] ...
    The store and products are chosen here; every other option goes to retina.engine.export."""
    ap = argparse.ArgumentParser(prog="retina export", description="Export a dataset (Parquet/CSV + data card).",
                                 epilog="Other options (--days, --seed, --csv, --name, --no-realism) are passed on.")
    ap.add_argument("--store", default=os.environ.get("RETINA_STORE", "generic_supermarket"))
    ap.add_argument("--products", metavar="CSV")
    ap.add_argument("--products-mode", choices=("only", "add"), default="only")
    ap.add_argument("--skip-bad-rows", action="store_true")
    args, rest = ap.parse_known_args(argv)
    os.environ["RETINA_STORE"] = resolve_store(args.store)
    if args.products:
        use_products(args.products, args.products_mode, args.skip_bad_rows)
    from retina.engine import export
    export.main(rest)


def collect_main(argv):
    """retina collect [--decisions 3000] [--repeat 100] [--status] [--wait] [--provider groq|fake]"""
    ap = argparse.ArgumentParser(prog="retina collect", description="Teacher labels for the student model "
                                 "(Groq openai/gpt-oss-120b only; stops at 80% of the free day; resumes).")
    ap.add_argument("--decisions", type=int, default=3000, help="valid labels to reach (default 3000)")
    ap.add_argument("--repeat", type=int, help="ask this many labelled items again (teacher self-agreement)")
    ap.add_argument("--status", action="store_true", help="what is collected so far, and today's Groq use")
    ap.add_argument("--wait", action="store_true", help="when today's share is used: wait for tomorrow and go on")
    ap.add_argument("--provider", choices=("groq", "fake"), default="groq",
                    help="groq (the teacher, default) or fake (fixed rules, a dry run - never trains the real student)")
    ap.add_argument("--compare", choices=("gemini",), help="ask a second LLM the situations Groq labelled; "
                    "report agreement (uses only that LLM's free quota)")
    ap.add_argument("--items", type=int, default=200, help="situations to compare (default 200)")
    ap.add_argument("--export", type=int, metavar="N", help="write N situations (prompt + shuffled options, no answers) as JSONL")
    ap.add_argument("--which", choices=("new", "labelled"), default="new",
                    help="--export: new = not yet labelled by Groq (default); labelled = the ones Groq labelled (compare)")
    ap.add_argument("--out", help="--export: file (default data/teacher/export_<which>_<N>.jsonl)")
    ap.add_argument("--import", dest="import_file", metavar="JSONL", help="read another teacher's answers (checked like Groq's)")
    ap.add_argument("--compare-with", metavar="NAME/MODEL", help="report Groq vs this teacher (no new questions)")
    ap.add_argument("--tokens", action="store_true", help="tokens per label: old prompt vs compact prompt (no LLM)")
    args = ap.parse_args(argv)
    os.environ["RETINA_STORE"] = "generic_supermarket"           # situations come from the supermarket
    from retina.student import collect as C
    from retina.student.labels import LabelStore
    store = LabelStore()
    if args.status:
        C.status(store)
    elif args.tokens:
        C.token_report(C.load_pool(), store, real_per_label=240)
    elif args.export:
        out = args.out or C.TEACHER_DIR / f"export_{args.which}_{args.export}.jsonl"
        C.export_situations(out, args.export, args.which, store=store)
    elif args.import_file:
        C.import_labels(args.import_file, store=store)
    elif args.compare_with:
        provider, _, model = args.compare_with.partition("/")
        C.compare_report(store, provider, model or None)
    elif args.compare:
        C.compare(args.compare, args.items, store=store)
    elif args.repeat:
        C.collect(args.repeat, repeat=1, provider=args.provider, wait=args.wait, store=store)
        rate, n = C.self_agreement(store, *(("fake", None) if args.provider == "fake" else C.TEACHER))
        if rate is not None:
            print(f"Teacher self-agreement: {rate:.0%} on {n} items (the ceiling for the student)")
    else:
        C.collect(args.decisions, provider=args.provider, wait=args.wait, store=store)


def train_main(argv):
    ap = argparse.ArgumentParser(prog="retina train", description="Train, compare and export the student model.")
    ap.add_argument("--out", help="folder for the model (default models/student)")
    ap.add_argument("--consensus", metavar="NAME/MODEL", help="weight up decisions where this second teacher agrees")
    ap.add_argument("--consensus-weight", type=float, default=2.0, help="weight of an agreed decision (default 2)")
    ap.add_argument("--teachers", nargs="+", default=[], metavar="NAME/MODEL",
                    help="extra teachers whose labels are added to TRAINING only (e.g. claude-code/opus-5-5)")
    args = ap.parse_args(argv)
    os.environ["RETINA_STORE"] = "generic_supermarket"
    from retina.student.train import train, MODEL_DIR
    train(out=args.out or MODEL_DIR, consensus=args.consensus, consensus_weight=args.consensus_weight,
          teachers=args.teachers)


def release_main(argv):
    ap = argparse.ArgumentParser(prog="retina release", description="Build the sample datasets of a release "
                                 "(GitHub Actions runs this on every version tag).")
    ap.add_argument("--version", required=True, help="e.g. v2.0.0-alpha.1")
    ap.add_argument("--days", type=int, help="days per store (default: the profile's 365)")
    ap.add_argument("--stores", help="comma list, e.g. supermarket,convenience (default: all four)")
    ap.add_argument("--out", help="folder (default dist/<version>)")
    ap.add_argument("--realism", action="store_true", help="also run the realism test (needs the raw Instacart files)")
    ap.add_argument("--no-student", action="store_true", help="leave out the thinking-NPC (student) dataset")
    args = ap.parse_args(argv)
    from retina.release import release
    stores = [resolve_store(s.strip()) for s in args.stores.split(",")] if args.stores else None
    release(args.version, args.days, stores, args.out, args.realism, not args.no_student)


def app_main(argv):
    ap = argparse.ArgumentParser(prog="retina app", description="The local RETINA app (FastAPI) - your PC only.")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (default: this PC only); Docker uses 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    import uvicorn
    from retina.app.server import app
    print(f"RETINA app on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port} "
          f"(API docs: /docs). Ctrl+C stops it.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    commands = {"products": products_main, "export": export_main, "collect": collect_main, "train": train_main,
                "release": release_main, "app": app_main}
    if argv[:1] and argv[0] in commands:
        commands[argv[0]](argv[1:])
        return
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
    if args.products:
        use_products(args.products, args.products_mode, args.skip_bad_rows)
    from retina.engine import live
    try:
        live.main(live_args(args))
    except KeyboardInterrupt:
        print("\nStopped. Everything up to the last finished day is saved - run again to continue.")


if __name__ == "__main__":
    main()
