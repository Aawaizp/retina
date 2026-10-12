"""Glue between a live run (retina.engine.live) and the mind: attach it, show the ESTIMATE before
running, save each day's decisions, and print the report at the end."""

import json
import math

import numpy as np
import pandas as pd

from retina.engine import llm, traffic
from retina.engine.csvsafe import to_csv
from retina.mind.mind import Mind, BATCH
from retina.mind.teacher import MAX_REQUEST_TOKENS

# ESTIMATED prompt sizes (tokens) from the prompt layout in teacher.py; first real Groq run (12 Oct 2026):
# 2 requests, 70 items, 14,349 tokens = ~205 per item - these give ~205 per item, so they agree
TOKENS = {"fixed": 230, "per_npc": 110, "per_item": 30, "per_option": 28, "reply_per_item": 35}
TEACH_SHARE_PRIOR = 0.8       # share of a thinking NPC's items sent to the teacher in mix mode
                              # (ESTIMATED from a 3-day supermarket run: 111 of 138; "new on the list" dominates)


def attach(sim, mode, run_dir, panel, provider, model):
    mind = Mind(sim, mode, run_dir, panel, provider, model)
    who = f"panel of {len(mind.panel)} thinking NPCs" if mind.mode == "mix" else "every NPC thinks"
    print(f"Mind: {mind.mode} ({who}; teacher: {'fixed rules, NO LLM' if provider == 'fake' else provider})")
    return mind


def estimate(sim, days):
    """What a run of `days` will ask from the LLM - before anything runs."""
    mind, sh = sim.mind, sim.shoppers
    start = sim.day
    trips = sum(traffic.expected_npcs(d) for d in range(start, start + days)) if traffic.enabled() else \
        float(np.isin(sh.next_visit_day, range(start, start + days)).sum())
    pool = len(sh.shopper_id)
    ids = np.arange(pool) if mind.mode == "llm" else np.array(sorted(mind.panel))
    thinking_trips = trips * len(ids) / pool
    items = float(sh.basket_mean[ids].mean())
    past = mind.memory.decisions()
    if mind.mode == "llm":
        share, source = 1.0, "every item"
    elif len(past) >= 50:
        share, source = float(np.mean([d["mind"] != "autopilot" for d in past])), "this run so far"
    else:
        share, source = TEACH_SHARE_PRIOR, "ESTIMATED"
    teach = thinking_trips * items * share
    asked_trips = thinking_trips * (1 - (1 - share) ** max(items, 1))     # trips with at least one teacher item
    calls = math.ceil(asked_trips / BATCH) if teach > 0 else 0
    per_item = TOKENS["per_item"] + 5 * TOKENS["per_option"] + TOKENS["reply_per_item"]
    tokens = int(calls * TOKENS["fixed"] + asked_trips * TOKENS["per_npc"] + teach * per_item)
    calls = max(calls, math.ceil(tokens / MAX_REQUEST_TOKENS))          # big batches are split in two
    tokens = int(calls * TOKENS["fixed"] + asked_trips * TOKENS["per_npc"] + teach * per_item)
    return {"days": days, "trips": trips, "thinking_trips": thinking_trips, "items": thinking_trips * items,
            "teach": teach, "share": share, "share_source": source, "calls": calls, "tokens": tokens}


def providers_for(teacher):
    """The providers this run may use, with what is still free today."""
    if teacher.name == "fake":
        return []
    names = llm.CHAIN if teacher.provider == "auto" else [teacher.provider]
    out = []
    for p in names:
        if not llm.ready(p):
            continue
        req, tok = llm.PACER.left_today(p)
        out.append({"provider": p, "requests_left": req, "tokens_left": tok, "limits": llm.limits(p)})
    return out


def check_estimate(sim, days, yes):
    """Print the estimate. False = do not run (above today's free limits without --yes, or no LLM)."""
    mind = sim.mind
    e = estimate(sim, days)
    print(f"\nEstimate before running (ESTIMATED): {days} day(s), ~{e['trips']:,.0f} trips, "
          f"~{e['thinking_trips']:,.0f} by thinking NPCs")
    print(f"  ~{e['items']:,.0f} item choices by thinking NPCs: ~{e['items'] - e['teach']:,.0f} autopilot (free), "
          f"~{e['teach']:,.0f} for the LLM teacher ({e['share']:.0%}, {e['share_source']})")
    print(f"  ~{e['calls']:,} LLM requests (up to {BATCH} NPCs each), ~{e['tokens']:,} tokens. "
          f"Cached answers (reruns) cost nothing.")
    if mind.mode == "llm":
        print("  WARNING: --mind llm sends EVERY choice of EVERY NPC to the LLM. Meant for small runs only "
              "(a few days, a small store); free tiers run out fast.")
    if mind.teacher.name == "fake":
        print("  Teacher: fixed rules (--provider fake) - no LLM calls, data is NOT LLM-made.")
        return True
    usable = providers_for(mind.teacher)
    if not usable:
        print("  No LLM provider is ready (no key, or resting). Set $env:GROQ_API_KEY or $env:GEMINI_API_KEY,"
              " or start Ollama and use --provider ollama.\n  For a dry run without any LLM: --provider fake")
        return False
    free = 0
    for u in usable:
        lim = u["limits"]
        left = "no daily limit" if u["tokens_left"] is None else f"{u['tokens_left']:,} tokens left today"
        pace = []
        if lim["rpm"]:
            pace.append(e["calls"] / lim["rpm"])
        if lim["tpm"]:
            pace.append(e["tokens"] / lim["tpm"])
        minutes = f", at its free pace ~{max(pace):.0f} min" if pace else ""
        print(f"  {u['provider']}: {left}{minutes}")
        free = math.inf if u["tokens_left"] is None else free + u["tokens_left"]
    if e["tokens"] > free and not yes:
        print(f"  Above today's free LLM limits (~{e['tokens']:,} needed). Run fewer days, or add --yes to run "
              f"anyway (items the LLM cannot answer fall back to the engine).")
        return False
    return True


def day_done(mind, day, run_dir):
    """End of a simulated day: make the memory permanent, append decisions.csv, one summary line."""
    rows = mind.end_day(day)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["options"] = df["options"].map(json.dumps)
    df["data_label"] = "SIMULATED"
    path = run_dir / "decisions.csv"
    to_csv(df, path, mode="a", header=not path.exists(), index=False)
    kinds = df["mind"].value_counts()
    cached = int(df.loc[df["mind"].isin(["llm", "llm_fallback"]), "cached"].sum())
    print(f"  mind: {len(df)} choices by thinking NPCs | {kinds.get('autopilot', 0)} autopilot | "
          f"{kinds.get('llm', 0)} LLM ({cached} from cache) | {kinds.get('llm_fallback', 0)} fell back to the engine")


def finish(mind, run_dir):
    from retina.mind.report import build_report
    text = build_report(mind, run_dir)
    print("\n" + text)
