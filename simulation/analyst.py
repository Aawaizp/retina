"""RETINA analyst: ask questions about the simulated store in plain English.

How it works (a small "agent loop"):
  1. The LLM reads your question and picks a TOOL (a Python function below) to look up data.
  2. Python runs the tool on the simulated output files and sends the result back.
  3. Repeat (max 4 lookups), then the LLM answers using ONLY those numbers.
The LLM never sees the raw files and never runs code - it can only call these tools.

Run:  python simulation\\analyst.py            (uses Groq, or $env:RETINA_LLM)
      python simulation\\analyst.py gemini     (choose the provider)
"""

import json
import sys

import pandas as pd

from config import OUTPUT_DIR
from llm import DEFAULT_PROVIDER, chat

MAX_LOOKUPS = 4        # tool calls per question (keeps free-tier usage low)
MAX_RESULT_CHARS = 3000  # tool results are cut to this size before going to the LLM


# ================================================================== data
def load_outputs():
    """Read what retina_mart.py saved. Adds a 'month' column (2025-09, 2025-10, ...)."""
    files = {"sales": "sales_log", "stockouts": "truth_stockouts", "waste": "store_waste",
             "effects": "truth_promo_effects"}
    data = {k: pd.read_parquet(OUTPUT_DIR / f"{f}.parquet") for k, f in files.items()}
    for df in data.values():
        df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    return data


def matching(df, name):
    """Rows for products named exactly `name` (any capitals, also without a plural 's');
    if there are none, rows whose product name contains it. "bananas" -> Banana, not Banana Chips."""
    names = df["product_name"].str.lower()
    name = name.lower().strip()
    for candidate in (name, name.removesuffix("es"), name.removesuffix("s")):
        exact = names == candidate
        if exact.any():
            return df[exact]
    return df[names.str.contains(name.removesuffix("s"), regex=False, na=False)]


def matched_names(df):
    """Which products a lookup used (so the model and you can check it is the right one)."""
    top = df["product_name"].value_counts().head(5)
    return "Products used: " + ", ".join(f"{n} ({c:,} rows)" for n, c in top.items()) + "\n"

# ================================================================= tools
def overview(data):
    """Whole-store totals and month-by-month trips, items, revenue, gross margin."""
    s = data["sales"]
    monthly = s.groupby("month").agg(trips=("trip_id", "nunique"), items=("quantity", "sum"),
                                     revenue=("line_total", "sum"))
    cost = (s["quantity"] * s["unit_cost"]).groupby(s["month"]).sum()
    waste = data["waste"].groupby("month")["total_cost"].sum()
    monthly["gross_margin_%"] = ((monthly["revenue"] - cost - waste.reindex(monthly.index).fillna(0))
                                 / monthly["revenue"] * 100)
    lost = data["stockouts"].query("lost_sale")["unit_price"].sum()
    head = (f"Year total: {s['trip_id'].nunique():,} trips, {s['quantity'].sum():,} items, "
            f"${s['line_total'].sum():,.0f} revenue, ${lost:,.0f} lost to empty shelves, "
            f"${data['waste']['total_cost'].sum():,.0f} thrown away.\n")
    return head + monthly.round(1).to_string()


def product_report(data, name):
    """Month by month for a product (exact name, else names containing it): units, revenue,
    average price, share sold on promotion, times found out of stock, units thrown away."""
    s = matching(data["sales"], name)
    if s.empty:
        return f"No product name contains '{name}'. Try a shorter word."
    monthly = s.groupby("month").agg(units=("quantity", "sum"), revenue=("line_total", "sum"),
                                     avg_price=("unit_price", "mean"), promo_pct=("on_promo", "mean"))
    monthly["promo_pct"] *= 100                          # % of sales lines bought on promotion
    monthly["out_of_stock"] = matching(data["stockouts"], name).groupby("month").size()
    monthly["thrown_away"] = matching(data["waste"], name).groupby("month")["units"].sum()
    return matched_names(s) + monthly.fillna(0).round(1).to_string()

def department_report(data, department):
    """Month by month for one department: units, revenue, promo share, out-of-stock, waste."""
    s = data["sales"][data["sales"]["department"] == department]
    if s.empty:
        return "Unknown department. Choose one of: " + ", ".join(sorted(data["sales"]["department"].unique()))
    monthly = s.groupby("month").agg(units=("quantity", "sum"), revenue=("line_total", "sum"),
                                     promo_pct=("on_promo", "mean"))
    monthly["promo_pct"] *= 100                          # % of sales lines bought on promotion
    so = data["stockouts"]
    monthly["out_of_stock"] = so[so["department"] == department].groupby("month").size()
    w = data["waste"]
    monthly["thrown_away"] = w[w["department"] == department].groupby("month")["units"].sum()
    return monthly.fillna(0).round(1).to_string()


def top_products(data, by="units", department="", month="", n=10):
    """Top n products by 'units', 'revenue', 'out_of_stock' or 'thrown_away',
    optionally inside one department and/or one month (YYYY-MM)."""
    sources = {"units": ("sales", "quantity"), "revenue": ("sales", "line_total"),
               "out_of_stock": ("stockouts", None), "thrown_away": ("waste", "units")}
    if by not in sources:
        return "by must be one of: " + ", ".join(sources)
    table, column = sources[by]
    df = data[table]
    if department:
        df = df[df["department"] == department]
    if month:
        df = df[df["month"] == month]
    grouped = df.groupby("product_name")
    result = grouped.size() if column is None else grouped[column].sum()
    return result.nlargest(int(n)).round(2).to_string() if len(result) else "No rows for that filter."


def promo_report(data, name=""):
    """Money on promoted vs normal lines, and where promotion sales came from (brand switching /
    stocking up / unplanned buying, 'had at home' skips), for the store or one product."""
    s = data["sales"] if not name else matching(data["sales"], name)
    e = data["effects"] if not name else matching(data["effects"], name)
    if s.empty:
        return "No sales found."
    lines = [matched_names(s).strip()] if name else []
    for label, part in [("promoted", s[s["on_promo"]]), ("normal price", s[~s["on_promo"]])]:
        revenue = part["line_total"].sum()
        profit = revenue - (part["quantity"] * part["unit_cost"]).sum()
        margin = profit / revenue * 100 if revenue else 0
        lines.append(f"{label}: {part['quantity'].sum():,} units, revenue ${revenue:,.0f}, "
                     f"gross profit ${profit:,.0f} ({margin:.1f}% margin)")
    effects = e.groupby("effect")["units"].sum().to_string() if len(e) else "none"
    return ("\n".join(lines) + "\n\nWhy promoted units sold (units): brand_switch = taken from other "
            "brands, stock_up = borrowed from future weeks, impulse = truly extra, had_at_home = later "
            "skips, bought_together = partner items\n" + effects)


def stockout_report(data, name=""):
    """Empty-shelf moments: what shoppers did (real research shares) and lost sales by month,
    for the whole store or products whose name contains `name`."""
    so = data["stockouts"] if not name else matching(data["stockouts"], name)
    if so.empty:
        return "No out-of-stock moments found."
    meaning = {"other_store": "bought it at another store (lost sale)",
               "other_brand": "switched to another brand", "same_brand": "took another size/flavour of the same brand",
               "delay": "will buy it later (lost today)", "skip": "did not buy it (lost sale)"}
    reactions = so["reaction"].value_counts()
    reaction_lines = "\n".join(f"  {meaning[r]}: {c:,} times" for r, c in reactions.items())
    monthly = so.groupby("month").agg(times=("reaction", "size"),
                                      lost_sales_usd=("unit_price", lambda p: p[so.loc[p.index, "lost_sale"]].sum()))
    head = matched_names(so) if name else ""
    lost_total = so.loc[so["lost_sale"], "unit_price"].sum()
    busiest = monthly["times"].nlargest(3)
    return (head + f"Empty shelf {len(so):,} times, lost sales total ${lost_total:,.0f}. Busiest months: "
            + ", ".join(f"{m} ({t:,} times)" for m, t in busiest.items())
            + "\nWhat shoppers did:\n" + reaction_lines + "\n\nBy month:\n" + monthly.round(0).to_string())

TOOLS = {"overview": overview, "product_report": product_report, "department_report": department_report,
         "top_products": top_products, "promo_report": promo_report, "stockout_report": stockout_report}

TOOL_HELP = """overview() - store totals and monthly trips, items, revenue, gross margin
product_report(name) - monthly units, revenue, price, promo share, out-of-stock, waste for one product (or names containing `name`)
department_report(department) - the same, for one department
top_products(by, department, month, n) - by = units | revenue | out_of_stock | thrown_away; department, month (YYYY-MM) optional
promo_report(name) - profit on promoted vs normal-price sales, and where promotion sales came from; name optional
stockout_report(name) - empty-shelf reactions and lost sales by month; name optional"""


# ============================================================ agent loop
def system_prompt(data):
    s = data["sales"]
    return f"""You are the data analyst for RETINA Mart, a SIMULATED supermarket (not real sales).
Data: {s['date'].min():%d %b %Y} to {s['date'].max():%d %b %Y}, 2,000 simulated shoppers.
Departments: {', '.join(sorted(s['department'].unique()))}.

You can only learn facts by calling tools. Reply with ONE JSON object and nothing else:
  {{"tool": "<tool name>", "args": {{...}}}}   to look something up, or
  {{"answer": "<your answer>"}}                when you can answer.

Tools:
{TOOL_HELP}

Rules:
- ALWAYS call at least one tool before answering. Use only numbers that appear in tool results.
- Never invent numbers or write placeholders. If a number is missing, call another tool.
  - Avoid your own big calculations - quote tool numbers; simple differences and % changes are fine.
  - Check "Products used" in tool results is the product the user meant.
- Explain WHY using the evidence (out-of-stock, promotions, prices, waste, number of days in a month).
- At most {MAX_LOOKUPS} tool calls, then answer. Keep answers under 150 words.
- Mention once that the data is simulated."""


def parse_reply(text):
    """Find the JSON object in the model's reply (models sometimes add ```json fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"answer": text.strip()}
    try:
        return json.loads(text[start:end + 1], strict=False)   # strict=False: allow line breaks
    except json.JSONDecodeError:
        return {"answer": text.strip()}


def ask(question, data, provider):
    """Run the agent loop for one question and return the final answer.
    The code checks every reply: empty replies, answers without any lookup, and unknown tools
    are sent back to the model with a short correction (models make these mistakes)."""
    messages = [{"role": "system", "content": system_prompt(data)},
                {"role": "user", "content": question}]
    lookups = 0
    for _ in range(MAX_LOOKUPS + 3):                  # a few extra turns for corrections
        reply = chat(messages, provider)
        step = parse_reply(reply)
        if step.get("tool") in TOOLS and lookups < MAX_LOOKUPS:
            args = step.get("args") or {}
            print(f"  -> looking up {step['tool']}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            try:
                result = TOOLS[step["tool"]](data, **args)
            except (TypeError, ValueError) as error:  # wrong arguments from the model
                result = f"Tool error: {error}. Check the tool list."
            lookups += 1
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "TOOL RESULT:\n" + result[:MAX_RESULT_CHARS]})
            continue
        if step.get("answer") and lookups > 0:
            return step["answer"]
        if not reply:
            problem = "Your reply was empty."
        elif lookups == 0:
            problem = "You have not looked anything up yet. Call a tool first - never answer from memory."
        elif lookups >= MAX_LOOKUPS:
            problem = "No more lookups. Answer now using the tool results above."
        else:
            problem = "That tool does not exist. Use one from the tool list."
        print(f"  (correcting the model: {problem})")
        messages.append({"role": "assistant", "content": reply or "(empty reply)"})
        messages.append({"role": "user", "content": problem + ' Reply with ONE JSON object: '
                         '{"tool": ..., "args": {...}} or {"answer": "..."}.'})
    return "Sorry, the model did not give a usable answer. Try again, or try another provider (gemini)."


def main():
    provider = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROVIDER
    print("Loading simulated store data...")
    data = load_outputs()
    print(f"RETINA analyst ({provider}). Ask about the SIMULATED store. Press Enter on an empty line to quit.")
    print("Examples: Why did banana sales change in February?  |  Which department wastes the most?")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):         # Ctrl+C or Ctrl+Z also quit cleanly
            break



        if not question:
            break
        try:
            print("\nAnalyst:", ask(question, data, provider))
        except RuntimeError as error:                # network / key / rate-limit problems
            print("Problem:", error)


if __name__ == "__main__":
    main()