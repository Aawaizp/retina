"""Check an uploaded product file, row by row, and say exactly what is wrong.

Columns (any order; headers like "Product Name" are fine):
  required  name, brand, category, price, pack_size
  optional  aisle          exact aisle name - wins over category (see `retina products --template`)
            promo_price    the price when it is on a deal (below price; deals deeper than 50% are capped)
            popularity     any number, bigger = sells more (relative within your file)
            shelf_life_days  days a unit stays sellable (0 = does not spoil)
            product_id     your own id (must be unique)
Prices are US dollars per item (Phase 3: USD only - NPC budgets come from US data).

ERRORS stop the upload (or, with skip_bad_rows, drop that row); WARNINGS never stop it.
"category" must name one of the 124 aisles. If it does not, the message SUGGESTS close aisles
("did you mean ...") - nothing is changed by itself: write the aisle in an `aisle` column.
"""

import re
from dataclasses import dataclass, field

from retina.products import aisles as A
from retina.products.packsize import parse_pack
from retina.products.reader import read_products

REQUIRED = ["name", "brand", "price", "pack_size"]
OPTIONAL = ["category", "aisle", "promo_price", "popularity", "shelf_life_days", "product_id"]
MAX_PRICE = 5_000.0
MAX_DEAL = 0.50                      # same cap as the US catalog clean-up (loaders/clean_catalog.py)
MAX_NAME = 120
OTHER_MONEY = re.compile(r"(₹|€|£|¥|\brs\.?|\binr\b|\beur\b|\bgbp\b)", re.I)


@dataclass
class Check:
    file: str
    info: dict = field(default_factory=dict)
    products: list = field(default_factory=list)
    errors: list = field(default_factory=list)       # (row, message); row 0 = the whole file
    warnings: list = field(default_factory=list)
    dropped: int = 0

    @property
    def ok(self):
        return not self.errors


def money(text):
    """'$1,299.50' -> 1299.5; returns (value, problem)."""
    t = str(text).strip()
    if not t:
        return None, "is empty"
    if OTHER_MONEY.search(t):
        return None, f"'{t}' is not in US dollars (Phase 3 takes USD prices only)"
    t = re.sub(r"^(usd|us\$|\$)\s*|\s*(usd)$", "", t, flags=re.I).strip()
    if re.fullmatch(r"\d+,\d{1,2}", t):
        return None, f"'{text}' uses a comma for decimals - write {t.replace(',', '.')}"
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    try:
        value = float(t)
    except ValueError:
        return None, f"'{text}' is not a number"
    return value, None


def resolve_aisle(row):
    """(aisle, error) from the row's aisle or category - exact names only."""
    given = row.get("aisle", "")
    if given:
        hit = A.find(given)
        if hit:
            return hit, None
        tips = A.suggest(given)
        return None, (f"aisle '{given}' is not one of the 124 aisles"
                      + (f" - did you mean {' / '.join(repr(t) for t in tips)}?" if tips else ""))
    category = row.get("category", "")
    if not category:
        return None, "has no category (or aisle)"
    hit = A.find(category)
    if hit:
        return hit, None
    in_dept = A.department_aisles(category)
    if in_dept:
        return None, (f"category '{category}' is a whole department - put the exact aisle in an 'aisle' column, "
                      f"one of: {', '.join(in_dept[:8])}{' ...' if len(in_dept) > 8 else ''}")
    tips = A.suggest(category)
    hint = f" - did you mean {' / '.join(repr(t) for t in tips)}?" if tips else ""
    return None, (f"category '{category}' is not one of the 124 aisles{hint} "
                  f"Add an 'aisle' column with the exact aisle name (list: retina products --template).")


def check_file(path, skip_bad_rows=False):
    from retina.products.reader import UploadError
    result = Check(file=str(path))
    try:
        columns, rows, info = read_products(path)
    except UploadError as error:
        result.errors.append((0, str(error)))
        return result
    result.info = info
    missing = [c for c in REQUIRED if c not in columns]
    if "category" not in columns and "aisle" not in columns:
        missing.append("category (or aisle)")
    if missing:
        result.errors.append((0, f"missing column(s): {', '.join(missing)}. Found: {', '.join(c for c in columns if c)}"))
        return result
    unknown = [c for c in columns if c and c not in REQUIRED + OPTIONAL]
    if unknown:
        result.warnings.append((0, f"ignored column(s): {', '.join(unknown)}"))
    if not rows:
        result.errors.append((0, "no products in the file (only a header)"))
        return result

    table, seen, ids = A.aisle_table(), {}, {}
    for number, row in rows:
        errors, warns = [], []
        if row.get("__extra__"):
            warns.append("has more cells than the header - a name with a comma needs \"quotes\"")
        name = row.get("name", "")
        if not name:
            errors.append("name is empty")
        elif len(name) > MAX_NAME:
            warns.append(f"name is longer than {MAX_NAME} characters - shortened")
            name = name[:MAX_NAME].rstrip()
        if name.startswith(("=", "+", "-", "@")):
            warns.append(f"name starts with '{name[0]}' - written as text in RETINA's files (spreadsheet safety)")
        brand = row.get("brand", "")
        if not brand:
            warns.append("brand is empty - same-brand switching at empty shelves will not apply")
        price, problem = money(row.get("price", ""))
        if problem:
            errors.append(f"price {problem}")
        elif price <= 0:
            errors.append(f"price must be above 0 (got {price:g})")
        elif price > MAX_PRICE:
            errors.append(f"price ${price:,.2f} is above ${MAX_PRICE:,.0f} - is it in another currency? (USD only)")
        aisle, problem = resolve_aisle(row)
        if problem:
            errors.append(problem)
        pack_text = row.get("pack_size", "")
        pack = parse_pack(pack_text)
        if not pack_text:
            errors.append("pack_size is empty (e.g. 500 g, 12 x 330 ml, 6 ct)")
        elif pack is None:
            warns.append(f"pack_size '{pack_text}' could not be read - kept as written")
        promo = None
        if row.get("promo_price"):
            value, problem = money(row["promo_price"])
            if problem:
                warns.append(f"promo_price {problem} - ignored")
            elif price and not errors and value >= price:
                warns.append(f"promo_price ${value:.2f} is not below the price ${price:.2f} - ignored")
            elif price and value > 0:
                promo = value
                if promo < price * (1 - MAX_DEAL):
                    warns.append(f"deal deeper than {MAX_DEAL:.0%} - capped at {MAX_DEAL:.0%} off")
                    promo = round(price * (1 - MAX_DEAL), 2)
        popularity = None
        if row.get("popularity"):
            try:
                popularity = float(row["popularity"].replace(",", ""))
                if popularity <= 0:
                    raise ValueError
            except ValueError:
                warns.append(f"popularity '{row['popularity']}' is not a positive number - ignored")
                popularity = None
        life = None
        if row.get("shelf_life_days"):
            try:
                life = int(float(row["shelf_life_days"]))
                if not 0 <= life <= 3650:
                    raise ValueError
            except ValueError:
                warns.append(f"shelf_life_days '{row['shelf_life_days']}' is not 0-3650 - the department's default is used")
                life = None
        pid = row.get("product_id", "")
        if pid and pid in ids:
            errors.append(f"product_id '{pid}' is used again (first in row {ids[pid]})")
        if aisle and price and not errors:
            typical = float(table.at[aisle, "median_price"])
            if price > 6 * typical or price < typical / 6:
                warns.append(f"price ${price:.2f} is far from typical for '{aisle}' (US catalog median ${typical:.2f})")
        key = (A.norm(name), A.norm(brand), (pack or {}).get("text", A.norm(pack_text)))
        if not errors and key in seen:
            warns.append(f"same product as row {seen[key]} - this row is skipped")
            result.warnings += [(number, w) for w in warns]
            result.dropped += 1
            continue
        result.warnings += [(number, w) for w in warns]
        if errors:
            if skip_bad_rows:
                result.warnings += [(number, f"SKIPPED: {e}") for e in errors]
                result.dropped += 1
            else:
                result.errors += [(number, e) for e in errors]
            continue
        seen[key] = number
        if pid:
            ids[pid] = number
        result.products.append({"row": number, "product_id": pid, "name": name, "brand": brand,
                                "category": row.get("category", ""), "aisle": aisle,
                                "department": table.at[aisle, "department"], "price": round(price, 2),
                                "promo_price": promo, "pack_size": pack["text"] if pack else pack_text,
                                "popularity": popularity, "shelf_life_days": life})
    if not result.products and not result.errors:
        result.errors.append((0, "no usable products left"))
    return result


def report_text(check, max_lines=40):
    """The check as readable text (errors first)."""
    out = [f"Product check: {check.file}"]
    if check.info:
        out.append(f"  read as {check.info['encoding']}, {check.info['separator']}-separated")
    if check.products:
        aisles = {p["aisle"] for p in check.products}
        depts = {p["department"] for p in check.products}
        out.append(f"  {len(check.products):,} products OK in {len(aisles)} aisles / {len(depts)} departments"
                   + (f" | {check.dropped} row(s) skipped" if check.dropped else ""))
    for title, items in (("ERRORS (fix these first)", check.errors), ("Warnings", check.warnings)):
        if items:
            out.append(f"\n{title}: {len(items)}")
            for row, message in items[:max_lines]:
                out.append(f"  {'file' if row == 0 else f'row {row}'}: {message}")
            if len(items) > max_lines:
                out.append(f"  ... and {len(items) - max_lines} more (see check_report.txt)")
    if check.ok:
        out.append("\nResult: OK - ready to run")
    elif any(row > 0 for row, _ in check.errors):           # bad rows: they can be left out
        out.append("\nResult: NOT OK - fix the errors above (or use --skip-bad-rows to leave those rows out)")
    else:                                                    # the file itself: skipping rows cannot help
        out.append("\nResult: NOT OK - fix the file problem above")
    return "\n".join(out)
