"""Read an uploaded product file the way people really save them.

  - text: UTF-8 (with or without the BOM Excel adds), else Windows-1252 (old Excel "CSV" on Windows)
  - separator: comma, semicolon (Excel in many countries) or tab - detected from the file
  - quotes: "Cookies, Chocolate Chip" and "12"" Pizza" are read as one cell (standard CSV rules)
  - headers: case, spaces and a few common names do not matter ("Product Name" = name)
  - limit: MAX_ROWS products per file (bigger files are refused with a clear message)
Returns the header and the rows as dicts with their row number as a spreadsheet shows it
(row 1 = the header, row 2 = the first product).
"""

import csv
import io
import re
from pathlib import Path

MAX_ROWS = 50_000
MAX_BYTES = 50_000_000
ALIASES = {"product": "name", "product_name": "name", "item": "name", "item_name": "name", "title": "name",
           "brand_name": "brand", "manufacturer": "brand",
           "type": "category", "product_category": "category",
           "unit_price": "price", "shelf_price": "price", "price_usd": "price",
           "sale_price": "promo_price", "deal_price": "promo_price", "offer_price": "promo_price",
           "size": "pack_size", "pack": "pack_size", "package_size": "pack_size", "quantity": "pack_size",
           "id": "product_id", "sku": "product_id", "shelf_life": "shelf_life_days"}


class UploadError(ValueError):
    """The file cannot be read at all (not one bad row - the whole file)."""


def column_name(raw):
    key = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    return ALIASES.get(key, key)


def decode(data):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UploadError("The file is not UTF-8 or Windows-1252 text. Save it as 'CSV UTF-8' and try again.")


def read_products(path):
    """(header, rows, info) - rows = [(row number, {column: text})], empty lines skipped."""
    path = Path(path)
    if not path.exists():
        raise UploadError(f"No file '{path}'.")
    if path.stat().st_size > MAX_BYTES:
        raise UploadError(f"The file is {path.stat().st_size / 1e6:.0f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB.")
    if path.suffix.lower() in (".xlsx", ".xls", ".ods"):
        raise UploadError("That is a spreadsheet file. In Excel: File > Save As > 'CSV UTF-8 (comma delimited)'.")
    text, encoding = decode(path.read_bytes())
    if not text.strip():
        raise UploadError("The file is empty.")
    sample = text[:20_000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, quotechar='"', doublequote=True)
    try:
        header = next(reader)
    except StopIteration:
        raise UploadError("The file has no header row.")
    columns = [column_name(h) for h in header]
    seen = [c for i, c in enumerate(columns) if c and c in columns[:i]]
    if seen:
        raise UploadError(f"Column '{seen[0]}' appears twice in the header.")
    rows = []
    try:
        for record in reader:
            if not any(cell.strip() for cell in record):
                continue                                         # empty line
            if len(rows) >= MAX_ROWS:
                raise UploadError(f"More than {MAX_ROWS:,} products. Split the file, or keep your best sellers.")
            number = reader.line_num                             # last line of this record (quotes may span lines)
            cells = record + [""] * (len(columns) - len(record))
            row = {c: cells[i].strip() for i, c in enumerate(columns) if c}
            if len(record) > len(columns) and any(x.strip() for x in record[len(columns):]):
                row["__extra__"] = "yes"                         # more cells than headers: a missing quote?
            rows.append((number, row))
    except csv.Error as error:
        raise UploadError(f"The file could not be read near line {reader.line_num}: {error}. "
                          "Check for a missing closing quote (\").")
    info = {"encoding": "UTF-8" if encoding == "utf-8-sig" else "Windows-1252",
            "separator": {",": "comma", ";": "semicolon", "\t": "tab"}[delimiter], "columns": columns}
    return columns, rows, info
