"""Pack sizes as people write them -> one tidy form.

  "500 g" / "500g" / "500 grams"        -> 500 g
  "12 x 330 ml" / "12x330ml" / "12 * 330 mL" -> 12 x 330 ml
  "1.5 L" / "1,5 l" / "2 litres"         -> 1.5 l
  "16 oz" / "16 fl oz" / "2 lb"          -> 16 oz / 16 fl oz / 2 lb
  "6 pack" / "pack of 6" / "6 ct" / "6 pcs" / "each" -> 6 ct ... 1 ct
Returns None when it cannot be read (the checker then keeps the text as given, with a warning).
"""

import re

UNITS = {"g": "g", "gr": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
         "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
         "mg": "mg", "ml": "ml", "mls": "ml", "millilitre": "ml", "milliliter": "ml", "millilitres": "ml",
         "milliliters": "ml", "cl": "cl", "l": "l", "lt": "l", "ltr": "l", "ltrs": "l", "litre": "l", "liter": "l",
         "litres": "l", "liters": "l", "oz": "oz", "ounce": "oz", "ounces": "oz", "floz": "fl oz", "fl oz": "fl oz",
         "fl. oz": "fl oz", "fluid ounce": "fl oz", "fluid ounces": "fl oz", "lb": "lb", "lbs": "lb", "pound": "lb",
         "pounds": "lb", "ct": "ct", "count": "ct", "pc": "ct", "pcs": "ct", "piece": "ct", "pieces": "ct",
         "each": "ct", "ea": "ct", "unit": "ct", "units": "ct", "pack": "ct", "pk": "ct", "packs": "ct",
         "sheets": "ct", "rolls": "ct", "roll": "ct", "bags": "ct", "bag": "ct", "tablets": "ct", "capsules": "ct",
         "gal": "gal", "gallon": "gal", "gallons": "gal", "qt": "qt", "quart": "qt", "quarts": "qt",
         "pt": "pt", "pint": "pt", "pints": "pt", "slice": "ct", "slices": "ct", "pk.": "ct"}
NUM = r"(\d+(?:[.,]\d+)?)"
UNIT = r"(fl\.?\s?oz|fluid ounces?|[a-z]+)"


def number(text):
    return float(text.replace(",", "."))


def unit(text):
    return UNITS.get(re.sub(r"\s+", " ", text.lower().replace("fl.", "fl")).strip())


def fmt(x):
    return f"{x:g}"


def parse_pack(text):
    """-> {"count", "amount", "unit", "text"} or None. Also: "1 lb 11.2 oz" -> 27.2 oz; two sizes
    ("7 oz, 198 g", "750 ml 25.3 fl oz") -> the first one; "40 pk. 16.9oz" -> 40 x 16.9 oz."""
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    if not t:
        return None
    one = parse_one(t)
    if one:
        return one
    m = re.fullmatch(rf"(?:(\d+)\s*[x×*-]\s*)?(\d+)\s*lbs?\.?\s*{NUM}\s*oz\.?", t)       # 1 lb 11.2 oz
    if m:
        n, oz = int(m.group(1) or 1), int(m.group(2)) * 16 + number(m.group(3))
        text = f"{n} x {fmt(oz)} oz" if n > 1 else f"{fmt(oz)} oz"
        return {"count": n, "amount": oz, "unit": "oz", "text": text}
    m = re.fullmatch(rf"(\d+)\s*(?:pk|pack|ct)\.?\s*{NUM}\s*{UNIT}", t)                 # 40 pk. 16.9oz
    if m and unit(m.group(3)):
        n, amount, u = int(m.group(1)), number(m.group(2)), unit(m.group(3))
        return {"count": n, "amount": amount, "unit": u, "text": f"{n} x {fmt(amount)} {u}"}
    for part in re.split(r"[,/;()]", t):                                                # 7 oz, 198 g
        one = parse_one(part.strip())
        if one:
            return one
    m = re.match(rf"{NUM}\s*{UNIT}(?=\s|$)", t)                                         # 750 ml 25.3 fl oz
    if m and unit(m.group(2)):
        return parse_one(m.group(0))
    return None


def parse_one(t):
    if not t:
        return None
    if t in ("each", "ea", "1 each", "single"):
        return {"count": 1, "amount": 1.0, "unit": "ct", "text": "1 ct"}
    m = re.fullmatch(r"(\d+)?\s*dozen", t)                                   # eggs: "1 dozen", "dozen"
    if m:
        n = 12 * int(m.group(1) or 1)
        return {"count": 1, "amount": float(n), "unit": "ct", "text": f"{n} ct"}
    m = re.fullmatch(rf"(?:pack of|case of|box of)\s*(\d+)", t)
    if m:
        return {"count": 1, "amount": float(m.group(1)), "unit": "ct", "text": f"{m.group(1)} ct"}
    m = re.fullmatch(rf"(\d+)\s*[x×*]\s*{NUM}\s*-?\s*{UNIT}", t)           # 12 x 330 ml
    if m and unit(m.group(3)):
        n, amount, u = int(m.group(1)), number(m.group(2)), unit(m.group(3))
        return {"count": n, "amount": amount, "unit": u, "text": f"{n} x {fmt(amount)} {u}"}
    m = re.fullmatch(rf"{NUM}\s*-?\s*{UNIT}\.?", t)                         # 500 g, 6-pack, 2 lbs
    if m and unit(m.group(2)):
        amount, u = number(m.group(1)), unit(m.group(2))
        return {"count": 1, "amount": amount, "unit": u, "text": f"{fmt(amount)} {u}"}
    m = re.fullmatch(rf"{NUM}\s*-?\s*{UNIT}\s*[x×*]\s*(\d+)", t)            # 330 ml x 12
    if m and unit(m.group(2)):
        n, amount, u = int(m.group(3)), number(m.group(1)), unit(m.group(2))
        return {"count": n, "amount": amount, "unit": u, "text": f"{n} x {fmt(amount)} {u}"}
    return None
