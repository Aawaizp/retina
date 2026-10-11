"""RETINA web-scraping module: collects product data from the
webscraper.io e-commerce test site and saves it as CSV."""

import csv
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------- settings
BASE_URL = "https://webscraper.io"
HEADERS = {"User-Agent": "RETINA-Scraper/0.1 (learning project)"}

# (category, subcategory, listing path)
CATEGORY_PAGES = [
    ("Computers", "Laptops", "/test-sites/e-commerce/static/computers/laptops"),
    ("Computers", "Tablets", "/test-sites/e-commerce/static/computers/tablets"),
    ("Phones", "Touch", "/test-sites/e-commerce/static/phones/touch"),
]

DELAY_SECONDS = 1.0  # pause between requests so we don't hammer the server
MAX_PAGES = 50       # safety limit in case the stop condition never triggers

SOURCE = "webscraper.io"
CURRENCY = "USD"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "products.csv"

# Final column order of the dataset
FIELDS = [
    "product_id", "product_name", "brand", "category", "subcategory",
    "price", "original_price", "discount_percentage", "currency",
    "availability", "rating", "review_count", "product_url",
    "image_url", "source", "scraped_at",
]

# Keyword found in product name (lowercase) -> brand
BRAND_KEYWORDS = {
    "asus": "Asus", "vivobook": "Asus", "zenbook": "Asus", "rog": "Asus", "memo pad": "Asus",
    "acer": "Acer", "aspire": "Acer", "swift": "Acer", "predator": "Acer", "nitro": "Acer",
    "travelmate": "Acer", "extensa": "Acer", "iconia": "Acer",
    "lenovo": "Lenovo", "thinkpad": "Lenovo", "ideapad": "Lenovo", "ideatab": "Lenovo",
    "yoga": "Lenovo", "legion": "Lenovo",
    "hp": "HP", "packard": "HP", "probook": "HP", "elitebook": "HP", "pavilion": "HP",
    "envy": "HP", "omen": "HP", "spectre": "HP",
    "dell": "Dell", "inspiron": "Dell", "latitude": "Dell", "vostro": "Dell",
    "xps": "Dell", "alienware": "Dell",
    "apple": "Apple", "macbook": "Apple", "ipad": "Apple", "iphone": "Apple",
    "msi": "MSI", "toshiba": "Toshiba", "satellite": "Toshiba",
    "samsung": "Samsung", "galaxy": "Samsung",
    "nokia": "Nokia", "lumia": "Nokia", "sony": "Sony", "xperia": "Sony",
    "lg": "LG", "optimus": "LG", "motorola": "Motorola",
    "huawei": "Huawei", "kindle": "Amazon", "ubuntu": "Canonical",
    "prestigio": "Prestigio", "microsoft": "Microsoft", "surface": "Microsoft",
}


# ----------------------------------------------------------------- helpers
def to_float(text):
    """Pull the first number out of text like '$1,178.99' -> 1178.99."""
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def fetch_soup(session, url, params=None):
    response = session.get(url, params=params, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


# ----------------------------------------------------------------- parsing
def parse_card(card):
    """Turn one product card (a BeautifulSoup tag) into a dict."""
    title_tag = card.select_one("a.title")
    price_tag = card.select_one(".price")
    reviews_tag = card.select_one(".review-count")
    rating_tag = card.select_one("p[data-rating]")
    image_tag = card.select_one("img")

    product_url = urljoin(BASE_URL, title_tag["href"]) if title_tag else None

    if rating_tag is not None:
        rating = to_float(rating_tag.get("data-rating"))
    else:
        rating = float(len(card.select(".ws-icon-star, .glyphicon-star"))) or None

    return {
        "product_id": product_url.rstrip("/").split("/")[-1] if product_url else None,
        "product_name": title_tag.get("title", title_tag.get_text(strip=True)) if title_tag else None,
        "price": to_float(price_tag.get_text()) if price_tag else None,
        "rating": rating,
        "review_count": int(to_float(reviews_tag.get_text()) or 0) if reviews_tag else None,
        "product_url": product_url,
        "image_url": urljoin(BASE_URL, image_tag["src"]) if image_tag else None,
    }


# ---------------------------------------------------------------- crawling
def crawl_category(session, category, subcategory, path):
    """Visit page 1, 2, 3, ... of one subcategory until no new products appear."""
    products = {}  # product_id -> product dict (a dict prevents duplicates)

    for page in range(1, MAX_PAGES + 1):
        soup = fetch_soup(session, BASE_URL + path, params={"page": page})
        cards = soup.select("div.thumbnail")

        new_count = 0
        for card in cards:
            product = parse_card(card)
            pid = product["product_id"]
            if pid and pid not in products:
                product["category"] = category
                product["subcategory"] = subcategory
                products[pid] = product
                new_count += 1

        print(f"  {subcategory} page {page}: {len(cards)} cards, {new_count} new")

        if new_count == 0:  # empty page or a repeat of the last page -> done
            break
        time.sleep(DELAY_SECONDS)

    return list(products.values())


def make_session():
    """Session that retries failed requests up to 3 times (waits 1s, 2s, 4s)."""
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def scrape_all():
    """Crawl every subcategory and return one list of products."""
    all_products = []
    with make_session() as session:
        for category, subcategory, path in CATEGORY_PAGES:
            print(f"\nCrawling {category} > {subcategory}")
            items = crawl_category(session, category, subcategory, path)
            print(f"  -> {len(items)} products")
            all_products.extend(items)
    return all_products


# ------------------------------------------------------- enrich & save
def infer_brand(name):
    """Match whole-word keywords in the name, e.g. 'ThinkPad X240' -> 'Lenovo'."""
    lowered = (name or "").lower()
    for keyword, brand in BRAND_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return brand
    return "Unknown"


def finalize_product(product, scraped_at):
    """Add the remaining fields and return the product in FIELDS order."""
    product["brand"] = infer_brand(product["product_name"])
    product["original_price"] = None        # site shows no pre-discount price
    product["discount_percentage"] = None   # site shows no discounts
    product["currency"] = CURRENCY
    product["availability"] = "in_stock"    # site has no stock field; listed = buyable
    product["source"] = SOURCE
    product["scraped_at"] = scraped_at
    return {field: product.get(field) for field in FIELDS}
OPTIONAL_FIELDS = {"original_price", "discount_percentage"}  # empty on this site


def validate(products):
    """Print a data quality report. Returns True if no problems were found."""
    problems = 0
    print("\nData quality report")

    for field in FIELDS:
        if field in OPTIONAL_FIELDS:
            continue
        missing = sum(1 for p in products if p[field] in (None, ""))
        if missing:
            print(f"  missing {field}: {missing}")
            problems += missing

    ids = [p["product_id"] for p in products]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        print(f"  duplicate product_id: {duplicates}")
        problems += duplicates

    bad_prices = [p["product_id"] for p in products if not p["price"] or p["price"] <= 0]
    if bad_prices:
        print(f"  invalid price: {bad_prices}")
        problems += len(bad_prices)

    bad_ratings = [p["product_id"] for p in products
                   if p["rating"] is not None and not 0 <= p["rating"] <= 5]
    if bad_ratings:
        print(f"  rating outside 0-5: {bad_ratings}")
        problems += len(bad_ratings)

    prices = [p["price"] for p in products if p["price"]]
    if prices:
        print(f"  price range: {min(prices)} - {max(prices)} {CURRENCY}")

    print("  OK: no problems found" if problems == 0 else f"  {problems} problem(s) found")
    return problems == 0

def save_csv(products, path=OUTPUT_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(products)


# -------------------------------------------------------------------- main
def main():
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    products = [finalize_product(p, scraped_at) for p in scrape_all()]
    validate(products)
    save_csv(products)

    print(f"\nTotal products: {len(products)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Brands:", dict(Counter(p["brand"] for p in products).most_common()))

    unknown = [p["product_name"] for p in products if p["brand"] == "Unknown"]
    if unknown:
        print(f"Unknown brand ({len(unknown)}):", unknown)


if __name__ == "__main__":
    main()