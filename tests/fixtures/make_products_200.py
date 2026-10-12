"""Make tests/fixtures/products_200.csv: 200 real products from RETINA's US catalog, written the way
a user would upload them (name, brand, category, price, promo_price, pack_size, popularity).

25 aisles across the departments (fresh and shelf-stable), best sellers with a known brand and pack
size, taken rank by rank across the aisles (about 8 each) until there are 200. Popularity is given for every other row (the rest gets the aisle default), promo_price
where the catalog saw a discount. Product names, brands, sizes: Open Food Facts; prices: Open Prices
- both ODbL 1.0 (see README.md). Run:  python tests/fixtures/make_products_200.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AISLES = ["milk", "yogurt", "eggs", "packaged cheese", "butter", "fresh fruits", "fresh vegetables",
          "packaged vegetables fruits", "bread", "lunch meat", "chips pretzels", "cookies cakes", "crackers",
          "candy chocolate", "cereal", "dry pasta", "pasta sauce", "soup broth bouillon", "coffee", "tea",
          "soft drinks", "water seltzer sparkling water", "frozen pizza", "ice cream ice", "paper goods"]


def main():
    c = pd.read_csv(ROOT / "data" / "processed" / "catalog_us_clean.csv", dtype={"product_id": str}, low_memory=False)
    c = c[c["aisle"].isin(AISLES) & c["pack_size"].notna() & c["brand"].notna()]
    c = c.sort_values(["aisle", "popularity", "product_id"], ascending=[True, False, True])
    c = c.assign(rank=c.groupby("aisle").cumcount())             # 0 = the aisle's best seller
    picks = c.sort_values(["rank", "aisle"]).head(200).sort_values(["aisle", "rank"]).reset_index(drop=True)
    out = pd.DataFrame({
        "name": picks["product_name"], "brand": picks["brand"], "category": picks["aisle"],
        "price": picks["original_price"].round(2),
        "promo_price": picks["price"].where(picks["discount_percentage"] > 0).round(2),
        "pack_size": picks["pack_size"],
        "popularity": picks["popularity"].round(1).where(picks.index % 2 == 0),
    })
    assert len(out) == 200, len(out)
    out.to_csv(Path(__file__).parent / "products_200.csv", index=False, encoding="utf-8")
    print(f"wrote {len(out)} products in {out['category'].nunique()} aisles")


if __name__ == "__main__":
    main()
