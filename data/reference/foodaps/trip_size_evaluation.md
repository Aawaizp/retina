# NPC basket-size calibration trial

The optional supercenter trip-size prior uses paid, receipt-observed USDA FoodAPS super-store food purchases (2012–13). Households were split before fitting. The prior supplies a trip's desired SKU-line count; the local list student still chooses products. No remote LLM is called during shopping. The store still enforces stock and budgets.

## Actual checkout comparison

Two isolated days, 671 paid NPC baskets in each run, same settings and one NPC per ten estimated visitors. Only food departments are counted below. Neither run wrote to a saved live run.

| Food SKU lines per paid basket | Baseline student | Calibrated student | FoodAPS comparison households |
|---|---:|---:|---:|
| Mean | 9.52 | 8.11 | 9.51 |
| Median | 9 | 6 | 7 |
| 1–3 lines | 8.05% | 27.42% | 28.50% |
| 25+ lines | 0.00% | 1.64% | 10.75% |

The calibration materially improves small baskets and creates some large baskets, but it shifts the mean low and still misses most large trips. A separate one-day diagnostic found 39 NPCs planning 25+ lines, with 22.1 paid lines on average; they spent about 92% of their trip budgets on average. This points to the budget and product-price relationship as a remaining limit. The FoodAPS comparison households have been consulted during this iteration, so they are no longer an untouched final test set.

**Release decision:** keep `RETINA_TRIP_SIZE_CALIBRATION=1` optional for experiments on the generic supercenter; do not enable it by default or claim the baskets are retailer-validated. For a stronger release, calibrate household trip budgets and food prices against independent purchase data, then test again with fresh households or real store receipts.

Source: [USDA FoodAPS public-use data](https://www.ers.usda.gov/data-products/foodaps-national-household-food-acquisition-and-purchase-survey). FoodAPS is not Walmart transaction data and does not observe in-store routes or shelf skips.
