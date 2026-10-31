# Public grocery purchase benchmark

The observed records are USDA FoodAPS food purchases from 2012–13. The simulated records are one saved RETINA supercenter run. This is a diagnostic comparison, not validation of a particular retailer or of walking and shelf decisions.

**Sample:** 4,926 paid USDA super-store acquisition events with an observed itemized receipt and item records, from 2,613 households; 301 paid NPC baskets. FoodAPS household survey weights are applied per event. The NPC sample weight of 10 is not applied to per-basket metrics.

| Food per basket | FoodAPS super store | NPC run | NPC minus FoodAPS |
|---|---:|---:|---:|
| SKU lines, mean | 10.21 | 10.12 | -0.09 (-0.9%) |
| SKU lines, p10 | 2.00 | 5.00 | +3.00 (+150.0%) |
| SKU lines, p25 | 3.00 | 7.00 | +4.00 (+133.3%) |
| SKU lines, median | 7.00 | 10.00 | +3.00 (+42.9%) |
| SKU lines, p75 | 14.00 | 13.00 | -1.00 (-7.1%) |
| SKU lines, p90 | 24.00 | 16.00 | -8.00 (-33.3%) |
| Units, mean | 12.74 | 12.96 | +0.22 (+1.7%) |
| Units, p10 | 2.00 | 5.00 | +3.00 (+150.0%) |
| Units, p25 | 4.00 | 8.00 | +4.00 (+100.0%) |
| Units, median | 8.00 | 13.00 | +5.00 (+62.5%) |
| Units, p75 | 17.00 | 17.00 | +0.00 (+0.0%) |
| Units, p90 | 29.00 | 21.00 | -8.00 (-27.6%) |
| Food spend, roughly CPI adjusted to 2025-09 USD, mean | 44.68 | 61.93 | +17.25 (+38.6%) |
| Food spend, roughly CPI adjusted to 2025-09 USD, p10 | 5.84 | 24.21 | +18.37 (+314.6%) |
| Food spend, roughly CPI adjusted to 2025-09 USD, p25 | 13.08 | 39.28 | +26.20 (+200.3%) |
| Food spend, roughly CPI adjusted to 2025-09 USD, median | 27.76 | 57.84 | +30.08 (+108.4%) |
| Food spend, roughly CPI adjusted to 2025-09 USD, p75 | 59.18 | 81.88 | +22.70 (+38.4%) |
| Food spend, roughly CPI adjusted to 2025-09 USD, p90 | 108.11 | 102.93 | -5.18 (-4.8%) |

| Distribution share | FoodAPS super store | NPC run | Difference |
|---|---:|---:|---:|
| 1–3 food SKU lines | 27.22% | 4.98% | -22.24 pp |
| 1–3 food units | 22.43% | 4.32% | -18.11 pp |
| 25+ food units | 14.69% | 3.65% | -11.04 pp |
| single-unit food SKU lines | 84.67% | 78.65% | -6.02 pp |

## Sensitivity and coverage

USDA supermarkets (`PLACETYPE=122`, 4,412 events) have mean/median 9.91/6 food SKU lines and 12.50/8 food units. The small-basket finding is similar to super stores.
The primary USDA filter selected 4,926 of 5,489 super-store events; all selected events had item rows. Food spend used 4,918 complete-price events; 22 of 59,690 item rows lacked expenditure. 16 item rows had a negative net expense and were retained. 10,695 item rows (17.9%) had no usable barcode and were grouped by description. RETINA excluded 102 of 3,147 lines from nonfood, baby, pet and alcohol categories; 0 NPC baskets had no food lines.
Without barcode/description grouping, FoodAPS has 12.05 raw item records per basket (median 8; 1–3-row share 23.73%). This is a sensitivity check, since repeated units can occupy separate rows and raw row counts are not SKU lines.

## Method

The USDA filter is `PLACETYPE=121`, `FREE=0`, `RECEIPTOBSERVED=1`, `TOTALPAID>0`, a positive `HHWGT`, and at least one item row. An observed itemized receipt does not imply every item record came only from that receipt; USDA also integrated scanner and diary records. A USDA SKU line groups repeated item records by barcode within an event; rows without a barcode group by normalized description. Units sum `QUANTITY`. This line grouping is approximate. The synthetic filter includes every paid NPC basket and food categories listed in the JSON report.
Food spending sums `TOTITEMEXP` for USDA events with all item expenses present. Each 2012–13 event is multiplied by the [BLS food-at-home CPI](https://download.bls.gov/pub/time.series/cu/cu.data.0.Current) (`CUUR0000SAF11`) ratio to 2025-09 (index 316.042). Synthetic food `NET_SALES` is adjusted from each simulated basket month to the same target month. This is only a rough price-level comparison: store assortment, geography, household mix, product sizes and catalog price dates differ.

## Limits

FoodAPS is a household survey, not observed sales from this simulated retailer. Filtering to purchases with observed itemized receipts selects a subset, so these event quantiles should not be treated as nationally representative. HHWGT alone does not provide survey-design confidence intervals. FoodAPS covers food acquisitions and provides neither in-store paths nor rejected shelf offers or stockouts. This saved NPC run covers 1 simulated day; the comparison does not establish uncertainty across runs. Purchase-only records cannot teach a shelf SKIP decision directly.

## Sources and reproduction

- [USDA FoodAPS public-use CSV](https://www.ers.usda.gov/media/5594/csv-data-files.zip?v=20317) — local ZIP SHA256 `3cf2450b6190065d844b8a9e53faa9dd97cbdc67f3b70cf75bc6829be26487ba`.
- [USDA food-at-home event codebook](https://www.ers.usda.gov/sites/default/files/images/3_fah-events-codebook-puf.pdf) and [item codebook](https://www.ers.usda.gov/sites/default/files/images/4_fah-items-codebook-puf.pdf).
- [BLS CPI data](https://download.bls.gov/pub/time.series/cu/cu.data.0.Current) — local file SHA256 `70b9070ba38ebfd4cdf434aabd1be798cf2841e77fa940e99da37368950746c2`.
- From the repository root, run `.venv\Scripts\python.exe data\reference\foodaps\compare_saved_run.py`. The ignored USDA ZIP and existing BLS file are required; the command writes only this benchmark JSON and Markdown report.

```powershell
Invoke-WebRequest -Uri 'https://www.ers.usda.gov/media/5594/csv-data-files.zip?v=20317' -OutFile 'data\reference\foodaps\csv-data-files.zip'
.\.venv\Scripts\python.exe data\reference\foodaps\compare_saved_run.py
```
