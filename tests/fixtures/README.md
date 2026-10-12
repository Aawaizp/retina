# Test fixtures

`products_200.csv` — 200 real products (25 aisles) in the upload format of `retina --products`,
used by `tests/test_products.py` to run a store end to end with uploaded products.
Made from RETINA's US catalog by `make_products_200.py` (run it again to rebuild the file).

**Attribution and licence.** The product names, brands and pack sizes come from
[Open Food Facts](https://world.openfoodfacts.org) and the prices from
[Open Prices](https://prices.openfoodfacts.org), both © their contributors and made available
under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
This file is a derived database and is shared under the same ODbL 1.0 licence. The popularity
column is ESTIMATED by RETINA (Instacart-based statistics, aggregate only).
