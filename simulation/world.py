"""RETINA Mart world: load the data layer and build the store, shoppers, prices and stock."""

import json
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import *

# =========================================================== data loading
def load_inputs():
    """Read the 4 files the simulator needs from the data layer."""
    catalog = pd.read_csv(CATALOG_FILE, dtype={"product_id": str})
    cpi = pd.read_csv(US_CPI_FILE, parse_dates=["date"])
    with open(SHOPPER_PARAMS_FILE, encoding="utf-8") as f:
        params = json.load(f)
    with open(DEPARTMENTS_FILE, encoding="utf-8") as f:
        departments = json.load(f)["departments"]
    return catalog, cpi, params, departments


# ================================================================== store
@dataclass
class Department:
    name: str
    share: float              # share of all items bought here (real Instacart share)
    product_idx: np.ndarray   # row numbers of the products on this department's shelves
    pick_prob: np.ndarray     # chance of picking each of those products (popularity-weighted)
    cum_prob: np.ndarray      # running total of pick_prob (for fast random picks)
    aisle_names: list = None  # departments only: their aisles (substitute groups), fixed order
    aisle_cum: np.ndarray = None  # running total of the aisles' shares of the department

@dataclass
class Store:
    products: pd.DataFrame    # one row per product: names, prices, CPI code, popularity
    departments: dict         # department name -> Department
    dept_names: list          # department names, in a fixed order
    dept_shares: np.ndarray   # department shares in the same order (they add up to 1)
    partners: dict = None     # product -> (partner products, running total of extra chance)
    aisles: dict = None       # aisle name -> its products (substitute group)
    aisle_of: np.ndarray = None   # aisle name of every product
    dept_cum: np.ndarray = None   # running total of department shares

def build_store(catalog, departments):
    """Turn the catalog into shelves: which products each department sells, and how likely
    a shopper is to pick each one."""
    products = catalog.reset_index(drop=True).copy()
    missing = [c for c in CATALOG_LABEL_COLUMNS if c not in products.columns]
    if missing:   # never guess labels: unlabelled data could be mistaken for real data
        raise ValueError(f"Catalog has no data labels ({', '.join(missing)}). Run add_data_labels in loader.py.")
    # Regular shelf price = price without discount. A recorded discount means the product
    # is "promo-eligible": it sometimes goes on sale at its observed discounted price.
    products["regular_price"] = products["original_price"]
    products["promo_price"] = products["price"]
    products["promo_eligible"] = products["discount_percentage"].fillna(0) > 0
    products["promo_ratio"] = (products["promo_price"] / products["regular_price"]).where(
        products["promo_eligible"], 1.0)   # e.g. 0.8 = 20% off when on promotion
    # Cost the store pays per unit = regular price x (1 - its department's gross margin)
    margin = products["department"].map(GROSS_MARGIN).fillna(DEFAULT_MARGIN)
    products["unit_cost"] = (products["regular_price"] * (1 - margin)).round(2)

    depts = {}
    for name, info in departments.items():
        idx = np.flatnonzero((products["department"] == name).to_numpy())
        weights = products["popularity"].to_numpy()[idx]
        depts[name] = Department(
            name=name,
            share=info["share"],
            product_idx=idx,
            pick_prob=weights / weights.sum(),
            cum_prob=np.cumsum(weights / weights.sum()),
        )

    names = list(depts)
    shares = np.array([depts[n].share for n in names])
    return Store(products=products, departments=depts, dept_names=names,
                 dept_shares=shares / shares.sum())


def summarize_store(store):
    p = store.products
    print(f"RETINA Mart: {len(p):,} products | {len(store.departments)} departments")
    print(f"  regular price: median ${p['regular_price'].median():.2f} | "
          f"range ${p['regular_price'].min():.2f} - ${p['regular_price'].max():.2f}")
    print(f"  promo-eligible products: {p['promo_eligible'].sum():,} "
          f"({p['promo_eligible'].mean() * 100:.0f}%)")
    print(f"  department shares add up to: {store.dept_shares.sum():.3f}")

    print("\n  department        share  products  most likely pick")
    for name, share in sorted(zip(store.dept_names, store.dept_shares), key=lambda x: -x[1]):
        d = store.departments[name]
        top = p["product_name"].iat[d.product_idx[d.pick_prob.argmax()]]
        print(f"  {name:<16} {share * 100:5.1f}%  {len(d.product_idx):>8,}  {top[:40]}")


# =============================================================== shoppers
def sample_from(distribution, size, rng):
    """Draw random values from a {value: share} table, e.g. basket sizes {"8": 0.07, ...}."""
    values = np.array([float(v) for v in distribution.keys()])
    shares = np.array(list(distribution.values()), dtype=float)
    return rng.choice(values, size=size, p=shares / shares.sum())


@dataclass
class Shoppers:
    """All NPC shoppers, stored as columns (one array per habit) instead of one object per
    shopper. Same idea as a table: row i = shopper i. Much faster for thousands of NPCs."""
    shopper_id: np.ndarray         # 0, 1, 2, ...
    basket_mean: np.ndarray        # usual number of items per trip (real Instacart profiles)
    gap_mean: np.ndarray           # usual days between trips (real Instacart profiles)
    reorder_prob: np.ndarray       # chance an item is a repeat purchase (real profiles)
    price_sensitivity: np.ndarray  # ASSUMPTION: 1.0 = average reaction to price
    
    next_visit_day: np.ndarray     # simulation day of the next trip
    favorite_weekday: np.ndarray   # 0 = Monday ... 6 = Sunday (drawn from real Instacart days)
    sensitivity_group: np.ndarray  # 0 = least price-sensitive group ... 2 = most
    favorites: list                # per shopper: products bought before (filled while shopping)
    pantry: list                   # per shopper: {product: day their stocked-up supply runs out}
    favorite_cum: list = None      # per shopper: running total of re-buy weights of `favorites`
    home: list = None              # per shopper: {product: day it runs out at home}
    counts: list = None            # per shopper: {product: times bought}
    by_aisle: list = None          # per shopper: {aisle: products bought there}
    last_trip_day: np.ndarray = None   # day of each shopper's last trip (-1 = none yet)
    household_size: np.ndarray = None  # people the NPC shops for (households.py)
    kids: np.ndarray = None            # children at home
    income_quintile: np.ndarray = None # 0 = lowest fifth of incomes ... 4 = highest
    weekly_budget: np.ndarray = None   # food-at-home budget per week, dollars

def plan_visit(planned_day, earliest_day, favorite_weekday, rng):
    """Habit: most of the time, move a planned trip to the closest day that falls on the
    shopper's favourite weekday (at most 3 days earlier or later). If that day is too soon
    (frequent shoppers), keep the planned day. Moving both ways keeps the average gap the same."""
    if rng.random() >= WEEKDAY_HABIT:
        return planned_day
    shift = (favorite_weekday - (START_WEEKDAY + planned_day)) % 7   # 0..6 days later
    if shift > 3:
        shift -= 7                                                    # closer to go earlier
    new_day = planned_day + shift
    return new_day if new_day >= earliest_day else planned_day

def create_shoppers(params, n, rng):
    """Create n NPCs. Each NPC copies ONE real Instacart shopper's habits together - basket size,
    days between trips and re-buy rate - because they are linked (frequent shoppers re-buy more).
    Without the profiles file: each habit drawn separately from the real distributions."""
    if SHOPPER_PROFILES_FILE.exists():
        real = pd.read_csv(SHOPPER_PROFILES_FILE)
        pick = rng.choice(len(real), n, replace=True)
        basket_mean = np.clip(real["basket_size"].to_numpy()[pick], 1, 50)
        gap_mean = np.maximum(1, real["days_between_orders"].to_numpy()[pick])
        reorder_prob = real["rebuy_rate"].to_numpy()[pick]
    else:
        prof = params["shopper_profiles"]
        basket_mean = np.maximum(1, sample_from(prof["avg_basket_size"], n, rng))
        gap_mean = np.maximum(1, sample_from(prof["avg_days_between_orders"], n, rng))
        reorder_prob = sample_from(prof["reorder_pct"], n, rng) / 100
    price_sensitivity = rng.lognormal(mean=0.0, sigma=PRICE_SENSITIVITY_SPREAD, size=n)
    # Favourite shopping weekday, drawn from the real share of orders per weekday
    instacart_day = sample_from(params["orders_by_day_of_week"], n, rng).astype(int)
    favorite_weekday = (instacart_day + INSTACART_DAY0) % 7
    # First trip: somewhere within each shopper's usual gap, then moved to their favourite day
    first_day = (rng.random(n) * gap_mean).astype(int)
    next_visit_day = np.array([plan_visit(d, 0, w, rng) for d, w in zip(first_day, favorite_weekday)])
    # Split NPCs into equal-sized low / medium / high price-sensitivity groups
    cut_points = np.quantile(price_sensitivity, np.linspace(0, 1, SENSITIVITY_GROUPS + 1)[1:-1])
    sensitivity_group = np.searchsorted(cut_points, price_sensitivity)
    return Shoppers(
        shopper_id=np.arange(n),
        basket_mean=basket_mean,
        gap_mean=gap_mean,
        reorder_prob=reorder_prob,
        price_sensitivity=price_sensitivity,
        next_visit_day=next_visit_day,
        favorite_weekday=favorite_weekday,
        sensitivity_group=sensitivity_group,
        favorites=[[] for _ in range(n)],
        pantry=[{} for _ in range(n)],
        favorite_cum=[[] for _ in range(n)],
        home=[{} for _ in range(n)],
        counts=[{} for _ in range(n)],
        by_aisle=[{} for _ in range(n)],
        last_trip_day=np.full(n, -1),
    )


def summarize_shoppers(shoppers, params):
    prof = params["shopper_profiles"]

    def real_mean(dist):
        return sum(float(k) * v for k, v in dist.items())

    print(f"\nNPC shoppers: {len(shoppers.shopper_id):,}")
    print("  habit                     NPC average   real average (Instacart profiles)")
    print(f"  items per trip            {shoppers.basket_mean.mean():11.1f}   "
          f"{real_mean(prof['avg_basket_size']):11.1f}")
    print(f"  days between trips        {shoppers.gap_mean.mean():11.1f}   "
          f"{real_mean(prof['avg_days_between_orders']):11.1f}")
    print(f"  repeat-purchase chance    {shoppers.reorder_prob.mean() * 100:10.0f}%   "
          f"{real_mean(prof['reorder_pct']):10.0f}%")
    print(f"  price sensitivity         {shoppers.price_sensitivity.mean():11.2f}   (assumption)")
    print(f"  trips expected per week:  {(7 / shoppers.gap_mean).sum():,.0f}")
    print("\n  5 example NPCs:")
    for i in range(5):
        print(f"    NPC {i}: ~{shoppers.basket_mean[i]:.0f} items every ~{shoppers.gap_mean[i]:.0f} days, "
              f"{shoppers.reorder_prob[i] * 100:.0f}% repeats, price sensitivity "
              f"{shoppers.price_sensitivity[i]:.2f}, first visit on day {shoppers.next_visit_day[i]}")


# ================================================================= prices
@dataclass
class PriceBook:
    """Every product's price on every simulated day: regular price x real inflation,
    and a weekly promotion calendar."""
    dates: pd.DatetimeIndex     # one entry per simulated day
    month_of_day: np.ndarray    # which month (0, 1, 2, ...) each day falls in
    week_of_day: np.ndarray     # which week each day falls in
    inflation: np.ndarray       # [month, product] price multiplier from real CPI (1.03 = +3%)
    on_promo: np.ndarray        # [week, product] True if the product is on sale that week


def build_price_book(store, cpi, n_days, rng):
    """Inflation: each product's price moves with its own BLS CPI category, measured from
    the month its real price was observed. Promotions: random weeks (ASSUMPTION frequency),
    at the real observed discount depth."""
    p = store.products
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")
    months = dates.to_period("M")
    sim_months = months.unique()

    index = cpi.assign(month=cpi["date"].dt.to_period("M")).pivot_table(
        index="month", columns="item_code", values="index")
    index = index.reindex(index.index.union(sim_months)).ffill()   # after last CPI month: hold

    # Month each product's price was observed (clipped to the CPI range)
    observed = pd.to_datetime(p["price_date"], errors="coerce").dt.to_period("M")
    observed = observed.fillna(index.index.max()).clip(index.index.min(), index.index.max())

    codes = p["cpi_item_code"].where(p["cpi_item_code"].isin(index.columns), "SA0")
    base = index.stack().reindex(list(zip(observed, codes))).to_numpy()
    inflation = np.empty((len(sim_months), len(p)))
    for m, month in enumerate(sim_months):
        inflation[m] = index.loc[month].reindex(codes).to_numpy() / base
    inflation = np.nan_to_num(inflation, nan=1.0)

    n_weeks = n_days // 7 + 1
    on_promo = (rng.random((n_weeks, len(p))) < PROMO_WEEKLY_PROB) & p["promo_eligible"].to_numpy()

    return PriceBook(dates=dates,
                     month_of_day=np.searchsorted(sim_months, months),
                     week_of_day=np.arange(n_days) // 7,
                     inflation=inflation,
                     on_promo=on_promo)


def prices_on(day, store, book):
    """Shelf price and promo flag for every product on a given day."""
    p = store.products
    promo = book.on_promo[book.week_of_day[day]]
    ratio = np.where(promo, p["promo_ratio"].to_numpy(), 1.0)
    regular = p["regular_price"].to_numpy() * book.inflation[book.month_of_day[day]]
    return (regular * ratio).round(2), regular.round(2), promo


def choice_tables(store, vs_normal, vs_memory, group_sensitivity, available=None):
    """This week's price-aware pick probabilities, per aisle, for each price-sensitivity group.
    Two price effects (together = PRICE_ELASTICITY):
      - permanent: vs_normal = price / normal price (without any experiment or promotion)
      - surprise:  vs_memory = price / the price shoppers remember; fades as memory adapts, and
                   prices ABOVE memory ('losses') count LOSS_AVERSION times more than gains.
    vs_normal / vs_memory hold one array per group (a targeted deal is seen by one group only);
    available = False for products taken off the shelves (weight 0).
    weight = popularity x boost. 'inclusive' = how attractive the whole aisle feels vs normal
    (1.0 = normal); below 1 some shoppers walk away from the aisle (see fill_basket)."""
    e_memory = PRICE_ELASTICITY * REFERENCE_SHARE
    e_permanent = PRICE_ELASTICITY - e_memory
    tables = []
    for g, sens in enumerate(group_sensitivity):
        boost = (vs_normal[g] ** (-e_permanent * sens)
                 * vs_memory[g] ** (-e_memory * sens * np.where(vs_memory[g] > 1, LOSS_AVERSION, 1.0)))
        if available is not None:
            boost = boost * available
        cum, inclusive = {}, {}
        for name, a in store.aisles.items():
            w = a.pick_prob * boost[a.product_idx]
            total = w.sum()                         # pick_prob adds to 1, so this is 1.0 at normal prices
            cum[name] = np.cumsum(w / total) if total > 0 else None   # None = aisle emptied
            inclusive[name] = total
        tables.append({"cum": cum, "inclusive": inclusive, "boost": boost})
    return tables
# ================================================================= stock
@dataclass
class Inventory:
    """Units on the shelf for every product, plus the store's ordering rules."""
    stock: np.ndarray          # units on the shelf now
    forecast: np.ndarray       # the store's estimate of daily demand (units per day)
    reorder_point: np.ndarray  # s: when stock falls to this level, place an order
    order_up_to: np.ndarray    # S: order enough to bring stock back up to this level
    order_days: np.ndarray     # days of demand per order (fresh items: smaller, more often)
    on_order: np.ndarray       # units ordered but not yet delivered
    deliveries: dict           # {arrival day: list of (products, units)} - orders on their way
    sellable: np.ndarray       # True for products in the store's departments
    department: np.ndarray     # department of each product
    brand: np.ndarray          # lower-case brand ("" if unknown), for same-brand switching
    shelf_life: np.ndarray     # days a unit stays sellable (0 = never spoils)
    batches: dict              # fresh products: {product: deque of [expiry_day, units]}, oldest first
    safety_days: np.ndarray = None   # stock buffer in days of demand, per product

def set_order_levels(inv):
    """(s, S) from the forecast: s covers lead time + safety days, S adds one order's worth."""
    inv.reorder_point = np.ceil(inv.forecast * (LEAD_TIME_DAYS + inv.safety_days)).astype(int)
    inv.order_up_to = inv.reorder_point + np.maximum(1, np.ceil(inv.forecast * inv.order_days)).astype(int)


def expected_daily_demand(store, shoppers):
    """Units per day each product should sell: expected items per day x department share x
    popularity within the department."""
    items_per_day = (shoppers.basket_mean / shoppers.gap_mean).sum()
    demand = np.zeros(len(store.products))
    for name, share in zip(store.dept_names, store.dept_shares):
        d = store.departments[name]
        demand[d.product_idx] = items_per_day * share * d.pick_prob
    return demand


def plan_assortment(store, shoppers):
    """Fresh departments only stock products that sell fast enough not to rot; popularity is
    re-spread over the products that stay. Long-life products are all kept."""
    demand = expected_daily_demand(store, shoppers)
    print("\nAssortment: fresh products kept (expected to sell "
          f">= {FRESH_MIN_SALES} units within their shelf life)")
    for name, life in SHELF_LIFE_DAYS.items():
        d = store.departments.get(name)
        if d is None:
            continue
        keep = demand[d.product_idx] * life >= FRESH_MIN_SALES
        keep[np.argmax(d.pick_prob)] = True                  # always keep the top seller
        w = d.pick_prob[keep]
        print(f"  {name:<14} {keep.sum():>5,} of {len(keep):>5,} products | "
              f"{w.sum() * 100:5.1f}% of the department's demand")
        d.product_idx = d.product_idx[keep]
        d.pick_prob = w / w.sum()
        d.cum_prob = np.cumsum(d.pick_prob)


def build_aisles(store):
    """Split every department into aisles (substitute groups). A shopper first picks an aisle -
    by its share of the department's popularity, which prices never change - and then a product
    inside it. So a price change moves shoppers between SIMILAR products (Coke -> Pepsi), and a
    random draw that lands in another aisle is never disturbed by it."""
    p = store.products
    popularity = p["popularity"].to_numpy()
    store.aisle_of = p["aisle"].to_numpy()
    store.aisles = {}
    # Real share of each aisle inside its department (Instacart order lines) when available;
    # otherwise the summed popularity of the products we carry
    stats = pd.read_csv(AISLE_STATS_FILE).set_index("aisle") if AISLE_STATS_FILE.exists() else None
    # NEW picks follow the mix of real FIRST-TIME purchases (re-buys come from favourites, see
    # fill_basket): lines that were not re-bought
    real = (stats["lines"] - stats["rebought"]) if stats is not None else None
    for d in store.departments.values():
        names, weights = [], []
        for aisle in pd.unique(store.aisle_of[d.product_idx]):
            idx = d.product_idx[store.aisle_of[d.product_idx] == aisle]
            w = popularity[idx]
            store.aisles[aisle] = Department(name=aisle, share=w.sum(), product_idx=idx,
                                             pick_prob=w / w.sum(), cum_prob=np.cumsum(w / w.sum()))
            names.append(aisle)
            weights.append(float(real.get(aisle, 0)) if real is not None else w.sum())
        if sum(weights) == 0:                           # no real figures for this department
            weights = [store.aisles[a].share for a in names]
        d.aisle_names = names
        d.aisle_cum = np.cumsum(np.array(weights) / sum(weights))
    store.dept_cum = np.cumsum(store.dept_shares)
    store.product_rebuy = np.full(len(p), 0.5)          # how often real shoppers re-buy each product's aisle
    if stats is not None:
        rebuy = stats["rebuy_rate"]
        first = stats["lines"] * (1 - rebuy)            # first-time purchases per aisle
        # department mix of first-time purchases: overall share x (first-time / all) in that department
        share = []
        for name, s in zip(store.dept_names, store.dept_shares):
            aisles = [a for a in store.departments[name].aisle_names if a in stats.index]
            lines = stats.loc[aisles, "lines"].sum()
            share.append(s * (first[aisles].sum() / lines if lines else 1.0))
        store.dept_cum = np.cumsum(np.array(share) / sum(share))
        for a, d in store.aisles.items():
            if a in rebuy.index:
                store.product_rebuy[d.product_idx] = rebuy[a]
    # how many trips a product lasts at home before the same shopper buys it again (real, per aisle)
    store.product_cycle = np.full(len(p), 3.5)
    if AISLE_CYCLES_FILE.exists():
        cycles = pd.read_csv(AISLE_CYCLES_FILE).set_index("aisle")["trips_mean"]
        store.product_cycle = np.full(len(p), float(cycles.median()))
        for a, d in store.aisles.items():
            if a in cycles.index:
                store.product_cycle[d.product_idx] = cycles[a]
    print(f"\nAisles: {len(store.aisles)} substitute groups across {len(store.departments)} departments"
          + (" | aisle shares: real Instacart order lines" if real is not None else ""))

def add_favorites(store, shoppers, s, products):
    """Remember what a shopper bought. Which AISLE they re-buy from follows the aisle's real re-buy
    rate x how often they bought there (so the overall mix stays real); inside an aisle, the
    product they buy again and again (their milk, their yogurt) becomes a habit - see repeat_pick."""
    fav, cum, counts = shoppers.favorites[s], shoppers.favorite_cum[s], shoppers.counts[s]
    by_aisle = shoppers.by_aisle[s]
    for product in products:
        k = counts.get(product, 0) + 1
        counts[product] = k
        if k == 1:
            by_aisle.setdefault(store.aisle_of[product], []).append(product)
        fav.append(product)
        cum.append((cum[-1] if cum else 0.0) + store.product_rebuy[product])


def habit_aisle(store, shoppers, s, u):
    """The aisle of a usual item: aisles they bought from more often, and aisles real shoppers
    re-buy more (milk far more than spices), are chosen more often. Keeps the overall mix real."""
    fav, cum = shoppers.favorites[s], shoppers.favorite_cum[s]
    return shoppers.by_aisle[s][store.aisle_of[fav[min(bisect_right(cum, u * cum[-1]), len(cum) - 1)]]]


def habit_product(shoppers, s, group, u):
    """Inside the aisle: the product bought again and again (their milk) - weight = times
    bought ^ habit_power."""
    if len(group) == 1:
        return group[0]
    counts = shoppers.counts[s]
    w = [counts[p] ** HABIT_POWER for p in group]
    x = u * sum(w)
    for p, wp in zip(group, w):
        x -= wp
        if x <= 0:
            return p
    return group[-1]


def repeat_pick(store, shoppers, s, u_aisle, u_product):
    return habit_product(shoppers, s, habit_aisle(store, shoppers, s, u_aisle), u_product)


def seed_history(store, shoppers, rng):
    """Shoppers arrive with a shopping history, like real ones: each NPC lives through a few past
    trips before day 0. On each past trip some items are re-buys of what they bought before (their
    real re-buy rate; habits grow, see add_favorites) and the rest are new, drawn from what real
    shoppers RE-BUY (aisle mix of re-bought lines, products by popularity). Without it every NPC
    would start the year as a brand-new customer."""
    trips = HISTORY_TRIPS
    if trips <= 0 or not AISLE_STATS_FILE.exists():
        return
    stats = pd.read_csv(AISLE_STATS_FILE).set_index("aisle")
    names = [a for a in store.aisles if a in stats.index]
    w = stats.loc[names, "rebought"].to_numpy(float)
    cum = np.cumsum(w / w.sum())
    for s in range(len(shoppers.shopper_id)):
        k = max(1, int(round(shoppers.basket_mean[s])))
        fav, fc, home = shoppers.favorites[s], shoppers.favorite_cum[s], shoppers.home[s]
        for t in range(trips):
            bought = []
            for ur, ua, up, uh in rng.random((k, 4)):
                if fav and ur < shoppers.reorder_prob[s]:          # a re-buy (habits grow)
                    product = repeat_pick(store, shoppers, s, ua, up)
                else:
                    aisle = store.aisles[names[pick_index(cum, ua)]]
                    product = aisle.product_idx[min(int(np.searchsorted(aisle.cum_prob, up)),
                                                    len(aisle.product_idx) - 1)]
                bought.append(product)
                # bought (trips - t) trips ago: some of it may still be at home on day 0
                lasts = HOME_FACTOR * store.product_cycle[product] * shoppers.gap_mean[s] * (0.5 + uh)
                home[product] = max(home.get(product, -1), int(lasts - (trips - t) * shoppers.gap_mean[s]))
            add_favorites(store, shoppers, s, bought)


def build_inventory(store, shoppers, rng):
    """The store forecasts each product's daily demand (expected items per day x department
    share x popularity), with a random forecast error, and sets its (s, S) levels from that.
    Shelves start full; fresh stock starts at random ages. Promotions are NOT in the forecast."""
    p = store.products
    forecast = expected_daily_demand(store, shoppers) * rng.lognormal(0.0, FORECAST_ERROR, len(p))
    sellable = np.zeros(len(p), dtype=bool)
    for d in store.departments.values():
        sellable[d.product_idx] = True
    shelf_life = p["department"].map(SHELF_LIFE_DAYS).fillna(0).astype(int).to_numpy()
    # Fresh items are ordered every day, one day of demand at a time (like real stores)
    order_days = np.where(shelf_life > 0, 1, ORDER_DAYS)
    inv = Inventory(
        stock=np.zeros(len(p), dtype=int), forecast=forecast,
        reorder_point=None, order_up_to=None, order_days=order_days,
        on_order=np.zeros(len(p), dtype=int), deliveries={},
        sellable=sellable,
        department=p["department"].to_numpy(),
        brand=p["brand"].fillna("").astype(str).str.strip().str.lower().to_numpy(),
        shelf_life=shelf_life, batches={}, safety_days=np.full(len(p), float(SAFETY_DAYS)),
    )
    set_order_levels(inv)
    inv.stock = inv.order_up_to.copy()
    for i in np.flatnonzero(sellable & (shelf_life > 0)):
        inv.batches[i] = deque([[int(rng.integers(1, shelf_life[i] + 1)), int(inv.stock[i])]])
    return inv


def update_forecast(inv, units_sold_last_week):
    """Weekly: blend the old forecast with last week's sales (exponential smoothing) and reset
    the (s, S) levels. The store only sees SALES, not the demand lost to empty shelves, so items
    that often run out get under-forecast - a real retail problem called 'censored demand'."""
    inv.forecast = ((1 - FORECAST_LEARNING) * inv.forecast
                    + FORECAST_LEARNING * units_sold_last_week / 7)
    set_order_levels(inv)


def remove_spoiled(day, inv):
    """Morning: fresh units past their shelf life are thrown away. Returns {product: units}."""
    spoiled = {}
    for product, batches in inv.batches.items():
        while batches and batches[0][0] <= day:
            _, units = batches.popleft()
            inv.stock[product] -= units
            spoiled[product] = spoiled.get(product, 0) + units
    return spoiled


def receive_deliveries(day, inv):
    """Morning: orders due today are put on the shelves (fresh ones get an expiry date)."""
    for products, units in inv.deliveries.pop(day, []):
        inv.stock[products] += units
        inv.on_order[products] -= units
        for i, u in zip(products, units):
            if i in inv.batches:
                inv.batches[i].append([day + inv.shelf_life[i], int(u)])


def place_orders(day, inv, rng):
    """Evening: the store counts stock on the shelf + units already on the way, minus fresh
    units that will expire before a new delivery could arrive. Every product at or below its
    reorder point is ordered back up to its order-up-to level.
    Deliveries take LEAD_TIME_DAYS; some come late (supplier problems)."""
    position = inv.stock + inv.on_order
    for product, batches in inv.batches.items():
        for expiry, units in batches:
            if expiry > day + LEAD_TIME_DAYS:
                break                                 # batches are oldest first
            position[product] -= units
    need = np.flatnonzero(inv.sellable & (position <= inv.reorder_point))
    units = inv.order_up_to[need] - position[need]
    inv.on_order[need] += units
        # Draw late/extra days for EVERY product, then use only the ordered ones, so one product's
    # delivery luck does not depend on how many other products were ordered today
    late = (rng.random(len(inv.stock)) < SUPPLIER_LATE_PROB)[need]
    extra = rng.integers(1, SUPPLIER_LATE_DAYS + 1, len(inv.stock))[need]
    arrival = day + LEAD_TIME_DAYS + np.where(late, extra, 0)
    for a in np.unique(arrival):
        mask = arrival == a
        inv.deliveries.setdefault(int(a), []).append((need[mask], units[mask]))


def take_unit(product, inv):
    """Remove one unit from the shelf. Fresh items: shoppers get the oldest batch (FIFO)."""
    inv.stock[product] -= 1
    batches = inv.batches.get(product)
    if batches:
        batches[0][1] -= 1
        if batches[0][1] == 0:
            batches.popleft()


def pick_index(cum_prob, u):
    """Position in a running total for a random number u (0-1)."""
    return min(int(np.searchsorted(cum_prob, u * cum_prob[-1], side="right")), len(cum_prob) - 1)


OOS_NAMES = list(OOS_REACTIONS)
OOS_CUM = np.cumsum(list(OOS_REACTIONS.values()))


def take_from_shelf(wanted, store, inv, u_reaction, u_substitute):
    """Shopper reaches for `wanted`. In stock: take one. Empty shelf: react like real shoppers
    (Gruen 2002) - switch to another product of the same or another brand (first in the SAME
    AISLE, else the department), or leave without it. Returns (product taken or -1, reaction)."""
    if inv.stock[wanted] > 0:
        take_unit(wanted, inv)
        return wanted, None
    reaction = OOS_NAMES[pick_index(OOS_CUM, u_reaction)]
    if reaction not in ("same_brand", "other_brand"):
        return -1, reaction                                   # lost sale for this store
    for group in (store.aisles[store.aisle_of[wanted]], store.departments[inv.department[wanted]]):
        idx = group.product_idx
        ok = (inv.stock[idx] > 0) & (idx != wanted)
        same = (inv.brand[idx] == inv.brand[wanted]) & (inv.brand[wanted] != "")
        preferred = ok & same if reaction == "same_brand" else ok & ~same
        if preferred.any():
            ok = preferred                                    # else: any product in stock
        if ok.any():
            break
    else:
        return -1, reaction                                   # nothing to switch to
    product = idx[pick_index(np.cumsum(group.pick_prob * ok), u_substitute)]
    take_unit(product, inv)
    return product, reaction