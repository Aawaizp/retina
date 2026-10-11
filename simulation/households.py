"""RETINA households: who each NPC shops for, their income and their food budget.

Every NPC already copies one real Instacart shopper's habits (basket size, days between trips,
re-buy rate). This adds the household behind those habits:
  household size   PUBLIC_STATISTIC  EIA Residential Energy Consumption Survey 2024 (132.5 million homes)
  children         ESTIMATED         from BLS Consumer Expenditure Survey 2022 (children per household size)
  income quintile  PUBLIC_STATISTIC  20% of homes in each fifth, by definition
  food budget      PUBLIC_STATISTIC  BLS CE 2022 food-at-home spending by household size and income,
                                     moved to today's prices with the real CPI for food at home
Links between them are ESTIMATED: bigger baskets -> bigger households, bigger households ->
somewhat higher income, lower income -> more price-sensitive (the same sensitivity values as
before, only handed to different NPCs, so the store-wide price response does not change).
"""

import numpy as np
import pandas as pd

from config import HOUSEHOLDS, US_CPI_FILE, START_DATE, SENSITIVITY_GROUPS

CE_PERSONS_5PLUS = 5.7          # BLS CE 2022: average size of "five or more" households


def normal_scores(values, rng):
    """Rank-based normal scores (ties broken randomly) - the 'Gaussian copula' trick."""
    order = np.lexsort((rng.random(len(values)), values))
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    from statistics import NormalDist
    inv = NormalDist().inv_cdf
    return np.array([inv((r + 0.5) / len(values)) for r in ranks])


def linked(z, rho, rng):
    """A new normal score correlated rho with z."""
    return rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(z))


def by_shares(z, shares):
    """Cut scores into categories with the given shares (lowest scores -> first category)."""
    cut = np.quantile(z, np.cumsum(shares)[:-1] / np.sum(shares))
    return np.searchsorted(cut, z)


def cpi_factor():
    """Food-at-home prices in the simulated year vs the budget survey year (real CPI, BLS)."""
    cpi = pd.read_csv(US_CPI_FILE, usecols=["date", "item_code", "index"], parse_dates=["date"])
    f = cpi[cpi["item_code"] == "SAF11"]                      # CPI-U food at home
    then = f[f["date"].dt.year == HOUSEHOLDS["budget_dollars_year"]]["index"].mean()
    start = pd.Timestamp(START_DATE)
    now = f[f["date"] <= start]["index"].iloc[-12:].mean()   # last 12 months before the start
    return float(now / then)


def assign_households(shoppers, rng):
    h = HOUSEHOLDS
    n = len(shoppers.shopper_id)
    sizes = np.array([int(k) for k in h["size_shares"]])
    z_basket = normal_scores(shoppers.basket_mean, rng)
    z_house = linked(z_basket, h["link_basket_household"], rng)
    size = sizes[by_shares(z_house, list(h["size_shares"].values()))]
    z_income = linked(normal_scores(size + rng.random(n), rng), h["link_household_income"], rng)
    quintile = by_shares(z_income, [1] * 5)                  # 0 = lowest fifth ... 4 = highest
    kids_chance = np.array([h["kids_chance_by_size"][str(s)] for s in size])
    kids = np.where(rng.random(n) < kids_chance, [h["kids_count_by_size"][str(s)] for s in size], 0)
    # weekly food-at-home budget: by household size, then richer/poorer than that size's average
    x = np.array([1, 2, 3, 4, CE_PERSONS_5PLUS])
    by_size = np.array([h["food_at_home_by_size"][str(s)] for s in range(1, 6)], dtype=float)
    expected = np.interp(h["persons_by_income_quintile"], x, by_size)
    income_mult = np.array(h["food_at_home_by_income_quintile"]) / expected
    weekly = by_size[size - 1] * income_mult[quintile] / 52 * cpi_factor()
    # lower income -> more price-sensitive: hand the SAME sensitivity values out by income
    z_sens = linked(normal_scores(quintile + rng.random(n), rng), h["link_income_price_sensitivity"], rng)
    values = np.sort(shoppers.price_sensitivity)
    sens = np.empty(n)
    sens[np.argsort(z_sens)] = values
    cut = np.quantile(sens, np.linspace(0, 1, SENSITIVITY_GROUPS + 1)[1:-1])
    shoppers.price_sensitivity, shoppers.sensitivity_group = sens, np.searchsorted(cut, sens)
    shoppers.household_size, shoppers.kids, shoppers.income_quintile = size, kids, quintile
    shoppers.weekly_budget = weekly.round(2)


def summarize_households(shoppers):
    s = shoppers
    sizes = pd.Series(s.household_size).value_counts(normalize=True).sort_index()
    print("\nHouseholds (who each NPC shops for):")
    print("  people at home: " + " | ".join(f"{k}{'+' if k == 5 else ''}: {v:.0%}" for k, v in sizes.items())
          + f"   (real US homes: " + " | ".join(f"{k}: {v:.0%}" for k, v in HOUSEHOLDS["size_shares"].items()) + ")")
    print(f"  with children: {(s.kids > 0).mean():.0%} | weekly food budget: median ${np.median(s.weekly_budget):.0f} "
          f"(lowest income fifth ${np.median(s.weekly_budget[s.income_quintile == 0]):.0f}, "
          f"highest ${np.median(s.weekly_budget[s.income_quintile == 4]):.0f})")
    corr = np.corrcoef(s.basket_mean, s.household_size)[0, 1]
    sens = [s.price_sensitivity[s.income_quintile == q].mean() for q in (0, 4)]
    print(f"  bigger households, bigger baskets: correlation {corr:.2f} | price sensitivity lowest vs highest "
          f"income: {sens[0]:.2f} vs {sens[1]:.2f}")
