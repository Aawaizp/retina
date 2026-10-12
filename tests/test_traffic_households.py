"""Store traffic and households follow their real inputs."""

import numpy as np

from retina.engine import config
from retina.engine import traffic
from retina.engine.households import assign_households
from retina.engine.world import create_shoppers, load_inputs
from retina.engine.config import stream, STREAM_SHOPPERS


def test_hourly_shape():
    shape = traffic.hour_shape()
    open_h, close_h = config.TRAFFIC["open_hours"]
    assert abs(shape.sum() - 1) < 1e-9
    assert shape[:open_h].sum() == 0 and shape[close_h:].sum() == 0


def test_weekend_busier_and_scaled():
    days = range(7)
    real = [traffic.real_per_day(d) for d in days]
    weekend = [r for d, r in zip(days, real) if traffic.is_weekend(d)]
    weekday = [r for d, r in zip(days, real) if not traffic.is_weekend(d)]
    assert min(weekend) > max(weekday)
    assert abs(sum(real) - config.TRAFFIC["real_shoppers_per_week"]) < 1
    assert abs(traffic.expected_npcs(0) - real[0] * config.TRAFFIC["npc_scale"]) < 1e-9


def test_households_match_inputs_and_keep_price_sensitivity():
    _, _, params, _ = load_inputs()
    sh = create_shoppers(params, 3000, stream(STREAM_SHOPPERS))
    before = np.sort(sh.price_sensitivity.copy())
    assign_households(sh, stream(STREAM_SHOPPERS, 2))
    assert np.allclose(np.sort(sh.price_sensitivity), before)      # same values, handed out by income
    shares = {int(k): v for k, v in config.HOUSEHOLDS["size_shares"].items()}
    for size, share in shares.items():
        assert abs((sh.household_size == size).mean() - share) < 0.02
    assert (sh.kids[sh.household_size == 1] == 0).all()
    assert (sh.weekly_budget > 0).all()
    low, high = sh.price_sensitivity[sh.income_quintile == 0].mean(), sh.price_sensitivity[sh.income_quintile == 4].mean()
    assert low > high                                              # poorer homes are more price-sensitive
