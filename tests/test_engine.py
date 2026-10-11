"""The engine: same seed = same data, and stopping/restarting changes nothing."""

from conftest import run_days


def test_same_seed_gives_the_same_data():
    a, b = run_days(4), run_days(4)
    assert len(a) > 0
    assert a.equals(b)


def test_stop_and_continue_equals_one_run():
    """live.py runs the store in pieces; the data must equal one long run."""
    assert run_days(6, chunks=[2, 4]).equals(run_days(6))


def test_every_day_has_shoppers(pipeline):
    sales = pipeline["results"].sales
    assert set(sales["day"]) == set(range(sales["day"].max() + 1))
