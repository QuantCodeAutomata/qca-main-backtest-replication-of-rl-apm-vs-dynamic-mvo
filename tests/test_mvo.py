import pytest
import numpy as np
import pandas as pd
from src.mvo import DynamicMVO, equal_weight_portfolio
from src.config import MVO_ESTIMATION_WINDOW


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_prices(n_days: int, n_assets: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic price DataFrame with log-normal paths."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    prices = np.cumprod(1 + log_returns, axis=0) * 100.0
    tickers = [f"A{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    return adj_close.pct_change().dropna()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_mvo_weights_sum_to_one():
    """Max-Sharpe weights must be non-negative and sum to 1."""
    n_assets = 5
    prices = _make_prices(n_days=252, n_assets=n_assets)
    mvo = DynamicMVO(n_assets=n_assets)
    weights = mvo.compute_weights(prices)

    assert weights.shape == (n_assets,)
    assert np.all(weights >= 0), "All weights must be non-negative."
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0


def test_mvo_fallback_equal_weights():
    """Constant prices → singular covariance → fallback to equal weights."""
    n_assets = 5
    n_days = 252
    tickers = [f"A{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_days)
    # All prices identical: zero variance, covariance matrix is singular.
    prices = pd.DataFrame(
        np.full((n_days, n_assets), 100.0), index=dates, columns=tickers
    )
    mvo = DynamicMVO(n_assets=n_assets)
    weights = mvo.compute_weights(prices)

    expected = np.ones(n_assets) / n_assets
    np.testing.assert_array_almost_equal(weights, expected)


def test_mvo_run_returns_correct_length():
    """run() output length equals len(returns) - estimation_window."""
    n_assets = 5
    n_days = MVO_ESTIMATION_WINDOW + 21   # 21 post-warmup days of returns
    adj_close = _make_prices(n_days=n_days, n_assets=n_assets)
    returns = _make_returns(adj_close)    # n_days - 1 rows

    mvo = DynamicMVO(n_assets=n_assets)
    result = mvo.run(adj_close, returns)

    expected_length = len(returns) - MVO_ESTIMATION_WINDOW
    assert len(result['returns']) == expected_length
    assert len(result['weights']) == expected_length
    # Each weight vector has the right shape
    for w in result['weights']:
        assert w.shape == (n_assets,)


def test_equal_weight_portfolio():
    """equal_weight_portfolio produces 1/N weights for every date and correct returns."""
    n_assets = 5
    n_days = 100
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n_days)
    tickers = [f"A{i}" for i in range(n_assets)]
    rets = pd.DataFrame(
        rng.normal(0, 0.01, size=(n_days, n_assets)), index=dates, columns=tickers
    )

    result = equal_weight_portfolio(rets)

    assert len(result['returns']) == n_days
    assert len(result['weights']) == n_days

    expected_w = np.full(n_assets, 1.0 / n_assets)
    for w in result['weights']:
        np.testing.assert_array_almost_equal(w, expected_w)

    # Portfolio return on each day is the equal-weighted average of asset returns
    expected_port_returns = rets.values @ expected_w
    np.testing.assert_array_almost_equal(result['returns'], expected_port_returns)


def test_mvo_no_lookahead():
    """Modifying a future price row must not affect the first post-warmup weights."""
    n_assets = 5
    # n_days rows in adj_close → n_days-1 rows in returns
    # First post-warmup step i=MVO_ESTIMATION_WINDOW uses adj_close.iloc[0:MVO_ESTIMATION_WINDOW]
    # Modifying adj_close.iloc[MVO_ESTIMATION_WINDOW] (just outside that window)
    # must leave the weights at step 0 unchanged.
    n_days = MVO_ESTIMATION_WINDOW + 5
    adj_close_base = _make_prices(n_days=n_days, n_assets=n_assets)
    returns = _make_returns(adj_close_base)

    adj_close_modified = adj_close_base.copy()
    # Wildly inflate one future price point that would only appear if lookahead is used
    adj_close_modified.iloc[MVO_ESTIMATION_WINDOW] *= 50.0

    mvo = DynamicMVO(n_assets=n_assets)
    result_base = mvo.run(adj_close_base, returns)
    result_modified = mvo.run(adj_close_modified, returns)

    # The first post-warmup weight vector must be identical in both runs
    np.testing.assert_array_almost_equal(
        result_base['weights'][0],
        result_modified['weights'][0],
        err_msg="Lookahead detected: weights at step 0 changed when future price was modified.",
    )


def test_mvo_transaction_cost_applied():
    """Higher transaction cost rate must reduce the total portfolio return."""
    n_assets = 5
    n_days = MVO_ESTIMATION_WINDOW + 20
    adj_close = _make_prices(n_days=n_days, n_assets=n_assets)
    returns = _make_returns(adj_close)

    mvo_no_tc = DynamicMVO(n_assets=n_assets, transaction_cost_rate=0.0)
    mvo_high_tc = DynamicMVO(n_assets=n_assets, transaction_cost_rate=0.5)  # 50% per unit turnover

    result_no_tc = mvo_no_tc.run(adj_close, returns)
    result_high_tc = mvo_high_tc.run(adj_close, returns)

    sum_no_tc = sum(result_no_tc['returns'])
    sum_high_tc = sum(result_high_tc['returns'])

    # TC is always non-negative, so cumulative return with TC <= without TC
    assert sum_no_tc >= sum_high_tc, (
        f"Expected no-TC total ({sum_no_tc:.6f}) >= high-TC total ({sum_high_tc:.6f})"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
