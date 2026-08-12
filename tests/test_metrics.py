import pytest
import numpy as np
from src.metrics import (
    cumulative_return,
    annualized_volatility,
    sharpe_ratio,
    maximum_drawdown,
    compute_all_metrics,
    portfolio_value_series,
)
from src.config import RISK_FREE_DAILY


def test_cumulative_return_positive():
    """Known sequence [0.01, 0.02, -0.01] -> exact value."""
    returns = [0.01, 0.02, -0.01]
    expected = (1.01 * 1.02 * 0.99 - 1.0) * 100.0
    result = cumulative_return(returns)
    assert abs(result - expected) < 1e-10


def test_cumulative_return_zero_returns():
    """All-zero returns produce 0% cumulative return."""
    assert cumulative_return([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_annualized_volatility_known():
    """Constant (identical) returns have zero dispersion -> 0% volatility."""
    result = annualized_volatility([0.001] * 100)
    assert result == pytest.approx(0.0, abs=1e-10)


def test_annualized_volatility_formula():
    """Verify formula: sqrt(252) * std(returns) * 100."""
    returns = [0.01, -0.02, 0.015, -0.005, 0.008]
    expected = np.std(returns, ddof=1) * np.sqrt(252) * 100.0
    assert annualized_volatility(returns) == pytest.approx(expected, rel=1e-9)


def test_sharpe_ratio_positive_returns():
    """Consistently positive returns above risk-free rate produce a positive Sharpe."""
    returns = [0.002] * 252  # well above RISK_FREE_DAILY (~0.000179)
    assert sharpe_ratio(returns) > 0.0


def test_sharpe_ratio_zero_std():
    """Constant (identical) returns give std=0, so Sharpe should be 0."""
    assert sharpe_ratio([0.001] * 50) == pytest.approx(0.0)


def test_sharpe_ratio_formula():
    """Verify Sharpe formula: mean(excess) * sqrt(252) / std(returns)."""
    returns = [0.01, -0.005, 0.003, 0.007, -0.002]
    r = np.array(returns)
    excess = r - RISK_FREE_DAILY
    expected = np.mean(excess) * np.sqrt(252) / np.std(r, ddof=1)
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-9)


def test_maximum_drawdown_known():
    """Returns [0.1, -0.2, 0.1]: peak after step-1 is 1.1, then drops to 0.88."""
    returns = [0.1, -0.2, 0.1]
    # portfolio values: 1.1, 0.88, 0.968
    # drawdown at step 2: (0.88 - 1.1) / 1.1 = -0.2/1.1
    expected_mdd = (0.88 - 1.10) / 1.10 * 100.0
    result = maximum_drawdown(returns)
    assert result == pytest.approx(expected_mdd, rel=1e-6)


def test_maximum_drawdown_no_drawdown():
    """Monotonically increasing portfolio has MDD ≈ 0 (numerical precision only)."""
    returns = [0.005] * 200
    mdd = maximum_drawdown(returns)
    # With strictly positive returns, drawdown from any peak to the immediate
    # next point is zero (each value exceeds the previous peak), so MDD == 0.
    assert mdd == pytest.approx(0.0, abs=1e-6)


def test_compute_all_metrics_keys():
    """compute_all_metrics must return a dict with all five expected keys."""
    result = compute_all_metrics([0.01, -0.005, 0.003], label='test')
    required_keys = {'label', 'cumulative_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown'}
    assert required_keys.issubset(result.keys())
    assert result['label'] == 'test'


def test_compute_all_metrics_values_consistent():
    """Values in compute_all_metrics match individual function outputs."""
    returns = [0.01, -0.02, 0.015, -0.005, 0.008]
    metrics = compute_all_metrics(returns, label='consistency')
    assert metrics['cumulative_return'] == pytest.approx(cumulative_return(returns))
    assert metrics['annualized_volatility'] == pytest.approx(annualized_volatility(returns))
    assert metrics['sharpe_ratio'] == pytest.approx(sharpe_ratio(returns))
    assert metrics['max_drawdown'] == pytest.approx(maximum_drawdown(returns))


def test_portfolio_value_series_monotone_growth():
    """All-positive returns produce a strictly increasing value series."""
    returns = [0.001] * 100
    values = portfolio_value_series(returns)
    assert len(values) == 100
    assert np.all(np.diff(values) > 0)


def test_portfolio_value_series_initial_value():
    """Custom initial value scales the entire series proportionally."""
    returns = [0.01, 0.02, -0.01]
    default_vals = portfolio_value_series(returns, initial_value=1.0)
    scaled_vals = portfolio_value_series(returns, initial_value=100.0)
    np.testing.assert_allclose(scaled_vals, default_vals * 100.0, rtol=1e-12)


def test_portfolio_value_series_compounding():
    """Verify compounding: values[i] == initial * prod(1+r[:i+1])."""
    returns = [0.05, -0.03, 0.02]
    values = portfolio_value_series(returns, initial_value=1.0)
    assert values[0] == pytest.approx(1.05, rel=1e-12)
    assert values[1] == pytest.approx(1.05 * 0.97, rel=1e-12)
    assert values[2] == pytest.approx(1.05 * 0.97 * 1.02, rel=1e-12)


def test_metrics_edge_cases():
    """Single-element returns list: functions return a scalar without raising."""
    single = [0.05]
    assert isinstance(cumulative_return(single), float)
    # std of a single value is 0 (ddof=1 with n=1 gives NaN, handled by returning 0)
    vol = annualized_volatility(single)
    assert vol == pytest.approx(0.0)
    sr = sharpe_ratio(single)
    assert sr == pytest.approx(0.0)
    mdd = maximum_drawdown(single)
    assert mdd == pytest.approx(0.0)
    metrics = compute_all_metrics(single, label='edge')
    assert metrics['label'] == 'edge'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
