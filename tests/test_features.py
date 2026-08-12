"""
Tests for src/features.py and src/environment.py.

All tests use synthetic, deterministic data — no real network or API calls.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import compute_asset_features, build_state, build_all_features, N_ASSET_FEATURES
from src.environment import PortfolioEnv
from src.config import WARMUP_DAYS, RECENT_RETURN_WINDOW, REWARD_VOL_WINDOW, TRANSACTION_COST_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic-data factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with n trading-day rows."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)  # keep positive
    high = close * (1.0 + rng.uniform(0.0, 0.02, n))
    low = close * (1.0 - rng.uniform(0.0, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj Close": close,
        },
        index=idx,
    )


def _make_env(
    n_assets: int = 3,
    n_steps: int = 80,
    n_feat: int = 20,
    seed: int = 42,
) -> PortfolioEnv:
    """Return a PortfolioEnv backed entirely by synthetic data."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n_steps)
    returns = pd.DataFrame(rng.normal(0.001, 0.01, (n_steps, n_assets)), index=idx)
    features = pd.DataFrame(rng.standard_normal((n_steps, n_feat)), index=idx)
    market_returns = pd.Series(rng.normal(0.0005, 0.01, n_steps), index=idx)
    return PortfolioEnv(returns, features, market_returns, n_assets)


# ─────────────────────────────────────────────────────────────────────────────
# compute_asset_features tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_asset_features_shape():
    """Output must have exactly N_ASSET_FEATURES columns; no NaN after warmup."""
    ohlcv = _make_ohlcv(n=120, seed=42)
    feat = compute_asset_features(ohlcv, "TEST")

    # Column count
    assert feat.shape[1] == N_ASSET_FEATURES, (
        f"Expected {N_ASSET_FEATURES} feature columns, got {feat.shape[1]}"
    )

    # Row count equals the input
    assert feat.shape[0] == len(ohlcv)

    # After warmup, no column should contain NaN
    after_warmup = feat.iloc[WARMUP_DAYS:]
    assert not after_warmup.isna().any().any(), (
        "NaN values remain after warmup drop:\n"
        + str(after_warmup.isna().sum()[after_warmup.isna().any()])
    )


def test_compute_asset_features_sma_correctness():
    """SMA-5 column must match a manual pandas rolling mean of Adj Close."""
    ohlcv = _make_ohlcv(n=120, seed=7)
    feat = compute_asset_features(ohlcv, "TICK")

    expected_sma5 = ohlcv["Adj Close"].rolling(5).mean()
    computed_sma5 = feat["TICK_sma_5"]

    # After the window fills, values must agree to floating-point precision.
    mask = ~expected_sma5.isna()
    np.testing.assert_allclose(
        computed_sma5[mask].values,
        expected_sma5[mask].values,
        rtol=1e-6,
        err_msg="SMA-5 does not match manual rolling mean.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioEnv tests
# ─────────────────────────────────────────────────────────────────────────────

def test_portfolio_env_reset():
    """reset() must return an observation with the correct shape."""
    env = _make_env(n_assets=3, n_feat=20)
    obs, info = env.reset(seed=0)

    expected_dim = 20 + 3  # n_feat + n_assets
    assert obs.shape == (expected_dim,), (
        f"Expected observation shape ({expected_dim},), got {obs.shape}"
    )
    assert isinstance(info, dict)


def test_portfolio_env_step():
    """One step with equal-weight action must populate info correctly."""
    n_assets = 4
    env = _make_env(n_assets=n_assets, n_feat=15, n_steps=50, seed=0)
    env.reset(seed=0)

    # Equal-weight action: softmax of all-zeros → uniform
    action = np.zeros(n_assets, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert "portfolio_return" in info
    assert "weights" in info
    assert "turnover" in info
    assert "portfolio_value" in info

    assert info["turnover"] >= 0.0, "Turnover must be non-negative."
    assert isinstance(reward, float)
    assert obs.shape == env.observation_space.shape


def test_portfolio_env_weights_sum_to_one():
    """Softmax-normalised weights returned in info must sum to 1."""
    n_assets = 5
    env = _make_env(n_assets=n_assets, n_feat=10, n_steps=30, seed=1)
    env.reset()

    rng = np.random.default_rng(99)
    for _ in range(5):
        action = rng.standard_normal(n_assets).astype(np.float32)
        _, _, terminated, _, info = env.step(action)
        np.testing.assert_allclose(
            info["weights"].sum(),
            1.0,
            atol=1e-6,
            err_msg="Portfolio weights must sum to 1.",
        )
        if terminated:
            break


def test_portfolio_env_reward_components():
    """Reward should be positive when returns are consistently positive.

    With uniformly positive asset returns and no extreme volatility the
    REWARD_RETURN_COEF * portfolio_return term should dominate after
    enough steps, yielding a positive average reward.
    """
    n_assets = 3
    n_steps = 60
    # All assets earn +1 % every day (large positive returns)
    returns = pd.DataFrame(np.full((n_steps, n_assets), 0.01))
    features = pd.DataFrame(np.zeros((n_steps, 10)))
    market_returns = pd.Series(np.full(n_steps, 0.005))

    env = PortfolioEnv(returns, features, market_returns, n_assets)
    env.reset()

    action = np.zeros(n_assets, dtype=np.float32)  # uniform weights
    rewards = []
    for _ in range(min(REWARD_VOL_WINDOW + 5, n_steps - 1)):
        _, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
        if terminated:
            break

    # With all assets returning +1 % and very low volatility (constant
    # returns), the Sharpe / return terms should dominate.
    assert np.mean(rewards) > 0.0, (
        f"Expected positive average reward with positive returns, got {np.mean(rewards):.6f}"
    )


def test_portfolio_env_no_lookahead():
    """portfolio_return at step t must use only returns[t+1], not future data.

    We inject known values at returns.iloc[1] and returns.iloc[2], then
    verify that the info dict matches the expected dot-product at each step.
    """
    n_assets = 3
    n_steps = 20
    n_feat = 10

    rng = np.random.default_rng(42)
    returns_vals = rng.normal(0, 0.01, (n_steps, n_assets))

    # Inject deterministic return values at rows 1 and 2
    ret_day1 = np.array([0.03, 0.04, 0.05])
    ret_day2 = np.array([-0.01, 0.02, -0.03])
    returns_vals[1] = ret_day1
    returns_vals[2] = ret_day2

    returns = pd.DataFrame(returns_vals)
    features = pd.DataFrame(rng.standard_normal((n_steps, n_feat)))
    market_returns = pd.Series(rng.normal(0, 0.01, n_steps))

    env = PortfolioEnv(returns, features, market_returns, n_assets)
    env.reset()

    # ── Step 0: should see returns.iloc[1] = ret_day1 ────────────────────
    # Equal-weight action → softmax → [1/3, 1/3, 1/3] (same as init weights)
    action = np.zeros(n_assets, dtype=np.float32)
    _, _, _, _, info0 = env.step(action)

    expected_ret0 = float(np.dot(np.ones(n_assets) / n_assets, ret_day1))
    # TC = 0 since weights didn't change from initial equal weights
    assert abs(info0["portfolio_return"] - expected_ret0) < 1e-7, (
        f"Step 0: expected portfolio_return={expected_ret0:.6f}, "
        f"got {info0['portfolio_return']:.6f}. "
        "Possible lookahead bias or wrong return index."
    )

    # ── Step 1: should see returns.iloc[2] = ret_day2 ────────────────────
    _, _, _, _, info1 = env.step(action)

    expected_ret1 = float(np.dot(np.ones(n_assets) / n_assets, ret_day2))
    assert abs(info1["portfolio_return"] - expected_ret1) < 1e-7, (
        f"Step 1: expected portfolio_return={expected_ret1:.6f}, "
        f"got {info1['portfolio_return']:.6f}. "
        "Possible lookahead bias or wrong return index."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
