"""
Feature engineering for the RL portfolio management pipeline.

Provides per-asset technical indicator computation, a single-step state
builder, and a batch pre-builder that covers the full dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import ta
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler

from src.config import (
    SMA_WINDOWS,
    RSI_WINDOW,
    BB_WINDOW,
    BB_STD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    VOLATILITY_WINDOW,
    RECENT_RETURN_WINDOW,
    WARMUP_DAYS,
)

# Number of raw (unscaled) feature columns produced per asset by
# compute_asset_features.  Update this constant whenever the feature set
# changes so that callers can infer state dimensions without running the
# pipeline first.
#   SMA_WINDOWS (4) + EMA (1) + RSI (1) + BB (4) + MACD (3)
#   + rolling_vol (1) + ROC (1) + ATR (1) + OBV (1)  = 17
N_ASSET_FEATURES = 17


def compute_asset_features(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute all technical indicators for a single asset.

    Requires the OHLCV DataFrame to contain at least the columns
    'Adj Close', 'High', 'Low', and 'Volume'.  'Open' is not used but may
    be present.

    Indicators computed:
        - SMA at windows [5, 10, 20, 50]
        - EMA at window 20
        - RSI at window 14
        - Bollinger Bands (window 20, std 2): upper, lower, middle, bandwidth
        - MACD (12, 26, 9): macd line, signal line, histogram
        - Rolling return standard deviation (window 20)
        - Rate of Change (ROC) at window 10
        - Average True Range (ATR) at window 14
        - On-Balance Volume (OBV)

    All column names are prefixed with ``<ticker>_``.

    Args:
        ohlcv: DataFrame with columns including 'Adj Close', 'High', 'Low',
            and 'Volume', indexed by trading date.
        ticker: Ticker symbol used as a column-name prefix.

    Returns:
        DataFrame of shape (len(ohlcv), 17) with named feature columns.
        Early rows will contain NaN where the rolling window has not yet
        filled; callers should apply a warmup drop before use.
    """
    close = ohlcv["Adj Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    features: Dict[str, pd.Series] = {}

    # ── Simple Moving Averages ────────────────────────────────────────────
    for w in SMA_WINDOWS:
        features[f"{ticker}_sma_{w}"] = SMAIndicator(
            close=close, window=w
        ).sma_indicator()

    # ── Exponential Moving Average ────────────────────────────────────────
    features[f"{ticker}_ema_20"] = EMAIndicator(
        close=close, window=20
    ).ema_indicator()

    # ── Relative Strength Index ───────────────────────────────────────────
    features[f"{ticker}_rsi_{RSI_WINDOW}"] = RSIIndicator(
        close=close, window=RSI_WINDOW
    ).rsi()

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = BollingerBands(close=close, window=BB_WINDOW, window_dev=BB_STD)
    features[f"{ticker}_bb_upper"] = bb.bollinger_hband()
    features[f"{ticker}_bb_lower"] = bb.bollinger_lband()
    features[f"{ticker}_bb_middle"] = bb.bollinger_mavg()
    features[f"{ticker}_bb_bandwidth"] = bb.bollinger_wband()

    # ── MACD ─────────────────────────────────────────────────────────────
    macd = MACD(
        close=close,
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
    )
    features[f"{ticker}_macd"] = macd.macd()
    features[f"{ticker}_macd_signal"] = macd.macd_signal()
    features[f"{ticker}_macd_diff"] = macd.macd_diff()

    # ── Rolling return volatility ─────────────────────────────────────────
    features[f"{ticker}_vol_{VOLATILITY_WINDOW}"] = (
        close.pct_change().rolling(VOLATILITY_WINDOW).std()
    )

    # ── Rate of Change ────────────────────────────────────────────────────
    features[f"{ticker}_roc_10"] = ROCIndicator(close=close, window=10).roc()

    # ── Average True Range ────────────────────────────────────────────────
    features[f"{ticker}_atr_14"] = AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # ── On-Balance Volume ─────────────────────────────────────────────────
    features[f"{ticker}_obv"] = OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()

    return pd.DataFrame(features, index=ohlcv.index)


def build_state(
    adj_close: pd.DataFrame,
    ohlcv_dict: Dict[str, pd.DataFrame],
    weights: np.ndarray,
    market_returns: pd.Series,
    scaler: Optional[StandardScaler],
    t: int,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """Construct state vector S_t = [F_t, W_t, C_t, R_t] for step t.

    This is an online builder that computes all features from scratch for a
    single timestep.  For full-dataset pre-computation, prefer
    :func:`build_all_features`.

    Components:
        F_t: Concatenated technical-indicator features for all assets at t.
        W_t: Current portfolio weight vector.
        C_t: Market context = [market_return_t, rolling_market_vol_20,
             market_sma5 / market_close].
        R_t: Trailing RECENT_RETURN_WINDOW-day return for each asset.

    F_t and C_t are normalised using ``scaler``.  Pass ``fit_scaler=True``
    to fit a fresh StandardScaler on the values at time t (useful for a
    single-sample sanity check; for production use, fit externally on the
    full training window and pass the fitted scaler).

    Args:
        adj_close: Adj Close DataFrame indexed by date, columns = tickers.
        ohlcv_dict: Mapping from ticker → OHLCV DataFrame.
        weights: Current portfolio weight vector (n_assets,).
        market_returns: Daily market return Series.
        scaler: Pre-fitted StandardScaler or None.
        t: Integer row index (0-based) into the DataFrames.
        fit_scaler: If True, fit a new scaler on the current observation.

    Returns:
        state: 1-D numpy array of the full state vector.
        scaler: The scaler used (fitted or unchanged).
    """
    tickers = list(ohlcv_dict.keys())

    # ── F_t: per-asset indicator features ────────────────────────────────
    f_parts: List[np.ndarray] = []
    for ticker in tickers:
        feat_df = compute_asset_features(ohlcv_dict[ticker].iloc[: t + 1], ticker)
        f_parts.append(feat_df.iloc[-1].values.astype(float))
    F_t = np.concatenate(f_parts)

    # ── C_t: market context ───────────────────────────────────────────────
    market_ret_t = float(market_returns.iloc[t]) if t < len(market_returns) else 0.0
    vol_slice = market_returns.iloc[max(0, t - 19) : t + 1]
    market_vol_20 = float(vol_slice.std()) if len(vol_slice) > 1 else 0.0

    # Reconstruct a price index from returns to compute SMA5/close ratio.
    price_index = (1.0 + market_returns.iloc[: t + 1]).cumprod()
    sma5_slice = price_index.iloc[max(0, t - 4) :]
    market_sma5_ratio = float(sma5_slice.mean() / (price_index.iloc[-1] + 1e-8))

    C_t = np.array([market_ret_t, market_vol_20, market_sma5_ratio], dtype=float)

    # ── R_t: trailing returns per asset ───────────────────────────────────
    R_t = np.array(
        [
            float(
                adj_close[tk].iloc[t] / adj_close[tk].iloc[max(0, t - RECENT_RETURN_WINDOW)] - 1
            )
            for tk in tickers
        ],
        dtype=float,
    )

    # ── Normalise F_t and C_t ─────────────────────────────────────────────
    raw = np.concatenate([F_t, C_t])
    nan_mask = np.isnan(raw)
    raw = np.where(nan_mask, 0.0, raw)  # replace NaN before scaling

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw.reshape(1, -1)).flatten()
    else:
        scaled = scaler.transform(raw.reshape(1, -1)).flatten()

    F_t_scaled = scaled[: len(F_t)]
    C_t_scaled = scaled[len(F_t) :]

    # ── W_t: portfolio weights (not scaled) ──────────────────────────────
    W_t = weights.astype(float)

    state = np.concatenate([F_t_scaled, W_t, C_t_scaled, R_t])
    return state, scaler


def build_all_features(
    adj_close: pd.DataFrame,
    ohlcv_dict: Dict[str, pd.DataFrame],
    market_returns: pd.Series,
) -> pd.DataFrame:
    """Pre-build all asset features for every date in the dataset.

    Assembles the feature block used by :class:`~src.environment.PortfolioEnv`
    (everything except the dynamic portfolio-weight vector W_t):

        [F_t (all assets) | C_t (market context) | R_t (trailing returns)]

    The first WARMUP_DAYS rows are dropped to avoid NaN-contaminated
    indicator values at the start of the series.

    Args:
        adj_close: Adj Close DataFrame indexed by date, columns = tickers.
        ohlcv_dict: Mapping from ticker → OHLCV DataFrame.
        market_returns: Daily market return Series aligned to adj_close.

    Returns:
        DataFrame with one row per trading day (after warmup) and
        N_ASSET_FEATURES * n_assets + 3 + n_assets columns:
            - per-asset indicator features (F_t)
            - 3 market context features (C_t)
            - per-asset trailing returns (R_t)
    """
    tickers = list(ohlcv_dict.keys())

    # ── F_t: per-asset technical indicators ──────────────────────────────
    asset_blocks: List[pd.DataFrame] = [
        compute_asset_features(ohlcv_dict[tk], tk) for tk in tickers
    ]
    asset_features = pd.concat(asset_blocks, axis=1)

    # ── C_t: market context features ─────────────────────────────────────
    price_index = (1.0 + market_returns).cumprod()
    context = pd.DataFrame(
        {
            "ctx_market_return": market_returns,
            "ctx_market_vol_20": market_returns.rolling(20).std(),
            "ctx_market_sma5_ratio": price_index.rolling(5).mean() / price_index,
        }
    )

    # ── R_t: trailing returns per asset ───────────────────────────────────
    trailing = pd.DataFrame(
        {
            f"trail_{tk}": adj_close[tk].pct_change(RECENT_RETURN_WINDOW)
            for tk in tickers
        }
    )

    # ── Combine and apply warmup drop ─────────────────────────────────────
    all_features = pd.concat([asset_features, context, trailing], axis=1)
    all_features = all_features.iloc[WARMUP_DAYS:]

    return all_features
