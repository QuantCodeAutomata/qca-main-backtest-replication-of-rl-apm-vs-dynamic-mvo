"""
Data loading and preprocessing utilities for the RL portfolio management pipeline.
"""

import logging
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.config import TRAIN_END, TEST_START


def download_data(
    tickers: List[str],
    market_proxy: str,
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Download daily OHLCV + Adj Close for all tickers and the market proxy.

    Uses yfinance with auto_adjust=False so 'Adj Close' is a separate column.
    All symbols are downloaded in a single call and returned as a MultiIndex
    DataFrame (level-0 = price field, level-1 = ticker).  The market proxy
    Adj Close is also returned as a convenience Series.

    Missing values are handled by forward-filling first; any row that still
    contains NaN in an Adj Close column is then dropped so that every date
    in the returned frame has a complete set of adjusted prices.

    Args:
        tickers: Portfolio asset ticker symbols.
        market_proxy: Market benchmark ticker (e.g. 'SPY').
        start: Download start date string (YYYY-MM-DD).
        end: Download end date string (YYYY-MM-DD).

    Returns:
        prices_df: MultiIndex-column DataFrame (field × ticker) for all
            downloaded symbols including the market proxy.
        spy_series: Adj Close Series for the market proxy.
    """
    # Deduplicate while preserving order; market proxy goes first for clarity.
    all_tickers = list(dict.fromkeys([market_proxy] + list(tickers)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw: pd.DataFrame = yf.download(
            all_tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

    # Forward-fill gaps (e.g. a ticker suspended for one day).
    raw = raw.ffill()

    # Drop tickers whose entire Adj Close column is NaN (failed downloads).
    adj_close_block: pd.DataFrame = raw["Adj Close"]
    fully_missing = adj_close_block.columns[adj_close_block.isna().all()].tolist()
    if fully_missing:
        logger.warning("Removing fully-missing tickers: %s", fully_missing)
        raw = raw.drop(columns=fully_missing, level=1)

    # Drop any remaining rows where at least one Adj Close is still NaN.
    adj_close_block = raw["Adj Close"]
    valid_rows = ~adj_close_block.isna().any(axis=1)
    raw = raw.loc[valid_rows]

    spy_series: pd.Series = raw["Adj Close"][market_proxy]

    return raw, spy_series


def compute_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns from an Adj Close DataFrame.

    The first row (which would be NaN after pct_change) is dropped.

    Args:
        adj_close: DataFrame whose columns are ticker symbols and whose
            index is a DatetimeIndex of trading days.

    Returns:
        DataFrame of daily simple returns with the same columns; the index
        starts one day after the first date in ``adj_close``.
    """
    return adj_close.pct_change().dropna()


def get_train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train and test subsets.

    The split dates are taken from the config constants TRAIN_END and
    TEST_START, so the two periods are non-overlapping by construction.

    Args:
        df: DataFrame with a DatetimeIndex sorted in ascending order.

    Returns:
        train: Rows with index <= TRAIN_END.
        test: Rows with index >= TEST_START.
    """
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:]
    return train, test


def prepare_experiment_data(
    tickers: List[str],
    market_proxy: str,
    start: str,
    end: str,
) -> Dict:
    """Download and structure all data needed for an experiment.

    Builds a single dict with adj_close, returns, market_returns, and
    per-ticker OHLCV DataFrames ready for feature engineering.

    Args:
        tickers: Asset ticker symbols (excluding market proxy).
        market_proxy: Benchmark ticker (e.g. 'SPY').
        start: Download start date (YYYY-MM-DD).
        end: Download end date (YYYY-MM-DD).

    Returns:
        dict with keys:
            'adj_close'      : pd.DataFrame, shape (dates, n_assets)
            'returns'        : pd.DataFrame, shape (dates-1, n_assets)
            'market_adj'     : pd.Series, market proxy Adj Close
            'market_returns' : pd.Series, market proxy daily returns
            'ohlcv_dict'     : Dict[str, pd.DataFrame], per-ticker OHLCV
            'raw'            : raw MultiIndex DataFrame from yfinance
    """
    raw, spy_series = download_data(tickers, market_proxy, start, end)

    # Restrict to tickers that were actually downloaded (removes delisted ones).
    available = [t for t in tickers if t in raw["Adj Close"].columns]
    if len(available) < len(tickers):
        dropped = set(tickers) - set(available)
        logger.warning("Tickers not in downloaded data (skipped): %s", dropped)
    tickers = available

    # Asset adjusted-close (tickers only, no market proxy)
    adj_close: pd.DataFrame = raw["Adj Close"][tickers]

    # Market series
    market_adj: pd.Series = raw["Adj Close"][market_proxy]
    market_returns: pd.Series = market_adj.pct_change().dropna()

    # Align adj_close to market_returns index (same set of dates)
    common_idx = adj_close.index.intersection(market_returns.index)
    adj_close = adj_close.loc[common_idx]
    market_returns = market_returns.loc[common_idx]

    returns: pd.DataFrame = compute_returns(adj_close)
    # Re-align after dropping the first row
    common_idx2 = returns.index.intersection(market_returns.index)
    returns = returns.loc[common_idx2]
    market_returns = market_returns.loc[common_idx2]
    adj_close = adj_close.loc[common_idx2]  # trim 1 row from start

    # Per-ticker OHLCV dicts (needed by feature engineering)
    ohlcv_dict: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ohlcv = pd.DataFrame({
            "Adj Close": raw["Adj Close"][ticker],
            "High":      raw["High"][ticker],
            "Low":       raw["Low"][ticker],
            "Volume":    raw["Volume"][ticker],
            "Open":      raw["Open"][ticker],
            "Close":     raw["Close"][ticker],
        })
        ohlcv_dict[ticker] = ohlcv.loc[common_idx2]

    return {
        "adj_close":      adj_close,
        "returns":        returns,
        "market_adj":     market_adj,
        "market_returns": market_returns,
        "ohlcv_dict":     ohlcv_dict,
        "raw":            raw,
    }


def save_data(data: dict, path: str) -> None:
    """Persist a data dictionary to disk.

    Files whose path ends with ``.json`` are serialised as JSON (values
    are coerced to strings for non-serialisable types).  All other paths
    are written as pickle.

    Args:
        data: Dictionary of data to save.
        path: Destination file path (directories must already exist).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w") as fh:
            json.dump(data, fh, default=str)
    else:
        with open(path, "wb") as fh:
            pickle.dump(data, fh)


def load_data(path: str) -> dict:
    """Load a data dictionary previously saved with :func:`save_data`.

    Args:
        path: Path to the file to load.

    Returns:
        The deserialised dictionary.
    """
    if path.endswith(".json"):
        with open(path, "r") as fh:
            return json.load(fh)
    else:
        with open(path, "rb") as fh:
            return pickle.load(fh)
