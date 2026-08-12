import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from src.config import RISK_FREE_DAILY


def portfolio_value_series(returns: List[float], initial_value: float = 1.0) -> np.ndarray:
    """Compute cumulative portfolio value from a sequence of period returns."""
    r = np.asarray(returns, dtype=float)
    return initial_value * np.cumprod(1.0 + r)


def cumulative_return(returns: List[float]) -> float:
    """Total cumulative return as percentage: (prod(1+r) - 1) * 100."""
    r = np.asarray(returns, dtype=float)
    return float((np.prod(1.0 + r) - 1.0) * 100.0)


def annualized_volatility(returns: List[float]) -> float:
    """Annualised volatility as percentage: sqrt(252) * std(returns) * 100."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(252) * 100.0)


def sharpe_ratio(returns: List[float], risk_free_daily: float = RISK_FREE_DAILY) -> float:
    """Annualised Sharpe ratio: mean(excess) * sqrt(252) / std(returns). Returns 0 if std=0."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_daily
    std = np.std(r, ddof=1)
    if std == 0.0:
        return 0.0
    return float(np.mean(excess) * np.sqrt(252) / std)


def maximum_drawdown(returns: List[float]) -> float:
    """Maximum peak-to-trough drawdown as percentage (negative value)."""
    values = portfolio_value_series(returns)
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / peaks
    return float(np.min(drawdowns) * 100.0)


def compute_all_metrics(returns: List[float], label: str = '') -> Dict:
    """Compute all standard performance metrics and return as a labelled dict."""
    return {
        'label': label,
        'cumulative_return': cumulative_return(returns),
        'annualized_volatility': annualized_volatility(returns),
        'sharpe_ratio': sharpe_ratio(returns),
        'max_drawdown': maximum_drawdown(returns),
    }
