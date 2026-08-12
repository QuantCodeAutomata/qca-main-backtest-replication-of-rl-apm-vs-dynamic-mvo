import numpy as np
import pandas as pd
from typing import Dict
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from src.config import MVO_ESTIMATION_WINDOW, TRANSACTION_COST_RATE
import warnings
warnings.filterwarnings('ignore')


class DynamicMVO:
    def __init__(
        self,
        n_assets: int,
        estimation_window: int = MVO_ESTIMATION_WINDOW,
        transaction_cost_rate: float = TRANSACTION_COST_RATE,
    ):
        self.n_assets = n_assets
        self.estimation_window = estimation_window
        self.transaction_cost_rate = transaction_cost_rate

    def compute_weights(self, price_history: pd.DataFrame) -> np.ndarray:
        tickers = list(price_history.columns)
        n = len(tickers)
        equal_w = np.ones(n) / n

        try:
            mu = expected_returns.mean_historical_return(price_history, frequency=252)
            S = risk_models.sample_cov(price_history, frequency=252)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.max_sharpe()
            cleaned = ef.clean_weights()
            weights = np.array([cleaned[t] for t in tickers], dtype=float)
            if weights.sum() == 0:
                return equal_w
            return weights
        except Exception:
            return equal_w

    def run(self, adj_close: pd.DataFrame, returns: pd.DataFrame) -> Dict:
        n_assets = len(adj_close.columns)
        prev_weights = np.ones(n_assets) / n_assets
        portfolio_returns = []
        all_weights = []

        for i in range(self.estimation_window, len(returns)):
            price_window = adj_close.iloc[i - self.estimation_window:i]
            weights = self.compute_weights(price_window)

            tc = self.transaction_cost_rate * np.abs(weights - prev_weights).sum()
            port_return = float(np.dot(weights, returns.iloc[i].values)) - tc

            portfolio_returns.append(port_return)
            all_weights.append(weights)
            prev_weights = weights

        return {'returns': portfolio_returns, 'weights': all_weights}


def equal_weight_portfolio(returns: pd.DataFrame) -> Dict:
    n_assets = len(returns.columns)
    weights = np.ones(n_assets) / n_assets
    port_returns = (returns.values @ weights).tolist()
    all_weights = [weights.copy() for _ in range(len(returns))]
    return {'returns': port_returns, 'weights': all_weights}
