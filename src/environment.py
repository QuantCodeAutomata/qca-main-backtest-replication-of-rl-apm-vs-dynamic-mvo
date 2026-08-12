"""
Custom Gymnasium trading environment for the RL portfolio management pipeline.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    TRANSACTION_COST_RATE,
    REWARD_RETURN_COEF,
    REWARD_SHARPE_COEF,
    REWARD_VOL_COEF,
    REWARD_TC_COEF,
    REWARD_VOL_WINDOW,
)


class PortfolioEnv(gym.Env):
    """Discrete-time portfolio allocation environment.

    At each step the agent observes a state vector derived from pre-built
    technical features and its current holdings, then outputs a raw logit
    vector that is softmax-normalised to valid portfolio weights.  The
    one-step portfolio return (net of proportional transaction costs) drives
    a composite reward that also rewards high rolling Sharpe and penalises
    excess volatility.

    Observation space: Box(-inf, inf, shape=(features.shape[1] + n_assets,))
    Action space:      Box(0, 1, shape=(n_assets,)) — softmax'd to weights

    Args:
        returns: Daily simple-return DataFrame (dates × assets), aligned with
            ``features`` (same index and same number of rows).
        features: Pre-built, normalised feature DataFrame (dates × features).
            Should NOT include the portfolio weight block W_t — that is
            appended dynamically by :meth:`_get_state`.
        market_returns: Daily market-proxy return Series (aligned to returns).
        n_assets: Number of investable assets.
        scaler: Optional fitted StandardScaler (stored for external reference;
            features are assumed to be already normalised).
        mode: ``'train'`` or ``'test'`` (informational; does not change logic).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        market_returns: pd.Series,
        n_assets: int,
        scaler=None,
        mode: str = "train",
    ) -> None:
        super().__init__()

        self.returns = returns.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.market_returns = market_returns.reset_index(drop=True)
        self.n_assets = n_assets
        self.scaler = scaler
        self.mode = mode

        state_dim = self.features.shape[1] + n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        # Internal state (initialised properly in reset())
        self.current_step: int = 0
        self.current_weights: np.ndarray = np.ones(n_assets) / n_assets
        self.portfolio_value: float = 1.0
        self.return_history: List[float] = []
        self.weight_history: List[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the start of the episode.

        Args:
            seed: Optional RNG seed (passed to the gymnasium base class).
            options: Unused; kept for API compatibility.

        Returns:
            obs: Initial observation vector.
            info: Empty info dictionary.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_weights = np.ones(self.n_assets, dtype=np.float64) / self.n_assets
        self.portfolio_value = 1.0
        self.return_history = []
        self.weight_history = []

        return self._get_state(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one trading day.

        Steps:
            1. Softmax the raw action to obtain valid portfolio weights w_t.
            2. Compute turnover = Σ|w_t − w_{t−1}|.
            3. Deduct proportional transaction cost on turnover.
            4. Apply w_t to the next day's asset returns to get the
               gross portfolio return.
            5. Build a composite reward from return, rolling Sharpe,
               rolling volatility, and transaction cost.
            6. Update internal state, advance the step counter.

        Args:
            action: Raw logit vector of shape (n_assets,); will be
                softmax-normalised inside this method.

        Returns:
            obs: Next observation vector.
            reward: Scalar composite reward.
            terminated: True when the episode has consumed all available data.
            truncated: Always False (no artificial time limit).
            info: Dictionary with keys ``portfolio_return``, ``weights``,
                ``turnover``, and ``portfolio_value``.
        """
        # 1. Weights via softmax
        exp_a = np.exp(action - np.max(action))  # numerically stable
        w_t = exp_a / exp_a.sum()

        # 2. Turnover
        turnover = float(np.sum(np.abs(w_t - self.current_weights)))

        # 3. Transaction cost
        transaction_cost = TRANSACTION_COST_RATE * turnover

        # 4. Next-day returns (current_step + 1 is the day after action)
        next_returns = self.returns.iloc[self.current_step + 1].values.astype(float)
        gross_return = float(np.dot(w_t, next_returns))
        portfolio_return = gross_return - transaction_cost

        # 5. Reward components
        self.return_history.append(portfolio_return)

        if len(self.return_history) >= REWARD_VOL_WINDOW:
            recent = self.return_history[-REWARD_VOL_WINDOW:]
        else:
            recent = self.return_history

        rolling_vol = float(np.std(recent)) if len(recent) > 1 else 0.0
        sharpe = float(np.mean(recent)) / (rolling_vol + 1e-8)

        reward = (
            REWARD_RETURN_COEF * portfolio_return
            + REWARD_SHARPE_COEF * sharpe
            - REWARD_VOL_COEF * rolling_vol
            - REWARD_TC_COEF * transaction_cost
        )

        # 6. Update state
        self.current_weights = w_t
        self.weight_history.append(w_t.copy())
        self.portfolio_value *= 1.0 + portfolio_return

        self.current_step += 1
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False

        info: Dict[str, Any] = {
            "portfolio_return": portfolio_return,
            "weights": w_t,
            "turnover": turnover,
            "portfolio_value": self.portfolio_value,
        }

        return self._get_state(), float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Return the current observation: pre-built features + weights.

        Fetches the feature row at ``current_step`` from the pre-built
        (and already normalised) feature DataFrame, then appends the current
        portfolio weight vector.

        Returns:
            Observation array of shape (features.shape[1] + n_assets,).
        """
        feature_row = self.features.iloc[self.current_step].values.astype(np.float32)
        weights = self.current_weights.astype(np.float32)
        return np.concatenate([feature_row, weights])

    def get_portfolio_returns(self) -> List[float]:
        """Return the complete list of per-step portfolio returns so far.

        Returns:
            List of floats, one entry per :meth:`step` call since the last
            :meth:`reset`.
        """
        return list(self.return_history)
