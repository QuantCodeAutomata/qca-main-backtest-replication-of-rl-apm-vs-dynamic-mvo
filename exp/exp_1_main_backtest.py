"""
Experiment 1 — Main Backtest: RL-APM (DDPG) vs Dynamic MVO
Train: 2015-2021 | Test: 2022-2024 | Universe: 20 S&P 500 large-caps
"""

import os
import sys
import json
import warnings
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

from src.config import (
    MAIN_UNIVERSE, MARKET_PROXY, FULL_START, FULL_END,
    TRAIN_END, TEST_START, TRAINING_EPISODES,
)
from src.data_loader import prepare_experiment_data, get_train_test_split
from src.features import build_all_features
from src.environment import PortfolioEnv
from src.ddpg import DDPGAgent
from src.mvo import DynamicMVO
from src.metrics import compute_all_metrics, portfolio_value_series

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


# ─── Plotting helpers ──────────────────────────────────────────────────────────

def _save_cumulative_returns_plot(
    rl_returns: np.ndarray,
    mvo_returns: np.ndarray,
    rl_metrics: Dict,
    mvo_metrics: Dict,
    dates=None,
    out_path: str = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    rl_values  = portfolio_value_series(rl_returns.tolist())
    mvo_values = portfolio_value_series(mvo_returns.tolist())

    x_rl  = dates[-len(rl_values):]  if (dates is not None and len(dates) >= len(rl_values))  else range(len(rl_values))
    x_mvo = dates[-len(mvo_values):] if (dates is not None and len(dates) >= len(mvo_values)) else range(len(mvo_values))

    axes[0].plot(x_rl,  rl_values,  label='RL-APM (DDPG)', linewidth=2, color='steelblue')
    axes[0].plot(x_mvo, mvo_values, label='MVO',            linewidth=2, color='darkorange')
    axes[0].set_title('Portfolio Value — Test Period (2022-2024)', fontsize=13)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value (start = 1.0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if dates is not None:
        fig.autofmt_xdate()

    metric_labels = ['Cum. Return (%)', 'Sharpe Ratio', 'Volatility (%)', 'Max DD (%)']
    metric_keys   = ['cumulative_return', 'sharpe_ratio', 'annualized_volatility', 'max_drawdown']
    rl_vals  = [rl_metrics[k]  for k in metric_keys]
    mvo_vals = [mvo_metrics[k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    w = 0.35
    axes[1].bar(x - w/2, rl_vals,  w, label='RL-APM', color='steelblue', alpha=0.85)
    axes[1].bar(x + w/2, mvo_vals, w, label='MVO',    color='darkorange', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_labels, rotation=15, ha='right')
    axes[1].set_title('Metrics Comparison — Test Period', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _save_weights_plot(weights: List[np.ndarray], asset_names: List[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    weights_arr = np.array(weights)
    if weights_arr.ndim == 2 and weights_arr.shape[1] > 0:
        ax.stackplot(range(len(weights_arr)), weights_arr.T, labels=asset_names, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', ncol=5, fontsize=7, framealpha=0.6)
        ax.set_title('RL-APM Portfolio Weights Evolution — Test Period', fontsize=13)
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Portfolio Weight')
    else:
        ax.text(0.5, 0.5, 'Weights data unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ─── Main experiment function ──────────────────────────────────────────────────

def run_exp1(seed: int = 42, n_episodes: int = TRAINING_EPISODES,
             save_results: bool = True) -> Dict:
    """Run the main backtest: RL-APM vs Dynamic MVO.

    Args:
        seed: Random seed for reproducibility.
        n_episodes: Number of DDPG training episodes over the training window.
        save_results: Whether to persist metrics JSON and plots.

    Returns:
        Nested dict {'RL-APM': metrics_dict, 'MVO': metrics_dict}.
    """
    np.random.seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("Downloading data for %d assets + %s …", len(MAIN_UNIVERSE), MARKET_PROXY)
    data = prepare_experiment_data(MAIN_UNIVERSE, MARKET_PROXY, FULL_START, FULL_END)

    adj_close      = data['adj_close']        # (dates, n_assets) DataFrame
    returns        = data['returns']           # (dates-1, n_assets) DataFrame
    market_returns = data['market_returns']    # Series
    ohlcv_dict     = data['ohlcv_dict']        # Dict[ticker -> OHLCV DataFrame]

    n_assets = len(MAIN_UNIVERSE)

    # Train / test splits
    train_adj,     test_adj     = get_train_test_split(adj_close)
    train_returns, test_returns = get_train_test_split(returns)
    train_market,  test_market  = get_train_test_split(market_returns.to_frame())
    train_market   = train_market.iloc[:, 0]
    test_market    = test_market.iloc[:, 0]

    # Per-ticker OHLCV splits (needed by build_all_features)
    train_ohlcv = {tk: ohlcv_dict[tk].loc[:TRAIN_END]  for tk in MAIN_UNIVERSE}
    test_ohlcv  = {tk: ohlcv_dict[tk].loc[TEST_START:] for tk in MAIN_UNIVERSE}

    logger.info("Train rows: %d | Test rows: %d", len(train_returns), len(test_returns))

    # ── 2. Features ───────────────────────────────────────────────────────────
    logger.info("Building features …")
    train_feat_df = build_all_features(train_adj, train_ohlcv, train_market).dropna()
    test_feat_df  = build_all_features(test_adj,  test_ohlcv,  test_market).dropna()

    # Align all DataFrames to common indices (features drive the calendar)
    train_common = train_feat_df.index.intersection(train_returns.index).intersection(train_market.index)
    test_common  = test_feat_df.index.intersection(test_returns.index).intersection(test_market.index)

    train_feat_df  = train_feat_df.loc[train_common]
    train_returns  = train_returns.loc[train_common]
    train_market   = train_market.loc[train_common]
    train_adj      = train_adj.loc[train_common]

    test_feat_df   = test_feat_df.loc[test_common]
    test_returns   = test_returns.loc[test_common]
    test_market    = test_market.loc[test_common]
    test_adj       = test_adj.loc[test_common]

    # Fit scaler on training features only (no look-ahead)
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_feat_df.values),
        index=train_feat_df.index, columns=train_feat_df.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_feat_df.values),
        index=test_feat_df.index, columns=test_feat_df.columns,
    )

    # state_dim = feature_cols + n_assets (weights appended in _get_state)
    state_dim = train_scaled.shape[1] + n_assets
    logger.info("n_assets=%d  feature_cols=%d  state_dim=%d",
                n_assets, train_scaled.shape[1], state_dim)

    # ── 3. Train RL-APM ───────────────────────────────────────────────────────
    logger.info("Training RL-APM for %d episodes …", n_episodes)
    train_env = PortfolioEnv(
        returns=train_returns,
        features=train_scaled,
        market_returns=train_market,
        n_assets=n_assets,
        scaler=scaler,
        mode='train',
    )

    agent = DDPGAgent(state_dim=state_dim, n_assets=n_assets, seed=seed)
    training_rewards: List[float] = []

    for ep in range(n_episodes):
        ep_stats = agent.train_episode(train_env, episode_idx=ep)
        training_rewards.append(ep_stats['total_reward'])
        avg_actor  = float(np.mean(ep_stats['actor_losses']))  if ep_stats['actor_losses']  else 0.0
        avg_critic = float(np.mean(ep_stats['critic_losses'])) if ep_stats['critic_losses'] else 0.0
        logger.info(
            "  Ep %d/%d | reward=%.4f | actor_loss=%.6f | critic_loss=%.6f | steps=%d",
            ep + 1, n_episodes, ep_stats['total_reward'], avg_actor, avg_critic, ep_stats['n_steps'],
        )

    # ── 4. MVO benchmark ──────────────────────────────────────────────────────
    logger.info("Running Dynamic MVO benchmark …")
    mvo = DynamicMVO(n_assets)

    # ── 5. Evaluate on test ───────────────────────────────────────────────────
    logger.info("Evaluating RL-APM on test period …")
    test_env = PortfolioEnv(
        returns=test_returns,
        features=test_scaled,
        market_returns=test_market,
        n_assets=n_assets,
        scaler=scaler,
        mode='test',
    )
    eval_result    = agent.evaluate(test_env)
    rl_returns_arr = np.asarray(eval_result['returns'], dtype=float)
    rl_weights     = eval_result['weights']

    logger.info("Evaluating MVO on test period …")
    mvo_result     = mvo.run(test_adj, test_returns)
    mvo_returns_arr = np.asarray(mvo_result['returns'], dtype=float)
    mvo_weights    = mvo_result['weights']

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    rl_metrics  = compute_all_metrics(rl_returns_arr.tolist(),  label='RL-APM')
    mvo_metrics = compute_all_metrics(mvo_returns_arr.tolist(), label='MVO')

    logger.info("=== Test Period Results ===")
    for m in ('cumulative_return', 'sharpe_ratio', 'annualized_volatility', 'max_drawdown'):
        logger.info("  %-28s  RL-APM=%8.4f  MVO=%8.4f", m, rl_metrics[m], mvo_metrics[m])

    results = {'RL-APM': rl_metrics, 'MVO': mvo_metrics}

    # ── 7. Save ───────────────────────────────────────────────────────────────
    if save_results:
        with open(os.path.join(RESULTS_DIR, 'exp1_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)

        _save_cumulative_returns_plot(
            rl_returns_arr, mvo_returns_arr, rl_metrics, mvo_metrics,
            dates=test_returns.index,
            out_path=os.path.join(RESULTS_DIR, 'exp1_cumulative_returns.png'),
        )
        _save_weights_plot(
            rl_weights, MAIN_UNIVERSE,
            out_path=os.path.join(RESULTS_DIR, 'exp1_weights_evolution.png'),
        )

    return results


if __name__ == '__main__':
    results = run_exp1()
    print(json.dumps(results, indent=2))
