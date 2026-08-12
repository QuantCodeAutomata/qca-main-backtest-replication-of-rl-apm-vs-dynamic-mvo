"""
Experiment 2 — Sector-wise Robustness: RL-APM vs Dynamic MVO
Five sectors: TECH, HLTH, ENRG, FIN, CNSMR
Train: 2015-2021 | Test: 2022-2024
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
    SECTOR_UNIVERSES, SECTOR_ETFS, MARKET_PROXY,
    FULL_START, FULL_END, TRAIN_END, TEST_START,
    TRAINING_EPISODES,
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

METRIC_KEYS = ['cumulative_return', 'sharpe_ratio', 'annualized_volatility', 'max_drawdown']
METRIC_LABELS = ['Cum. Return (%)', 'Sharpe Ratio', 'Volatility (%)', 'Max DD (%)']


def run_sector(sector_name: str, tickers: List[str], sector_etf: str,
               seed: int = 42, n_episodes: int = TRAINING_EPISODES) -> Dict:
    """Train RL-APM and Dynamic MVO for one sector universe; return metrics dict.

    Args:
        sector_name: Human-readable sector label (e.g. 'TECH').
        tickers: List of constituent tickers for this sector.
        sector_etf: Sector ETF ticker used as the market-context proxy.
        seed: Random seed.

    Returns:
        Dict {'RL-APM': metrics, 'MVO': metrics}.
    """
    logger.info("=== Sector: %s | %d assets | ETF: %s ===", sector_name, len(tickers), sector_etf)
    np.random.seed(seed)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    # Use sector ETF as the market context proxy (instead of SPY)
    data = prepare_experiment_data(tickers, sector_etf, FULL_START, FULL_END)

    adj_close      = data['adj_close']
    returns        = data['returns']
    market_returns = data['market_returns']
    ohlcv_dict     = data['ohlcv_dict']
    n_assets       = len(tickers)

    train_adj,     test_adj     = get_train_test_split(adj_close)
    train_returns, test_returns = get_train_test_split(returns)
    train_mkt_df,  test_mkt_df  = get_train_test_split(market_returns.to_frame())
    train_market   = train_mkt_df.iloc[:, 0]
    test_market    = test_mkt_df.iloc[:, 0]

    train_ohlcv = {tk: ohlcv_dict[tk].loc[:TRAIN_END]  for tk in tickers}
    test_ohlcv  = {tk: ohlcv_dict[tk].loc[TEST_START:] for tk in tickers}

    # ── 2. Features ───────────────────────────────────────────────────────────
    train_feat_df = build_all_features(train_adj, train_ohlcv, train_market).dropna()
    test_feat_df  = build_all_features(test_adj,  test_ohlcv,  test_market).dropna()

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

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_feat_df.values),
        index=train_feat_df.index, columns=train_feat_df.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_feat_df.values),
        index=test_feat_df.index, columns=test_feat_df.columns,
    )

    state_dim = train_scaled.shape[1] + n_assets

    # ── 3. Train RL-APM ───────────────────────────────────────────────────────
    train_env = PortfolioEnv(
        returns=train_returns,
        features=train_scaled,
        market_returns=train_market,
        n_assets=n_assets,
        scaler=scaler,
        mode='train',
    )
    agent = DDPGAgent(state_dim=state_dim, n_assets=n_assets, seed=seed)

    for ep in range(n_episodes):
        ep_stats  = agent.train_episode(train_env, episode_idx=ep)
        avg_actor = float(np.mean(ep_stats['actor_losses'])) if ep_stats['actor_losses'] else 0.0
        logger.info(
            "  [%s] Ep %d/%d | reward=%.4f | actor_loss=%.6f",
            sector_name, ep + 1, n_episodes, ep_stats['total_reward'], avg_actor,
        )

    # ── 4. MVO benchmark ──────────────────────────────────────────────────────
    mvo = DynamicMVO(n_assets)

    # ── 5. Evaluate on test ───────────────────────────────────────────────────
    test_env = PortfolioEnv(
        returns=test_returns,
        features=test_scaled,
        market_returns=test_market,
        n_assets=n_assets,
        scaler=scaler,
        mode='test',
    )
    eval_result     = agent.evaluate(test_env)
    rl_returns_arr  = np.asarray(eval_result['returns'], dtype=float)

    mvo_result      = mvo.run(test_adj, test_returns)
    mvo_returns_arr = np.asarray(mvo_result['returns'], dtype=float)

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    rl_metrics  = compute_all_metrics(rl_returns_arr.tolist(),  label='RL-APM')
    mvo_metrics = compute_all_metrics(mvo_returns_arr.tolist(), label='MVO')

    logger.info(
        "  [%s] RL-APM Sharpe=%.3f | MVO Sharpe=%.3f",
        sector_name, rl_metrics['sharpe_ratio'], mvo_metrics['sharpe_ratio'],
    )
    return {'RL-APM': rl_metrics, 'MVO': mvo_metrics}


def _save_sector_comparison_plots(all_results: Dict, out_path: str):
    sectors = list(all_results.keys())
    n_metrics = len(METRIC_KEYS)

    # ── Figure 1: grouped bar charts per metric ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    colors = {'RL-APM': 'steelblue', 'MVO': 'darkorange'}

    for ax, metric_key, metric_label in zip(axes, METRIC_KEYS, METRIC_LABELS):
        x = np.arange(len(sectors))
        width = 0.35
        rl_vals = [all_results[s]['RL-APM'][metric_key] for s in sectors]
        mvo_vals = [all_results[s]['MVO'][metric_key] for s in sectors]
        ax.bar(x - width / 2, rl_vals, width, label='RL-APM', color=colors['RL-APM'], alpha=0.85)
        ax.bar(x + width / 2, mvo_vals, width, label='MVO', color=colors['MVO'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sectors, rotation=15, ha='right')
        ax.set_title(metric_label, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    fig.suptitle('Sector-Wise RL-APM vs MVO Comparison (Test Period)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved sector comparison plot → {out_path}")

    # ── Figure 2: heatmap of metrics by sector and method ────────────────────
    heatmap_path = out_path.replace('exp2_sector_comparison.png', 'exp2_sector_heatmap.png')
    rows = []
    for sector in sectors:
        for method in ('RL-APM', 'MVO'):
            row = {'Sector': sector, 'Method': method}
            row.update({k: all_results[sector][method][k] for k in METRIC_KEYS})
            rows.append(row)
    df_heat = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, max(4, len(sectors) * 0.8 + 2)))
    for ax, metric_key, metric_label in zip(axes, METRIC_KEYS, METRIC_LABELS):
        pivot = df_heat.pivot(index='Sector', columns='Method', values=metric_key)
        sns.heatmap(pivot, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                    linewidths=0.5, cbar=True)
        ax.set_title(metric_label, fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('' if ax != axes[0] else 'Sector')

    fig.suptitle('Heatmap: Metrics by Sector and Method (Test Period)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved sector heatmap → {heatmap_path}")


def run_exp2(n_episodes: int = TRAINING_EPISODES, save_results: bool = True) -> Dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results: Dict = {}

    for sector_name, tickers in SECTOR_UNIVERSES.items():
        sector_etf = SECTOR_ETFS[sector_name]
        sector_results = run_sector(sector_name, tickers, sector_etf, n_episodes=n_episodes)
        all_results[sector_name] = sector_results

    if save_results:
        metrics_path = os.path.join(RESULTS_DIR, 'exp2_sector_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved sector metrics → {metrics_path}")

        plot_path = os.path.join(RESULTS_DIR, 'exp2_sector_comparison.png')
        _save_sector_comparison_plots(all_results, plot_path)

    return all_results


if __name__ == '__main__':
    results = run_exp2()
    print(json.dumps(results, indent=2))
