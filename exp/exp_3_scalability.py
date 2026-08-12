"""
Experiment 3 — Scalability: RL-APM vs PPO vs SAC vs Dynamic MVO on 100-asset universe
Train: 2015-2021 | Test: 2022-2024
"""

import os
import sys
import time
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
    SCALABILITY_UNIVERSE, MARKET_PROXY,
    FULL_START, FULL_END, TRAIN_END, TEST_START,
    TRAINING_EPISODES,
)
from src.data_loader import prepare_experiment_data, get_train_test_split
from src.features import build_all_features
from src.environment import PortfolioEnv
from src.ddpg import DDPGAgent
from src.mvo import DynamicMVO
from src.rl_benchmarks import train_ppo, train_sac, evaluate_sb3_agent, SoftmaxPortfolioWrapper
from src.metrics import compute_all_metrics

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Reduced SB3 timesteps for feasibility on the large universe
PPO_TIMESTEPS = 2_000   # base per-episode; scaled by n_episodes at runtime
SAC_TIMESTEPS = 2_000   # base per-episode; scaled by n_episodes at runtime

METRIC_KEYS   = ['cumulative_return', 'sharpe_ratio', 'annualized_volatility', 'max_drawdown']
METRIC_LABELS = ['Cum. Return (%)', 'Sharpe Ratio', 'Volatility (%)', 'Max DD (%)']


def _save_scalability_plot(all_metrics: Dict, out_path: str) -> None:
    methods = list(all_metrics.keys())
    fig, axes = plt.subplots(1, len(METRIC_KEYS), figsize=(20, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(methods)))

    for ax, metric_key, metric_label in zip(axes, METRIC_KEYS, METRIC_LABELS):
        vals = [all_metrics[m][metric_key] for m in methods]
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_title(metric_label, fontsize=12)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Scalability: 100-Asset Portfolio (Test Period 2022-2024)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _print_scalability_table(all_metrics: Dict) -> None:
    hdr = f"{'Method':<16} {'Return(%)':>10} {'Sharpe':>8} {'Vol(%)':>8} {'MaxDD(%)':>10} {'Runtime(s)':>12}"
    logger.info("\n%s\n%s", hdr, "-" * len(hdr))
    for method, m in all_metrics.items():
        logger.info(
            "%-16s %10.2f %8.3f %8.2f %10.2f %12.1f",
            method, m['cumulative_return'], m['sharpe_ratio'],
            m['annualized_volatility'], m['max_drawdown'], m.get('runtime', 0.0),
        )


def run_exp3(seed: int = 42, n_episodes: int = 3, save_results: bool = True) -> Dict:
    """Scalability experiment: 100-asset universe, compare DDPG/PPO/SAC/MVO.

    Args:
        seed: Random seed.
        n_episodes: DDPG training episodes (reduced from TRAINING_EPISODES for feasibility).
        save_results: Persist JSON metrics and plots.

    Returns:
        Dict mapping method name to metrics dict.
    """
    np.random.seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("Downloading data for %d assets + %s …", len(SCALABILITY_UNIVERSE), MARKET_PROXY)
    data = prepare_experiment_data(SCALABILITY_UNIVERSE, MARKET_PROXY, FULL_START, FULL_END)

    adj_close      = data['adj_close']
    returns        = data['returns']
    market_returns = data['market_returns']
    ohlcv_dict     = data['ohlcv_dict']

    # ── 2. Drop assets with >5 % missing values ───────────────────────────────
    missing_pct = adj_close.isnull().mean()
    valid_assets = missing_pct[missing_pct <= 0.05].index.tolist()
    dropped = set(SCALABILITY_UNIVERSE) - set(valid_assets)
    if dropped:
        logger.warning("Dropped %d assets with >5%% missing data.", len(dropped))

    adj_close  = adj_close[valid_assets].ffill().bfill()
    returns    = returns[valid_assets].ffill().bfill()
    ohlcv_dict = {tk: ohlcv_dict[tk] for tk in valid_assets}
    n_assets   = len(valid_assets)
    logger.info("Proceeding with %d valid assets.", n_assets)

    # ── 3. Train / test split ─────────────────────────────────────────────────
    train_adj,     test_adj     = get_train_test_split(adj_close)
    train_returns, test_returns = get_train_test_split(returns)
    train_mkt_df,  test_mkt_df  = get_train_test_split(market_returns.to_frame())
    train_market   = train_mkt_df.iloc[:, 0]
    test_market    = test_mkt_df.iloc[:, 0]

    train_ohlcv = {tk: ohlcv_dict[tk].loc[:TRAIN_END]  for tk in valid_assets}
    test_ohlcv  = {tk: ohlcv_dict[tk].loc[TEST_START:] for tk in valid_assets}

    # ── 4. Features ───────────────────────────────────────────────────────────
    logger.info("Building features for %d assets …", n_assets)
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
    logger.info("n_assets=%d  feature_cols=%d  state_dim=%d",
                n_assets, train_scaled.shape[1], state_dim)

    # ── Environment factories (return fresh instances) ────────────────────────
    def make_train_env() -> PortfolioEnv:
        return PortfolioEnv(
            returns=train_returns,
            features=train_scaled,
            market_returns=train_market,
            n_assets=n_assets,
            scaler=scaler,
            mode='train',
        )

    def make_test_env() -> PortfolioEnv:
        return PortfolioEnv(
            returns=test_returns,
            features=test_scaled,
            market_returns=test_market,
            n_assets=n_assets,
            scaler=scaler,
            mode='test',
        )

    all_metrics: Dict = {}

    # ── 5a. RL-APM / DDPG ────────────────────────────────────────────────────
    logger.info("Training DDPG (%d episodes) …", n_episodes)
    t0 = time.time()
    agent_ddpg = DDPGAgent(state_dim=state_dim, n_assets=n_assets, seed=seed)
    train_env  = make_train_env()
    for ep in range(n_episodes):
        ep_stats = agent_ddpg.train_episode(train_env, episode_idx=ep)
        logger.info("  DDPG Ep %d/%d | reward=%.4f", ep + 1, n_episodes, ep_stats['total_reward'])
    ddpg_runtime = time.time() - t0

    eval_result  = agent_ddpg.evaluate(make_test_env())
    ddpg_rets    = np.asarray(eval_result['returns'], dtype=float)
    ddpg_metrics = compute_all_metrics(ddpg_rets.tolist(), label='RL-APM (DDPG)')
    ddpg_metrics['runtime'] = ddpg_runtime
    all_metrics['RL-APM (DDPG)'] = ddpg_metrics
    logger.info("  DDPG Sharpe=%.3f  runtime=%.1fs", ddpg_metrics['sharpe_ratio'], ddpg_runtime)

    # ── 5b. PPO ───────────────────────────────────────────────────────────────
    ppo_ts = max(1_000, PPO_TIMESTEPS * n_episodes)
    logger.info("Training PPO (%d timesteps) …", ppo_ts)
    t0 = time.time()
    ppo_agent   = train_ppo(SoftmaxPortfolioWrapper(make_train_env()),
                            total_timesteps=ppo_ts, seed=seed, n_assets=n_assets)
    ppo_runtime = time.time() - t0

    ppo_result  = evaluate_sb3_agent(ppo_agent, make_test_env(), n_assets)
    ppo_rets    = np.asarray(ppo_result['returns'], dtype=float)
    ppo_metrics = compute_all_metrics(ppo_rets.tolist(), label='PPO')
    ppo_metrics['runtime'] = ppo_runtime
    all_metrics['PPO'] = ppo_metrics
    logger.info("  PPO  Sharpe=%.3f  runtime=%.1fs", ppo_metrics['sharpe_ratio'], ppo_runtime)

    # ── 5c. SAC ───────────────────────────────────────────────────────────────
    sac_ts = max(1_000, SAC_TIMESTEPS * n_episodes)
    logger.info("Training SAC (%d timesteps) …", sac_ts)
    t0 = time.time()
    sac_agent   = train_sac(SoftmaxPortfolioWrapper(make_train_env()),
                            total_timesteps=sac_ts, seed=seed, n_assets=n_assets)
    sac_runtime = time.time() - t0

    sac_result  = evaluate_sb3_agent(sac_agent, make_test_env(), n_assets)
    sac_rets    = np.asarray(sac_result['returns'], dtype=float)
    sac_metrics = compute_all_metrics(sac_rets.tolist(), label='SAC')
    sac_metrics['runtime'] = sac_runtime
    all_metrics['SAC'] = sac_metrics
    logger.info("  SAC  Sharpe=%.3f  runtime=%.1fs", sac_metrics['sharpe_ratio'], sac_runtime)

    # ── 5d. Dynamic MVO ───────────────────────────────────────────────────────
    logger.info("Running Dynamic MVO …")
    t0 = time.time()
    mvo        = DynamicMVO(n_assets)
    mvo_result = mvo.run(test_adj, test_returns)
    mvo_runtime = time.time() - t0

    mvo_rets    = np.asarray(mvo_result['returns'], dtype=float)
    mvo_metrics = compute_all_metrics(mvo_rets.tolist(), label='MVO')
    mvo_metrics['runtime'] = mvo_runtime
    all_metrics['MVO'] = mvo_metrics
    logger.info("  MVO  Sharpe=%.3f  runtime=%.1fs", mvo_metrics['sharpe_ratio'], mvo_runtime)

    # ── 6. Print summary table ────────────────────────────────────────────────
    _print_scalability_table(all_metrics)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    if save_results:
        with open(os.path.join(RESULTS_DIR, 'exp3_scalability_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)

        plot_metrics = {m: {k: v for k, v in vals.items() if k != 'runtime'}
                        for m, vals in all_metrics.items()}
        _save_scalability_plot(
            plot_metrics,
            out_path=os.path.join(RESULTS_DIR, 'exp3_scalability_comparison.png'),
        )

    return all_metrics


if __name__ == '__main__':
    results = run_exp3()
    print(json.dumps(results, indent=2))
