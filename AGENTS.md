# Project: RL-APM — DDPG Portfolio Management

## Overview
Replication of RL-based Adaptive Portfolio Management (RL-APM) using DDPG.
Compares against a Dynamic MVO benchmark. Python 3.12, PyTorch 2.13, NumPy 1.26.

## Repository Layout
```
src/config.py          — All hyperparameters and universe definitions (read first)
src/data_loader.py     — download_data, compute_returns, get_train_test_split, save/load_data
src/features.py        — compute_asset_features, build_state, build_all_features
src/environment.py     — PortfolioEnv (Gymnasium Env)
src/ddpg.py            — DDPG agent: OUNoise, ReplayBuffer, Actor, Critic, DDPGAgent
src/metrics.py         — Performance metrics: cumulative_return, annualized_volatility,
                         sharpe_ratio, maximum_drawdown, compute_all_metrics,
                         portfolio_value_series
src/mvo.py             — DynamicMVO, equal_weight_portfolio
src/rl_benchmarks.py   — SoftmaxPortfolioWrapper, train_ppo, train_sac, evaluate_sb3_agent
tests/test_features.py — 7 pytest unit tests for features + environment (all pass)
tests/test_ddpg.py     — 9 pytest unit tests for all DDPG components
tests/test_metrics.py  — 15 pytest unit tests for metrics (all pass)
exp/exp_1_main_backtest.py  — Main backtest: DDPG vs MVO on MAIN_UNIVERSE
exp/exp_2_sector_wise.py    — Sector-wise: DDPG vs MVO per sector (SECTOR_UNIVERSES)
exp/exp_3_scalability.py    — Scalability: DDPG+PPO+SAC+MVO on 100-asset universe
data/                  — raw price data (populated externally)
results/               — evaluation outputs (JSON metrics + PNG plots)
```

## Key Constants (src/config.py)
| Symbol | Value | Meaning |
|---|---|---|
| ACTOR_LR | 1e-4 | Adam lr for policy network |
| CRITIC_LR | 1e-3 | Adam lr for Q-network |
| BATCH_SIZE | 64 | Mini-batch size for DDPG updates |
| REPLAY_BUFFER | 100 000 | Circular buffer capacity |
| TAU | 0.001 | Soft-update coefficient |
| GAMMA | 0.99 | Discount factor |
| OU_THETA / OU_SIGMA | 0.15 / 0.2 | Ornstein-Uhlenbeck noise params |
| TRAINING_EPISODES | 10 | Passes over training data |

## Architecture (Paper Tables 4 & 5)
**Actor**: `Linear(state,256,ReLU) → Linear(256,256,ReLU) → Linear(256,128,ReLU) → Linear(128,n_assets) → Softmax`
**Critic state branch**: `Linear(state,256,ReLU) → Linear(256,128,ReLU)`
**Critic action branch**: `Linear(n_assets,128,ReLU)`
**Critic merge**: concat → `Linear(256,128,ReLU) → Linear(128,1)`

## Design Decisions
- `select_action` adds OU noise to softmax output then re-normalises with numerically stable softmax (`exp(x - max(x))/sum(...)`) to keep actions on the probability simplex.
- Replay buffer uses `collections.deque(maxlen=capacity)` for O(1) circular eviction.
- Target network parameters have `requires_grad = False` after init.
- `save`/`load` persist only `actor` and `critic` state_dicts under a single `.pt` checkpoint; targets are synced from loaded weights on `load`.
- `train_episode` calls `env.reset()` → `env.step(action)` returning `(next_state, reward, done, info)` (standard Gym interface).

## Running Tests
```bash
cd /workspace/project
python -m pytest tests/test_features.py -v  # 7 tests, ~2 s
python -m pytest tests/test_ddpg.py -v      # 9 tests, ~5 s
python -m pytest tests/test_mvo.py -v       # 6 tests, ~2 s
python -m pytest tests/test_metrics.py -v   # 15 tests, ~1 s
```
All tests use synthetic/deterministic data with small `state_dim` and `n_assets`.

## Data Pipeline (src/data_loader.py, src/features.py, src/environment.py)

### src/data_loader.py
- **`download_data(tickers, market_proxy, start, end)`**: calls `yf.download(..., auto_adjust=False, progress=False)`; returns `(prices_df, spy_series)` where prices_df has MultiIndex columns `(field, ticker)`. Forward-fills NaN then drops rows with any remaining NaN in Adj Close.
- **`compute_returns(adj_close)`**: `pct_change().dropna()` — simple daily returns.
- **`get_train_test_split(df)`**: slices `df.loc[:TRAIN_END]` and `df.loc[TEST_START:]`.
- **`save_data(data, path)` / `load_data(path)`**: pickle for `.pkl` paths, JSON for `.json` paths.

### src/features.py
- **`N_ASSET_FEATURES = 17`** per-asset feature columns: SMA×4, EMA, RSI, BB×4, MACD×3, rolling_vol, ROC, ATR, OBV.
- **`compute_asset_features(ohlcv, ticker)`**: requires columns `Adj Close`, `High`, `Low`, `Volume`. Returns DataFrame shape `(len(ohlcv), 17)` with `<ticker>_` prefix. Early rows have NaN (rolling windows not yet filled).
- **`build_all_features(adj_close, ohlcv_dict, market_returns)`**: pre-builds all asset features + 3 context cols + n_assets trailing-return cols. Drops first `WARMUP_DAYS=60` rows. Output shape: `(T - 60, 17*n_assets + 3 + n_assets)`.
- **`build_state(adj_close, ohlcv_dict, weights, market_returns, scaler, t, fit_scaler=False)`**: online single-step builder; returns `(state_vec, scaler)`. State = [F_t | W_t | C_t | R_t].

### src/environment.py — PortfolioEnv
- action_space: `Box(0, 1, (n_assets,))` — raw logits, internally softmax-normalised.
- observation_space: `Box(-inf, inf, (features.shape[1] + n_assets,))`.
- `reset(seed=None)` → `(obs, {})`.  Initial weights = 1/n_assets.
- `step(action)` → `(obs, reward, terminated, truncated, info)`.  info keys: `portfolio_return`, `weights`, `turnover`, `portfolio_value`.
- Reward = `REWARD_RETURN_COEF*ret + REWARD_SHARPE_COEF*sharpe - REWARD_VOL_COEF*vol - REWARD_TC_COEF*tc`.
- `_get_state()` = `features.iloc[current_step]` ∥ `current_weights`.
- Termination: `current_step >= len(returns) - 1` (one fewer valid step than rows since step uses `returns.iloc[step+1]`).
- `features` and `returns` passed to `__init__` must have the same number of rows and aligned integer index (reset via `reset_index(drop=True)` internally).

### Feature → Env integration pattern
```python
features = build_all_features(adj_close, ohlcv_dict, market_returns)
# Align returns to the post-warmup window
returns_aligned = returns.loc[features.index].reset_index(drop=True)
features_aligned = features.reset_index(drop=True)
# Fit scaler on train portion, transform both
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(features_aligned.loc[:train_cutoff])
features_scaled = pd.DataFrame(scaler.transform(features_aligned), columns=features_aligned.columns)
env = PortfolioEnv(returns_aligned, features_scaled, market_returns_aligned, n_assets)
```

## Benchmark Implementations

### src/mvo.py — Dynamic MVO
- **`DynamicMVO(n_assets, estimation_window=252, transaction_cost_rate=0.001)`**
  - `compute_weights(price_history: pd.DataFrame) → np.ndarray`: fits EfficientFrontier max-Sharpe on a trailing window; falls back to equal weights on any exception (singular covariance, negative Sharpe, etc.).
  - `run(adj_close, returns) → {'returns': list, 'weights': list}`: loops `i in range(estimation_window, len(returns))`; at each step uses `adj_close.iloc[i-window:i]` (no lookahead); subtracts `tc = tc_rate × Σ|w_t - w_{t-1}|`.
- **`equal_weight_portfolio(returns) → {'returns': list, 'weights': list}`**: 1/N across all dates, no warmup.
- Indexing convention: `adj_close` and `returns` are passed separately; `returns` typically has one fewer row than `adj_close` (pct_change().dropna()); output length = `len(returns) - estimation_window`.

### src/rl_benchmarks.py — PPO & SAC Benchmarks (stable-baselines3)
- **`SoftmaxPortfolioWrapper(gym.Wrapper)`**: overrides `action_space` to `Box(-3, 3, n_assets)` for logits; `step()` applies numerically stable softmax before forwarding to inner env.
- **`PortfolioCallback(BaseCallback)`**: accumulates `episode_returns` from `info['portfolio_return']` in `_on_step`.
- **`train_ppo(env, total_timesteps, seed, n_assets) → PPO`**: wraps env, uses `net_arch=dict(pi=[256,256,128], vf=[256,256,128])`, `n_steps=2048`. If `total_timesteps` is None, tries `len(env.returns) * TRAINING_EPISODES`.
- **`train_sac(env, total_timesteps, seed, n_assets) → SAC`**: similar; uses `net_arch=dict(pi=[256,256,128], qf=[256,256,128])`.
- **`evaluate_sb3_agent(model, env, n_assets) → {'returns': list, 'weights': list}`**: runs one deterministic episode on the raw (non-wrapped) env; manually applies softmax to model outputs before stepping.
- Installed packages: `stable-baselines3==2.9.0`, `pyportfolioopt==1.6.0`.

## Experiment Design & Data Flow

### Common pipeline (exp_1, exp_2, exp_3)
1. `download_data(tickers, start, end) → {'adj_close': DataFrame, ...}`
2. `compute_returns(adj_close) → DataFrame`
3. `get_train_test_split(data, returns, train_start, train_end, test_start, test_end) → (train_data, test_data)` — each dict has `'adj_close'` and `'returns'` keys.
4. `build_all_features(adj_close, market_returns) → DataFrame` — may have NaN in first WARMUP_DAYS rows (SMA-50); always call `_align_features_to_returns()` to drop NaN and intersect indices before scaling.
5. Fit `StandardScaler` on train features only; transform both train and test.
6. `state_dim = train_features.shape[1] + 2 * n_assets + 3` (feature_cols×n_assets + weights + context3 + recent_per_asset).

### Key API invariants
- `DynamicMVO(n_assets)` — n_assets required as first positional arg.
- `DynamicMVO.run(adj_close, returns)` — output `{'returns': list, 'weights': list}`; output length = `len(returns) - MVO_ESTIMATION_WINDOW`.
- `train_ppo(env, total_timesteps, seed, n_assets)` / `train_sac(...)` — SB3 model; env must be `SoftmaxPortfolioWrapper`-wrapped.
- `evaluate_sb3_agent(model, env, n_assets)` — env must be the **raw** `PortfolioEnv` (not wrapped); returns `{'returns': list, 'weights': list}`.
- `DDPGAgent.train_episode(env, episode_idx) → dict` — keys: `total_reward`, `actor_losses`, `critic_losses`, `n_steps`.
- `DDPGAgent.evaluate(env) → dict` — keys: `returns` (step rewards), `weights`, `total_reward`.

### Output files
| Experiment | JSON | Plots |
|---|---|---|
| exp_1 | results/exp1_metrics.json | exp1_cumulative_returns.png, exp1_weights_evolution.png |
| exp_2 | results/exp2_sector_metrics.json | exp2_sector_comparison.png, exp2_sector_heatmap.png |
| exp_3 | results/exp3_scalability_metrics.json | exp3_scalability_comparison.png |

### Running tests
```bash
cd /workspace/project
python -m pytest tests/test_metrics.py -v   # 15 tests, all pass
```
