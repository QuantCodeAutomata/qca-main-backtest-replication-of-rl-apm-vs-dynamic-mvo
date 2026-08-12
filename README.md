# RL-APM vs Dynamic MVO — Replication Repository

A full replication of the paper's core claim that a **DDPG-based Reinforcement
Learning Asset/Portfolio Manager (RL-APM)** with a custom multi-objective reward
outperforms a **Dynamic Mean-Variance Optimization (MVO)** benchmark on daily
equity allocation.

---

## Experiments

| ID | Title | Description |
|----|-------|-------------|
| exp_1 | Main Backtest | RL-APM vs Dynamic MVO on 20 S&P 500 large-caps, daily 2015-2024 |
| exp_2 | Sector-Wise Robustness | Independent RL-APM vs MVO runs on TECH, HLTH, ENRG, FIN, CNSMR |
| exp_3 | Scalability | RL-APM vs PPO vs SAC vs MVO on a 100-asset universe |

---

## Repository Structure

```
.
├── data/                        # (empty at commit; populated at runtime by yfinance)
├── src/
│   ├── config.py                # All replication assumptions & hyperparameters
│   ├── data_loader.py           # yfinance download, alignment, train/test split
│   ├── features.py              # Technical indicators + state construction
│   ├── environment.py           # gymnasium PortfolioEnv
│   ├── ddpg.py                  # Custom PyTorch DDPG (Actor/Critic/OUNoise/Replay)
│   ├── mvo.py                   # Dynamic MVO via PyPortfolioOpt / cvxpy
│   ├── rl_benchmarks.py         # PPO & SAC wrappers (Stable-Baselines3)
│   └── metrics.py               # Cumulative return, volatility, Sharpe, MDD
├── exp/
│   ├── exp_1_main_backtest.py
│   ├── exp_2_sector_wise.py
│   └── exp_3_scalability.py
├── tests/
│   ├── test_ddpg.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_mvo.py
├── results/
│   └── RESULTS.md               # Auto-generated experiment summary
├── run_all_experiments.py       # Master runner
├── requirements.txt
└── README.md
```

---

## Key Methodology

### Reward Function (exact paper pseudocode)
```
reward = 0.6 * return
       + 0.3 * sharpe
       - 0.1 * volatility
       - 0.05 * transaction_cost
```

### DDPG Architecture (Tables 4 & 5)
**Actor:** `state_dim → 256(ReLU) → 256(ReLU) → 128(ReLU) → n_assets → Softmax`
**Critic:** state branch `256(ReLU)→128(ReLU)` + action branch `128(ReLU)` → concat → `256→128(ReLU)→1`

### Hyperparameters (Table 6)
| Parameter | Value |
|-----------|-------|
| Actor LR | 1e-4 |
| Critic LR | 1e-3 |
| Batch size | 64 |
| Replay buffer | 100 000 |
| τ (soft-update) | 0.001 |
| γ (discount) | 0.99 |
| OU noise σ | 0.2 |
| Transaction cost | 10 bps / unit turnover |

---

## Replication Assumptions (documented in `src/config.py`)

Because the paper leaves several details unspecified, the following choices
were made and documented:

- **Universe (exp_1):** 20 liquid S&P 500 large-caps (AAPL, MSFT, AMZN, GOOGL, …)
- **Sector universes (exp_2):** ~10 large-cap constituents per GICS sector
- **Scalability universe (exp_3):** 100 liquid S&P 500 equities
- **MVO estimation window:** 252 trading days
- **Risk-free rate:** constant 4.5 % p.a. ÷ 252 (daily)
- **Missing data:** forward-fill then drop rows with any remaining NaN
- **Normalization:** training-set StandardScaler (no look-ahead)
- **Recent-return block:** 30-day trailing daily returns

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys (optional — only needed if using Massive API)
```bash
export MASSIVE_API_KEY=your_key_here
```

### 3. Run full experiment suite (quick demo, ~15 min)
```bash
python run_all_experiments.py --quick
```

### 4. Run full-convergence version (several hours per experiment)
```bash
python run_all_experiments.py
```

### 5. Run individual experiments
```bash
python exp/exp_1_main_backtest.py --episodes 20
python exp/exp_2_sector_wise.py   --episodes 10
python exp/exp_3_scalability.py   --episodes 5
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full metrics tables and plots.

**Note:** Quick-mode runs (3/2/1 episodes) produce unconverged RL agents.
For paper-matching results, run with ≥ 50 episodes for exp_1/exp_2 and ≥ 20
for exp_3.

---

## Fidelity Notes

- Paper leaves the exact asset universe unspecified → fixed list documented in `src/config.py`.
- Paper's DDPG is described for TensorFlow 2.13; this replication uses **PyTorch** for framework simplicity while preserving exact architecture and hyperparameters.
- PPO/SAC (exp_3) use **Stable-Baselines3** with a custom gymnasium environment and softmax action wrapper.
- Survivorship bias: the universe uses current-day large-caps; future work should use point-in-time S&P 500 membership.

---

## Citation

If you use this replication codebase, please cite the original paper and note
this is an independent replication with documented assumptions.
