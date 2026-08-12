"""
Configuration for RL-APM vs Dynamic MVO replication experiments.

All replication assumptions are documented here as per paper methodology.
"""

# ─── Training / Test Split ─────────────────────────────────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-12-31"
FULL_START  = "2015-01-01"
FULL_END    = "2024-12-31"

# ─── Warm-up ───────────────────────────────────────────────────────────────
WARMUP_DAYS = 60  # max indicator window (SMA-50) + buffer

# ─── Risk-free rate (daily, annualised ~4.5% ÷ 252) ───────────────────────
RISK_FREE_DAILY = 0.045 / 252

# ─── Transaction cost model ────────────────────────────────────────────────
# Proportional turnover cost: 10 bps per unit turnover (paper unspecified)
TRANSACTION_COST_RATE = 0.001   # 10 bps

# ─── MVO benchmark rolling estimation window ───────────────────────────────
MVO_ESTIMATION_WINDOW = 252   # 1 calendar year of trading days

# ─── Reward coefficients (exact paper pseudocode) ──────────────────────────
REWARD_RETURN_COEF   = 0.6
REWARD_SHARPE_COEF   = 0.3
REWARD_VOL_COEF      = 0.1
REWARD_TC_COEF       = 0.05
REWARD_VOL_WINDOW    = 30   # rolling window for reward volatility / Sharpe

# ─── DDPG Hyperparameters (Table 6 primary) ───────────────────────────────
ACTOR_LR        = 1e-4
CRITIC_LR       = 1e-3
BATCH_SIZE      = 64
REPLAY_BUFFER   = 100_000
TAU             = 0.001
GAMMA           = 0.99
OU_SIGMA        = 0.2    # Ornstein-Uhlenbeck exploration noise sigma
OU_THETA        = 0.15   # OU mean-reversion speed (standard default)
TRAINING_EPISODES = 10   # passes over training data (paper ambiguous)

# ─── Technical indicator windows ──────────────────────────────────────────
SMA_WINDOWS     = [5, 10, 20, 50]
RSI_WINDOW      = 14
BB_WINDOW       = 20
BB_STD          = 2
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
VOLATILITY_WINDOW = 20  # rolling vol feature window
RECENT_RETURN_WINDOW = 30  # R_t block

# ─── Exp-1 Main Universe (20 liquid S&P 500 stocks) ───────────────────────
# Assumption: fixed 20 large-cap S&P 500 constituents with full 2015-2024
# data coverage; broad sector representation.
MAIN_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "JPM",  "BAC",  "WFC",  "GS",   "V",
    "JNJ",  "PFE",  "UNH",  "MRK",  "ABT",
    "XOM",  "CVX",  "COP",  "SLB",  "EOG",
]
MARKET_PROXY = "SPY"

# ─── Exp-2 Sector Universes ────────────────────────────────────────────────
# Assumption: 10 liquid large-cap constituents per sector (S&P 500 GICS)
SECTOR_UNIVERSES = {
    "TECH": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "INTC", "AMD"],
    "HLTH": ["JNJ",  "UNH",  "PFE",  "ABT",  "MRK",  "LLY",  "TMO",  "DHR",  "AMGN", "GILD"],
    "ENRG": ["XOM",  "CVX",  "COP",  "SLB",  "EOG",  "MPC",  "PSX",  "VLO",  "HAL",  "BKR"],
    "FIN":  ["JPM",  "BAC",  "WFC",  "GS",   "MS",   "C",    "BLK",  "SCHW", "AXP",  "PNC"],
    "CNSMR":["AMZN", "HD",   "MCD",  "NKE",  "SBUX", "TGT",  "LOW",  "TJX",  "COST", "YUM"],
}
SECTOR_ETFS = {
    "TECH":  "XLK",
    "HLTH":  "XLV",
    "ENRG":  "XLE",
    "FIN":   "XLF",
    "CNSMR": "XLY",
}

# ─── Exp-3 Scalability – 100-asset universe ────────────────────────────────
# Assumption: 100 liquid S&P 500 large-cap tickers with broad sector coverage
# Survivorship bias disclaimer: list uses present-day large-caps.
SCALABILITY_UNIVERSE = [
    # Technology
    "AAPL","MSFT","NVDA","AVGO","ORCL","CSCO","ADBE","CRM","INTC","AMD",
    "QCOM","TXN","AMAT","MU","KLAC","LRCX","ADI","MRVL","NXPI","CDNS",
    # Healthcare
    "JNJ","UNH","PFE","ABT","MRK","LLY","TMO","DHR","AMGN","GILD",
    "ISRG","BSX","MDT","SYK","EW","ZTS","REGN","VRTX","HCA","CI",
    # Energy
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","HAL","BKR",
    # Financials
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","PNC",
    "USB","TFC","COF","MTB","FITB","KEY","RF","HBAN","CFG","ZION",
    # Consumer Discretionary
    "AMZN","HD","MCD","NKE","SBUX","TGT","LOW","TJX","COST","YUM",
    # Consumer Staples
    "PG","KO","PEP","WMT","PM","MO","CL","GIS","KHC","CAG",
    # Industrials
    "HON","UPS","CAT","DE","MMM","GE","RTX","LMT","BA","NOC",
    # Materials
    "LIN","APD","SHW","ECL","NEM","FCX","NUE","VMC","MLM","PPG",
    # Communication Services
    "GOOGL","META","NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","PARA",
    # Real Estate / Utilities
    "AMT","PLD","EQIX","CCI","SPG",
]
# Trim to exactly 100
SCALABILITY_UNIVERSE = SCALABILITY_UNIVERSE[:100]
