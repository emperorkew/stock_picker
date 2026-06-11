"""Central configuration for the stock picker."""

# Universe
SP500_SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "WMT",
]

# Data fetching. The model's features look back up to 252 bars, so the live
# window must be comfortably longer than a year.
HISTORY_PERIOD = "2y"
FETCH_RETRIES = 3
FETCH_BACKOFF_FACTOR = 0.5
FUNDAMENTALS_CACHE_PATH = "data/fundamentals_cache.json"
FUNDAMENTALS_MAX_AGE_DAYS = 7

# Signal thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
PE_CHEAP = 20
PE_EXPENSIVE = 40
FCF_YIELD_GOOD = 0.05
VOLUME_SPIKE = 1.5
VOLUME_DRY = 0.5
BUY_SCORE_THRESHOLD = 4
SELL_SCORE_THRESHOLD = 4

# Simulated portfolio
STARTING_CASH = 10_000.0
INVESTMENT_PER_TRADE = 1_000.0

# Scheduling
DEFAULT_INTERVAL_HOURS = 24

# Backtesting
BACKTEST_PERIOD = "5y"
WARMUP_BARS = 200              # bars of history required before signals count
TRANSACTION_FEE_RATE = 0.001   # 0.1% per trade side

# ML model (LightGBM cross-sectional return ranker)
MODEL_PATH = "data/lgbm_model.txt"
MODEL_META_PATH = "data/lgbm_meta.json"
TARGET_HORIZON_DAYS = 20       # predict forward 20-day excess return
MIN_CROSS_SECTION = 5          # min tickers per date for ranking/IC
MODEL_BUY_QUANTILE = 0.8       # score above this per-date quantile -> Buy
MODEL_SELL_QUANTILE = 0.2      # score below this per-date quantile -> Sell
WALK_FORWARD_MIN_TRAIN_BARS = 504   # ~2 years before the first test fold
WALK_FORWARD_TEST_BARS = 63         # ~1 quarter per fold
WALK_FORWARD_EMBARGO_BARS = 21      # gap so the target horizon can't leak
