"""Central configuration for the stock picker."""

# Universe
SP500_SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "WMT",
]

# Data fetching
HISTORY_PERIOD = "1y"
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
