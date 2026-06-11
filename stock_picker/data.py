"""Loading and fetching stock data."""

import io
import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from stock_picker import config
from stock_picker.db import get_client

logger = logging.getLogger(__name__)

# Only these keys are used downstream; caching the full yfinance info blob
# would balloon the cache file.
FUNDAMENTAL_KEYS = (
    "forwardPE", "trailingPE", "freeCashflow", "marketCap", "longName", "shortName",
)


def load_table(table: str) -> pd.DataFrame:
    """Load a full Supabase table into a DataFrame."""
    response = get_client().table(table).select("*").execute()
    return pd.DataFrame(response.data)


def get_universe() -> List[str]:
    """Return the S&P 500 tickers, falling back to a small default universe."""
    try:
        request = urllib.request.Request(
            config.SP500_SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        html = urllib.request.urlopen(request, timeout=30).read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html), flavor="html5lib")
        symbols = tables[0]["Symbol"].tolist()
        # Yahoo Finance uses hyphens for share classes (BRK.B -> BRK-B)
        return [str(symbol).replace(".", "-") for symbol in symbols]
    except Exception:
        logger.exception(
            "Failed to fetch S&P 500 tickers; falling back to the "
            f"{len(config.DEFAULT_UNIVERSE)}-ticker default universe"
        )
        return list(config.DEFAULT_UNIVERSE)


def _load_fundamentals_cache() -> Dict[str, Any]:
    path = Path(config.FUNDAMENTALS_CACHE_PATH)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read fundamentals cache, starting fresh: {e}")
    return {}


def _save_fundamentals_cache(cache: Dict[str, Any]) -> None:
    path = Path(config.FUNDAMENTALS_CACHE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    except OSError as e:
        logger.warning(f"Could not write fundamentals cache: {e}")


def _get_fundamentals(ticker: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch fundamentals for one ticker, using the cache when fresh enough.

    yfinance's .info is a slow scrape and heavily rate-limited, so it is only
    refreshed every FUNDAMENTALS_MAX_AGE_DAYS; prices come from the fast bulk
    download instead.
    """
    now = datetime.now(timezone.utc)
    entry = cache.get(ticker)
    if entry:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if now - fetched_at < timedelta(days=config.FUNDAMENTALS_MAX_AGE_DAYS):
            return entry["info"]

    try:
        info = yf.Ticker(ticker).info or {}
        subset = {key: info.get(key) for key in FUNDAMENTAL_KEYS}
        cache[ticker] = {"fetched_at": now.isoformat(), "info": subset}
        return subset
    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        return entry["info"] if entry else {}


def _download_history(tickers: List[str], period: str) -> Optional[pd.DataFrame]:
    """Bulk-download price history for all tickers in one batched request."""
    for attempt in range(config.FETCH_RETRIES):
        try:
            return yf.download(
                tickers,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.warning(f"History download attempt {attempt + 1} failed: {e}")
            if attempt < config.FETCH_RETRIES - 1:
                time.sleep(config.FETCH_BACKOFF_FACTOR * (2 ** attempt))
    logger.error(f"Failed to download history after {config.FETCH_RETRIES} attempts.")
    return None


def fetch_market_data(
    ticker: str,
    period: str = config.HISTORY_PERIOD,
    retries: int = config.FETCH_RETRIES,
    backoff_factor: float = config.FETCH_BACKOFF_FACTOR,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fetch historical data and fundamentals for a single stock using yfinance.

    Returns a tuple of (history DataFrame, fundamentals dict), or (None, {})
    if fetching fails.
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                # Forward-fill only: back-filling would leak future bars into
                # the past and poison any backtest built on this data.
                hist = hist.ffill()
            info = stock.info
            return hist, info
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                logger.error(f"Failed to fetch data for {ticker} after {retries} retries.")

    return None, {}


def fetch_universe_data(
    tickers: List[str], period: str = config.HISTORY_PERIOD
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Fetch market data for a universe of stocks.

    Prices come from a single batched download; fundamentals are fetched per
    ticker but cached on disk for FUNDAMENTALS_MAX_AGE_DAYS.
    """
    if not tickers:
        return {}

    batch = _download_history(tickers, period)
    if batch is None or batch.empty:
        return {}

    cache = _load_fundamentals_cache()
    results = {}
    for ticker in tickers:
        if isinstance(batch.columns, pd.MultiIndex):
            if ticker not in batch.columns.get_level_values(0):
                continue
            hist = batch[ticker]
        else:
            hist = batch

        hist = hist.dropna(how="all")
        if hist.empty:
            continue
        hist = hist.ffill()  # forward-fill only; never fill from future bars

        info = _get_fundamentals(ticker, cache)
        results[ticker] = (hist, info)

    _save_fundamentals_cache(cache)
    return results
