"""Loading and fetching stock data."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from stock_picker.db import get_client

logger = logging.getLogger(__name__)


def load_table(table: str) -> pd.DataFrame:
    """Load a full Supabase table into a DataFrame."""
    response = get_client().table(table).select("*").execute()
    return pd.DataFrame(response.data)


def fetch_market_data(
    ticker: str,
    period: str = "1y",
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fetch historical data and fundamentals for a given stock using yfinance.

    Args:
        ticker: The stock ticker symbol.
        period: The period for historical data (e.g., "1y", "6mo").
        retries: Number of retries in case of network errors.
        backoff_factor: Multiplier for exponential backoff between retries.

    Returns:
        A tuple containing the historical price DataFrame and a dictionary of fundamentals.
        Returns (None, {}) if fetching fails.
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)

            # Fetch historical market data
            hist = stock.history(period=period)

            # Handle missing data
            if not hist.empty:
                hist = hist.ffill().bfill()

            # Fetch fundamentals
            info = stock.info

            return hist, info
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} failed for {ticker}: {e}"
            )
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2**attempt))
            else:
                logger.error(f"Failed to fetch data for {ticker} after {retries} retries.")

    return None, {}

def fetch_universe_data(
    tickers: List[str], period: str = "1y"
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Fetch market data for a universe of stocks.

    Args:
        tickers: A list of stock ticker symbols.
        period: The period for historical data.

    Returns:
        A dictionary mapping tickers to their historical data and fundamentals.
    """
    results = {}
    for ticker in tickers:
        hist, info = fetch_market_data(ticker, period=period)
        if hist is not None and not hist.empty:
            results[ticker] = (hist, info)
    return results
