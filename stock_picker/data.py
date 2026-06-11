"""Loading and storing stock data (Supabase, yfinance, ...)."""

import logging
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import numpy as np

from stock_picker.db import get_client

logger = logging.getLogger(__name__)

def load_table(table: str) -> pd.DataFrame:
    """Load a full Supabase table into a DataFrame."""
    response = get_client().table(table).select("*").execute()
    return pd.DataFrame(response.data)

def fetch_market_data(ticker: str) -> dict:
    """Fetch real-time market data and calculate indicators using yfinance."""
    try:
        # Fetch 1 year of daily data to ensure enough history for EMA_200
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="1y")

        if history.empty:
            logger.warning(f"No historical data found for {ticker}")
            return {}

        # Calculate indicators
        # EMAs
        history['EMA_50'] = history['Close'].ewm(span=50, adjust=False).mean()
        history['EMA_200'] = history['Close'].ewm(span=200, adjust=False).mean()

        # RSI 14
        delta = history['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        history['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = history['Close'].ewm(span=12, adjust=False).mean()
        exp2 = history['Close'].ewm(span=26, adjust=False).mean()
        history['MACD_line'] = exp1 - exp2
        history['MACD_signal'] = history['MACD_line'].ewm(span=9, adjust=False).mean()

        # Volume avg 20d
        history['Volume_avg_20d'] = history['Volume'].rolling(window=20).mean()

        # Get latest row
        latest = history.iloc[-1]

        # Fetch fundamentals
        info = ticker_obj.info

        pe_ratio = info.get('trailingPE')
        # yfinance doesn't always have fcf yield, calculate it or fallback to missing
        free_cash_flow = info.get('freeCashflow')
        market_cap = info.get('marketCap')
        fcf_yield = None
        if free_cash_flow and market_cap:
            fcf_yield = free_cash_flow / market_cap

        return {
            'price': float(latest['Close']) if not pd.isna(latest['Close']) else None,
            'volume': float(latest['Volume']) if not pd.isna(latest['Volume']) else None,
            'volume_avg_20d': float(latest['Volume_avg_20d']) if not pd.isna(latest['Volume_avg_20d']) else None,
            'pe_ratio': pe_ratio if pe_ratio is not None else None,
            'fcf_yield': fcf_yield if fcf_yield is not None else None,
            'ema_50': float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else None,
            'ema_200': float(latest['EMA_200']) if not pd.isna(latest['EMA_200']) else None,
            'rsi_14': float(latest['RSI_14']) if not pd.isna(latest['RSI_14']) else None,
            'macd_line': float(latest['MACD_line']) if not pd.isna(latest['MACD_line']) else None,
            'macd_signal': float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None,
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return {}

def insert_market_snapshot(asset_id: str, metrics: dict) -> None:
    """Write the current metrics as a new row into market_snapshots."""
    if not metrics:
        return

    client = get_client()
    row = {
        'asset_id': asset_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **metrics
    }

    # Handle NaN values to None for JSON serialization
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            row[k] = None

    client.table('market_snapshots').insert(row).execute()
    logger.debug(f"Inserted snapshot for asset {asset_id}")
