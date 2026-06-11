"""Screening and stock-picking logic."""

import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from stock_picker.db import get_client

logger = logging.getLogger(__name__)


def evaluate_factors(metrics: dict, predicted_return_pct: float) -> tuple[float, str, float]:
    """
    Evaluate the Multi-Factor Confluence Score.
    Returns:
        S: Confluence Score
        rationale: Explanation of the score
        T: Trend factor
    """
    # Initialize base weights
    w_t, w_m, w_p, w_v, w_f = 0.30, 0.25, 0.20, 0.15, 0.10

    # Trend (T): Price > EMA_50 > EMA_200
    price = metrics.get('price')
    ema_50 = metrics.get('ema_50')
    ema_200 = metrics.get('ema_200')

    if price is not None and ema_50 is not None and ema_200 is not None:
        T = 1.0 if (price > ema_50 and ema_50 > ema_200) else 0.0
    else:
        T = 0.0

    # Momentum (M): MACD cross & RSI bound
    macd_line = metrics.get('macd_line')
    macd_signal = metrics.get('macd_signal')
    rsi_14 = metrics.get('rsi_14')

    if macd_line is not None and macd_signal is not None and rsi_14 is not None:
        M = 1.0 if (macd_line > macd_signal and 30 <= rsi_14 <= 70) else 0.0
    else:
        M = 0.0

    # Predicted Return Factor (P)
    alpha = 0.5  # 0.5% threshold
    P = 1.0 if predicted_return_pct > alpha else 0.0

    # Volume Ratio (V) >= 1.5
    volume = metrics.get('volume')
    volume_avg_20d = metrics.get('volume_avg_20d')
    if volume is not None and volume_avg_20d is not None and volume_avg_20d > 0:
        V = 1.0 if (volume / volume_avg_20d) >= 1.5 else 0.0
    else:
        V = 0.0

    # Fundamental Health (F)
    pe_ratio = metrics.get('pe_ratio')
    fcf_yield = metrics.get('fcf_yield')

    F = None
    if pe_ratio is not None and fcf_yield is not None:
        # Example condition: PE < 25 and FCF Yield > 2%
        F = 1.0 if (0 < pe_ratio < 25 and fcf_yield > 0.02) else 0.0

    # Dynamic Weight Re-normalization
    if F is None:
        F = 0.0
        w_f_missing = w_f
        w_f = 0.0
        # Re-distribute the missing weight to other factors proportionally
        total_remaining_weight = w_t + w_m + w_p + w_v
        w_t += w_f_missing * (w_t / total_remaining_weight)
        w_m += w_f_missing * (w_m / total_remaining_weight)
        w_p += w_f_missing * (w_p / total_remaining_weight)
        w_v += w_f_missing * (w_v / total_remaining_weight)

    S = (w_t * T) + (w_m * M) + (w_p * P) + (w_v * V) + (w_f * F)

    rationale = f"S={S:.2f} (T={T}, M={M}, P={P}, V={V}, F={F})"

    return S, rationale, T


def generate_signal(S: float, T: float) -> str:
    """Generate execution signal based on score and trend."""
    if S >= 0.70:
        return 'BUY'
    elif S <= 0.30 or T == 0.0:
        return 'SELL'
    else:
        return 'HOLD'


def execute_trade(asset_id: str, signal: str, price: float) -> None:
    """Simulate execution within portfolio_ledger."""
    if signal not in ['BUY', 'SELL'] or price is None:
        return

    client = get_client()

    # Assume fixed quantity for simulation
    quantity = 10.0

    row = {
        'asset_id': asset_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': signal,
        'quantity': quantity,
        'price_per_unit': price,
        'transaction_fee': 0.0
    }

    try:
        client.table('portfolio_ledger').insert(row).execute()
        logger.info(f"Executed {signal} for asset {asset_id} at {price}")
    except Exception as e:
        logger.error(f"Failed to execute trade for {asset_id}: {e}")


def log_signal(asset_id: str, signal: str, score: float, rationale: str, predicted_return_pct: float) -> None:
    """Log the decision into trading_signals."""
    client = get_client()

    row = {
        'asset_id': asset_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'signal': signal,
        'confidence_score': score,
        'rationale': rationale,
        'predicted_return': predicted_return_pct,
        'forecast_horizon_minutes': 1440  # 1 day
    }

    try:
        client.table('trading_signals').insert(row).execute()
    except Exception as e:
        logger.error(f"Failed to log signal for {asset_id}: {e}")

def pick_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Apply screening rules and return the selected stocks.

    Placeholder for backward compatibility. Real logic runs in interval loop.
    """
    return df
