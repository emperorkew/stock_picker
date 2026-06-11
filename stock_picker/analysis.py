"""Screening and stock-picking logic."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def calculate_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators: EMA 50, EMA 200, RSI, MACD, and Volume anomalies."""
    df = hist.copy()

    if len(df) < 200:
        return df

    # EMA 50 and EMA 200
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Fill initial NaN RSI values using an exponential moving average approach
    gain_ema = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss_ema = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs_ema = gain_ema / loss_ema
    df['RSI'] = df['RSI'].fillna(100 - (100 / (1 + rs_ema)))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volume Anomaly
    df['Volume_20SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Anomaly'] = df['Volume'] / df['Volume_20SMA']

    return df


def generate_signals(hist: pd.DataFrame, info: Dict[str, Any]) -> str:
    """
    Generate an algorithmic decision signal ('Buy', 'Hold', or 'Sell')
    based on technicals and fundamentals.
    """
    if hist.empty or len(hist) < 200:
        return 'Hold'

    # Get latest technicals
    latest = hist.iloc[-1]

    ema_50 = latest.get('EMA_50', 0)
    ema_200 = latest.get('EMA_200', 0)
    rsi = latest.get('RSI', 50)
    macd = latest.get('MACD', 0)
    signal_line = latest.get('Signal_Line', 0)
    volume_anomaly = latest.get('Volume_Anomaly', 1)

    # Fundamentals
    pe_ratio = info.get('forwardPE') or info.get('trailingPE') or 50 # Default to high if missing
    fcf = info.get('freeCashflow')
    market_cap = info.get('marketCap')
    fcf_yield = (fcf / market_cap) if fcf and market_cap else 0

    buy_score = 0
    sell_score = 0

    # Technical Rules
    if ema_50 > ema_200:
        buy_score += 1
    elif ema_50 < ema_200:
        sell_score += 1

    if rsi < 30:
        buy_score += 1 # Oversold
    elif rsi > 70:
        sell_score += 1 # Overbought

    if macd > signal_line:
        buy_score += 1
    elif macd < signal_line:
        sell_score += 1

    if volume_anomaly > 1.5:
        buy_score += 0.5 # High volume interest
    elif volume_anomaly < 0.5:
        sell_score += 0.5 # Drying up

    # Fundamental Rules
    if pe_ratio < 20:
        buy_score += 1
    elif pe_ratio > 40:
        sell_score += 1

    if fcf_yield > 0.05:
        buy_score += 1
    elif fcf_yield < 0:
        sell_score += 1

    if buy_score >= 4 and buy_score > sell_score + 1:
        return 'Buy'
    elif sell_score >= 4 and sell_score > buy_score + 1:
        return 'Sell'
    else:
        return 'Hold'


def pick_stocks(universe_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Takes the fetched universe data, calculates indicators, generates signals,
    and returns a DataFrame of the recommendations and metrics.
    """
    records = []

    for ticker, (hist, info) in universe_data.items():
        if hist is None or hist.empty:
            continue

        hist_with_indicators = calculate_indicators(hist)
        signal = generate_signals(hist_with_indicators, info)

        latest = hist_with_indicators.iloc[-1]

        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        fcf = info.get('freeCashflow')
        market_cap = info.get('marketCap')
        fcf_yield = (fcf / market_cap) if fcf and market_cap else None

        records.append({
            'Ticker': ticker,
            'Signal': signal,
            'Price': latest.get('Close'),
            'EMA_50': latest.get('EMA_50'),
            'EMA_200': latest.get('EMA_200'),
            'RSI': latest.get('RSI'),
            'MACD': latest.get('MACD'),
            'P/E': pe_ratio,
            'FCF_Yield': fcf_yield
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('Ticker', inplace=True)
    return df
