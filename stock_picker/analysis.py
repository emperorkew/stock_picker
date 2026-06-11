"""Screening and stock-picking logic."""

from typing import Any, Dict, Tuple

import pandas as pd

from stock_picker import config


def calculate_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators: EMA 50, EMA 200, RSI, MACD, and Volume anomalies."""
    df = hist.copy()

    if len(df) < 200:
        return df

    # EMA 50 and EMA 200
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI (14-day, Wilder's smoothing throughout)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volume vs the average of the 20 *prior* bars, so a partially traded
    # current bar doesn't drag down its own baseline.
    df['Volume_20SMA'] = df['Volume'].rolling(window=20).mean().shift(1)
    df['Volume_Anomaly'] = df['Volume'] / df['Volume_20SMA']

    return df


def _value(latest: pd.Series, key: str, default: float) -> float:
    """Like Series.get, but treats NaN as missing too."""
    value = latest.get(key, default)
    return default if pd.isna(value) else float(value)


def score_signals(
    latest: pd.Series, info: Dict[str, Any], use_fundamentals: bool = True
) -> Tuple[float, float]:
    """Score the latest indicator row plus fundamentals; returns (buy_score, sell_score).

    use_fundamentals=False skips the P/E and FCF rules entirely — used by the
    backtest, where historical fundamentals aren't known point-in-time.
    """
    ema_50 = _value(latest, 'EMA_50', 0)
    ema_200 = _value(latest, 'EMA_200', 0)
    rsi = _value(latest, 'RSI', 50)
    macd = _value(latest, 'MACD', 0)
    signal_line = _value(latest, 'Signal_Line', 0)
    volume_anomaly = _value(latest, 'Volume_Anomaly', 1)

    buy_score = 0.0
    sell_score = 0.0

    # Technical Rules
    if ema_50 > ema_200:
        buy_score += 1
    elif ema_50 < ema_200:
        sell_score += 1

    if rsi < config.RSI_OVERSOLD:
        buy_score += 1  # Oversold
    elif rsi > config.RSI_OVERBOUGHT:
        sell_score += 1  # Overbought

    if macd > signal_line:
        buy_score += 1
    elif macd < signal_line:
        sell_score += 1

    if volume_anomaly > config.VOLUME_SPIKE:
        buy_score += 0.5  # High volume interest
    elif volume_anomaly < config.VOLUME_DRY:
        sell_score += 0.5  # Drying up

    if use_fundamentals:
        # Negative P/E means negative earnings, not "cheap".
        pe_ratio = info.get('forwardPE') or info.get('trailingPE') or 50  # Default to high if missing
        fcf = info.get('freeCashflow')
        market_cap = info.get('marketCap')
        fcf_yield = (fcf / market_cap) if fcf and market_cap else 0

        if 0 < pe_ratio < config.PE_CHEAP:
            buy_score += 1
        elif pe_ratio > config.PE_EXPENSIVE:
            sell_score += 1

        if fcf_yield > config.FCF_YIELD_GOOD:
            buy_score += 1
        elif fcf_yield < 0:
            sell_score += 1

    return buy_score, sell_score


def signal_from_scores(buy_score: float, sell_score: float) -> str:
    """Map (buy_score, sell_score) to a 'Buy'/'Hold'/'Sell' decision."""
    if buy_score >= config.BUY_SCORE_THRESHOLD and buy_score > sell_score + 1:
        return 'Buy'
    if sell_score >= config.SELL_SCORE_THRESHOLD and sell_score > buy_score + 1:
        return 'Sell'
    return 'Hold'


def generate_signals(hist: pd.DataFrame, info: Dict[str, Any]) -> str:
    """
    Generate an algorithmic decision signal ('Buy', 'Hold', or 'Sell')
    based on technicals and fundamentals.
    """
    if hist.empty or len(hist) < 200:
        return 'Hold'

    buy_score, sell_score = score_signals(hist.iloc[-1], info)
    return signal_from_scores(buy_score, sell_score)


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
            'Name': info.get('longName') or info.get('shortName'),
            'Signal': signal,
            'Price': latest.get('Close'),
            'Volume': latest.get('Volume'),
            'Volume_20SMA': latest.get('Volume_20SMA'),
            'EMA_50': latest.get('EMA_50'),
            'EMA_200': latest.get('EMA_200'),
            'RSI': latest.get('RSI'),
            'MACD': latest.get('MACD'),
            'Signal_Line': latest.get('Signal_Line'),
            'P/E': pe_ratio,
            'FCF_Yield': fcf_yield
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('Ticker', inplace=True)
    return df
