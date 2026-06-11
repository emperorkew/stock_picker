import numpy as np
import pandas as pd
import pytest

from stock_picker.analysis import calculate_indicators, generate_signals, score_signals


def test_calculate_indicators_empty():
    df = pd.DataFrame()
    res = calculate_indicators(df)
    assert res.empty


def test_calculate_indicators_short():
    df = pd.DataFrame({'Close': [1, 2, 3], 'Volume': [100, 200, 300]})
    res = calculate_indicators(df)
    assert 'EMA_50' not in res.columns


def test_generate_signals_hold():
    df = pd.DataFrame()
    signal = generate_signals(df, {})
    assert signal == 'Hold'


def test_generate_signals_buy():
    # Construct a DataFrame that guarantees a buy signal
    # EMA_50 > EMA_200 (+1), RSI < 30 (+1), MACD > Signal (+1), Volume > 1.5 (+0.5)
    # Total Technical: 3.5
    # P/E < 20 (+1), FCF Yield > 0.05 (+1)
    # Total Buy: 5.5 (vs Sell 0) -> 'Buy'

    # Needs to be length >= 200 to pass the length check
    hist = pd.DataFrame({
        'EMA_50': [100] * 200,
        'EMA_200': [50] * 200,
        'RSI': [20] * 200,
        'MACD': [2] * 200,
        'Signal_Line': [1] * 200,
        'Volume_Anomaly': [2] * 200,
    })

    info = {
        'trailingPE': 15,
        'freeCashflow': 100,
        'marketCap': 1000  # Yield = 0.1
    }

    signal = generate_signals(hist, info)
    assert signal == 'Buy'


def test_generate_signals_sell():
    # Construct a DataFrame that guarantees a sell signal
    # EMA_50 < EMA_200 (Sell +1), RSI > 70 (Sell +1), MACD < Signal (Sell +1), Volume < 0.5 (Sell +0.5)
    # Total Technical: 3.5 Sell
    # P/E > 40 (Sell +1), FCF Yield < 0 (Sell +1)
    # Total Sell: 5.5 (vs Buy 0) -> 'Sell'

    hist = pd.DataFrame({
        'EMA_50': [50] * 200,
        'EMA_200': [100] * 200,
        'RSI': [80] * 200,
        'MACD': [1] * 200,
        'Signal_Line': [2] * 200,
        'Volume_Anomaly': [0.1] * 200,
    })

    info = {
        'trailingPE': 50,
        'freeCashflow': -100,
        'marketCap': 1000  # Yield = -0.1
    }

    signal = generate_signals(hist, info)
    assert signal == 'Sell'


def test_rsi_extremes():
    rising = pd.DataFrame({'Close': np.linspace(100, 300, 250), 'Volume': [1e6] * 250})
    assert calculate_indicators(rising)['RSI'].iloc[-1] > 99

    falling = pd.DataFrame({'Close': np.linspace(300, 100, 250), 'Volume': [1e6] * 250})
    assert calculate_indicators(falling)['RSI'].iloc[-1] < 1


def test_indicators_flat_price():
    flat = pd.DataFrame({'Close': [100.0] * 250, 'Volume': [1e6] * 250})
    res = calculate_indicators(flat)
    assert res['EMA_50'].iloc[-1] == pytest.approx(100)
    assert res['EMA_200'].iloc[-1] == pytest.approx(100)
    assert res['MACD'].iloc[-1] == pytest.approx(0)
    assert res['Volume_Anomaly'].iloc[-1] == pytest.approx(1)


def test_negative_pe_is_not_cheap():
    latest = pd.Series({'RSI': 50.0})
    buy, sell = score_signals(latest, {'trailingPE': -10})
    assert buy == 0

    buy_cheap, _ = score_signals(latest, {'trailingPE': 10})
    assert buy_cheap == 1


def test_nan_indicators_score_nothing():
    latest = pd.Series({
        'EMA_50': float('nan'), 'EMA_200': float('nan'), 'RSI': float('nan'),
        'MACD': float('nan'), 'Signal_Line': float('nan'), 'Volume_Anomaly': float('nan'),
    })
    buy, sell = score_signals(latest, {'trailingPE': 25})
    assert buy == 0
    assert sell == 0
