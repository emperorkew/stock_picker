import pandas as pd
import pytest

from stock_picker.analysis import calculate_indicators, generate_signals


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
