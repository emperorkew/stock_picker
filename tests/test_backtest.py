"""Tests for the backtest engine's accounting and point-in-time discipline."""

import pandas as pd
import pytest

from stock_picker import config
from stock_picker.backtest import run_backtest


def _flat_history(days: int, price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=days, freq="B")
    return pd.DataFrame(
        {"Open": price, "High": price, "Low": price, "Close": price, "Volume": 1e6},
        index=index,
    )


def test_buy_sell_accounting():
    """Buy fills at next open with fees; sell returns proceeds minus fees."""
    hist = _flat_history(config.WARMUP_BARS + 10)
    calls = {"n": 0}

    def scripted(row, info):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Buy"
        if calls["n"] == 3:
            return "Sell"
        return "Hold"

    result = run_backtest(
        {"TEST": (hist, {})},
        starting_cash=10_000,
        investment_per_trade=1_000,
        fee_rate=0.001,
        signal_fn=scripted,
    )

    assert len(result.closed_trades) == 1
    trade = result.closed_trades[0]
    assert trade.entry_price == pytest.approx(100)
    assert trade.shares == pytest.approx(10)
    # Flat prices: the round trip loses exactly the two 0.1% fees.
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000 - 2 * 1.0)
    assert result.metrics["trades_open"] == 0


def test_no_signals_before_warmup():
    """The signal function must never see a bar before WARMUP_BARS of history."""
    hist = _flat_history(config.WARMUP_BARS + 5)
    seen = []

    def recorder(row, info):
        seen.append(row)
        return "Hold"

    run_backtest({"TEST": (hist, {})}, signal_fn=recorder)
    assert len(seen) == 5 + 1  # bars WARMUP_BARS..end only


def test_buy_executes_at_next_bar_not_same_bar():
    """A signal at bar t must fill at bar t+1's open price."""
    days = config.WARMUP_BARS + 10
    hist = _flat_history(days)
    # Make every open differ from every close so a same-bar fill is detectable.
    hist["Open"] = 200.0
    fired = {"done": False}

    def buy_once(row, info):
        if not fired["done"]:
            fired["done"] = True
            return "Buy"
        return "Hold"

    result = run_backtest(
        {"TEST": (hist, {})}, investment_per_trade=1_000, fee_rate=0.0, signal_fn=buy_once
    )
    assert len(result.open_trades) == 1
    assert result.open_trades[0].entry_price == pytest.approx(200.0)  # the *open*, not 100


def test_cash_constraint_blocks_buys():
    hist = _flat_history(config.WARMUP_BARS + 5)

    result = run_backtest(
        {"TEST": (hist, {})},
        starting_cash=500,  # less than one trade's size
        investment_per_trade=1_000,
        signal_fn=lambda row, info: "Buy",
    )
    assert result.open_trades == []
    assert result.closed_trades == []
    assert result.equity_curve.iloc[-1] == pytest.approx(500)


def test_too_short_history_raises():
    hist = _flat_history(50)
    with pytest.raises(ValueError):
        run_backtest({"TEST": (hist, {})})
