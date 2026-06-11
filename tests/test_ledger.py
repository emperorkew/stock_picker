"""Tests for the ledger replay logic."""

import pytest

from stock_picker.ledger import replay_ledger


def test_replay_empty():
    holdings, cash, realized = replay_ledger([], 10_000)
    assert holdings == {}
    assert cash == 10_000
    assert realized == 0


def test_replay_buy_and_partial_sell():
    rows = [
        {'ticker': 'AAPL', 'action': 'BUY', 'quantity': 10, 'price_per_unit': 100},
        {'ticker': 'AAPL', 'action': 'BUY', 'quantity': 10, 'price_per_unit': 200},
        {'ticker': 'AAPL', 'action': 'SELL', 'quantity': 5, 'price_per_unit': 300},
    ]
    holdings, cash, realized = replay_ledger(rows, 10_000)
    # Average cost is 150, so selling 5 shares at 300 realizes 5 * 150 = 750.
    assert realized == pytest.approx(750)
    assert holdings['AAPL']['shares'] == pytest.approx(15)
    assert holdings['AAPL']['cost'] == pytest.approx(15 * 150)
    assert cash == pytest.approx(10_000 - 1_000 - 2_000 + 1_500)


def test_replay_sell_without_position_is_ignored():
    rows = [{'ticker': 'AAPL', 'action': 'SELL', 'quantity': 5, 'price_per_unit': 100}]
    holdings, cash, realized = replay_ledger(rows, 1_000)
    assert holdings == {}
    assert cash == 1_000
    assert realized == 0


def test_replay_sell_clamps_to_held_shares():
    rows = [
        {'ticker': 'AAPL', 'action': 'BUY', 'quantity': 5, 'price_per_unit': 100},
        {'ticker': 'AAPL', 'action': 'SELL', 'quantity': 50, 'price_per_unit': 110},
    ]
    holdings, cash, realized = replay_ledger(rows, 1_000)
    assert 'AAPL' not in holdings
    assert realized == pytest.approx(5 * 10)
    assert cash == pytest.approx(1_000 - 500 + 550)
