"""Tests for the LightGBM pipeline: dataset, splits, and learnability."""

import numpy as np
import pandas as pd
import pytest

from stock_picker import config
from stock_picker.model import (
    FEATURES,
    build_dataset,
    build_features,
    latest_signals,
    make_signal_fn,
    train_final,
    walk_forward_evaluate,
    walk_forward_splits,
)

# Small/fast LightGBM settings for synthetic data.
TEST_PARAMS = {
    'objective': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 7,
    'min_data_in_leaf': 20,
    'verbosity': -1,
    'seed': 42,
}


def _synthetic_universe(n_tickers: int = 6, days: int = 700):
    """Tickers with distinct persistent drifts: momentum features genuinely
    predict cross-sectional ranking, so a working pipeline must find it."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2018-01-02", periods=days, freq="B")
    drifts = np.linspace(-0.002, 0.002, n_tickers)
    universe = {}
    for i, drift in enumerate(drifts):
        noise = rng.normal(0, 0.002, days)
        close = 100 * np.cumprod(1 + drift + noise)
        universe[f"T{i}"] = (
            pd.DataFrame(
                {
                    "Open": close,
                    "High": close,
                    "Low": close,
                    "Close": close,
                    "Volume": rng.integers(900_000, 1_100_000, days).astype(float),
                },
                index=index,
            ),
            {},
        )
    return universe


def test_build_features_causal_columns():
    universe = _synthetic_universe(n_tickers=1)
    feats = build_features(universe["T0"][0])
    assert list(FEATURES) == [c for c in feats.columns if c in FEATURES]
    # Warmup region is NaN, the tail is fully populated (no forward data used).
    assert feats[FEATURES].iloc[-1].notna().all()


def test_build_dataset_shape_and_target():
    universe = _synthetic_universe()
    panel = build_dataset(universe)
    assert set(FEATURES).issubset(panel.columns)
    assert not panel[FEATURES + ['excess_fwd_return']].isna().any().any()
    # Cross-sectional excess returns are centered: median per date is ~0.
    medians = panel.groupby('date')['excess_fwd_return'].median()
    assert abs(medians).max() == pytest.approx(0, abs=1e-12)
    # Every kept date has a full cross-section.
    assert panel.groupby('date')['ticker'].size().min() >= config.MIN_CROSS_SECTION


def test_walk_forward_splits_respect_embargo():
    splits = walk_forward_splits(1000)
    assert splits, "expected at least one fold"
    for train_end, test_start, test_end in splits:
        assert test_start - train_end >= config.WALK_FORWARD_EMBARGO_BARS
        assert test_end > test_start


def test_model_learns_planted_signal():
    universe = _synthetic_universe()
    panel = build_dataset(universe)
    summary = walk_forward_evaluate(panel, params=TEST_PARAMS, num_rounds=60, min_train=150)
    # Drift-driven momentum is strong by construction; a correct pipeline
    # should rank tickers far better than chance.
    assert summary['mean_ic'] > 0.3
    assert summary['pct_positive_ic_days'] > 0.7


def test_make_signal_fn_ranks_extremes():
    universe = _synthetic_universe()
    panel = build_dataset(universe)
    booster = train_final(panel, params=TEST_PARAMS, num_rounds=60)
    signal_fn = make_signal_fn(booster, universe)

    last_date = panel['date'].max()
    # Highest-drift ticker should be a Buy, lowest-drift a Sell, on a date
    # late enough that all features exist.
    n = len(universe) - 1
    assert signal_fn({'_ticker': f'T{n}', '_date': last_date}, {}) == 'Buy'
    assert signal_fn({'_ticker': 'T0', '_date': last_date}, {}) == 'Sell'
    # Unknown (ticker, date) combinations fall back to Hold.
    assert signal_fn({'_ticker': 'NOPE', '_date': last_date}, {}) == 'Hold'


def test_latest_signals_live_ranking():
    universe = _synthetic_universe()
    panel = build_dataset(universe)
    booster = train_final(panel, params=TEST_PARAMS, num_rounds=60)

    signals = latest_signals(booster, universe)
    assert set(signals.columns) == {'Score', 'Confidence', 'Signal'}
    assert len(signals) == len(universe)
    assert signals['Confidence'].between(0, 1).all()
    # Highest drift ranks top -> Buy; lowest ranks bottom -> Sell.
    n = len(universe) - 1
    assert signals.loc[f'T{n}', 'Signal'] == 'Buy'
    assert signals.loc['T0', 'Signal'] == 'Sell'


def test_latest_signals_small_cross_section_is_empty():
    universe = _synthetic_universe(n_tickers=2)
    panel_universe = _synthetic_universe()  # train on a full universe
    booster = train_final(build_dataset(panel_universe), params=TEST_PARAMS, num_rounds=60)
    assert latest_signals(booster, universe).empty
