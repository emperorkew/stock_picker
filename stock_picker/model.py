"""LightGBM cross-sectional return model.

Predicts each stock's forward TARGET_HORIZON_DAYS return *relative to the
universe median* ("which stocks beat the others"), trained and evaluated
walk-forward with an embargo gap so the forward-looking target can never
leak into training.

Run from the command line:

    python -m stock_picker.model                          # default universe, 10y
    python -m stock_picker.model --sp500 --limit 100      # broader cross-section

This evaluates walk-forward (the honest numbers), then trains a final model
on all data and saves it to data/lgbm_model.txt for use going forward:

    python -m stock_picker.backtest --model data/lgbm_model.txt

NOTE: backtesting a final model over the same window it was trained on is
in-sample and optimistic — trust the walk-forward metrics instead.
"""

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import pandas as pd
from scipy.stats import spearmanr

from stock_picker import analysis, config, data

logger = logging.getLogger(__name__)

FEATURES = [
    'rsi_14',
    'macd_hist',
    'ema50_gap',
    'ema200_gap',
    'ema_trend',
    'volume_anomaly',
    'mom_21',
    'mom_63',
    'mom_126',
    'mom_252',
    'vol_21',
    'drawdown_252',
]

# Longest feature lookback (252) + target horizon + slack.
MIN_HISTORY_BARS = 300

DEFAULT_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1,
    'seed': 42,
}
DEFAULT_NUM_ROUNDS = 300


def build_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker causal features: every value at date t uses only bars <= t."""
    if hist is None or len(hist) < MIN_HISTORY_BARS:
        return pd.DataFrame()

    df = analysis.calculate_indicators(hist)
    close = df['Close']
    returns = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    feats['rsi_14'] = df['RSI']
    feats['macd_hist'] = (df['MACD'] - df['Signal_Line']) / close
    feats['ema50_gap'] = close / df['EMA_50'] - 1
    feats['ema200_gap'] = close / df['EMA_200'] - 1
    feats['ema_trend'] = df['EMA_50'] / df['EMA_200'] - 1
    feats['volume_anomaly'] = df['Volume_Anomaly']
    for window in (21, 63, 126, 252):
        feats[f'mom_{window}'] = close.pct_change(window)
    feats['vol_21'] = returns.rolling(21).std()
    feats['drawdown_252'] = close / close.rolling(252).max() - 1
    return feats


def build_dataset(
    universe_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
    horizon: int = config.TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    """Panel of (date, ticker, features, target).

    Target = forward `horizon`-day return minus the same-date universe median,
    so the model learns cross-sectional ranking, not market direction.
    """
    frames = []
    for ticker, (hist, _info) in universe_data.items():
        feats = build_features(hist)
        if feats.empty:
            continue
        close = hist['Close']
        feats['fwd_return'] = close.shift(-horizon) / close - 1
        feats['ticker'] = ticker
        frames.append(feats)

    if not frames:
        raise ValueError(
            f"No ticker has the {MIN_HISTORY_BARS}+ bars needed to build features."
        )

    panel = pd.concat(frames)
    panel.index.name = 'date'
    panel = panel.reset_index()
    panel['excess_fwd_return'] = (
        panel['fwd_return']
        - panel.groupby('date')['fwd_return'].transform('median')
    )
    panel = panel.dropna(subset=FEATURES + ['excess_fwd_return'])

    # Ranking needs a real cross-section on each date.
    counts = panel.groupby('date')['ticker'].transform('size')
    panel = panel[counts >= config.MIN_CROSS_SECTION]
    return panel.sort_values(['date', 'ticker']).reset_index(drop=True)


def _train(train: pd.DataFrame, params: Optional[dict] = None,
           num_rounds: int = DEFAULT_NUM_ROUNDS) -> lgb.Booster:
    dataset = lgb.Dataset(train[FEATURES], label=train['excess_fwd_return'])
    return lgb.train(params or DEFAULT_PARAMS, dataset, num_boost_round=num_rounds)


def walk_forward_splits(
    n_dates: int,
    min_train: int = config.WALK_FORWARD_MIN_TRAIN_BARS,
    step: int = config.WALK_FORWARD_TEST_BARS,
    gap: int = config.WALK_FORWARD_EMBARGO_BARS,
) -> List[Tuple[int, int, int]]:
    """(train_end, test_start, test_end) index triples over the date axis.

    test_start - train_end >= the embargo, so no training target overlaps
    the test window.
    """
    splits = []
    train_end = min_train
    while train_end + gap < n_dates:
        test_start = train_end + gap
        test_end = min(test_start + step, n_dates)
        splits.append((train_end, test_start, test_end))
        train_end += step
    return splits


def walk_forward_evaluate(
    panel: pd.DataFrame,
    params: Optional[dict] = None,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    min_train: int = config.WALK_FORWARD_MIN_TRAIN_BARS,
) -> Dict[str, Any]:
    """Train/predict over rolling folds; returns rank-IC and spread metrics."""
    dates = sorted(panel['date'].unique())
    splits = walk_forward_splits(len(dates), min_train=min_train)
    if not splits:
        raise ValueError(
            f"Need more than {min_train + config.WALK_FORWARD_EMBARGO_BARS} "
            f"distinct dates to walk forward; got {len(dates)}."
        )

    daily = []
    for fold, (train_end, test_start, test_end) in enumerate(splits, start=1):
        train_dates = set(dates[:train_end])
        test_dates = set(dates[test_start:test_end])
        train = panel[panel['date'].isin(train_dates)]
        test = panel[panel['date'].isin(test_dates)].copy()
        if train.empty or test.empty:
            continue

        booster = _train(train, params, num_rounds)
        test['pred'] = booster.predict(test[FEATURES])

        for date, group in test.groupby('date'):
            if len(group) < config.MIN_CROSS_SECTION:
                continue
            ic = spearmanr(group['pred'], group['excess_fwd_return']).statistic
            top = group['pred'] >= group['pred'].quantile(config.MODEL_BUY_QUANTILE)
            bottom = group['pred'] <= group['pred'].quantile(config.MODEL_SELL_QUANTILE)
            spread = (
                group.loc[top, 'excess_fwd_return'].mean()
                - group.loc[bottom, 'excess_fwd_return'].mean()
            )
            daily.append({'date': date, 'ic': ic, 'spread': spread})
        logger.info(f"Fold {fold}/{len(splits)} done "
                    f"({len(train)} train rows, {len(test)} test rows)")

    if not daily:
        raise ValueError("Walk-forward produced no scored dates.")

    results = pd.DataFrame(daily)
    ic = results['ic'].dropna()
    # Daily ICs overlap within the 20-day horizon, so this t-stat is
    # optimistic; treat it as a rough scale, not a significance test.
    t_stat = ic.mean() / (ic.std() / math.sqrt(len(ic))) if ic.std() > 0 else float('nan')
    return {
        'folds': len(splits),
        'scored_dates': len(results),
        'start': results['date'].min(),
        'end': results['date'].max(),
        'mean_ic': ic.mean(),
        'ic_t_stat': t_stat,
        'pct_positive_ic_days': (ic > 0).mean(),
        'mean_spread': results['spread'].mean(),
    }


def train_final(panel: pd.DataFrame, params: Optional[dict] = None,
                num_rounds: int = DEFAULT_NUM_ROUNDS) -> lgb.Booster:
    """Train on the full panel — the model to use going forward (live)."""
    return _train(panel, params, num_rounds)


def save_model(booster: lgb.Booster, summary: Dict[str, Any],
               model_path: str = config.MODEL_PATH,
               meta_path: str = config.MODEL_META_PATH) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    meta = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'features': FEATURES,
        'target_horizon_days': config.TARGET_HORIZON_DAYS,
        'walk_forward': {
            key: str(value) if isinstance(value, pd.Timestamp) else value
            for key, value in summary.items()
        },
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2))


def load_model(model_path: str = config.MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path))


def score_history(
    booster: lgb.Booster,
    universe_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
) -> pd.DataFrame:
    """Model scores for every (date, ticker) with complete features."""
    frames = []
    for ticker, (hist, _info) in universe_data.items():
        feats = build_features(hist)
        if feats.empty:
            continue
        feats = feats.dropna(subset=FEATURES)
        if feats.empty:
            continue
        frames.append(pd.DataFrame({
            'date': feats.index,
            'ticker': ticker,
            'score': booster.predict(feats[FEATURES]),
        }))
    if not frames:
        return pd.DataFrame(columns=['date', 'ticker', 'score'])
    return pd.concat(frames, ignore_index=True)


def predict_latest(
    booster: lgb.Booster,
    universe_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
) -> pd.Series:
    """Latest-date score per ticker, for live ranking."""
    scores = score_history(booster, universe_data)
    if scores.empty:
        return pd.Series(dtype=float)
    latest = scores.loc[scores.groupby('ticker')['date'].idxmax()]
    return latest.set_index('ticker')['score'].sort_values(ascending=False)


def latest_signals(
    booster: lgb.Booster,
    universe_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
    buy_quantile: float = config.MODEL_BUY_QUANTILE,
    sell_quantile: float = config.MODEL_SELL_QUANTILE,
) -> pd.DataFrame:
    """Live signals from each ticker's latest bar.

    Returns a DataFrame indexed by ticker with Score (raw model output),
    Confidence (cross-sectional rank percentile, 0..1), and Signal
    ('Buy' for the top quantile, 'Sell' for the bottom, else 'Hold').
    Empty if the cross-section is too small to rank.
    """
    scores = predict_latest(booster, universe_data)
    if len(scores) < config.MIN_CROSS_SECTION:
        return pd.DataFrame(columns=['Score', 'Confidence', 'Signal'])

    rank = scores.rank(pct=True)
    signal = pd.Series('Hold', index=scores.index)
    signal[rank >= buy_quantile] = 'Buy'
    signal[rank <= sell_quantile] = 'Sell'
    return pd.DataFrame({'Score': scores, 'Confidence': rank, 'Signal': signal})


def make_signal_fn(
    booster: lgb.Booster,
    universe_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
    buy_quantile: float = config.MODEL_BUY_QUANTILE,
    sell_quantile: float = config.MODEL_SELL_QUANTILE,
):
    """Backtest signal function: Buy the top score quantile per date,
    Sell the bottom quantile, Hold in between."""
    scores = score_history(booster, universe_data)
    if scores.empty:
        return lambda row, info: 'Hold'

    grouped = scores.groupby('date')['score']
    hi = grouped.transform(lambda s: s.quantile(buy_quantile))
    lo = grouped.transform(lambda s: s.quantile(sell_quantile))
    eligible = grouped.transform('size') >= config.MIN_CROSS_SECTION

    signals: Dict[Tuple[Any, Any], str] = {}
    buys = scores[eligible & (scores['score'] >= hi)]
    sells = scores[eligible & (scores['score'] <= lo)]
    signals.update({(t, d): 'Buy' for t, d in zip(buys['ticker'], buys['date'])})
    signals.update({(t, d): 'Sell' for t, d in zip(sells['ticker'], sells['date'])})

    def signal_fn(row, info):
        return signals.get((row.get('_ticker'), row.get('_date')), 'Hold')

    return signal_fn


def print_evaluation(summary: Dict[str, Any], booster: Optional[lgb.Booster] = None) -> None:
    print("\n" + "=" * 72)
    print(f"WALK-FORWARD EVALUATION  {summary['start'].date()} -> {summary['end'].date()}")
    print("=" * 72)
    print(f"Folds:                    {summary['folds']} "
          f"({summary['scored_dates']} scored dates)")
    print(f"Mean rank IC:             {summary['mean_ic']:+.4f}")
    print(f"IC t-stat (optimistic):   {summary['ic_t_stat']:+.2f}")
    print(f"Positive-IC days:         {summary['pct_positive_ic_days']:.1%}")
    print(f"Top-bottom 20d spread:    {summary['mean_spread']:+.4%}")
    print("\nInterpretation: mean IC of +0.02..0.05 is a realistic usable edge; "
          "\nnear zero (or negative) means the features carry no signal yet.")
    if booster is not None:
        importance = sorted(
            zip(FEATURES, booster.feature_importance(importance_type='gain')),
            key=lambda pair: pair[1], reverse=True,
        )
        print("\nFEATURE IMPORTANCE (gain):")
        total = sum(gain for _, gain in importance) or 1
        for name, gain in importance:
            print(f"  {name:<16} {gain / total:6.1%}")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the LightGBM cross-sectional model"
    )
    parser.add_argument("--tickers", nargs="*", help="Explicit tickers")
    parser.add_argument("--sp500", action="store_true", help="Use the live S&P 500 universe")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max tickers when using --sp500 (default 100)")
    parser.add_argument("--period", default="10y", help="History window (default 10y)")
    parser.add_argument("--output", default=config.MODEL_PATH,
                        help=f"Where to save the trained model (default {config.MODEL_PATH})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.tickers:
        tickers = args.tickers
    elif args.sp500:
        tickers = data.get_universe()[: args.limit]
    else:
        tickers = list(config.DEFAULT_UNIVERSE)

    logger.info(f"Fetching {args.period} of history for {len(tickers)} tickers...")
    universe_data = data.fetch_universe_data(
        tickers, period=args.period, with_fundamentals=False
    )
    if not universe_data:
        raise SystemExit("No market data could be fetched.")

    logger.info("Building dataset...")
    panel = build_dataset(universe_data)
    logger.info(f"Dataset: {len(panel)} rows, "
                f"{panel['ticker'].nunique()} tickers, "
                f"{panel['date'].nunique()} dates")

    logger.info("Walk-forward evaluation...")
    summary = walk_forward_evaluate(panel)

    logger.info("Training final model on all data...")
    booster = train_final(panel)
    save_model(booster, summary, model_path=args.output)
    print_evaluation(summary, booster)
    print(f"Model saved to {args.output} (metadata: {config.MODEL_META_PATH})")
    print("Backtest it with:  python -m stock_picker.backtest --model "
          f"{args.output}  (in-sample over the training window — "
          "trust the walk-forward numbers above)")


if __name__ == "__main__":
    main()
