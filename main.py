"""Entry point for the stock picker."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import schedule

from stock_picker import analysis, config, data
from stock_picker.ledger import Ledger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _fmt(spec: str):
    """NaN-safe column formatter for DataFrame.to_string."""
    return lambda value: spec.format(value) if pd.notna(value) else '—'


def print_dashboard(recommendations, portfolio_summary, cash=None, realized_pnl=None,
                    signal_source='rules'):
    """Print a structured dashboard to the console."""
    print("\n" + "=" * 80)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"STOCK PICKER DASHBOARD - {now}  [signals: {signal_source}]")
    print("=" * 80)

    print("\nCURRENT RECOMMENDATIONS (Showing Top 20):")
    print("-" * 80)
    if not recommendations.empty:
        display_cols = ['Signal', 'Score', 'Price', 'P/E', 'FCF_Yield', 'RSI', 'MACD']
        display_cols = [c for c in display_cols if c in recommendations.columns]

        # Buy/Sell first; within each signal class, best model score first.
        display_df = recommendations.copy()
        display_df['Signal_Rank'] = display_df['Signal'].map({'Buy': 0, 'Sell': 1, 'Hold': 2})
        sort_cols, ascending = ['Signal_Rank'], [True]
        if 'Score' in display_df.columns:
            sort_cols.append('Score')
            ascending.append(False)
        display_df = (
            display_df.sort_values(sort_cols, ascending=ascending)
            .drop(columns=['Signal_Rank'])
            .head(20)
        )

        formatters = {
            'Score': _fmt('{:+.4f}'),
            'Price': _fmt('{:.2f}'),
            'P/E': _fmt('{:.2f}'),
            'FCF_Yield': _fmt('{:.4f}'),
            'RSI': _fmt('{:.2f}'),
            'MACD': _fmt('{:.4f}'),
        }

        print(display_df[display_cols].to_string(formatters=formatters))
        if len(recommendations) > 20:
            print(f"... and {len(recommendations) - 20} more hold signals.")
    else:
        print("No recommendations available.")

    print("\nTHEORETICAL PORTFOLIO PERFORMANCE:")
    print("-" * 80)
    if not portfolio_summary.empty:
        summary = portfolio_summary.copy()

        if not recommendations.empty:
            current_prices = recommendations['Price'].to_dict()
            summary['current_price'] = summary['ticker'].map(current_prices)
        else:
            summary['current_price'] = float('nan')

        summary['cost_basis'] = summary['shares_owned'] * summary['avg_price']
        summary['total_value'] = summary['shares_owned'] * summary['current_price']
        summary['unrealized_pnl'] = summary['total_value'] - summary['cost_basis']
        summary['pnl_percent'] = summary['unrealized_pnl'] / summary['cost_basis'].where(
            summary['cost_basis'] > 0
        ) * 100

        print(summary.to_string(
            columns=['ticker', 'shares_owned', 'avg_price', 'current_price',
                     'total_value', 'unrealized_pnl', 'pnl_percent'],
            formatters={
                'shares_owned': _fmt('{:.4f}'),
                'avg_price': _fmt('${:.2f}'),
                'current_price': _fmt('${:.2f}'),
                'total_value': _fmt('${:.2f}'),
                'unrealized_pnl': _fmt('${:.2f}'),
                'pnl_percent': _fmt('{:.2f}%'),
            },
            index=False
        ))

        # Totals only over positions with a known current price, stated honestly.
        priced = summary.dropna(subset=['current_price'])
        unpriced = len(summary) - len(priced)
        total_value = priced['total_value'].sum()
        total_cost = priced['cost_basis'].sum()
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        print("\nPORTFOLIO TOTALS:")
        print(f"Total Cost Basis: ${total_cost:.2f}")
        print(f"Total Value:      ${total_value:.2f}")
        print(f"Unrealized P&L:   ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        if unpriced:
            print(f"({unpriced} position(s) excluded from totals — no current price)")
    else:
        print("Portfolio is empty.")

    if realized_pnl is not None:
        print(f"Realized P&L:     ${realized_pnl:.2f}")
    if cash is not None:
        print(f"Cash available:   ${cash:.2f}")

    print("=" * 80 + "\n")


def _load_model_or_none():
    """The trained LightGBM model, or None to fall back to the rule scorecard."""
    if not Path(config.MODEL_PATH).exists():
        logger.warning(
            f"No trained model at {config.MODEL_PATH} — falling back to rule-based "
            "signals. Train one with: python -m stock_picker.model --sp500 --limit 100"
        )
        return None
    from stock_picker import model as ml  # lazy: lightgbm import is slow
    return ml.load_model(config.MODEL_PATH)


def _apply_model_signals(recommendations, universe_data, booster):
    """Replace the rule-based Signal column with model rankings.

    Adds Score (raw prediction) and Confidence (rank percentile); tickers the
    model can't score (not enough history) stay 'Hold'. Returns the signal
    source actually used.
    """
    from stock_picker import model as ml

    model_signals = ml.latest_signals(booster, universe_data)
    if model_signals.empty:
        logger.warning(
            "Model produced no scores (cross-section too small or history too "
            "short) — using rule-based signals."
        )
        return recommendations, 'rules'

    recommendations = recommendations.join(model_signals[['Score', 'Confidence']])
    recommendations['Signal'] = 'Hold'
    common = recommendations.index.intersection(model_signals.index)
    recommendations.loc[common, 'Signal'] = model_signals.loc[common, 'Signal']
    return recommendations, 'lgbm-model'


def job():
    """Main automated job to fetch data, analyze, log, and print dashboard."""
    logger.info("Starting stock picker job...")

    universe = data.get_universe()
    logger.info(f"Fetching market data for universe of {len(universe)} stocks...")
    universe_data = data.fetch_universe_data(universe)

    if not universe_data:
        logger.error("Failed to fetch any market data. Aborting job.")
        return

    logger.info("Analyzing data and generating signals...")
    recommendations = analysis.pick_stocks(universe_data)

    signal_source = 'rules'
    booster = _load_model_or_none()
    if booster is not None:
        recommendations, signal_source = _apply_model_signals(
            recommendations, universe_data, booster
        )
    logger.info(f"Signal source: {signal_source} "
                f"({(recommendations['Signal'] == 'Buy').sum()} buys, "
                f"{(recommendations['Signal'] == 'Sell').sum()} sells, "
                f"{(recommendations['Signal'] == 'Hold').sum()} holds)")

    portfolio_summary = pd.DataFrame(columns=['ticker', 'shares_owned', 'avg_price'])
    cash = realized_pnl = None
    try:
        logger.info("Logging signals and executing simulated trades in Supabase...")
        run_time = datetime.now(timezone.utc)
        ledger = Ledger()
        ledger.log_signals(recommendations, run_time, rationale=signal_source)
        ledger.log_snapshots(recommendations, run_time)
        ledger.execute_trades(recommendations, run_time)

        holdings, cash, realized_pnl = ledger.get_state()
        portfolio_summary = ledger.get_portfolio_summary(holdings)
    except Exception:
        logger.exception("Failed to persist results to Supabase; showing recommendations only")

    print_dashboard(recommendations, portfolio_summary, cash, realized_pnl, signal_source)
    logger.info("Job completed.")


def _model_age_days():
    """Days since the model was trained, or None if there is no model/metadata."""
    meta_path = Path(config.MODEL_META_PATH)
    if not Path(config.MODEL_PATH).exists() or not meta_path.exists():
        return None
    try:
        trained_at = datetime.fromisoformat(json.loads(meta_path.read_text())['trained_at'])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return (datetime.now(timezone.utc) - trained_at).days


def retrain_job():
    """Retrain the model; the next daily job picks the new file up from disk."""
    from stock_picker import model as ml  # lazy: lightgbm import is slow

    logger.info("Starting model retrain...")
    _, summary = ml.retrain()
    logger.info(f"Retrain complete (mean walk-forward IC {summary['mean_ic']:+.4f}); "
                "the next daily job will use the new model.")


def _run_job_safely():
    """One failed run must not kill the scheduler loop."""
    try:
        job()
    except Exception:
        logger.exception("Job failed; will retry at the next scheduled run")


def _run_retrain_safely():
    try:
        retrain_job()
    except Exception:
        logger.exception("Model retrain failed; keeping the previous model")


def run_scheduler(interval_hours=config.DEFAULT_INTERVAL_HOURS):
    """Run the daily job on a schedule, retraining the model monthly."""
    # Retrain first if the model is missing or stale, then run once immediately.
    age = _model_age_days()
    if age is None or age >= config.RETRAIN_INTERVAL_DAYS:
        logger.info("Model missing or stale "
                    f"({'no model' if age is None else f'{age} days old'}) — retraining first...")
        _run_retrain_safely()
    _run_job_safely()

    logger.info(f"Scheduling job every {interval_hours} hours and a model retrain "
                f"every {config.RETRAIN_INTERVAL_DAYS} days...")
    schedule.every(interval_hours).hours.do(_run_job_safely)
    schedule.every(config.RETRAIN_INTERVAL_DAYS).days.do(_run_retrain_safely)

    while True:
        schedule.run_pending()
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Stock Picker Engine")
    parser.add_argument("--run-once", action="store_true", help="Run the job once and exit")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the model once and exit")
    parser.add_argument("--interval", type=int, default=config.DEFAULT_INTERVAL_HOURS,
                        help="Interval in hours for scheduled runs")

    args = parser.parse_args()

    if args.retrain:
        retrain_job()
    elif args.run_once:
        job()
    else:
        run_scheduler(interval_hours=args.interval)


if __name__ == "__main__":
    main()
