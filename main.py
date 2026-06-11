"""Entry point for the stock picker."""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

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


def print_dashboard(recommendations, portfolio_summary, cash=None, realized_pnl=None):
    """Print a structured dashboard to the console."""
    print("\n" + "=" * 80)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"STOCK PICKER DASHBOARD - {now}")
    print("=" * 80)

    print("\nCURRENT RECOMMENDATIONS (Showing Top 20):")
    print("-" * 80)
    if not recommendations.empty:
        display_cols = ['Signal', 'Price', 'P/E', 'FCF_Yield', 'RSI', 'MACD']
        display_cols = [c for c in display_cols if c in recommendations.columns]

        # Sort so we see Buy/Sell signals first
        display_df = recommendations.copy()
        display_df['Signal_Rank'] = display_df['Signal'].map({'Buy': 0, 'Sell': 1, 'Hold': 2})
        display_df = display_df.sort_values('Signal_Rank').drop(columns=['Signal_Rank']).head(20)

        formatters = {
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

    portfolio_summary = pd.DataFrame(columns=['ticker', 'shares_owned', 'avg_price'])
    cash = realized_pnl = None
    try:
        logger.info("Logging signals and executing simulated trades in Supabase...")
        run_time = datetime.now(timezone.utc)
        ledger = Ledger()
        ledger.log_signals(recommendations, run_time)
        ledger.log_snapshots(recommendations, run_time)
        ledger.execute_trades(recommendations, run_time)

        holdings, cash, realized_pnl = ledger.get_state()
        portfolio_summary = ledger.get_portfolio_summary(holdings)
    except Exception:
        logger.exception("Failed to persist results to Supabase; showing recommendations only")

    print_dashboard(recommendations, portfolio_summary, cash, realized_pnl)
    logger.info("Job completed.")


def _run_job_safely():
    """One failed run must not kill the scheduler loop."""
    try:
        job()
    except Exception:
        logger.exception("Job failed; will retry at the next scheduled run")


def run_scheduler(interval_hours=config.DEFAULT_INTERVAL_HOURS):
    """Run the job on a schedule."""
    # Run once immediately
    _run_job_safely()

    logger.info(f"Scheduling job to run every {interval_hours} hours...")
    schedule.every(interval_hours).hours.do(_run_job_safely)

    while True:
        schedule.run_pending()
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Stock Picker Engine")
    parser.add_argument("--run-once", action="store_true", help="Run the job once and exit")
    parser.add_argument("--interval", type=int, default=config.DEFAULT_INTERVAL_HOURS,
                        help="Interval in hours for scheduled runs")

    args = parser.parse_args()

    if args.run_once:
        job()
    else:
        run_scheduler(interval_hours=args.interval)


if __name__ == "__main__":
    main()
