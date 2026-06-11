"""Entry point for the stock picker."""

import argparse
import logging
import sys
import time
from datetime import datetime

import schedule

from stock_picker import analysis, data
from stock_picker.ledger import Ledger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Pre-defined universe of stocks
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

def print_dashboard(recommendations, portfolio_summary):
    """Print a structured dashboard to the console."""
    print("\n" + "="*80)
    print(f"STOCK PICKER DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\nCURRENT RECOMMENDATIONS:")
    print("-" * 80)
    if not recommendations.empty:
        # Format the dataframe for display
        display_cols = ['Signal', 'Price', 'P/E', 'FCF_Yield', 'RSI', 'MACD']
        # Filter only existing columns just in case
        display_cols = [c for c in display_cols if c in recommendations.columns]

        # Format floats
        formatters = {
            'Price': '{:.2f}'.format,
            'P/E': '{:.2f}'.format,
            'FCF_Yield': '{:.4f}'.format,
            'RSI': '{:.2f}'.format,
            'MACD': '{:.4f}'.format,
        }

        print(recommendations[display_cols].to_string(formatters=formatters))
    else:
        print("No recommendations available.")

    print("\nTHEORETICAL PORTFOLIO PERFORMANCE:")
    print("-" * 80)
    if not portfolio_summary.empty:
        # We need current prices to show performance
        # For simplicity, if we have recommendations, we use those prices
        summary_display = portfolio_summary.copy()

        # Simple performance calc
        if not recommendations.empty:
            current_prices = recommendations['Price'].to_dict()
            summary_display['current_price'] = summary_display['ticker'].map(current_prices)

            # Fill NA with avg_price if we couldn't fetch current price
            summary_display['current_price'] = summary_display['current_price'].fillna(summary_display['avg_price'])

            summary_display['total_value'] = summary_display['shares_owned'] * summary_display['current_price']
            summary_display['cost_basis'] = summary_display['shares_owned'] * summary_display['avg_price']
            summary_display['unrealized_pnl'] = summary_display['total_value'] - summary_display['cost_basis']
            summary_display['pnl_percent'] = (summary_display['unrealized_pnl'] / summary_display['cost_basis']) * 100

            # Format
            print(summary_display.to_string(
                columns=['ticker', 'shares_owned', 'avg_price', 'current_price', 'total_value', 'unrealized_pnl', 'pnl_percent'],
                formatters={
                    'shares_owned': '{:.4f}'.format,
                    'avg_price': '${:.2f}'.format,
                    'current_price': '${:.2f}'.format,
                    'total_value': '${:.2f}'.format,
                    'unrealized_pnl': '${:.2f}'.format,
                    'pnl_percent': '{:.2f}%'.format,
                },
                index=False
            ))

            total_value = summary_display['total_value'].sum()
            total_cost = summary_display['cost_basis'].sum()
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            print("\nPORTFOLIO TOTALS:")
            print(f"Total Cost Basis: ${total_cost:.2f}")
            print(f"Total Value:      ${total_value:.2f}")
            print(f"Unrealized P&L:   ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        else:
            print(summary_display.to_string(index=False))

    else:
        print("Portfolio is empty.")

    print("="*80 + "\n")


def job():
    """Main automated job to fetch data, analyze, log, and print dashboard."""
    logger.info("Starting scheduled stock picker job...")

    # 1. Fetch Data
    logger.info(f"Fetching market data for universe: {UNIVERSE}")
    universe_data = data.fetch_universe_data(UNIVERSE)

    if not universe_data:
        logger.error("Failed to fetch any market data. Aborting job.")
        return

    # 2. Analyze
    logger.info("Analyzing data and generating signals...")
    recommendations = analysis.pick_stocks(universe_data)

    # 3. Log and Execute
    logger.info("Logging signals and executing simulated trades...")
    ledger = Ledger()
    ledger.log_signals(recommendations)
    ledger.execute_trades(recommendations)

    # 4. Print Dashboard
    portfolio_summary = ledger.get_portfolio_summary()
    print_dashboard(recommendations, portfolio_summary)

    logger.info("Job completed successfully.")

def run_scheduler(interval_hours=24):
    """Run the job on a schedule."""
    # Run once immediately
    job()

    logger.info(f"Scheduling job to run every {interval_hours} hours...")
    schedule.every(interval_hours).hours.do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Stock Picker Engine")
    parser.add_argument("--run-once", action="store_true", help="Run the job once and exit")
    parser.add_argument("--interval", type=int, default=24, help="Interval in hours for scheduled runs")

    args = parser.parse_args()

    if args.run_once:
        job()
    else:
        run_scheduler(interval_hours=args.interval)

if __name__ == "__main__":
    main()
