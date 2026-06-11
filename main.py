"""Entry point for the stock picker."""

import logging
import time
import schedule
import torch

from stock_picker import analysis, data
from stock_picker.model import StockLSTM, get_historical_data, preprocess_data, log_return_to_percentage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Features to use for sequence prediction
FEATURE_COLS = ['price', 'volume', 'volume_avg_20d', 'pe_ratio', 'fcf_yield', 'ema_50', 'ema_200', 'rsi_14', 'macd_line', 'macd_signal']

# Define and load model (in production, weights would be loaded here)
# For now, we use an uninitialized model for sequence prediction loop
MODEL = StockLSTM(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2)
MODEL.eval()

def run_orchestration_loop() -> None:
    """Run the main interval-driven execution loop."""
    logger.info("Starting orchestration loop iteration.")
    try:
        # Load active assets
        assets = data.load_table("assets")
        active_assets = assets[assets['is_active'] == True]

        for _, asset in active_assets.iterrows():
            asset_id = asset['id']
            ticker = asset['ticker']
            logger.info(f"Processing asset: {ticker} ({asset_id})")

            # Ingest real-time data
            metrics = data.fetch_market_data(ticker)
            if not metrics:
                continue

            # Write current metrics
            data.insert_market_snapshot(asset_id, metrics)

            # Fetch historical sequence for PyTorch model
            hist_df = get_historical_data(asset_id, limit=30)

            predicted_return_pct = 0.0
            if len(hist_df) == 30:
                # Preprocess
                tensor_data, _ = preprocess_data(hist_df, FEATURE_COLS)

                # Predict
                with torch.no_grad():
                    log_return_pred = MODEL(tensor_data).item()

                # Convert
                predicted_return_pct = log_return_to_percentage(log_return_pred)
                logger.info(f"Predicted return for {ticker}: {predicted_return_pct:.4f}%")
            else:
                logger.warning(f"Not enough historical data for {ticker}. Found {len(hist_df)} snapshots. Skipping prediction.")

            # Evaluate factors
            score, rationale, trend = analysis.evaluate_factors(metrics, predicted_return_pct)

            # Generate signals
            signal = analysis.generate_signal(score, trend)

            # Log signal
            analysis.log_signal(asset_id, signal, score, rationale, predicted_return_pct)

            # Execute trade if action required
            if signal in ['BUY', 'SELL']:
                price = metrics.get('price')
                if price:
                    analysis.execute_trade(asset_id, signal, price)

    except Exception as e:
        logger.error(f"Error in orchestration loop: {e}", exc_info=True)
    logger.info("Finished orchestration loop iteration.")

def main() -> None:
    """Run the engine as a background daemon using schedule."""
    logger.info("Starting automated stock picking and forecasting engine.")

    # Run once immediately
    run_orchestration_loop()

    # Schedule to run every 4 hours
    schedule.every(4).hours.do(run_orchestration_loop)

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
