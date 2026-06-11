# Stock Picker

Fetch market data, screen stocks, and plot the results.

## Project layout

```
main.py                 Entry point: fetch -> analyze -> log to Supabase -> dashboard
stock_picker/           The application package
    config.py           Tunables: universe, thresholds, trade sizing, schedule
    db.py               Supabase client setup (+ connection check)
    data.py             S&P 500 universe, batched price downloads, cached fundamentals
    analysis.py         Indicators (EMA/RSI/MACD/volume) and signal scoring
    model.py            LightGBM cross-sectional return model (train/evaluate/score)
    backtest.py         Walk-forward backtest harness
    ledger.py           Supabase-backed ledger: signals, snapshots, simulated trades
    plotting.py         Matplotlib charts
tests/                  Pytest tests
notebooks/              Jupyter notebooks for exploration
data/                   Local caches (not committed)
supabase/migrations/    Database schema + pg_cron data-retention jobs
```

Old data is purged automatically by `pg_cron` jobs in the database:
market snapshots after 2 years, trading signals after 1 year. The trade
ledger is kept forever.

## Setup

1. Create/activate the conda environment and install dependencies:

   ```bash
   conda activate stock-picker
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your Supabase credentials
   (Supabase dashboard > Project Settings > API).

3. Test the database connection:

   ```bash
   python -m stock_picker.db
   ```

## Run

```bash
python main.py              # scheduled runs (every 24h)
python main.py --run-once   # single run
```

## Backtest

Replays the live signal rules over history with next-day-open fills and
transaction costs, and compares against an equal-weight buy-and-hold
benchmark:

```bash
python -m stock_picker.backtest                      # default universe, 5y
python -m stock_picker.backtest --sp500 --limit 50   # wider universe
python -m stock_picker.backtest --tickers AAPL MSFT --period 10y
```

## ML model

Trains a LightGBM model that ranks stocks by predicted forward 20-day
return relative to the universe median. Evaluation is walk-forward with an
embargo gap (the honest numbers — mean rank IC, hit rate, decile spread);
the final model is saved to `data/lgbm_model.txt`:

```bash
python -m stock_picker.model --sp500 --limit 100 --period 10y
python -m stock_picker.backtest --model data/lgbm_model.txt   # in-sample demo
```

## Tests

```bash
pytest
```