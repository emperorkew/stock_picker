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
python main.py
```

## Tests

```bash
pytest
```