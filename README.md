# Stock Picker

Fetch market data, screen stocks, and plot the results.

## Project layout

```
main.py                 Entry point: load data -> pick stocks -> plot
stock_picker/           The application package
    db.py               Supabase client setup (+ connection check)
    data.py             Loading/storing stock data
    analysis.py         Screening and stock-picking logic
    plotting.py         Matplotlib charts
tests/                  Pytest tests
notebooks/              Jupyter notebooks for exploration
data/                   Local data files (not committed)
```

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