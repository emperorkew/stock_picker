"""Loading and storing stock data (Supabase, CSV files in data/, ...)."""

import pandas as pd

from stock_picker.db import get_client


def load_table(table: str) -> pd.DataFrame:
    """Load a full Supabase table into a DataFrame."""
    response = get_client().table(table).select("*").execute()
    return pd.DataFrame(response.data)
