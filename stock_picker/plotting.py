"""Matplotlib charts for picked stocks."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_prices(df: pd.DataFrame, title: str = "Prices") -> None:
    """Plot a DataFrame of price series (one column per ticker)."""
    df.plot(title=title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()