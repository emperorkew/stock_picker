"""Persistent ledger using SQLite to log signals and simulated portfolio."""

import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd


class Ledger:
    def __init__(self, db_path: str = "data/ledger.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize the database tables if they do not exist."""
        query = """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            signal TEXT,
            price REAL,
            pe_ratio REAL,
            fcf_yield REAL
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            action TEXT,
            price REAL,
            shares REAL
        );
        """
        with self._get_connection() as conn:
            conn.executescript(query)

    def log_signals(self, df: pd.DataFrame) -> None:
        """Log the generated signals from the analysis into the database."""
        if df.empty:
            return

        timestamp = datetime.now().isoformat()
        records = []

        for ticker, row in df.iterrows():
            records.append((
                timestamp,
                ticker,
                row['Signal'],
                row['Price'],
                row['P/E'],
                row['FCF_Yield']
            ))

        query = """
        INSERT INTO signals (timestamp, ticker, signal, price, pe_ratio, fcf_yield)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        with self._get_connection() as conn:
            conn.executemany(query, records)

    def execute_trades(self, df: pd.DataFrame, investment_amount: float = 1000.0) -> None:
        """Simulate transaction executions based on signals."""
        if df.empty:
            return

        timestamp = datetime.now().isoformat()
        records = []

        for ticker, row in df.iterrows():
            signal = row['Signal']
            price = row['Price']

            if pd.isna(price) or price <= 0:
                continue

            if signal == 'Buy':
                shares = investment_amount / price
                records.append((timestamp, ticker, 'Buy', price, shares))
            elif signal == 'Sell':
                # For simplicity, sell all holding of this ticker if it's 'Sell'
                # In a real system, you'd check current portfolio holdings
                # Here we just log the intent to sell
                records.append((timestamp, ticker, 'Sell', price, 0.0))

        if not records:
            return

        query = """
        INSERT INTO portfolio (timestamp, ticker, action, price, shares)
        VALUES (?, ?, ?, ?, ?)
        """

        with self._get_connection() as conn:
            conn.executemany(query, records)

    def get_portfolio_summary(self) -> pd.DataFrame:
        """Return a simple summary of current holdings based on the portfolio table."""
        query = """
        SELECT ticker, action, shares, price
        FROM portfolio
        """
        try:
            with self._get_connection() as conn:
                df = pd.read_sql(query, conn)

            if df.empty:
                return pd.DataFrame(columns=['ticker', 'shares_owned', 'avg_price'])

            summary = []
            for ticker, group in df.groupby('ticker'):
                shares = 0.0
                total_cost = 0.0
                for _, row in group.iterrows():
                    if row['action'] == 'Buy':
                        shares += row['shares']
                        total_cost += row['shares'] * row['price']
                    elif row['action'] == 'Sell':
                        shares = 0.0 # simplified: selling all
                        total_cost = 0.0

                if shares > 0:
                    summary.append({
                        'ticker': ticker,
                        'shares_owned': shares,
                        'avg_price': total_cost / shares
                    })
            return pd.DataFrame(summary)
        except sqlite3.Error:
            return pd.DataFrame(columns=['ticker', 'shares_owned', 'avg_price'])
