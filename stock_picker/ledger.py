"""Supabase-backed ledger: logs signals, market snapshots, and simulated trades.

All writes go to the project's Supabase tables (assets, trading_signals,
market_snapshots, portfolio_ledger). Holdings, cash, and realized P&L are
derived by replaying the trade ledger in timestamp order.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from stock_picker import config
from stock_picker.db import get_client

logger = logging.getLogger(__name__)

SIGNAL_ENUM = {'Buy': 'BUY', 'Hold': 'HOLD', 'Sell': 'SELL'}

# Bounds of the NUMERIC columns in the schema; values outside are stored as NULL.
_PRICE_MAX = 99_999_999.9999   # NUMERIC(12, 4)
_RATIO_MAX = 9_999.99          # NUMERIC(6, 2)
_PERCENT_MAX = 999.99          # NUMERIC(5, 2)

Holdings = Dict[str, Dict[str, float]]


def _clean(value: Any, ndigits: int, max_abs: float) -> Optional[float]:
    """NaN-safe rounding that respects the DB column's NUMERIC bounds."""
    if value is None or pd.isna(value):
        return None
    rounded = round(float(value), ndigits)
    return rounded if abs(rounded) <= max_abs else None


def replay_ledger(
    rows: Iterable[Dict[str, Any]], starting_cash: float
) -> Tuple[Holdings, float, float]:
    """Replay BUY/SELL rows (already ordered by time) into portfolio state.

    Returns (holdings, cash, realized_pnl) where holdings maps ticker to
    {'shares': ..., 'cost': ...}. Sells without a position are ignored and
    sell quantities are clamped to the shares actually held.
    """
    holdings: Holdings = {}
    cash = float(starting_cash)
    realized = 0.0

    for row in rows:
        ticker = row['ticker']
        quantity = float(row['quantity'])
        price = float(row['price_per_unit'])

        if row['action'] == 'BUY':
            position = holdings.setdefault(ticker, {'shares': 0.0, 'cost': 0.0})
            position['shares'] += quantity
            position['cost'] += quantity * price
            cash -= quantity * price
        elif row['action'] == 'SELL':
            position = holdings.get(ticker)
            if not position or position['shares'] <= 0:
                continue
            quantity = min(quantity, position['shares'])
            avg_price = position['cost'] / position['shares']
            realized += quantity * (price - avg_price)
            position['shares'] -= quantity
            position['cost'] -= quantity * avg_price
            cash += quantity * price
            if position['shares'] <= 1e-9:
                del holdings[ticker]

    return holdings, cash, realized


class Ledger:
    def __init__(
        self,
        client=None,
        starting_cash: float = config.STARTING_CASH,
        investment_per_trade: float = config.INVESTMENT_PER_TRADE,
    ):
        self.client = client or get_client()
        self.starting_cash = starting_cash
        self.investment_per_trade = investment_per_trade

    @staticmethod
    def _timestamp(timestamp: Optional[datetime]) -> str:
        return (timestamp or datetime.now(timezone.utc)).isoformat()

    def _asset_ids(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map tickers in df's index to asset UUIDs, creating missing assets."""
        tickers = [str(t) for t in df.index]
        existing = (
            self.client.table('assets')
            .select('id, ticker')
            .in_('ticker', tickers)
            .execute()
            .data
        )
        ids = {row['ticker']: row['id'] for row in existing}

        missing = [t for t in tickers if t not in ids]
        if missing:
            rows = []
            for ticker in missing:
                name = df.loc[ticker].get('Name') if 'Name' in df.columns else None
                if not isinstance(name, str) or not name.strip():
                    name = ticker
                rows.append({'ticker': ticker, 'name': name})
            inserted = self.client.table('assets').insert(rows).execute().data
            ids.update({row['ticker']: row['id'] for row in inserted})

        return ids

    def log_signals(self, df: pd.DataFrame, timestamp: Optional[datetime] = None) -> None:
        """Log the generated signals into trading_signals."""
        if df.empty:
            return
        ts = self._timestamp(timestamp)
        ids = self._asset_ids(df)

        rows = [
            {'asset_id': ids[ticker], 'timestamp': ts, 'signal': SIGNAL_ENUM[row['Signal']]}
            for ticker, row in df.iterrows()
            if ticker in ids and row['Signal'] in SIGNAL_ENUM
        ]
        if rows:
            self.client.table('trading_signals').insert(rows).execute()

    def log_snapshots(self, df: pd.DataFrame, timestamp: Optional[datetime] = None) -> None:
        """Log indicator/fundamental snapshots into market_snapshots."""
        if df.empty:
            return
        ts = self._timestamp(timestamp)
        ids = self._asset_ids(df)

        rows = []
        for ticker, row in df.iterrows():
            price = _clean(row.get('Price'), 4, _PRICE_MAX)
            volume = row.get('Volume')
            if ticker not in ids or price is None or pd.isna(volume):
                continue  # price and volume are NOT NULL in the schema
            volume_avg = row.get('Volume_20SMA')
            rows.append({
                'asset_id': ids[ticker],
                'timestamp': ts,
                'price': price,
                'volume': int(volume),
                'volume_avg_20d': None if pd.isna(volume_avg) else int(volume_avg),
                'pe_ratio': _clean(row.get('P/E'), 2, _RATIO_MAX),
                'fcf_yield': _clean(row.get('FCF_Yield'), 2, _PERCENT_MAX),
                'ema_50': _clean(row.get('EMA_50'), 4, _PRICE_MAX),
                'ema_200': _clean(row.get('EMA_200'), 4, _PRICE_MAX),
                'rsi_14': _clean(row.get('RSI'), 2, _PERCENT_MAX),
                'macd_line': _clean(row.get('MACD'), 4, _PRICE_MAX),
                'macd_signal': _clean(row.get('Signal_Line'), 4, _PRICE_MAX),
            })
        if rows:
            # Upsert so re-running a job for the same timestamp can't duplicate.
            self.client.table('market_snapshots').upsert(
                rows, on_conflict='asset_id,timestamp'
            ).execute()

    def execute_trades(self, df: pd.DataFrame, timestamp: Optional[datetime] = None) -> None:
        """Simulate trades: buy only with available cash and no open position,
        sell the actually held quantity."""
        if df.empty:
            return
        ts = self._timestamp(timestamp)
        holdings, cash, _ = self.get_state()
        ids = self._asset_ids(df)

        trades = []
        for ticker, row in df.iterrows():
            price = row.get('Price')
            if ticker not in ids or pd.isna(price) or price <= 0:
                continue

            if row['Signal'] == 'Buy':
                if ticker in holdings or cash < self.investment_per_trade:
                    continue
                quantity = round(self.investment_per_trade / price, 6)
                cash -= quantity * price
                trades.append({
                    'asset_id': ids[ticker],
                    'timestamp': ts,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price_per_unit': round(float(price), 4),
                })
            elif row['Signal'] == 'Sell':
                position = holdings.pop(ticker, None)
                if not position:
                    continue
                trades.append({
                    'asset_id': ids[ticker],
                    'timestamp': ts,
                    'action': 'SELL',
                    'quantity': round(position['shares'], 6),
                    'price_per_unit': round(float(price), 4),
                })

        if trades:
            self.client.table('portfolio_ledger').insert(trades).execute()

    def _ledger_rows(self) -> List[Dict[str, Any]]:
        """All trades joined with their ticker, ordered by time, paged past
        PostgREST's 1000-row response limit."""
        rows: List[Dict[str, Any]] = []
        page_size = 1000
        start = 0
        while True:
            page = (
                self.client.table('portfolio_ledger')
                .select('timestamp, action, quantity, price_per_unit, assets(ticker)')
                .order('timestamp')
                .range(start, start + page_size - 1)
                .execute()
                .data
            )
            for row in page:
                ticker = (row.get('assets') or {}).get('ticker')
                if ticker:
                    rows.append({
                        'ticker': ticker,
                        'action': row['action'],
                        'quantity': row['quantity'],
                        'price_per_unit': row['price_per_unit'],
                    })
            if len(page) < page_size:
                return rows
            start += page_size

    def get_state(self) -> Tuple[Holdings, float, float]:
        """Return (holdings, cash, realized_pnl) replayed from the ledger."""
        return replay_ledger(self._ledger_rows(), self.starting_cash)

    def get_portfolio_summary(self, holdings: Optional[Holdings] = None) -> pd.DataFrame:
        """Current holdings as a DataFrame of ticker, shares_owned, avg_price."""
        if holdings is None:
            holdings, _, _ = self.get_state()
        if not holdings:
            return pd.DataFrame(columns=['ticker', 'shares_owned', 'avg_price'])
        return pd.DataFrame([
            {
                'ticker': ticker,
                'shares_owned': position['shares'],
                'avg_price': position['cost'] / position['shares'],
            }
            for ticker, position in holdings.items()
        ])
