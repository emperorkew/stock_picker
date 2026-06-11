"""Walk-forward backtest of the signal rules.

Run from the command line:

    python -m stock_picker.backtest                       # default universe, 5y
    python -m stock_picker.backtest --tickers AAPL MSFT   # specific tickers
    python -m stock_picker.backtest --sp500 --limit 50    # first 50 S&P 500 names

Point-in-time discipline:
- All indicators are causal (EWMs and rolling windows), so the value at bar t
  only uses bars <= t.
- A signal computed at bar t's close executes at bar t+1's open — never the
  same bar.
- Fundamentals are NOT known point-in-time (yfinance only exposes today's
  P/E and FCF), so by default the backtest scores technicals only.
  --fundamentals static applies *today's* fundamentals to all of history,
  which is look-ahead bias — use it only as a sensitivity check.
"""

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from stock_picker import analysis, config, data

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

    @property
    def closed(self) -> bool:
        return self.exit_price is not None

    @property
    def pnl(self) -> Optional[float]:
        if not self.closed:
            return None
        return (self.exit_price - self.entry_price) * self.shares


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    closed_trades: List[Trade]
    open_trades: List[Trade]
    signal_stats: pd.DataFrame
    metrics: Dict[str, Any]


SignalFn = Callable[[Dict[str, Any], Dict[str, Any]], str]


def _prepare(universe_data, use_fundamentals):
    """Indicator frames, per-date row dicts, and forward returns per ticker."""
    records: Dict[str, Dict[pd.Timestamp, Dict[str, Any]]] = {}
    infos: Dict[str, Dict[str, Any]] = {}
    for ticker, (hist, info) in universe_data.items():
        if hist is None or len(hist) < config.WARMUP_BARS + 2:
            continue
        df = analysis.calculate_indicators(hist)
        # Forward returns are for *evaluation* only; the signal function
        # never reads them.
        df = df.assign(
            _fwd_5d=df['Close'].shift(-5) / df['Close'] - 1,
            _fwd_20d=df['Close'].shift(-20) / df['Close'] - 1,
        )
        records[ticker] = dict(zip(df.index, df.to_dict('records')))
        infos[ticker] = info if use_fundamentals else {}
    return records, infos


def run_backtest(
    universe_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
    starting_cash: float = config.STARTING_CASH,
    investment_per_trade: float = config.INVESTMENT_PER_TRADE,
    fee_rate: float = config.TRANSACTION_FEE_RATE,
    use_fundamentals: bool = False,
    signal_fn: Optional[SignalFn] = None,
) -> BacktestResult:
    """Walk forward day by day, trading the same rules the live system uses."""
    records, infos = _prepare(universe_data, use_fundamentals)
    if not records:
        raise ValueError(
            f"No ticker has the {config.WARMUP_BARS + 2}+ bars of history needed to backtest."
        )

    if signal_fn is None:
        def signal_fn(row, info):
            buy, sell = analysis.score_signals(row, info, use_fundamentals=use_fundamentals)
            return analysis.signal_from_scores(buy, sell)

    calendar = sorted(set().union(*(set(r) for r in records.values())))
    n_tickers = len(records)
    bench_slice = starting_cash / n_tickers

    cash = starting_cash
    positions: Dict[str, Trade] = {}
    closed: List[Trade] = []
    pending: Dict[str, str] = {}
    bars_seen = {ticker: 0 for ticker in records}
    last_close: Dict[str, float] = {}

    bench_cash = starting_cash
    bench_shares: Dict[str, float] = {}

    equity_dates: List[pd.Timestamp] = []
    equity_values: List[float] = []
    bench_values: List[float] = []
    exposure: List[float] = []
    signal_log: List[Dict[str, Any]] = []

    for date in calendar:
        # 1) Execute orders queued at earlier closes, at today's open.
        for ticker in list(pending):
            row = records[ticker].get(date)
            if row is None:
                continue  # ticker didn't trade today; order stays queued
            open_price = row.get('Open')
            if open_price is None or pd.isna(open_price) or open_price <= 0:
                continue
            action = pending.pop(ticker)
            if action == 'BUY' and ticker not in positions:
                cost = investment_per_trade
                if cash < cost * (1 + fee_rate):
                    continue
                shares = cost / open_price
                cash -= cost * (1 + fee_rate)
                positions[ticker] = Trade(ticker, date, open_price, shares)
            elif action == 'SELL' and ticker in positions:
                trade = positions.pop(ticker)
                proceeds = trade.shares * open_price
                cash += proceeds * (1 - fee_rate)
                trade.exit_date = date
                trade.exit_price = open_price
                closed.append(trade)

        # 2) Generate signals at today's close (queued for tomorrow's open).
        for ticker, ticker_records in records.items():
            row = ticker_records.get(date)
            if row is None:
                continue
            last_close[ticker] = row['Close']
            bars_seen[ticker] += 1
            if bars_seen[ticker] < config.WARMUP_BARS:
                continue
            if bars_seen[ticker] == config.WARMUP_BARS and ticker not in bench_shares:
                # Benchmark buys and holds this ticker from the same moment
                # the strategy is first allowed to trade it.
                bench_shares[ticker] = bench_slice / row['Close']
                bench_cash -= bench_slice

            signal = signal_fn(row, infos[ticker])
            signal_log.append({
                'signal': signal,
                'fwd_5d': row.get('_fwd_5d'),
                'fwd_20d': row.get('_fwd_20d'),
            })
            if signal == 'Buy' and ticker not in positions:
                pending[ticker] = 'BUY'
            elif signal == 'Sell' and ticker in positions:
                pending[ticker] = 'SELL'

        # 3) Mark to market at the close, once any ticker is past warmup.
        if not bench_shares:
            continue
        invested = sum(t.shares * last_close[t.ticker] for t in positions.values())
        equity = cash + invested
        bench = bench_cash + sum(
            shares * last_close[ticker] for ticker, shares in bench_shares.items()
        )
        equity_dates.append(date)
        equity_values.append(equity)
        bench_values.append(bench)
        exposure.append(invested / equity if equity > 0 else 0.0)

    if not equity_dates:
        raise ValueError("Backtest window is shorter than the warmup period.")

    equity_curve = pd.Series(equity_values, index=equity_dates, name='equity')
    benchmark_curve = pd.Series(bench_values, index=equity_dates, name='benchmark')
    signal_stats = _signal_stats(signal_log)
    metrics = _metrics(equity_curve, benchmark_curve, closed, positions, exposure)

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        closed_trades=closed,
        open_trades=list(positions.values()),
        signal_stats=signal_stats,
        metrics=metrics,
    )


def _signal_stats(signal_log: List[Dict[str, Any]]) -> pd.DataFrame:
    """Average forward returns and hit rates per signal class.

    If Buy signals don't show better forward returns than Hold, the rules
    carry no predictive information regardless of portfolio performance.
    """
    df = pd.DataFrame(signal_log)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby('signal').agg(
        count=('signal', 'size'),
        avg_fwd_5d=('fwd_5d', 'mean'),
        hit_rate_5d=('fwd_5d', lambda s: (s > 0).mean()),
        avg_fwd_20d=('fwd_20d', 'mean'),
        hit_rate_20d=('fwd_20d', lambda s: (s > 0).mean()),
    )
    order = [s for s in ('Buy', 'Hold', 'Sell') if s in grouped.index]
    return grouped.loc[order]


def _metrics(equity, benchmark, closed, positions, exposure) -> Dict[str, Any]:
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((equity.index[-1] - equity.index[0]).days, 1) / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    volatility = returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        returns.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        if returns.std() > 0 else float('nan')
    )
    max_drawdown = (equity / equity.cummax() - 1).min()

    wins = [t for t in closed if t.pnl > 0]
    benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1

    return {
        'start': equity.index[0].date(),
        'end': equity.index[-1].date(),
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades_closed': len(closed),
        'trades_open': len(positions),
        'win_rate': len(wins) / len(closed) if closed else float('nan'),
        'avg_trade_pnl': sum(t.pnl for t in closed) / len(closed) if closed else float('nan'),
        'avg_exposure': sum(exposure) / len(exposure) if exposure else 0.0,
        'benchmark_return': benchmark_return,
        'excess_return': total_return - benchmark_return,
    }


def print_report(result: BacktestResult) -> None:
    m = result.metrics
    pct = '{:+.2%}'.format
    print("\n" + "=" * 72)
    print(f"BACKTEST REPORT  {m['start']} -> {m['end']}")
    print("=" * 72)
    print(f"Strategy total return:   {pct(m['total_return'])}   (CAGR {pct(m['cagr'])})")
    print(f"Buy & hold benchmark:    {pct(m['benchmark_return'])}")
    print(f"Excess vs benchmark:     {pct(m['excess_return'])}")
    print(f"Sharpe ratio:            {m['sharpe']:.2f}")
    print(f"Annualized volatility:   {m['volatility']:.2%}")
    print(f"Max drawdown:            {m['max_drawdown']:.2%}")
    print(f"Avg invested exposure:   {m['avg_exposure']:.2%}")
    print(f"Trades: {m['trades_closed']} closed ({m['win_rate']:.0%} winners, "
          f"avg P&L ${m['avg_trade_pnl']:.2f}), {m['trades_open']} still open"
          if m['trades_closed'] else
          f"Trades: 0 closed, {m['trades_open']} still open")
    print("\nSIGNAL QUALITY (forward returns after each signal):")
    if result.signal_stats.empty:
        print("  no signals generated")
    else:
        print(result.signal_stats.to_string(float_format='{:.4f}'.format))
        print("\n  A predictive system needs Buy rows to beat Hold rows; "
              "hit rates near 0.5 mean coin-flip.")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the stock picker's signal rules")
    parser.add_argument("--tickers", nargs="*", help="Explicit tickers to test")
    parser.add_argument("--sp500", action="store_true", help="Use the live S&P 500 universe")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max tickers when using --sp500 (default 50)")
    parser.add_argument("--period", default=config.BACKTEST_PERIOD,
                        help=f"History window (default {config.BACKTEST_PERIOD})")
    parser.add_argument("--fundamentals", choices=["none", "static"], default="none",
                        help="'static' applies today's fundamentals to all of history "
                             "(look-ahead bias; sensitivity check only)")
    parser.add_argument("--cash", type=float, default=config.STARTING_CASH)
    parser.add_argument("--fee", type=float, default=config.TRANSACTION_FEE_RATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.tickers:
        tickers = args.tickers
    elif args.sp500:
        tickers = data.get_universe()[: args.limit]
    else:
        tickers = list(config.DEFAULT_UNIVERSE)

    if args.fundamentals == "static":
        logger.warning("Static fundamentals apply TODAY's P/E and FCF to all of history "
                       "— results will be optimistic.")

    logger.info(f"Fetching {args.period} of history for {len(tickers)} tickers...")
    universe_data = data.fetch_universe_data(tickers, period=args.period)
    if not universe_data:
        raise SystemExit("No market data could be fetched.")

    result = run_backtest(
        universe_data,
        starting_cash=args.cash,
        fee_rate=args.fee,
        use_fundamentals=args.fundamentals == "static",
    )
    print_report(result)


if __name__ == "__main__":
    main()
