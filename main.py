"""Entry point for the stock picker."""

from stock_picker import analysis, data, plotting


def main() -> None:
    stocks = data.load_table("stocks")
    picks = analysis.pick_stocks(stocks)
    print(picks)
    if not picks.empty:
        plotting.plot_prices(picks)


if __name__ == "__main__":
    main()