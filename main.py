"""Entry point for the stock picker."""

from stock_picker import analysis, data, plotting


def main() -> None:
    assets = data.load_table("assets")
    picks = analysis.pick_stocks(assets)
    print(picks)
    if not picks.empty:
        plotting.plot_prices(picks)


if __name__ == "__main__":
    main()