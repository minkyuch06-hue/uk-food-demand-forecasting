from src.forecasting import (
    load_series,
    make_backtest,
    score_backtest,
    plot_history,
    plot_backtest,
    plot_errors,
    forecast_next_12,
    plot_outlook,
)
from src.waste_model import waste_scenarios


def main():
    y = load_series()
    bt = make_backtest(y)
    metrics = score_backtest(bt)

    print(f"Series start: {y.index.min()}")
    print(f"Series end:   {y.index.max()}")
    print(f"Observations: {len(y)}\n")

    print(bt.head(), "\n")
    print(metrics, "\n")

    best = metrics.loc["ridge", "MAE"]
    seasonal = metrics.loc["seasonal_naive", "MAE"]
    err_reduction = 1 - best / seasonal

    print(f"MAE reduction vs seasonal naive: {err_reduction:.1%}\n")

    scenarios = waste_scenarios(err_reduction)
    print("=== Waste scenarios ===")
    print(scenarios, "\n")

    fc = forecast_next_12(y)
    print("=== Next 12 months forecast ===")
    print(fc.head(), "\n")

    plot_history(y)
    plot_backtest(bt)
    plot_errors(metrics)
    plot_outlook(y, fc)

    bt.to_csv("outputs/backtest_predictions.csv")
    metrics.to_csv("outputs/metrics.csv")
    scenarios.to_csv("outputs/waste_scenarios.csv")
    fc.to_csv("outputs/forecast_next_12m.csv", header=["forecast"])


if __name__ == "__main__":
    main()