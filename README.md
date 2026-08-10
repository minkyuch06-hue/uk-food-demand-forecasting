# UK Food Demand Forecasting

Forecasted monthly UK food retail demand using the ONS EAGW series. A 48-month backtest showed that ridge regression with engineered features beat seasonal naive by 15.9% MAE, and the improvement translated into rough annual savings of £7.9k to £18.4k per store.

## Outputs

- [Backtest predictions CSV](outputs/backtest_predictions.csv)
- [Metrics CSV](outputs/metrics.csv)
- [Waste scenarios CSV](outputs/waste_scenarios.csv)
- [Next 12 months forecast CSV](outputs/forecast_next_12m.csv)

### Charts

![History chart](outputs/fig1_history.png)

![Backtest chart](outputs/fig2_backtest.png)

![Error chart](outputs/fig3_errors.png)

![Outlook chart](outputs/fig4_outlook.png)

## Method

I loaded the ONS monthly food-store series, built lag and calendar features, and ran an expanding-window 48-month backtest. I compared naive, seasonal naive, and ridge regression forecasts, then translated forecast improvement into waste-savings scenarios.

## Results

Ridge regression was the best model in the backtest. It reduced MAE by 15.9% versus seasonal naive.

## Business Value

The forecast improvement suggests rough annual savings of £7.9k to £18.4k per store under low, central, and high assumptions.

## Limitations

This is a simplified forecasting project. The waste numbers are scenario estimates, not measured savings, and the forecast is built from a single monthly retail series.

## How to run

```bash
python run.py