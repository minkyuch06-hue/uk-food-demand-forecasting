from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.easter import easter
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/ons_eagw_food_stores_volume_nsa.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

TEST_START = "2022-05-01"
TEST_END = "2026-04-01"
MONTH_RE = r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b"


def load_series(path: Path = DATA_PATH) -> pd.Series:
    raw = pd.read_csv(path, skiprows=7, names=["period", "value"])

    raw["period"] = raw["period"].astype(str).str.strip()
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    monthly = raw[raw["period"].str.contains(MONTH_RE, na=False, regex=True)].copy()
    monthly["date"] = pd.to_datetime(monthly["period"], format="%Y %b", errors="coerce")
    monthly = monthly.dropna(subset=["date", "value"]).sort_values("date")

    series = monthly.set_index("date")["value"].astype(float).asfreq("MS")
    return series


def build_feature_frame(y: pd.Series) -> pd.DataFrame:
    df = y.rename("value").to_frame()
    df["t"] = np.arange(len(df))
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["lag1"] = df["value"].shift(1)
    df["lag3"] = df["value"].shift(3)
    df["lag12"] = df["value"].shift(12)
    df["roll3"] = df["value"].shift(1).rolling(3).mean()
    df["roll12"] = df["value"].shift(1).rolling(12).mean()
    df["gap12"] = df["value"] - df["value"].shift(12)

    def easter_month_flag(dt):
        return int(dt.month == easter(dt.year).month)

    def easter_shift_flag(dt):
        return int(easter(dt.year).month != easter(dt.year - 1).month)

    df["easter_month"] = [easter_month_flag(dt) for dt in df.index]
    df["easter_shift"] = [easter_shift_flag(dt) for dt in df.index]
    df["covid_flag"] = ((df.index >= "2020-03-01") & (df.index <= "2021-12-01")).astype(int)

    return df


def make_backtest(y: pd.Series, test_start: str = TEST_START, test_end: str = TEST_END) -> pd.DataFrame:
    ft = build_feature_frame(y)
    test_index = ft.loc[test_start:test_end].index

    feature_cols = [
        "t",
        "month_sin",
        "month_cos",
        "lag1",
        "lag3",
        "lag12",
        "roll3",
        "roll12",
        "easter_month",
        "easter_shift",
        "covid_flag",
    ]

    rows = []
    for dt in test_index:
        train = ft.loc[ft.index < dt].copy()
        train = train.dropna(subset=feature_cols + ["gap12"])

        X_train = train[feature_cols]
        y_train = train["gap12"]

        X_test = ft.loc[[dt], feature_cols]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        pred_gap = model.predict(X_test)[0]
        ridge_forecast = ft.loc[dt, "lag12"] + pred_gap

        rows.append(
            {
                "date": dt,
                "actual": ft.loc[dt, "value"],
                "naive": y.shift(1).loc[dt],
                "seasonal_naive": y.shift(12).loc[dt],
                "ridge": ridge_forecast,
            }
        )

    bt = pd.DataFrame(rows).set_index("date").dropna()
    return bt


def score_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["naive", "seasonal_naive", "ridge"]:
        err = bt["actual"] - bt[col]
        rows.append(
            {
                "model": col,
                "MAE": err.abs().mean(),
                "RMSE": np.sqrt((err ** 2).mean()),
                "MAPE_pct": (err.abs() / bt["actual"]).mean() * 100,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def plot_history(y: pd.Series, filename: str = "fig1_history.png") -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y.index, y.values, color="#1d70b8", lw=1)
    ax.set_title("GB food store sales volumes, monthly")
    ax.set_ylabel("Index")
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)


def plot_backtest(bt: pd.DataFrame, filename: str = "fig2_backtest.png") -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(bt.index, bt["actual"], color="black", lw=1.6, marker="o", ms=2.5, label="Actual")
    ax.plot(bt.index, bt["seasonal_naive"], color="#e55000", lw=1.4, marker="o", ms=2.5,
            label="Seasonal naive (same month last year)")
    ax.plot(bt.index, bt["ridge"], color="#1d70b8", lw=1.4, marker="o", ms=2.5,
            label="Ridge + engineered features")
    ax.set_title("48-month backtest")
    ax.set_ylabel("Index")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)

def plot_errors(metrics: pd.DataFrame, filename: str = "fig3_errors.png") -> None:
    order = ["naive", "seasonal_naive", "ridge"]
    labels = ["Naive\n(last month)", "Seasonal naive\n(last year)", "Ridge\n(features)"]
    vals = metrics.loc[order, "MAPE_pct"]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = ["#505a5f", "#e55000", "#1d70b8"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.9)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%",
                ha="center", fontsize=9, fontweight="bold")

    imp = (1 - vals["ridge"] / vals["seasonal_naive"]) * 100
    ax.set_title(f"Backtest error (MAPE): ridge cuts seasonal-naive error by {imp:.0f}%")
    ax.set_ylabel("MAPE, %")
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)

def forecast_next_12(y: pd.Series) -> pd.Series:
    """
    Simple 12-month outlook using ridge refit on all history.
    """
    ft = build_feature_frame(y)
    feature_cols = [
        "t",
        "month_sin",
        "month_cos",
        "lag1",
        "lag3",
        "lag12",
        "roll3",
        "roll12",
        "easter_month",
        "easter_shift",
        "covid_flag",
    ]

    train = ft.dropna(subset=feature_cols + ["gap12"]).copy()
    X_train = train[feature_cols]
    y_train = train["gap12"]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    future_index = pd.date_range(y.index.max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future_rows = []

    history = y.copy()
    for dt in future_index:
        temp = pd.concat([history, pd.Series([np.nan], index=[dt])])
        temp_ft = build_feature_frame(temp)
        row = temp_ft.loc[[dt], feature_cols].copy()

        # fill any needed lag features recursively from history
        if row[["lag1", "lag3", "lag12", "roll3", "roll12"]].isna().any(axis=None):
            row = row.fillna(method="ffill", axis=0)

        pred_gap = model.predict(row)[0]
        pred = temp_ft.loc[dt, "lag12"] + pred_gap

        future_rows.append((dt, pred))
        history.loc[dt] = pred

    return pd.Series(
        [v for _, v in future_rows],
        index=[d for d, _ in future_rows],
        name="forecast",
    )

def plot_outlook(y: pd.Series, fc: pd.Series, filename: str = "fig4_outlook.png") -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    hist = y.loc["2023":]
    ax.plot(hist.index, hist.values, color="#1d70b8", lw=1.3, label="Actual")
    link = pd.concat([hist.tail(1), fc])
    ax.plot(link.index, link.values, color="#e55000", lw=1.3, ls="--", marker="o",
            ms=3, label="12-month outlook")
    ax.set_title("Where the model thinks food volumes go next")
    ax.set_ylabel("Index")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)