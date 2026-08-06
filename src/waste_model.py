from __future__ import annotations

import pandas as pd


def waste_scenarios(err_reduction: float) -> pd.DataFrame:
    """
    Translate forecast improvement into rough annual savings scenarios.
    err_reduction should be a decimal, e.g. 0.16 for 16% MAE reduction.
    """
    base_waste_pool = 166000  # illustrative annual waste pool per store (£)
    scenarios = {
        "low": 0.30,
        "central": 0.50,
        "high": 0.70,
    }

    rows = []
    for label, forecast_share in scenarios.items():
        saving = base_waste_pool * forecast_share * err_reduction
        rows.append(
            {
                "scenario": label,
                "share_of_waste_forecast_driven": forecast_share,
                "annual_saving_per_store_gbp": round(saving, 0),
            }
        )

    return pd.DataFrame(rows).set_index("scenario")