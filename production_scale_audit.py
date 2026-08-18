from __future__ import annotations

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.production_forecast_router import (
    ProductionForecastRouter,
    build_routing_table,
)
from finemed_ai.demand_forecasting.production_forecast_service import (
    ProductionForecastService,
)


HISTORY_PATH = (
    "data/04_silver/demand_forecasting/daily_demand.parquet"
)

ROBUSTNESS_PATH = (
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/medicine_model_robustness.parquet"
)


def main() -> None:

    # ---------------------------------------------------------------
    # Load production history
    # ---------------------------------------------------------------

    history = pd.read_parquet(HISTORY_PATH)

    history = history.rename(
        columns={
            "MDCODE": "Medicine_ID",
            "INVDT": "timestamp",
            "Demand_Qty": "target",
        }
    )

    history["Medicine_ID"] = history["Medicine_ID"].astype(str)
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["target"] = pd.to_numeric(
        history["target"],
        errors="raise",
    )

    # ---------------------------------------------------------------
    # Build frozen validation routing
    # ---------------------------------------------------------------

    robustness = pd.read_parquet(
        ROBUSTNESS_PATH
    )

    routing = build_routing_table(
        robustness
    )

    # ---------------------------------------------------------------
    # Production service
    # ---------------------------------------------------------------

    service = ProductionForecastService(
        router=ProductionForecastRouter()
    )

    rows = []
    failed = []

    print("=" * 100)
    print("PRODUCTION FORECAST SCALE AUDIT")
    print("=" * 100)
    print()
    print(f"History rows       : {len(history):,}")
    print(f"History medicines  : {history['Medicine_ID'].nunique()}")
    print(f"Routing medicines  : {len(routing)}")
    print(
        "Chronos routed     : "
        f"{(routing['Selected_Model'] == 'chronos-2-P50').sum()}"
    )
    print(
        "TSB routed         : "
        f"{(routing['Selected_Model'] == 'tsb').sum()}"
    )
    print()

    # ---------------------------------------------------------------
    # Forecast every routed medicine
    # ---------------------------------------------------------------

    for medicine_id in routing["Medicine_ID"].astype(str):

        try:

            result = service.forecast_medicine(
                medicine_id=medicine_id,
                history_df=history,
                routing_table=routing,
            )

            medicine_history = history[
                history["Medicine_ID"]
                == medicine_id
            ].copy()

            # Recent 90-day demand scale.
            last_date = medicine_history["timestamp"].max()

            recent = medicine_history[
                medicine_history["timestamp"]
                >= last_date - pd.Timedelta(days=89)
            ]

            recent_total = float(
                recent["target"].sum()
            )

            recent_mean = (
                recent_total / 90.0
            )

            forecast_mean = float(
                np.mean(result.predicted_demand)
            )

            forecast_total = float(
                np.sum(result.predicted_demand)
            )

            p50_mean = (
                float(np.mean(result.p50))
                if result.p50 is not None
                else np.nan
            )

            p90_mean = (
                float(np.mean(result.p90))
                if result.p90 is not None
                else np.nan
            )

            scale_ratio = (
                forecast_mean / recent_mean
                if recent_mean > 0
                else np.nan
            )

            rows.append(
                {
                    "Medicine_ID": medicine_id,
                    "Selected_Model": result.selected_model,
                    "Validation_Advantage_Pct": (
                        result.routing_advantage_pct
                    ),
                    "Recent_90D_Total": recent_total,
                    "Recent_90D_Daily_Mean": recent_mean,
                    "Forecast_30D_Total": forecast_total,
                    "Forecast_Daily_Mean": forecast_mean,
                    "P50_Daily_Mean": p50_mean,
                    "P90_Daily_Mean": p90_mean,
                    "Forecast_to_Recent_Ratio": scale_ratio,
                    "Forecast_Min": min(
                        result.predicted_demand
                    ),
                    "Forecast_Max": max(
                        result.predicted_demand
                    ),
                }
            )

        except Exception as exc:

            failed.append(
                (
                    medicine_id,
                    str(exc),
                )
            )

    audit = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Overall results
    # ---------------------------------------------------------------

    print("=" * 100)
    print("EXECUTION")
    print("=" * 100)

    print(
        f"Successful medicines : {audit['Medicine_ID'].nunique()}"
    )
    print(
        f"Failed medicines     : {len(failed)}"
    )
    print()

    if failed:

        print("FAILED:")
        for medicine_id, error in failed:
            print(
                f"  {medicine_id}: {error}"
            )
        print()

    # ---------------------------------------------------------------
    # Model-level scale statistics
    # ---------------------------------------------------------------

    print("=" * 100)
    print("MODEL SCALE SUMMARY")
    print("=" * 100)

    summary = (
        audit
        .groupby("Selected_Model")
        [
            [
                "Recent_90D_Daily_Mean",
                "Forecast_Daily_Mean",
                "Forecast_to_Recent_Ratio",
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(summary.to_string())
    print()

    # ---------------------------------------------------------------
    # Chronos detailed audit
    # ---------------------------------------------------------------

    chronos = audit[
        audit["Selected_Model"]
        == "chronos-2-P50"
    ].copy()

    print("=" * 100)
    print("CHRONOS SCALE AUDIT")
    print("=" * 100)

    if chronos.empty:

        print("No Chronos-routed medicines.")
        print()

    else:

        display_columns = [
            "Medicine_ID",
            "Validation_Advantage_Pct",
            "Recent_90D_Daily_Mean",
            "Forecast_Daily_Mean",
            "P50_Daily_Mean",
            "P90_Daily_Mean",
            "Forecast_to_Recent_Ratio",
            "Forecast_30D_Total",
        ]

        print(
            chronos[
                display_columns
            ]
            .sort_values(
                "Forecast_to_Recent_Ratio"
            )
            .to_string(index=False)
        )

        print()

        print(
            "Chronos ratio < 0.10 : "
            f"{(chronos['Forecast_to_Recent_Ratio'] < 0.10).sum()}"
        )

        print(
            "Chronos ratio < 0.25 : "
            f"{(chronos['Forecast_to_Recent_Ratio'] < 0.25).sum()}"
        )

        print(
            "Chronos ratio < 0.50 : "
            f"{(chronos['Forecast_to_Recent_Ratio'] < 0.50).sum()}"
        )

        print(
            "Chronos ratio >= 0.50: "
            f"{(chronos['Forecast_to_Recent_Ratio'] >= 0.50).sum()}"
        )

        print()

    # ---------------------------------------------------------------
    # Near-zero forecast detection
    # ---------------------------------------------------------------

    print("=" * 100)
    print("NEAR-ZERO FORECAST DETECTION")
    print("=" * 100)

    near_zero = audit[
        (
            audit["Recent_90D_Daily_Mean"] > 0
        )
        &
        (
            audit["Forecast_to_Recent_Ratio"] < 0.10
        )
    ]

    print(
        f"Medicines with forecast <10% of recent demand: "
        f"{len(near_zero)}"
    )

    if not near_zero.empty:

        print()

        print(
            near_zero[
                [
                    "Medicine_ID",
                    "Selected_Model",
                    "Recent_90D_Daily_Mean",
                    "Forecast_Daily_Mean",
                    "Forecast_to_Recent_Ratio",
                ]
            ]
            .sort_values(
                "Forecast_to_Recent_Ratio"
            )
            .to_string(index=False)
        )

    print()

    # ---------------------------------------------------------------
    # Save audit
    # ---------------------------------------------------------------

    output_path = (
        "data/05_gold/demand_forecasting/"
        "production_scale_audit.parquet"
    )

    audit.to_parquet(
        output_path,
        index=False,
    )

    print("=" * 100)
    print("AUDIT ARTIFACT")
    print("=" * 100)
    print(output_path)
    print()

    print("NEXT:")
    print(
        "Use this audit to determine whether the Chronos "
        "under-scaling is isolated or systematic."
    )


if __name__ == "__main__":
    main()