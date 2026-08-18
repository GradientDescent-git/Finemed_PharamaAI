from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from finemed_ai.demand_forecasting.backtest import (
    BacktestConfig,
    run_backtest,
)
from finemed_ai.demand_forecasting.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
)
from finemed_ai.demand_forecasting.predictor_service import (
    PredictorService,
)

logger = logging.getLogger(__name__)


CONTEXT_LENGTHS = (
    365,
    540,
    730,
    1095,
    1460,
)

DEFAULT_SERIES_PATH = Path(
    "data/04_silver/demand_forecasting/chronos_series.parquet"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/context_optimization"
)


def run_context_optimization(
    series_path: str | Path = DEFAULT_SERIES_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Evaluate Chronos-2 across multiple context lengths.

    The experiment keeps the forecasting horizon and quantile configuration
    fixed while changing only context_length.

    Context lengths:
        365
        540
        730
        1095
        1460

    The existing production backtest implementation is reused so that
    context-length experiments remain directly comparable with the
    established baseline evaluation.
    """

    series_path = Path(series_path)
    output_dir = Path(output_dir)

    if not series_path.exists():
        raise FileNotFoundError(
            f"Forecasting series not found: {series_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_results = []

    print("=" * 80)
    print("FINEMED CHRONOS-2 CONTEXT-LENGTH OPTIMIZATION")
    print("=" * 80)

    print(
        "Context lengths:",
        ", ".join(str(x) for x in CONTEXT_LENGTHS),
    )

    print(
        "Prediction length:",
        DEFAULT_CONFIG.prediction_length,
    )

    print(
        "Point forecast baseline:",
        f"P{int(DEFAULT_CONFIG.point_quantile * 100)}",
    )

    print()

    for context_length in CONTEXT_LENGTHS:

        print("=" * 80)
        print(
            f"RUNNING CONTEXT LENGTH: {context_length} DAYS"
        )
        print("=" * 80)

        # -----------------------------------------------------------
        # IMPORTANT:
        # PredictorService is process-wide singleton.
        # Reset it so the new ForecastConfig is actually used.
        # -----------------------------------------------------------

        PredictorService.reset_instance()

        config = ForecastConfig(
            model_id=DEFAULT_CONFIG.model_id,
            context_length=context_length,
            prediction_length=DEFAULT_CONFIG.prediction_length,
            quantile_levels=DEFAULT_CONFIG.quantile_levels,
            point_quantile=DEFAULT_CONFIG.point_quantile,
            apply_bias_correction=DEFAULT_CONFIG.apply_bias_correction,
            id_column=DEFAULT_CONFIG.id_column,
            timestamp_column=DEFAULT_CONFIG.timestamp_column,
            target_column=DEFAULT_CONFIG.target_column,
            output_id_column=DEFAULT_CONFIG.output_id_column,
            output_date_column=DEFAULT_CONFIG.output_date_column,
            output_point_column=DEFAULT_CONFIG.output_point_column,
        )

        context_output_dir = (
            output_dir
            / f"context_{context_length}"
        )

        # Reuse exactly the existing six-cutoff backtest.
        result_df = run_backtest(
            series_path=series_path,
            output_dir=context_output_dir,
            forecast_config=config,
            backtest_config=BacktestConfig(
                horizon=DEFAULT_CONFIG.prediction_length,
                min_history=730,
                seasonal_period=7,
                min_medicine_history=760,
            ),
        )

        # -----------------------------------------------------------
        # Keep only Chronos results for context comparison.
        # Baselines are already evaluated by the normal backtest.
        # -----------------------------------------------------------

        chronos_results = result_df[
            result_df["Model"].str.startswith(
                "chronos-2-"
            )
        ].copy()

        if chronos_results.empty:
            logger.warning(
                "No Chronos results for context=%d",
                context_length,
            )
            continue

        chronos_results["Context_Length"] = (
            context_length
        )

        # -----------------------------------------------------------
        # Aggregate context-level metrics.
        # -----------------------------------------------------------

        for model, group in chronos_results.groupby(
            "Model",
            sort=True,
        ):

            total_actual = group[
                "Total_Actual"
            ].sum()

            total_absolute_error = group[
                "Total_Absolute_Error"
            ].sum()

            wape = (
                total_absolute_error
                / total_actual
                * 100.0
                if total_actual != 0
                else float("nan")
            )

            experiment_results.append(
                {
                    "Context_Length": context_length,
                    "Model": model,
                    "Medicines": group[
                        "Medicine_ID"
                    ].nunique(),
                    "Windows": group[
                        "Cutoff_Date"
                    ].nunique(),
                    "Samples": group[
                        "Sample_Count"
                    ].sum(),
                    "Total_Actual": total_actual,
                    "Total_Predicted": group[
                        "Total_Predicted"
                    ].sum(),
                    "Total_Absolute_Error": total_absolute_error,
                    "WAPE_Pct": wape,
                    "MAE": group["MAE"].mean(),
                    "sMAPE_Pct": group[
                        "sMAPE_Pct"
                    ].mean(),
                    "MBE": group["MBE"].mean(),
                    "P10_P90_Coverage_Pct": group[
                        "P10_P90_Coverage_Pct"
                    ].mean(),
                }
            )

        print()
        print(
            "Context results:"
        )

        print(
            chronos_results
            .groupby("Model")
            .agg(
                Medicines=("Medicine_ID", "nunique"),
                WAPE_Pct=(
                    "Total_Absolute_Error",
                    lambda x: (
                        x.sum()
                        / chronos_results.loc[
                            x.index,
                            "Total_Actual",
                        ].sum()
                        * 100.0
                    ),
                ),
                MAE=("MAE", "mean"),
                sMAPE_Pct=("sMAPE_Pct", "mean"),
                MBE=("MBE", "mean"),
                Coverage=("P10_P90_Coverage_Pct", "mean"),
            )
            .to_string()
        )

    # ---------------------------------------------------------------
    # Final experiment dataframe
    # ---------------------------------------------------------------

    if not experiment_results:
        raise RuntimeError(
            "Context optimization produced no Chronos results."
        )

    result = pd.DataFrame(
        experiment_results
    )

    result = result.sort_values(
        [
            "Model",
            "WAPE_Pct",
            "Context_Length",
        ]
    ).reset_index(drop=True)

    result_path = (
        output_dir
        / "context_optimization_results.parquet"
    )

    result.to_parquet(
        result_path,
        index=False,
    )

    csv_path = (
        output_dir
        / "context_optimization_results.csv"
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # P50 summary — primary production candidate
    # ---------------------------------------------------------------

    p50 = result[
        result["Model"] == "chronos-2-P50"
    ].copy()

    p50 = p50.sort_values(
        "WAPE_Pct"
    )

    print()
    print("=" * 80)
    print("P50 CONTEXT-LENGTH COMPARISON")
    print("=" * 80)

    print(
        p50[
            [
                "Context_Length",
                "Medicines",
                "Windows",
                "WAPE_Pct",
                "MAE",
                "sMAPE_Pct",
                "MBE",
                "P10_P90_Coverage_Pct",
            ]
        ].to_string(index=False)
    )

    if not p50.empty:

        best = p50.iloc[0]

        print()
        print("=" * 80)
        print("BEST P50 CONTEXT")
        print("=" * 80)

        print(
            f"Context length : "
            f"{int(best['Context_Length'])} days"
        )

        print(
            f"WAPE           : "
            f"{best['WAPE_Pct']:.3f}%"
        )

        print(
            f"MAE            : "
            f"{best['MAE']:.3f}"
        )

        print(
            f"sMAPE          : "
            f"{best['sMAPE_Pct']:.3f}%"
        )

        print(
            f"MBE            : "
            f"{best['MBE']:.3f}"
        )

    print()
    print(
        f"Results saved : {result_path}"
    )

    print(
        f"CSV saved     : {csv_path}"
    )

    return result


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    run_context_optimization()