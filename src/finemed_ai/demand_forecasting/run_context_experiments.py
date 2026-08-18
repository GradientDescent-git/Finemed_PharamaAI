from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from finemed_ai.demand_forecasting.backtest import (
    BacktestConfig,
    run_backtest,
)
from finemed_ai.demand_forecasting.config import ForecastConfig
from finemed_ai.demand_forecasting.predictor_service import PredictorService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


SERIES_PATH = Path(
    "data/04_silver/demand_forecasting/chronos_series.parquet"
)

BASE_OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/context_experiments"
)

CONTEXT_LENGTHS = (
    90,
    180,
    365,
    730,
)


def run_context_experiments() -> None:
    if not SERIES_PATH.exists():
        raise FileNotFoundError(
            f"Series file not found: {SERIES_PATH}"
        )

    BASE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_summaries = []

    for context_length in CONTEXT_LENGTHS:

        print()
        print("=" * 80)
        print(
            f"CONTEXT-LENGTH EXPERIMENT: {context_length} DAYS"
        )
        print("=" * 80)

        forecast_config = ForecastConfig(
            context_length=context_length,
        )
        PredictorService.reset_instance()

        output_dir = (
            BASE_OUTPUT_DIR
            / f"context_{context_length}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Running Chronos-2 with context_length=%d",
            context_length,
        )

        run_backtest(
            series_path=SERIES_PATH,
            output_dir=output_dir,
            forecast_config=forecast_config,
            backtest_config=BacktestConfig(),
        )

        summary_path = (
            output_dir
            / "backtest_summary.parquet"
        )

        if not summary_path.exists():
            raise RuntimeError(
                f"Expected summary not found: {summary_path}"
            )

        summary = pd.read_parquet(
            summary_path
        )

        chronos = summary[
            summary["Model"] == "chronos-2-P50"
        ].copy()

        if chronos.empty:
            logger.warning(
                "No Chronos P50 result for context=%d",
                context_length,
            )
            continue

        chronos.insert(
            0,
            "Context_Length",
            context_length,
        )

        experiment_summaries.append(
            chronos
        )

    if not experiment_summaries:
        raise RuntimeError(
            "No context experiment results were produced."
        )

    comparison = pd.concat(
        experiment_summaries,
        ignore_index=True,
    )

    columns = [
        "Context_Length",
        "Model",
        "Windows",
        "Medicines",
        "Samples",
        "Total_Actual",
        "Total_Predicted",
        "Total_Absolute_Error",
        "WAPE_Pct",
        "MAE",
        "sMAPE_Pct",
        "MBE",
        "P10_P90_Coverage_Pct",
    ]

    comparison = comparison[
        [
            column
            for column in columns
            if column in comparison.columns
        ]
    ].sort_values(
        "Context_Length"
    )

    comparison_path = (
        BASE_OUTPUT_DIR
        / "context_comparison.parquet"
    )

    comparison.to_parquet(
        comparison_path,
        index=False,
    )

    csv_path = (
        BASE_OUTPUT_DIR
        / "context_comparison.csv"
    )

    comparison.to_csv(
        csv_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("CHRONOS-2 CONTEXT-LENGTH COMPARISON")
    print("=" * 80)

    print(
        comparison[
            [
                "Context_Length",
                "WAPE_Pct",
                "MAE",
                "sMAPE_Pct",
                "MBE",
                "P10_P90_Coverage_Pct",
            ]
        ].to_string(
            index=False
        )
    )

    best = comparison.loc[
        comparison["WAPE_Pct"].idxmin()
    ]

    print()
    print("=" * 80)
    print("CURRENT BEST CONTEXT")
    print("=" * 80)

    print(
        f"Context length : "
        f"{int(best['Context_Length'])} days"
    )

    print(
        f"WAPE           : "
        f"{best['WAPE_Pct']:.2f}%"
    )

    print(
        f"MAE            : "
        f"{best['MAE']:.2f}"
    )

    print(
        f"sMAPE          : "
        f"{best['sMAPE_Pct']:.2f}%"
    )

    print(
        f"MBE            : "
        f"{best['MBE']:.2f}"
    )

    print(
        f"Coverage       : "
        f"{best['P10_P90_Coverage_Pct']:.2f}%"
    )

    print()
    print(
        f"Comparison saved : {comparison_path}"
    )

    print(
        f"CSV saved        : {csv_path}"
    )


if __name__ == "__main__":
    run_context_experiments()