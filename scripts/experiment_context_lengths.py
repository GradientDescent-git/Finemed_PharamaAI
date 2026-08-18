from __future__ import annotations

from pathlib import Path

from finemed_ai.demand_forecasting.backtest import (
    BacktestConfig,
    run_backtest,
)
from finemed_ai.demand_forecasting.config import ForecastConfig
from finemed_ai.demand_forecasting.predictor_service import PredictorService


SERIES_PATH = Path(
    "data/04_silver/demand_forecasting/chronos_series.parquet"
)

OUTPUT_ROOT = Path(
    "data/05_gold/demand_forecasting/context_experiments_fair"
)

CONTEXT_LENGTHS = (
    365,
    730,
    1095,
    1460,
)


def run_experiment() -> None:
    for context_length in CONTEXT_LENGTHS:

        print()
        print("=" * 80)
        print(f"CONTEXT LENGTH EXPERIMENT: {context_length}")
        print("=" * 80)

        # Make sure no PredictorService from the previous experiment
        # survives into this configuration.
        PredictorService.reset_instance()

        config = ForecastConfig(
            context_length=context_length,
            prediction_length=30,
            quantile_levels=(
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ),
            point_quantile=0.5,
            apply_bias_correction=False,
        )

        output_dir = (
            OUTPUT_ROOT
            / f"context_{context_length}"
        )

        run_backtest(
            series_path=SERIES_PATH,
            output_dir=output_dir,
            forecast_config=config,
            backtest_config=BacktestConfig(
                horizon=30,
                min_history=1460,
                min_medicine_history=760,
                )
        )

        PredictorService.reset_instance()


if __name__ == "__main__":
    run_experiment()