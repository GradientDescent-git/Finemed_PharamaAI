from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
)
from finemed_ai.demand_forecasting.evaluation import compute_metrics
from finemed_ai.demand_forecasting.predictor_service import PredictorService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 30

    # Number of historical rolling evaluation windows.
    n_windows: int = 4

    # Minimum history required before creating a backtest.
    min_history: int = 730

    # Seasonal period for seasonal-naive baseline.
    seasonal_period: int = 7

    # Only evaluate medicines having enough history.
    min_medicine_history: int = 760


@dataclass
class BacktestResult:
    model: str
    cutoff_date: pd.Timestamp
    medicine_id: str
    sample_count: int
    total_actual: float
    total_predicted: float
    wape_pct: float
    mae: float
    smape_pct: float
    mbe: float
    coverage_pct: float

def _validate_series(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "MDCODE",
        "INVDT",
        "Demand_Qty",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Backtest dataset missing columns: {sorted(missing)}"
        )

    df = df[
        ["MDCODE", "INVDT", "Demand_Qty"]
    ].copy()

    df["MDCODE"] = (
        df["MDCODE"]
        .astype(str)
        .str.strip()
    )

    df["INVDT"] = pd.to_datetime(
        df["INVDT"],
        errors="coerce",
    )

    df["Demand_Qty"] = pd.to_numeric(
        df["Demand_Qty"],
        errors="coerce",
    )

    if df["INVDT"].isna().any():
        raise ValueError(
            "Backtest dataset contains invalid dates."
        )

    if df["Demand_Qty"].isna().any():
        raise ValueError(
            "Backtest dataset contains NULL demand."
        )

    if (df["Demand_Qty"] < 0).any():
        raise ValueError(
            "Backtest dataset contains negative demand."
        )

    duplicate_count = df.duplicated(
        ["MDCODE", "INVDT"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Backtest dataset contains {duplicate_count} "
            "duplicate medicine/date rows."
        )

    return (
        df
        .sort_values(["MDCODE", "INVDT"])
        .reset_index(drop=True)
    )


def _make_rolling_cutoffs(
    medicine_dates: pd.Series,
    config: BacktestConfig,
) -> List[pd.Timestamp]:

    dates = (
        pd.to_datetime(medicine_dates)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) < (
        config.min_history
        + config.horizon
    ):
        return []

    latest_possible_cutoff_index = (
        len(dates) - config.horizon - 1
    )

    earliest_cutoff_index = (
        config.min_history - 1
    )

    available = (
        latest_possible_cutoff_index
        - earliest_cutoff_index
        + 1
    )

    if available <= 0:
        return []

    window_count = min(
        config.n_windows,
        available,
    )

    indices = np.linspace(
        earliest_cutoff_index,
        latest_possible_cutoff_index,
        num=window_count,
        dtype=int,
    )

    return [
        pd.Timestamp(dates.iloc[index])
        for index in sorted(set(indices))
    ]


def _naive_forecast(
    history: pd.Series,
    horizon: int,
) -> np.ndarray:

    if history.empty:
        return np.zeros(horizon)

    value = float(history.iloc[-1])

    return np.repeat(
        max(value, 0.0),
        horizon,
    )


def _seasonal_naive_forecast(
    history: pd.Series,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:

    if history.empty:
        return np.zeros(horizon)

    if len(history) < seasonal_period:
        return _naive_forecast(
            history,
            horizon,
        )

    last_values = (
        history
        .iloc[-seasonal_period:]
        .to_numpy(dtype=float)
    )

    repeats = int(
        np.ceil(
            horizon / seasonal_period
        )
    )

    forecast = np.tile(
        last_values,
        repeats,
    )[:horizon]

    return np.maximum(
        forecast,
        0.0,
    )


def _evaluate_predictions(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:

    metrics = compute_metrics(
        actuals=actual,
        predictions=prediction,
    )

    return metrics


def _chronos_forecast(
    predictor: PredictorService,
    history: pd.DataFrame,
    medicine_id: str,
    config: ForecastConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    result = predictor.forecast_medicine(
        medicine_id,
        history,
    )

    predictions = np.array(
        [
            day.predicted_demand
            for day in result.days
        ],
        dtype=float,
    )

    p10 = np.array(
        [
            day.quantiles.p10
            for day in result.days
        ],
        dtype=float,
    )

    p90 = np.array(
        [
            day.quantiles.p90
            for day in result.days
        ],
        dtype=float,
    )

    expected_length = config.prediction_length

    if len(predictions) != expected_length:
        raise ValueError(
            f"Chronos returned {len(predictions)} "
            f"days instead of {expected_length}."
        )

    return predictions, p10, p90


def run_backtest(
    series_path: str | Path,
    output_dir: str | Path,
    forecast_config: ForecastConfig = DEFAULT_CONFIG,
    backtest_config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:

    series_path = Path(series_path)
    output_dir = Path(output_dir)

    if not series_path.exists():
        raise FileNotFoundError(
            f"Backtest series not found: {series_path}"
        )

    df = pd.read_parquet(series_path)

    df = _validate_series(df)

    if (
        forecast_config.prediction_length
        != backtest_config.horizon
    ):
        raise ValueError(
            "Backtest horizon must match "
            "ForecastConfig.prediction_length."
        )

    logger.info(
        "Loaded backtest dataset: %d rows, %d medicines",
        len(df),
        df["MDCODE"].nunique(),
    )

    # ---------------------------------------------------------------
    # Determine medicines eligible for backtesting
    # ---------------------------------------------------------------

    eligible_medicines = []

    for medicine_id, group in df.groupby(
        "MDCODE",
        sort=True,
    ):

        if len(group) < backtest_config.min_medicine_history:
            continue

        eligible_medicines.append(
            medicine_id
        )

    if not eligible_medicines:
        raise ValueError(
            "No medicines have enough history "
            "for backtesting."
        )

    logger.info(
        "Eligible medicines for backtest: %d",
        len(eligible_medicines),
    )

    # ---------------------------------------------------------------
    # Predictor
    # ---------------------------------------------------------------

    predictor = PredictorService.get_instance(
        forecast_config
    )

    results: List[BacktestResult] = []

    # ---------------------------------------------------------------
    # Rolling backtest
    # ---------------------------------------------------------------

    for medicine_id in eligible_medicines:

        medicine_df = (
            df[df["MDCODE"] == medicine_id]
            .sort_values("INVDT")
            .reset_index(drop=True)
        )

        # Rename into Chronos schema.
        chronos_df = medicine_df.rename(
            columns={
                "MDCODE": "item_id",
                "INVDT": "timestamp",
                "Demand_Qty": "target",
            }
        )

        cutoffs = _make_rolling_cutoffs(
            chronos_df["timestamp"],
            backtest_config,
        )

        if not cutoffs:
            continue

        logger.info(
            "Backtesting medicine %s: %d windows",
            medicine_id,
            len(cutoffs),
        )

        for cutoff in cutoffs:

            history = chronos_df[
                chronos_df["timestamp"] <= cutoff
            ].copy()

            future = chronos_df[
                chronos_df["timestamp"] > cutoff
            ].head(
                backtest_config.horizon
            ).copy()

            if len(future) < backtest_config.horizon:
                continue

            actual = (
                future["target"]
                .to_numpy(dtype=float)
            )

            # -------------------------------------------------------
            # Naive
            # -------------------------------------------------------

            naive_prediction = _naive_forecast(
                history["target"],
                backtest_config.horizon,
            )

            metrics = _evaluate_predictions(
                actual,
                naive_prediction,
            )

            results.append(
                BacktestResult(
                    model="naive",
                    cutoff_date=cutoff,
                    medicine_id=medicine_id,
                    sample_count=len(actual),
                    total_actual=metrics["total_actual"],
                    total_predicted=metrics["total_predicted"],
                    wape_pct=metrics["wape_pct"],
                    mae=metrics["mae"],
                    smape_pct=metrics["smape_pct"],
                    mbe=metrics["mbe"],
                    coverage_pct=0.0,
                )
            )

            # -------------------------------------------------------
            # Seasonal naive
            # -------------------------------------------------------

            seasonal_prediction = (
                _seasonal_naive_forecast(
                    history["target"],
                    backtest_config.horizon,
                    backtest_config.seasonal_period,
                )
            )

            metrics = _evaluate_predictions(
                actual,
                seasonal_prediction,
            )

            results.append(
                BacktestResult(
                    model="seasonal_naive",
                    cutoff_date=cutoff,
                    medicine_id=medicine_id,
                    sample_count=len(actual),
                    total_actual=metrics["total_actual"],
                    total_predicted=metrics["total_predicted"],
                    wape_pct=metrics["wape_pct"],
                    mae=metrics["mae"],
                    smape_pct=metrics["smape_pct"],
                    mbe=metrics["mbe"],
                    coverage_pct=0.0,
                )
            )

            # -------------------------------------------------------
            # Chronos-2
            # -------------------------------------------------------

            try:

                predictions, p10, p90 = (
                    _chronos_forecast(
                        predictor,
                        history,
                        medicine_id,
                        forecast_config,
                    )
                )

                metrics = compute_metrics(
                    actuals=actual,
                    predictions=predictions,
                    p10s=p10,
                    p90s=p90,
                )

                results.append(
                    BacktestResult(
                        model="chronos-2",
                        cutoff_date=cutoff,
                        medicine_id=medicine_id,
                        sample_count=len(actual),
                        total_actual=metrics["total_actual"],
                        total_predicted=metrics["total_predicted"],
                        wape_pct=metrics["wape_pct"],
                        mae=metrics["mae"],
                        smape_pct=metrics["smape_pct"],
                        mbe=metrics["mbe"],
                        coverage_pct=metrics[
                            "coverage_pct"
                        ],
                    )
                )

            except Exception:

                logger.exception(
                    "Chronos backtest failed for "
                    "medicine=%s cutoff=%s",
                    medicine_id,
                    cutoff,
                )

    if not results:
        raise RuntimeError(
            "Backtest produced no evaluation results."
        )

    result_df = pd.DataFrame(
        [
            {
                "Model": r.model,
                "Cutoff_Date": r.cutoff_date,
                "Medicine_ID": r.medicine_id,
                "Sample_Count": r.sample_count,
                "Total_Actual": r.total_actual,
                "Total_Predicted": r.total_predicted,
                "WAPE_Pct": r.wape_pct,
                "WAPE_Pct": r.wape_pct,
                "MAE": r.mae,
                "sMAPE_Pct": r.smape_pct,
                "MBE": r.mbe,
                "P10_P90_Coverage_Pct": r.coverage_pct,
            }
            for r in results
        ]
    )

    # ---------------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "backtest_results.parquet"
    )

    result_df.to_parquet(
        raw_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Aggregate report
    # ---------------------------------------------------------------

    summary = (
        result_df
        .groupby("Model")
        .agg(
            Windows=(
                "Cutoff_Date",
                "nunique",
            ),
            Medicines=(
                "Medicine_ID",
                "nunique",
            ),
            Samples=(
                "Sample_Count",
                "sum",
            ),
            WAPE_Pct=(
                "WAPE_Pct",
                "mean",
            ),
            MAE=(
                "MAE",
                "mean",
            ),
            sMAPE_Pct=(
                "sMAPE_Pct",
                "mean",
            ),
            MBE=(
                "MBE",
                "mean",
            ),
            P10_P90_Coverage_Pct=(
                "P10_P90_Coverage_Pct",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        output_dir
        / "backtest_summary.parquet"
    )

    summary.to_parquet(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------

    print("=" * 80)
    print("FINEMED DEMAND FORECASTING BACKTEST")
    print("=" * 80)

    print(
        f"Medicines evaluated : "
        f"{result_df['Medicine_ID'].nunique()}"
    )

    print(
        f"Backtest windows    : "
        f"{result_df['Cutoff_Date'].nunique()}"
    )

    print(
        f"Results             : "
        f"{len(result_df):,}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(f"Raw results saved : {raw_path}")
    print(f"Summary saved     : {summary_path}")

    return result_df


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

    run_backtest(
        series_path=(
            "data/04_silver/"
            "demand_forecasting/"
            "chronos_series.parquet"
        ),
        output_dir=(
            "data/05_gold/"
            "demand_forecasting/"
            "backtest"
        ),
    )