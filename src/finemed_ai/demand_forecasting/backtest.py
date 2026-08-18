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

    # Primary evaluation cutoffs.
    evaluation_cutoffs: tuple[str, ...] = (
        "2025-11-30",
        "2025-12-31",
        "2026-01-31",
        "2026-02-28",
        "2026-03-31",
        "2026-04-30",
    )

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
    total_absolute_error: float
    absolute_error_sum: float
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

def _make_fixed_cutoffs(
    medicine_dates: pd.Series,
    config: BacktestConfig,
) -> List[pd.Timestamp]:

    dates = (
        pd.to_datetime(medicine_dates)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if dates.empty:
        return []

    configured_cutoffs = [
        pd.Timestamp(value)
        for value in config.evaluation_cutoffs
    ]

    valid_cutoffs = []

    for cutoff in configured_cutoffs:

        history_count = (
            dates <= cutoff
        ).sum()

        future_count = (
            dates > cutoff
        ).sum()

        if history_count < config.min_history:
            continue

        if future_count < config.horizon:
            continue

        valid_cutoffs.append(cutoff)

    return valid_cutoffs


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

def _croston_forecast(
    history: pd.Series,
    horizon: int,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Croston's method for intermittent demand.

    Separately estimates:
    - demand size when demand occurs
    - interval between non-zero demands
    """

    y = history.to_numpy(dtype=float)

    if len(y) == 0:
        return np.zeros(horizon)

    y = np.maximum(y, 0.0)

    non_zero = np.flatnonzero(y > 0)

    if len(non_zero) == 0:
        return np.zeros(horizon)

    first = non_zero[0]

    demand_estimate = float(y[first])
    interval_estimate = float(first + 1)

    previous = first

    for index in non_zero[1:]:
        demand = float(y[index])
        interval = float(index - previous)

        demand_estimate += alpha * (
            demand - demand_estimate
        )

        interval_estimate += alpha * (
            interval - interval_estimate
        )

        previous = index

    if interval_estimate <= 0:
        return np.zeros(horizon)

    forecast_value = (
        demand_estimate / interval_estimate
    )

    return np.full(
        horizon,
        max(forecast_value, 0.0),
        dtype=float,
    )


def _sba_forecast(
    history: pd.Series,
    horizon: int,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Syntetos-Boylan Approximation (SBA).

    SBA applies a bias correction to Croston's forecast.
    """

    croston = _croston_forecast(
        history,
        horizon,
        alpha,
    )

    correction = 1.0 - (alpha / 2.0)

    return np.maximum(
        croston * correction,
        0.0,
    )


def _tsb_forecast(
    history: pd.Series,
    horizon: int,
    alpha_demand: float = 0.1,
    alpha_probability: float = 0.1,
) -> np.ndarray:
    """
    Teunter-Syntetos-Babai (TSB) method.

    Estimates:
    - probability of demand occurrence
    - size of demand when it occurs
    """

    y = history.to_numpy(dtype=float)

    if len(y) == 0:
        return np.zeros(horizon)

    y = np.maximum(y, 0.0)

    non_zero = np.flatnonzero(y > 0)

    if len(non_zero) == 0:
        return np.zeros(horizon)

    first = non_zero[0]

    demand_estimate = float(y[first])

    probability = (
        1.0 / float(first + 1)
        if first >= 0
        else 1.0
    )

    for index, demand in enumerate(y):

        occurrence = 1.0 if demand > 0 else 0.0

        probability += alpha_probability * (
            occurrence - probability
        )

        if occurrence > 0:
            demand_estimate += alpha_demand * (
                float(demand) - demand_estimate
            )

    forecast_value = (
        probability * demand_estimate
    )

    return np.full(
        horizon,
        max(forecast_value, 0.0),
        dtype=float,
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
) -> Dict[str, np.ndarray]:

    result = predictor.forecast_medicine(
        medicine_id,
        history,
    )

    quantiles = {}

    for level in (
        "p10", "p20", "p30", "p40", "p50",
        "p60", "p70", "p80", "p90",
    ):
        quantiles[level] = np.array(
            [
                getattr(day.quantiles, level)
                for day in result.days
            ],
            dtype=float,
        )

    expected_length = config.prediction_length

    for level, values in quantiles.items():
        if len(values) != expected_length:
            raise ValueError(
                f"Chronos {level} returned {len(values)} "
                f"days instead of {expected_length}."
            )

    return quantiles


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

        cutoffs = _make_fixed_cutoffs(
            chronos_df["timestamp"],
            backtest_config)

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
                    total_absolute_error=metrics["absolute_error_sum"],
                    absolute_error_sum=metrics["absolute_error_sum"],
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
                    total_absolute_error=metrics["absolute_error_sum"],
                    absolute_error_sum=metrics["absolute_error_sum"],
                    wape_pct=metrics["wape_pct"],
                    mae=metrics["mae"],
                    smape_pct=metrics["smape_pct"],
                    mbe=metrics["mbe"],
                    coverage_pct=0.0,
                    )
            )

            # -------------------------------------------------------
            # Croston
            # # -------------------------------------------------------
            croston_prediction = _croston_forecast(
                history["target"],
                backtest_config.horizon)

            metrics = _evaluate_predictions(
                actual,
                croston_prediction)

            results.append(
                BacktestResult(
                    model="croston",
                    cutoff_date=cutoff,
                    medicine_id=medicine_id,
                    sample_count=len(actual),
                    total_actual=metrics["total_actual"],
                    total_predicted=metrics["total_predicted"],
                    total_absolute_error=metrics["absolute_error_sum"],
                    absolute_error_sum=metrics["absolute_error_sum"],
                    wape_pct=metrics["wape_pct"],
                    mae=metrics["mae"],
                    smape_pct=metrics["smape_pct"],
                    mbe=metrics["mbe"],
                    coverage_pct=0.0))

            # -------------------------------------------------------
            # SBA
            # -------------------------------------------------------
            sba_prediction = _sba_forecast(
                history["target"],
                backtest_config.horizon)

            metrics = _evaluate_predictions(
                actual,
                sba_prediction)

            results.append(
                BacktestResult(
                    model="sba",
                    cutoff_date=cutoff,
                    medicine_id=medicine_id,
                    sample_count=len(actual),
                    total_actual=metrics["total_actual"],
                    total_predicted=metrics["total_predicted"],
                    total_absolute_error=metrics["absolute_error_sum"],
                    absolute_error_sum=metrics["absolute_error_sum"],
                    wape_pct=metrics["wape_pct"],
                    mae=metrics["mae"],
                    smape_pct=metrics["smape_pct"],
                    mbe=metrics["mbe"],
                    coverage_pct=0.0))

            # -------------------------------------------------------
            # TSB
            # -------------------------------------------------------
            tsb_prediction = _tsb_forecast(
                history["target"],
                backtest_config.horizon)

            metrics = _evaluate_predictions(
                actual,
                tsb_prediction)

            results.append(
                BacktestResult(
                    model="tsb",
                    cutoff_date=cutoff,
                    medicine_id=medicine_id,
                    sample_count=len(actual),
                    total_actual=metrics["total_actual"],
                    total_predicted=metrics["total_predicted"],
                    total_absolute_error=metrics["absolute_error_sum"],
                    absolute_error_sum=metrics["absolute_error_sum"],
                    wape_pct=metrics["wape_pct"],
                    mae=metrics["mae"],
                    smape_pct=metrics["smape_pct"],
                    mbe=metrics["mbe"],
                    coverage_pct=0.0))

            # -------------------------------------------------------
            # Chronos-2
            # -------------------------------------------------------
            try:
                q = _chronos_forecast(
                    predictor,
                    history,
                    medicine_id,
                    forecast_config)

                # Evaluate every quantile as a possible point forecast.
                for level in (
                    "p10",
                    "p20",
                    "p30",
                    "p40",
                    "p50",
                    "p60",
                    "p70",
                    "p80",
                    "p90"):

                    predictions = q[level]
                    metrics = compute_metrics(
                        actuals=actual,
                        predictions=predictions,
                        p10s=q["p10"],
                        p90s=q["p90"])

                    results.append(
                        BacktestResult(
                            model=f"chronos-2-{level.upper()}",
                            cutoff_date=cutoff,
                            medicine_id=medicine_id,
                            sample_count=len(actual),
                            total_actual=metrics["total_actual"],
                            total_predicted=metrics["total_predicted"],
                            total_absolute_error=metrics["absolute_error_sum"],
                            absolute_error_sum=metrics["absolute_error_sum"],
                            wape_pct=metrics["wape_pct"],
                            mae=metrics["mae"],
                            smape_pct=metrics["smape_pct"],
                            mbe=metrics["mbe"],
                            coverage_pct=metrics["coverage_pct"]))
            except Exception:
                logger.exception(
                    "Chronos backtest failed for "
                    "medicine=%s cutoff=%s",
                    medicine_id,
                    cutoff)
                

    # ---------------------------------------------------------------
    # Aggregate report
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Build raw results DataFrame
    # ---------------------------------------------------------------

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
                "Total_Absolute_Error": r.absolute_error_sum,
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
            Windows=("Cutoff_Date", "size"),
            Medicines=("Medicine_ID", "nunique"),
            Samples=("Sample_Count", "sum"),
            Total_Actual=("Total_Actual", "sum"),
            Total_Predicted=("Total_Predicted", "sum"),
            Total_Absolute_Error=("Total_Absolute_Error", "sum"),
            MAE=("MAE", "mean"),
            sMAPE_Pct=("sMAPE_Pct", "mean"),
            MBE=("MBE", "mean"),
            P10_P90_Coverage_Pct=(
                "P10_P90_Coverage_Pct",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["WAPE_Pct"] = (
        summary["Total_Absolute_Error"]
        / summary["Total_Actual"]
        * 100.0
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
