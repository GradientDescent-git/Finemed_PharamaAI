from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd

from finemed_ai.automation.alert_engine import (
    AlertEngine,
    AlertStore,
)
from finemed_ai.config.settings import Settings
from finemed_ai.demand_forecasting.config import ForecastConfig
from finemed_ai.demand_forecasting.data_preparation import (
    prepare_demand_data as _prepare_demand_data,
)
from finemed_ai.demand_forecasting.evaluation import (
    ForecastEvaluator,
    OverallEvaluation,
)
from finemed_ai.demand_forecasting.pipeline import (
    run_monthly_forecast,
)
from finemed_ai.pipeline.run_pipeline import (
    run_pipeline,
)
from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)

T = TypeVar("T")


class MonthlyPipelineError(RuntimeError):
    """
    Raised when a critical stage of the monthly production pipeline fails.
    """


class MonthlyPipeline:
    """
    End-to-end monthly production pipeline.

    Execution order:

        1. Run ETL pipeline
        2. Refresh forecasting demand history
        3. Evaluate the previously published forecast
        4. Generate, publish, and validate a new production forecast
        5. Generate operational alerts
        6. Complete downstream application handoff

    Production guarantees:

        - Stage 2 produces the canonical demand artifact used by all
          downstream stages in the current run.
        - Previous forecast evaluation occurs before a new forecast can
          replace the currently published forecast.
        - A forecast is usable only when its publication gate passes.
        - Forecast artifacts are structurally and numerically validated.
        - Duplicate medicine/date forecast records are rejected.
        - Invalid, non-finite, or negative predictions are rejected.
        - Forecast horizon consistency is validated.
        - The live published artifact is verified before downstream use.
        - No downstream stage consumes an unpublished forecast.
        - Every pipeline failure contains stage context.
    """

    REQUIRED_FORECAST_COLUMNS = {
        "Medicine_ID",
        "Forecast_Date",
        "Predicted_Demand",
    }

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.forecast_dir = Path(
            self.settings.FORECAST_DIR
        )

        self.evaluation_dir = Path(
            self.settings.FORECAST_EVALUATION_DIR
        )

        self.alert_dir = Path(
            self.settings.FORECAST_ALERT_DIR
        )

        # Initial configured demand artifact.
        #
        # After Stage 2 succeeds, this is replaced with the exact output
        # artifact generated during the current production run.
        self.demand_file = Path(
            self.settings.DEMAND_FILE
        )

        self.latest_forecast_file = Path(
            self.settings.LATEST_FORECAST_FILE
        )

        self._ensure_directories()

        self.forecast_config = ForecastConfig(
            model_id=self.settings.CHRONOS_MODEL_NAME,
            context_length=self.settings.CONTEXT_LENGTH,
            prediction_length=self.settings.PREDICTION_LENGTH,
            quantile_levels=tuple(
                self.settings.QUANTILES
            ),
        )

    # ================================================================
    # Initialization helpers
    # ================================================================

    def _ensure_directories(
        self,
    ) -> None:
        """
        Ensure all production artifact directories exist.
        """

        for directory in (
            self.forecast_dir,
            self.evaluation_dir,
            self.alert_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _require_file(
        self,
        path: Path,
        *,
        description: str,
    ) -> None:
        """
        Raise a clear production error when a required artifact is absent
        or is not a regular file.
        """

        if not path.exists():
            raise MonthlyPipelineError(
                f"{description} does not exist: {path}"
            )

        if not path.is_file():
            raise MonthlyPipelineError(
                f"{description} is not a regular file: {path}"
            )

    def _read_parquet(
        self,
        path: Path,
        *,
        description: str,
    ) -> pd.DataFrame:
        """
        Validate and read a parquet artifact.
        """

        self._require_file(
            path,
            description=description,
        )

        try:
            dataframe = pd.read_parquet(
                path
            )

        except Exception as exc:
            raise MonthlyPipelineError(
                f"Failed to read {description}: {path}"
            ) from exc

        return dataframe

    # ================================================================
    # Forecast artifact validation
    # ================================================================

    def _validate_forecast_artifact(
        self,
        path: Path,
        *,
        description: str,
        expected_horizon: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform strict production validation of a forecast artifact.

        Validation includes:

        - file existence
        - parquet readability
        - non-empty artifact
        - required columns
        - missing Medicine_ID rejection
        - missing Forecast_Date rejection
        - missing Predicted_Demand rejection
        - valid datetime parsing
        - numeric predictions
        - finite predictions
        - non-negative predictions
        - duplicate (Medicine_ID, Forecast_Date) rejection
        - per-medicine forecast horizon validation
        """

        forecast_df = self._read_parquet(
            path,
            description=description,
        )

        if forecast_df.empty:
            raise MonthlyPipelineError(
                f"{description} contains no rows."
            )

        missing_columns = (
            self.REQUIRED_FORECAST_COLUMNS
            - set(forecast_df.columns)
        )

        if missing_columns:
            raise MonthlyPipelineError(
                f"{description} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # ------------------------------------------------------------
        # Medicine_ID validation
        # ------------------------------------------------------------

        if forecast_df["Medicine_ID"].isna().any():
            raise MonthlyPipelineError(
                f"{description} contains missing Medicine_ID values."
            )

        medicine_ids = (
            forecast_df["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq("").any():
            raise MonthlyPipelineError(
                f"{description} contains empty Medicine_ID values."
            )

        # Normalize in-memory representation used by downstream checks.
        forecast_df = forecast_df.copy()

        forecast_df["Medicine_ID"] = medicine_ids

        # ------------------------------------------------------------
        # Forecast_Date validation
        # ------------------------------------------------------------

        if forecast_df["Forecast_Date"].isna().any():
            raise MonthlyPipelineError(
                f"{description} contains missing Forecast_Date values."
            )

        try:
            forecast_dates = pd.to_datetime(
                forecast_df["Forecast_Date"],
                errors="raise",
            )

        except Exception as exc:
            raise MonthlyPipelineError(
                f"{description} contains invalid Forecast_Date values."
            ) from exc

        if forecast_dates.isna().any():
            raise MonthlyPipelineError(
                f"{description} contains invalid Forecast_Date values."
            )

        # Normalize timestamps to calendar days.
        #
        # Forecasting is daily, so timestamps representing the same
        # calendar date must not be treated as separate forecast keys.
        forecast_df["Forecast_Date"] = (
            forecast_dates.dt.normalize()
        )

        # ------------------------------------------------------------
        # Predicted_Demand validation
        # ------------------------------------------------------------

        if forecast_df["Predicted_Demand"].isna().any():
            raise MonthlyPipelineError(
                f"{description} contains missing "
                "Predicted_Demand values."
            )

        predicted_values = pd.to_numeric(
            forecast_df["Predicted_Demand"],
            errors="coerce",
        )

        if predicted_values.isna().any():
            raise MonthlyPipelineError(
                f"{description} contains non-numeric "
                "Predicted_Demand values."
            )

        predicted_array = (
            predicted_values
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        if not np.isfinite(
            predicted_array
        ).all():
            raise MonthlyPipelineError(
                f"{description} contains non-finite "
                "Predicted_Demand values."
            )

        if (
            predicted_array < 0
        ).any():
            raise MonthlyPipelineError(
                f"{description} contains negative "
                "Predicted_Demand values."
            )

        forecast_df["Predicted_Demand"] = (
            predicted_values.astype(float)
        )

        # ------------------------------------------------------------
        # Duplicate key validation
        # ------------------------------------------------------------

        duplicate_mask = forecast_df.duplicated(
            subset=[
                "Medicine_ID",
                "Forecast_Date",
            ],
            keep=False,
        )

        if duplicate_mask.any():

            duplicate_count = int(
                duplicate_mask.sum()
            )

            raise MonthlyPipelineError(
                f"{description} contains duplicate "
                "(Medicine_ID, Forecast_Date) records. "
                f"Duplicate rows detected: {duplicate_count}."
            )

        # ------------------------------------------------------------
        # Forecast horizon validation
        # ------------------------------------------------------------

        if expected_horizon is not None:

            if expected_horizon <= 0:
                raise MonthlyPipelineError(
                    "Expected forecast horizon must be positive."
                )

            horizon_counts = (
                forecast_df
                .groupby(
                    "Medicine_ID"
                )["Forecast_Date"]
                .nunique()
            )

            invalid_horizons = horizon_counts[
                horizon_counts
                != expected_horizon
            ]

            if not invalid_horizons.empty:

                sample = (
                    invalid_horizons
                    .head(10)
                    .to_dict()
                )

                raise MonthlyPipelineError(
                    f"{description} contains medicines with "
                    "an invalid forecast horizon. "
                    f"Expected {expected_horizon} forecast days "
                    "per medicine. "
                    f"Examples: {sample}"
                )

        logger.info(
            "Forecast artifact validated | "
            "description=%s | "
            "path=%s | "
            "rows=%d | "
            "medicines=%d",
            description,
            path,
            len(forecast_df),
            forecast_df["Medicine_ID"].nunique(),
        )

        return forecast_df

    def _validate_live_publication(
        self,
        manifest: Any,
    ) -> pd.DataFrame:
        """
        Verify that the forecast accepted by the publication gate is
        also available at the configured live production location.

        This prevents downstream stages from assuming that a run-specific
        output artifact is the same artifact currently served by the
        application.
        """

        output_path_value = getattr(
            manifest,
            "output_path",
            None,
        )

        if output_path_value is None:
            raise MonthlyPipelineError(
                "Published forecast manifest does not contain "
                "output_path."
            )

        run_output_path = Path(
            output_path_value
        )

        expected_horizon = (
            self.forecast_config.prediction_length
        )

        # Validate the run-specific artifact first.
        run_forecast_df = (
            self._validate_forecast_artifact(
                run_output_path,
                description=(
                    "Published forecast artifact"
                ),
                expected_horizon=expected_horizon,
            )
        )

        # Validate the live artifact independently.
        live_forecast_df = (
            self._validate_forecast_artifact(
                self.latest_forecast_file,
                description=(
                    "Live published forecast artifact"
                ),
                expected_horizon=expected_horizon,
            )
        )

        # The live artifact should contain the same forecast key set
        # as the successfully published run.
        run_keys = (
            run_forecast_df[
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ]
            ]
            .sort_values(
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        live_keys = (
            live_forecast_df[
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ]
            ]
            .sort_values(
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if not run_keys.equals(
            live_keys
        ):
            raise MonthlyPipelineError(
                "Publication consistency check failed: "
                "the live forecast artifact does not contain "
                "the same forecast key set as the published run."
            )

        logger.info(
            "Live forecast publication verified | "
            "run_output=%s | live_output=%s | "
            "rows=%d",
            run_output_path,
            self.latest_forecast_file,
            len(live_forecast_df),
        )

        return live_forecast_df

    # ================================================================
    # Stage execution helper
    # ================================================================

    def _run_stage(
        self,
        stage_number: int,
        stage_name: str,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a pipeline stage with consistent logging and
        stage-specific failure context.
        """

        logger.info(
            "Stage %d/6 started: %s",
            stage_number,
            stage_name,
        )

        try:

            result = func(
                *args,
                **kwargs,
            )

        except MonthlyPipelineError as exc:

            logger.exception(
                "Stage %d/6 failed: %s",
                stage_number,
                stage_name,
            )

            message = str(exc)

            if message.startswith(
                f"Stage {stage_number}/6"
            ):
                raise

            raise MonthlyPipelineError(
                f"Stage {stage_number}/6 failed "
                f"({stage_name}): {message}"
            ) from exc

        except Exception as exc:

            logger.exception(
                "Stage %d/6 failed: %s",
                stage_number,
                stage_name,
            )

            raise MonthlyPipelineError(
                f"Stage {stage_number}/6 failed "
                f"({stage_name}): {exc}"
            ) from exc

        logger.info(
            "Stage %d/6 completed: %s",
            stage_number,
            stage_name,
        )

        return result

    # ================================================================
    # Stage 1: ETL
    # ================================================================

    def run_etl_pipeline(
        self,
    ) -> None:
        """
        Run the canonical ETL pipeline.
        """

        run_pipeline()

    # ================================================================
    # Stage 2: Demand preparation
    # ================================================================

    def prepare_demand_data(
        self,
    ) -> Path:
        """
        Refresh forecasting demand history.

        Returns
        -------
        Path
            The exact demand artifact generated by this production run.
        """

        output_path = Path(
            _prepare_demand_data()
        )

        self._require_file(
            output_path,
            description=(
                "Prepared demand-history artifact"
            ),
        )

        if output_path.stat().st_size <= 0:
            raise MonthlyPipelineError(
                "Prepared demand-history artifact is empty."
            )

        demand_df = self._read_parquet(
            output_path,
            description=(
                "Prepared demand-history artifact"
            ),
        )

        if demand_df.empty:
            raise MonthlyPipelineError(
                "Prepared demand-history artifact contains no rows."
            )

        logger.info(
            "Demand data prepared | "
            "path=%s | rows=%d",
            output_path,
            len(demand_df),
        )

        return output_path

    # ================================================================
    # Stage 3: Previous forecast evaluation
    # ================================================================

    def evaluate_forecast(
        self,
    ) -> Optional[OverallEvaluation]:
        """
        Evaluate the previously published forecast against newly
        available actual demand.

        This stage executes before a new forecast can replace the
        currently published forecast.
        """

        if not self.latest_forecast_file.exists():

            logger.info(
                "No previous published forecast exists. "
                "Skipping evaluation."
            )

            return None

        actuals_df = self._read_parquet(
            self.demand_file,
            description=(
                "Demand-history artifact"
            ),
        )

        forecast_df = self._read_parquet(
            self.latest_forecast_file,
            description=(
                "Previous production forecast"
            ),
        )

        if actuals_df.empty:
            raise MonthlyPipelineError(
                "Demand-history artifact is empty; "
                "cannot evaluate previous forecast."
            )

        if forecast_df.empty:

            logger.warning(
                "Previous production forecast is empty. "
                "Skipping evaluation."
            )

            return None

        evaluator = ForecastEvaluator(
            self.evaluation_dir
        )

        result = evaluator.evaluate(
            actuals_df,
            forecast_df,
        )

        if (
            result.total_medicines_evaluated
            == 0
        ):

            logger.info(
                "No overlapping actual demand is available "
                "for the previous forecast yet."
            )

        else:

            logger.info(
                "Previous forecast evaluation complete | "
                "medicines=%d | "
                "WAPE=%.2f%% | "
                "SMAPE=%.2f%% | "
                "coverage=%.2f%%",
                result.total_medicines_evaluated,
                result.overall_wape_pct,
                result.overall_smape_pct,
                result.overall_coverage_pct,
            )

        return result

    # ================================================================
    # Stage 4: Production forecast generation
    # ================================================================

    def generate_forecast(
        self,
    ) -> Any:
        """
        Generate, publish, and strictly validate the new production
        forecast.

        The forecast is accepted only when:

            - A manifest is returned.
            - The publication gate passes.
            - output_path exists.
            - The run artifact passes validation.
            - The live published artifact passes validation.
            - The run and live forecast key sets are consistent.
        """

        self._require_file(
            self.demand_file,
            description=(
                "Demand-history artifact"
            ),
        )

        manifest = run_monthly_forecast(
            forecasting_series_path=self.demand_file,
            output_dir=self.forecast_dir,
            config=self.forecast_config,
        )

        if manifest is None:
            raise MonthlyPipelineError(
                "Forecast generation completed without "
                "returning a manifest."
            )

        run_id = getattr(
            manifest,
            "run_id",
            "unknown",
        )

        published = bool(
            getattr(
                manifest,
                "published",
                False,
            )
        )

        publish_note = str(
            getattr(
                manifest,
                "publish_note",
                "",
            )
        )

        logger.info(
            "Forecast run completed | "
            "run_id=%s | "
            "successful=%s/%s | "
            "failed=%s | "
            "published=%s",
            run_id,
            getattr(
                manifest,
                "medicines_succeeded",
                "unknown",
            ),
            getattr(
                manifest,
                "medicines_requested",
                "unknown",
            ),
            getattr(
                manifest,
                "medicines_failed",
                "unknown",
            ),
            published,
        )

        if not published:

            raise MonthlyPipelineError(
                "Forecast publication gate failed. "
                f"Reason: "
                f"{publish_note or 'unknown'}"
            )

        # Strictly validate both the run-specific artifact and the
        # artifact currently exposed to production consumers.
        self._validate_live_publication(
            manifest
        )

        logger.info(
            "Forecast generation and publication "
            "validated successfully | run_id=%s",
            run_id,
        )

        return manifest

    # ================================================================
    # Stage 5: Operational alerts
    # ================================================================

    def run_alerts(
        self,
        manifest: Any,
    ) -> AlertStore:
        """
        Generate operational alerts from the validated live forecast.
        """

        published = bool(
            getattr(
                manifest,
                "published",
                False,
            )
        )

        if not published:
            raise MonthlyPipelineError(
                "Cannot generate alerts from an "
                "unpublished forecast."
            )

        # Use the validated live artifact rather than assuming that
        # the run-specific output is automatically the artifact
        # currently exposed to production consumers.
        forecast_df = (
            self._validate_live_publication(
                manifest
            )
        )

        actuals_df = self._read_parquet(
            self.demand_file,
            description=(
                "Demand-history artifact for alerts"
            ),
        )

        if actuals_df.empty:
            raise MonthlyPipelineError(
                "Cannot generate alerts from empty "
                "historical demand data."
            )

        engine = AlertEngine(
            self.alert_dir
        )

        store = engine.scan_forecasts(
            forecast_df,
            historical_demand_df=actuals_df,
        )

        if store is None:
            raise MonthlyPipelineError(
                "Alert engine completed without "
                "returning an alert store."
            )

        logger.info(
            "Alert scan complete | "
            "total=%d | "
            "critical=%d | "
            "warning=%d",
            store.total_alerts,
            store.critical_count,
            store.warning_count,
        )

        return store

    # ================================================================
    # Stage 6: Application handoff
    # ================================================================

    def refresh_llm(
        self,
    ) -> None:
        """
        Complete the downstream application handoff.

        The API/application layer is responsible for:

            - Reloading ForecastStore.
            - Invalidating stale cached forecast data.
            - Clearing or refreshing stale conversation context.

        This pipeline deliberately does not manipulate API process
        state directly.
        """

        logger.info(
            "Production pipeline completed successfully. "
            "Application layer may now reload ForecastStore "
            "and invalidate stale forecast sessions."
        )

    # ================================================================
    # Full monthly production run
    # ================================================================

    def run(
        self,
    ) -> dict[str, Any]:
        """
        Execute the complete monthly production pipeline.

        Execution:

            Stage 1 -> ETL
            Stage 2 -> Demand preparation
            Stage 3 -> Previous forecast evaluation
            Stage 4 -> Forecast generation and publication validation
            Stage 5 -> Operational alerts
            Stage 6 -> Application handoff

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

                manifest
                evaluation
                alerts

        Raises
        ------
        MonthlyPipelineError
            When a production stage fails.
        """

        logger.info(
            "Starting complete monthly production pipeline."
        )

        # ------------------------------------------------------------
        # Stage 1: ETL
        # ------------------------------------------------------------

        self._run_stage(
            1,
            "ETL pipeline",
            self.run_etl_pipeline,
        )

        # ------------------------------------------------------------
        # Stage 2: Demand preparation
        # ------------------------------------------------------------

        demand_output = self._run_stage(
            2,
            "Demand preparation",
            self.prepare_demand_data,
        )

        # The exact artifact produced in Stage 2 becomes the canonical
        # demand source for this specific production run.
        self.demand_file = demand_output

        logger.info(
            "Canonical forecasting demand source set to: %s",
            self.demand_file,
        )

        # ------------------------------------------------------------
        # Stage 3: Evaluate previous forecast
        # ------------------------------------------------------------

        evaluation = self._run_stage(
            3,
            "Previous forecast evaluation",
            self.evaluate_forecast,
        )

        # ------------------------------------------------------------
        # Stage 4: Generate and validate new forecast
        # ------------------------------------------------------------

        manifest = self._run_stage(
            4,
            "Production forecast generation",
            self.generate_forecast,
        )

        # ------------------------------------------------------------
        # Stage 5: Generate alerts
        # ------------------------------------------------------------

        alerts = self._run_stage(
            5,
            "Operational alert generation",
            self.run_alerts,
            manifest,
        )

        # ------------------------------------------------------------
        # Stage 6: Application handoff
        # ------------------------------------------------------------

        self._run_stage(
            6,
            "Application refresh handoff",
            self.refresh_llm,
        )

        logger.info(
            "Monthly production pipeline completed "
            "successfully | run_id=%s | published=%s",
            getattr(
                manifest,
                "run_id",
                "unknown",
            ),
            getattr(
                manifest,
                "published",
                False,
            ),
        )

        return {
            "manifest": manifest,
            "evaluation": evaluation,
            "alerts": alerts,
        }


def main() -> None:
    """
    CLI entry point for executing the complete monthly pipeline.
    """

    settings = Settings()

    pipeline = MonthlyPipeline(
        settings=settings
    )

    result = pipeline.run()

    manifest = result["manifest"]
    evaluation = result["evaluation"]
    alerts = result["alerts"]

    print("\n" + "=" * 70)
    print(
        "FINEMED PHARMAAI — "
        "MONTHLY PRODUCTION PIPELINE"
    )
    print("=" * 70)

    print(
        f"Run ID: "
        f"{getattr(manifest, 'run_id', 'unknown')}"
    )

    print(
        f"Published: "
        f"{getattr(manifest, 'published', False)}"
    )

    print(
        f"Forecast Output: "
        f"{getattr(manifest, 'output_path', 'unknown')}"
    )

    print(
        f"Live Forecast: "
        f"{pipeline.latest_forecast_file}"
    )

    if evaluation is None:

        print(
            "Previous Forecast Evaluation: "
            "No overlapping actuals available."
        )

    else:

        print(
            f"Previous Forecast WAPE: "
            f"{evaluation.overall_wape_pct:.2f}%"
        )

        print(
            f"Previous Forecast SMAPE: "
            f"{evaluation.overall_smape_pct:.2f}%"
        )

        print(
            f"Medicines Evaluated: "
            f"{evaluation.total_medicines_evaluated}"
        )

    if alerts is None:

        print(
            "Operational Alerts: "
            "No alert result returned."
        )

    else:

        print(
            f"Total Alerts: "
            f"{alerts.total_alerts}"
        )

        print(
            f"Critical Alerts: "
            f"{alerts.critical_count}"
        )

        print(
            f"Warning Alerts: "
            f"{alerts.warning_count}"
        )

    print("=" * 70)
    print(
        "MONTHLY PRODUCTION PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()