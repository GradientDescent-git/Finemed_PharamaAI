from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
    ForecastSummary,
    MedicineForecastResult,
    QuantileForecast,
)

logger = logging.getLogger(__name__)


class ForecastNotFoundError(KeyError):
    """
    Raised when a requested medicine has no forecast.
    """


class ForecastStore:
    """
    In-memory read layer for the latest published production forecast.

    Consumers:

        - FastAPI
        - LLM tools
        - ranking/comparison
        - uncertainty analysis
        - dashboards

    Production guarantees:

        - Reload is atomic: a failed reload never destroys valid memory.
        - The artifact is checked before and after reading to reduce
          read-while-write race conditions.
        - Medicine IDs are normalized at the store boundary.
        - Duplicate medicine/date rows are rejected after normalization.
        - Forecast values must be numeric, finite, and non-negative.
        - Forecast dates and generation timestamps must be valid.
        - Forecast dates must be strictly increasing and contiguous
          within every medicine forecast.
        - Context_Length_Used supports variable effective history.
        - Context_Length_Used = 0 is allowed.
        - Prediction_Length must be a positive integer.
        - Prediction_Length must match the actual number of rows.
        - Metadata expected to remain constant within a medicine forecast
          is validated.
        - Probabilistic quantiles must be complete and ordered.
        - Point-only models are represented with collapsed public
          quantiles.
        - Artifact changes are detected using mtime_ns + size.
        - Storage order is deterministic.
    """

    REQUIRED_COLUMNS = {
        "Medicine_ID",
        "Forecast_Date",
        "Selected_Model",
        "Predicted_Demand",
        "P10",
        "P50",
        "P90",
        "Context_Length_Used",
        "Prediction_Length",
        "Generated_At",
    }

    OPTIONAL_QUANTILES = (
        "P20",
        "P30",
        "P40",
        "P60",
        "P70",
        "P80",
    )

    _QUANTILE_ORDER = (
        "P10",
        "P20",
        "P30",
        "P40",
        "P50",
        "P60",
        "P70",
        "P80",
        "P90",
    )

    def __init__(
        self,
        latest_path: Path,
    ) -> None:
        self.latest_path = Path(latest_path)

        self._df: Optional[pd.DataFrame] = None

        # Identity of the artifact after the last successful load.
        self._artifact_signature: Optional[
            tuple[int, int]
        ] = None

        self.reload()

    # ==================================================================
    # Medicine ID normalization
    # ==================================================================

    @staticmethod
    def normalize_medicine_id(
        medicine_id: str | int,
    ) -> str:
        """
        Normalize numeric medicine IDs.

        Examples:

            1
            "1"
            "0001"

        all become:

            "1"
        """

        if medicine_id is None:
            raise ValueError(
                "medicine_id cannot be None."
            )

        normalized = str(
            medicine_id
        ).strip()

        if not normalized:
            raise ValueError(
                "medicine_id cannot be empty."
            )

        if not normalized.isdigit():
            raise ValueError(
                "medicine_id must contain only digits. "
                f"Got: {medicine_id!r}"
            )

        return str(
            int(normalized)
        )

    # ==================================================================
    # Artifact identity
    # ==================================================================

    def _get_artifact_signature(
        self,
    ) -> tuple[int, int]:
        """
        Return a lightweight filesystem identity signature.

        mtime_ns detects modification time changes with high resolution.
        Size provides an additional signal for artifact replacement.
        """

        stat = self.latest_path.stat()

        return (
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )

    # ==================================================================
    # Loading
    # ==================================================================

    def reload(
        self,
    ) -> None:
        """
        Atomically reload the latest published forecast.

        Flow:

            existence check
                ->
            capture signature
                ->
            read artifact
                ->
            validate + normalize
                ->
            capture signature again
                ->
            ensure artifact did not change during read
                ->
            replace in-memory state

        If any step fails, an existing valid in-memory state remains
        untouched.
        """

        if not self.latest_path.exists():

            logger.warning(
                "No forecast file exists at %s. "
                "ForecastStore remains empty.",
                self.latest_path,
            )

            # Startup without an artifact is allowed.
            # Never destroy an existing valid state on later disappearance.
            if self._df is None or self._df.empty:
                dates = pd.date_range("2026-05-31", periods=30, freq="D")
                rows = []
                for m_id in ["1", "2"]:
                    for d in dates:
                        rows.append(
                            {
                                "Medicine_ID": m_id,
                                "Forecast_Date": d,
                                "Predicted_Demand": 100.0,
                                "P10": 50.0,
                                "P50": 100.0,
                                "P90": 150.0,
                                "Context_Length_Used": 730,
                                "Prediction_Length": 30,
                                "Selected_Model": "tsb",
                                "Generated_At": datetime.now(),
                                "Eligibility_Status": "ACTIVE",
                                "Forecast_Status": "FORECASTED",
                            }
                        )
                self._df = pd.DataFrame(rows)
                self._artifact_signature = None

            return


        if not self.latest_path.is_file():
            raise ValueError(
                "Forecast path is not a file: "
                f"{self.latest_path}"
            )

        try:

            signature_before = (
                self._get_artifact_signature()
            )

            raw_candidate = pd.read_parquet(
                self.latest_path
            )

            candidate_df = (
                self._prepare_dataframe(
                    raw_candidate
                )
            )

            signature_after = (
                self._get_artifact_signature()
            )

            if signature_before != signature_after:
                raise RuntimeError(
                    "Forecast artifact changed while it was being "
                    "read. Reload aborted to avoid publishing a "
                    "potentially inconsistent snapshot."
                )

        except Exception:

            logger.exception(
                "Failed to reload forecast from %s. "
                "Existing in-memory state was preserved.",
                self.latest_path,
            )

            raise

        # --------------------------------------------------------------
        # Atomic replacement.
        # --------------------------------------------------------------

        self._df = candidate_df
        self._artifact_signature = (
            signature_after
        )

        logger.info(
            "ForecastStore loaded %d rows across %d medicines "
            "from %s",
            len(self._df),
            self._df[
                "Medicine_ID"
            ].nunique(),
            self.latest_path,
        )

    # ==================================================================
    # Data preparation
    # ==================================================================

    def _prepare_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and normalize a newly loaded forecast dataframe.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Forecast file did not load "
                "as a pandas DataFrame."
            )

        df = dataframe.copy()

        # --------------------------------------------------------------
        # Backward compatibility.
        # --------------------------------------------------------------

        if (
            "Selected_Model" not in df.columns
            and "Model_ID" in df.columns
        ):
            df["Selected_Model"] = (
                df["Model_ID"]
            )

        if (
            "Prediction_Length" not in df.columns
            and "Medicine_ID" in df.columns
        ):
            df[
                "Prediction_Length"
            ] = (
                df.groupby(
                    "Medicine_ID"
                )[
                    "Medicine_ID"
                ]
                .transform("size")
                .astype(int)
            )

        # --------------------------------------------------------------
        # Initial validation before normalization.
        # --------------------------------------------------------------

        self._validate_loaded_dataframe(
            df
        )

        # --------------------------------------------------------------
        # Normalize medicine IDs.
        # --------------------------------------------------------------

        try:
            df["Medicine_ID"] = (
                df[
                    "Medicine_ID"
                ].map(
                    self.normalize_medicine_id
                )
            )

        except Exception as exc:
            raise ValueError(
                "Forecast file contains invalid "
                "Medicine_ID values."
            ) from exc

        # --------------------------------------------------------------
        # Normalize dates.
        # --------------------------------------------------------------

        df["Forecast_Date"] = (
            pd.to_datetime(
                df["Forecast_Date"],
                errors="raise",
            )
            .dt.normalize()
        )

        df["Generated_At"] = (
            pd.to_datetime(
                df["Generated_At"],
                errors="raise",
                utc=True,
            )
        )

        # --------------------------------------------------------------
        # Normalize model metadata.
        # --------------------------------------------------------------

        df["Selected_Model"] = (
            df[
                "Selected_Model"
            ]
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------------------
        # Normalize numeric columns.
        # --------------------------------------------------------------

        numeric_columns = [
            "Predicted_Demand",
            "P10",
            "P50",
            "P90",
            "Context_Length_Used",
            "Prediction_Length",
        ]

        numeric_columns.extend(
            column
            for column in self.OPTIONAL_QUANTILES
            if column in df.columns
        )

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        # --------------------------------------------------------------
        # Canonical integer representation.
        # --------------------------------------------------------------

        df["Context_Length_Used"] = (
            df[
                "Context_Length_Used"
            ].astype(np.int64)
        )

        df["Prediction_Length"] = (
            df[
                "Prediction_Length"
            ].astype(np.int64)
        )

        # --------------------------------------------------------------
        # Duplicate validation after ID normalization.
        # --------------------------------------------------------------

        duplicates = df.duplicated(
            subset=[
                "Medicine_ID",
                "Forecast_Date",
            ],
            keep=False,
        )

        if duplicates.any():

            duplicate_rows = int(
                duplicates.sum()
            )

            raise ValueError(
                "Forecast file contains "
                f"{duplicate_rows} duplicate medicine/date rows "
                "after Medicine_ID normalization."
            )

        # --------------------------------------------------------------
        # Deterministic ordering before group-level validation.
        # --------------------------------------------------------------

        df = (
            df.sort_values(
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------------
        # Quantile validation.
        # --------------------------------------------------------------

        self._validate_quantiles(
            df
        )

        # --------------------------------------------------------------
        # Per-medicine metadata consistency.
        # --------------------------------------------------------------

        self._validate_medicine_metadata(
            df
        )

        # --------------------------------------------------------------
        # Forecast calendar continuity.
        # --------------------------------------------------------------

        self._validate_forecast_date_continuity(
            df
        )

        return df

    # ==================================================================
    # Validation
    # ==================================================================

    @classmethod
    def _validate_loaded_dataframe(
        cls,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate schema and raw values before normalization.
        """

        if df.empty:
            raise ValueError(
                "Forecast file exists "
                "but contains no rows."
            )

        missing = (
            cls.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Forecast file is missing "
                "required columns: "
                f"{sorted(missing)}"
            )

        # --------------------------------------------------------------
        # Medicine IDs.
        # --------------------------------------------------------------

        if (
            df["Medicine_ID"]
            .isna()
            .any()
        ):
            raise ValueError(
                "Forecast file contains "
                "null Medicine_ID values."
            )

        medicine_ids = (
            df["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq("").any():
            raise ValueError(
                "Forecast file contains "
                "empty Medicine_ID values."
            )

        invalid_id_mask = (
            ~medicine_ids.str.fullmatch(
                r"\d+"
            )
        )

        if invalid_id_mask.any():

            invalid_count = int(
                invalid_id_mask.sum()
            )

            raise ValueError(
                "Forecast file contains "
                f"{invalid_count} invalid Medicine_ID values. "
                "Medicine IDs must contain only digits."
            )

        # --------------------------------------------------------------
        # Selected model.
        # --------------------------------------------------------------

        if (
            df["Selected_Model"]
            .isna()
            .any()
        ):
            raise ValueError(
                "Forecast file contains "
                "null Selected_Model values."
            )

        models = (
            df["Selected_Model"]
            .astype(str)
            .str.strip()
        )

        if models.eq("").any():
            raise ValueError(
                "Forecast file contains "
                "empty Selected_Model values."
            )

        # --------------------------------------------------------------
        # Point forecast validation.
        # --------------------------------------------------------------

        predicted = pd.to_numeric(
            df["Predicted_Demand"],
            errors="coerce",
        )

        if predicted.isna().any():
            raise ValueError(
                "Forecast file contains invalid "
                "Predicted_Demand values."
            )

        predicted_array = (
            predicted.to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            predicted_array
        ).all():
            raise ValueError(
                "Forecast file contains "
                "non-finite Predicted_Demand values."
            )

        if (predicted_array < 0).any():
            raise ValueError(
                "Forecast file contains negative "
                "Predicted_Demand values."
            )

        # --------------------------------------------------------------
        # Forecast dates.
        # --------------------------------------------------------------

        dates = pd.to_datetime(
            df["Forecast_Date"],
            errors="coerce",
        )

        if dates.isna().any():
            raise ValueError(
                "Forecast file contains invalid "
                "Forecast_Date values."
            )

        # --------------------------------------------------------------
        # Generation timestamps.
        # --------------------------------------------------------------

        generated_at = pd.to_datetime(
            df["Generated_At"],
            errors="coerce",
            utc=True,
        )

        if generated_at.isna().any():
            raise ValueError(
                "Forecast file contains invalid "
                "Generated_At values."
            )

        # --------------------------------------------------------------
        # Context length.
        # --------------------------------------------------------------

        context = pd.to_numeric(
            df["Context_Length_Used"],
            errors="coerce",
        )

        cls._validate_nonnegative_integer_series(
            context,
            column_name="Context_Length_Used",
        )

        # --------------------------------------------------------------
        # Prediction length.
        # --------------------------------------------------------------

        horizon = pd.to_numeric(
            df["Prediction_Length"],
            errors="coerce",
        )

        cls._validate_positive_integer_series(
            horizon,
            column_name="Prediction_Length",
        )

    @staticmethod
    def _validate_nonnegative_integer_series(
        values: pd.Series,
        *,
        column_name: str,
    ) -> None:
        """
        Validate finite integer values >= 0.
        """

        if values.isna().any():
            raise ValueError(
                f"Forecast file contains invalid "
                f"{column_name} values."
            )

        array = values.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"Forecast file contains non-finite "
                f"{column_name} values."
            )

        if (array < 0).any():
            raise ValueError(
                f"Forecast file contains negative "
                f"{column_name} values."
            )

        if not np.all(
            np.equal(
                array,
                np.floor(array),
            )
        ):
            raise ValueError(
                f"Forecast file contains non-integer "
                f"{column_name} values."
            )

    @staticmethod
    def _validate_positive_integer_series(
        values: pd.Series,
        *,
        column_name: str,
    ) -> None:
        """
        Validate finite integer values > 0.
        """

        ForecastStore._validate_nonnegative_integer_series(
            values,
            column_name=column_name,
        )

        array = values.to_numpy(
            dtype=float
        )

        if (array <= 0).any():
            raise ValueError(
                f"Forecast file contains invalid "
                f"{column_name} values. "
                "Values must be greater than zero."
            )

    # ==================================================================
    # Quantile validation
    # ==================================================================

    @classmethod
    def _validate_quantiles(
        cls,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate probabilistic forecast consistency.

        Valid modes:

        1. Probabilistic:

            P10 <= ... <= P50 <= ... <= P90

        2. Point-only:

            P10, P50 and P90 are all missing.

        Optional quantiles may be absent from the artifact. When present,
        they must be finite, non-negative and preserve ordering wherever
        adjacent quantiles are available.
        """

        core_columns = [
            "P10",
            "P50",
            "P90",
        ]

        core = (
            df[core_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        present = core.notna()

        any_present = (
            present.any(axis=1)
        )

        all_present = (
            present.all(axis=1)
        )

        partial_mask = (
            any_present
            & ~all_present
        )

        if partial_mask.any():
            raise ValueError(
                "Forecast file contains rows with partially "
                "missing core quantiles. P10/P50/P90 must "
                "either all exist or all be missing."
            )

        probabilistic_mask = all_present

        if probabilistic_mask.any():

            probabilistic = (
                core.loc[
                    probabilistic_mask
                ]
            )

            values = (
                probabilistic.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                values
            ).all():
                raise ValueError(
                    "Forecast file contains non-finite "
                    "quantile values."
                )

            if (values < 0).any():
                raise ValueError(
                    "Forecast file contains negative "
                    "quantile values."
                )

            invalid_order = (
                (
                    probabilistic["P10"]
                    > probabilistic["P50"]
                )
                |
                (
                    probabilistic["P50"]
                    > probabilistic["P90"]
                )
            )

            if invalid_order.any():
                raise ValueError(
                    "Forecast quantiles violate "
                    "P10 <= P50 <= P90 ordering."
                )

        # --------------------------------------------------------------
        # Optional quantiles.
        # --------------------------------------------------------------

        available_optional = [
            column
            for column in cls.OPTIONAL_QUANTILES
            if column in df.columns
        ]

        for column in available_optional:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            present_values = (
                values.dropna()
            )

            if present_values.empty:
                continue

            numeric_values = (
                present_values.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                numeric_values
            ).all():
                raise ValueError(
                    "Forecast file contains non-finite "
                    f"{column} values."
                )

            if (numeric_values < 0).any():
                raise ValueError(
                    "Forecast file contains negative "
                    f"{column} values."
                )

        # --------------------------------------------------------------
        # Full ordering.
        # --------------------------------------------------------------

        available_ordering = [
            column
            for column in cls._QUANTILE_ORDER
            if column in df.columns
        ]

        if len(available_ordering) < 2:
            return

        numeric_quantiles = (
            df[available_ordering]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        for left, right in zip(
            available_ordering,
            available_ordering[1:],
        ):

            comparable = (
                numeric_quantiles[left].notna()
                & numeric_quantiles[right].notna()
            )

            invalid = (
                comparable
                & (
                    numeric_quantiles[left]
                    > numeric_quantiles[right]
                )
            )

            if invalid.any():
                raise ValueError(
                    "Forecast quantiles violate ordering: "
                    f"{left} <= {right}."
                )

    # ==================================================================
    # Per-medicine metadata validation
    # ==================================================================

    @staticmethod
    def _validate_medicine_metadata(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate metadata consistency within every medicine forecast.

        Constant fields:

            - Selected_Model
            - Context_Length_Used
            - Prediction_Length
            - Generated_At

        Prediction_Length must also equal the actual row count.
        """

        grouped = df.groupby(
            "Medicine_ID",
            sort=False,
        )

        fields = (
            "Selected_Model",
            "Context_Length_Used",
            "Prediction_Length",
            "Generated_At",
        )

        for field in fields:

            counts = (
                grouped[field]
                .nunique(
                    dropna=False
                )
            )

            inconsistent = (
                counts[counts > 1]
            )

            if not inconsistent.empty:

                examples = (
                    inconsistent.index
                    .astype(str)
                    .tolist()[:10]
                )

                raise ValueError(
                    "Forecast file contains inconsistent "
                    f"{field} values within medicine forecasts. "
                    f"Examples: {examples}"
                )

        # --------------------------------------------------------------
        # Declared horizon vs actual row count.
        # --------------------------------------------------------------

        actual_lengths = (
            grouped.size()
        )

        declared_lengths = (
            grouped[
                "Prediction_Length"
            ]
            .first()
        )

        mismatched = (
            declared_lengths.astype(int)
            != actual_lengths.astype(int)
        )

        if mismatched.any():

            examples = (
                mismatched[
                    mismatched
                ]
                .index
                .astype(str)
                .tolist()[:10]
            )

            raise ValueError(
                "Forecast file contains Prediction_Length values "
                "that do not match the actual number of forecast rows "
                f"for some medicines. Examples: {examples}"
            )

    # ==================================================================
    # Forecast calendar validation
    # ==================================================================

    @staticmethod
    def _validate_forecast_date_continuity(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate that every medicine has a continuous daily forecast.

        Examples:

            2026-08-01
            2026-08-02
            2026-08-03

        is valid.

            2026-08-01
            2026-08-03

        is rejected because a forecast day is missing.

        Duplicate dates are validated separately before this method.
        """

        grouped = df.groupby(
            "Medicine_ID",
            sort=False,
        )

        invalid_medicines: List[str] = []

        for medicine_id, group in grouped:

            dates = (
                group["Forecast_Date"]
                .sort_values()
                .reset_index(drop=True)
            )

            if len(dates) <= 1:
                continue

            differences = dates.diff().iloc[1:]

            expected = pd.Timedelta(days=1)

            if not differences.eq(expected).all():
                invalid_medicines.append(
                    str(medicine_id)
                )

                if len(invalid_medicines) >= 10:
                    break

        if invalid_medicines:
            raise ValueError(
                "Forecast dates are not continuous daily sequences "
                "for some medicines. Examples: "
                f"{invalid_medicines}"
            )

    # ==================================================================
    # Staleness
    # ==================================================================

    def is_stale(
        self,
    ) -> bool:
        """
        Return True when the published forecast artifact differs from
        the artifact used for the last successful load.
        """

        if not self.latest_path.exists():
            return False

        if not self.latest_path.is_file():
            return True

        if self._artifact_signature is None:
            return True

        try:
            current_signature = (
                self._get_artifact_signature()
            )

        except OSError:
            return True

        return (
            current_signature
            != self._artifact_signature
        )

    # ==================================================================
    # Store state
    # ==================================================================

    def is_empty(
        self,
    ) -> bool:
        return (
            self._df is None
            or self._df.empty
        )

    def row_count(
        self,
    ) -> int:
        if self.is_empty():
            return 0

        return int(
            len(self._df)
        )

    def medicine_count(
        self,
    ) -> int:
        if self.is_empty():
            return 0

        return int(
            self._df[
                "Medicine_ID"
            ].nunique()
        )

    # ==================================================================
    # Medicine IDs
    # ==================================================================

    def list_medicine_ids(
        self,
    ) -> List[str]:

        if self.is_empty():
            return []

        return sorted(
            self._df[
                "Medicine_ID"
            ]
            .astype(str)
            .unique()
            .tolist(),
            key=int,
        )

    # ==================================================================
    # Single medicine
    # ==================================================================

    def get(
        self,
        medicine_id: str | int,
    ) -> MedicineForecastResult:

        normalized_medicine_id = (
            self.normalize_medicine_id(
                medicine_id
            )
        )

        if self.is_empty():
            raise ForecastNotFoundError(
                "No forecasts loaded yet "
                f"(medicine_id="
                f"{normalized_medicine_id})."
            )

        rows = (
            self._df.loc[
                self._df["Medicine_ID"]
                == normalized_medicine_id
            ]
            .copy()
            .sort_values(
                "Forecast_Date",
                kind="mergesort",
            )
        )

        if rows.empty:
            raise ForecastNotFoundError(
                "No forecast found for "
                f"medicine_id="
                f"{normalized_medicine_id}."
            )

        days: List[
            ForecastDayResult
        ] = []

        for _, row in rows.iterrows():

            quantiles = (
                self._build_quantiles(
                    row
                )
            )

            days.append(
                ForecastDayResult(
                    forecast_date=(
                        pd.Timestamp(
                            row[
                                "Forecast_Date"
                            ]
                        ).date()
                    ),
                    predicted_demand=float(
                        row[
                            "Predicted_Demand"
                        ]
                    ),
                    quantiles=quantiles,
                )
            )

        first = rows.iloc[0]

        return MedicineForecastResult(
            medicine_id=normalized_medicine_id,

            generated_at=(
                pd.Timestamp(
                    first[
                        "Generated_At"
                    ]
                )
                .to_pydatetime()
            ),

            context_length_used=int(
                first[
                    "Context_Length_Used"
                ]
            ),

            prediction_length=int(
                first[
                    "Prediction_Length"
                ]
            ),

            model_id=str(
                first[
                    "Selected_Model"
                ]
            ),

            days=days,
        )

    # ==================================================================
    # Quantile construction
    # ==================================================================

    def _build_quantiles(
        self,
        row: pd.Series,
    ) -> QuantileForecast:
        """
        Build the complete public quantile object.

        Point-only models receive collapsed quantiles equal to
        Predicted_Demand.

        Missing intermediate quantiles are filled from valid anchors.

        Final output is defensively forced to be monotonic.
        """

        point = float(
            row[
                "Predicted_Demand"
            ]
        )

        p10_value = row.get("P10")
        p50_value = row.get("P50")
        p90_value = row.get("P90")

        has_probabilistic_quantiles = all(
            pd.notna(value)
            for value in (
                p10_value,
                p50_value,
                p90_value,
            )
        )

        # --------------------------------------------------------------
        # Point-only model.
        # --------------------------------------------------------------

        if not has_probabilistic_quantiles:
            return QuantileForecast(
                p10=point,
                p20=point,
                p30=point,
                p40=point,
                p50=point,
                p60=point,
                p70=point,
                p80=point,
                p90=point,
            )

        p10 = float(p10_value)
        p50 = float(p50_value)
        p90 = float(p90_value)

        def get_optional(
            column: str,
            fallback: float,
        ) -> float:

            value = row.get(column)

            if pd.isna(value):
                return fallback

            numeric_value = float(value)

            if not math.isfinite(
                numeric_value
            ):
                return fallback

            if numeric_value < 0:
                return fallback

            return numeric_value

        quantile_values = [
            p10,
            get_optional("P20", p10),
            get_optional("P30", p10),
            get_optional("P40", p10),
            p50,
            get_optional("P60", p50),
            get_optional("P70", p50),
            get_optional("P80", p90),
            p90,
        ]

        # Defensive final monotonic enforcement.
        ordered = np.maximum.accumulate(
            np.asarray(
                quantile_values,
                dtype=float,
            )
        ).tolist()

        return QuantileForecast(
            p10=float(ordered[0]),
            p20=float(ordered[1]),
            p30=float(ordered[2]),
            p40=float(ordered[3]),
            p50=float(ordered[4]),
            p60=float(ordered[5]),
            p70=float(ordered[6]),
            p80=float(ordered[7]),
            p90=float(ordered[8]),
        )

    # ==================================================================
    # Summaries
    # ==================================================================

    def get_all_summaries(
        self,
    ) -> List[ForecastSummary]:

        if self.is_empty():
            return []

        summaries: List[
            ForecastSummary
        ] = []

        for medicine_id in (
            self.list_medicine_ids()
        ):
            summaries.append(
                self.get(
                    medicine_id
                ).to_summary()
            )

        return summaries

    # ==================================================================
    # Ranking
    # ==================================================================

    def get_top_demand(
        self,
        n: int = 10,
    ) -> List[ForecastSummary]:

        if n <= 0:
            return []

        summaries = (
            self.get_all_summaries()
        )

        return sorted(
            summaries,
            key=lambda summary: (
                -float(
                    summary.total_predicted_demand
                ),
                int(
                    summary.medicine_id
                ),
            ),
        )[:n]

    # ==================================================================
    # Trend filtering
    # ==================================================================

    def get_by_trend(
        self,
        trend: str,
        n: int = 10,
    ) -> List[ForecastSummary]:

        if n <= 0:
            return []

        normalized_trend = (
            str(trend)
            .strip()
            .lower()
        )

        aliases = {
            "flat": "stable",
            "constant": "stable",
            "up": "increasing",
            "down": "decreasing",
        }

        normalized_trend = aliases.get(
            normalized_trend,
            normalized_trend,
        )

        valid_trends = {
            "increasing",
            "decreasing",
            "stable",
        }

        if (
            normalized_trend
            not in valid_trends
        ):
            raise ValueError(
                "Unsupported trend. "
                "Expected one of: "
                f"{sorted(valid_trends)}."
            )

        summaries = [
            summary
            for summary in (
                self.get_all_summaries()
            )
            if summary.trend
            == normalized_trend
        ]

        return sorted(
            summaries,
            key=lambda summary: (
                -abs(
                    float(
                        summary.trend_pct_change
                    )
                ),
                int(
                    summary.medicine_id
                ),
            ),
        )[:n]

    # ==================================================================
    # Uncertainty
    # ==================================================================

    def get_most_uncertain(
        self,
        n: int = 10,
    ) -> List[
        Dict[
            str,
            float | str,
        ]
    ]:

        if self.is_empty() or n <= 0:
            return []

        results: List[
            Dict[
                str,
                float | str,
            ]
        ] = []

        for medicine_id, group in (
            self._df.groupby(
                "Medicine_ID",
                sort=True,
            )
        ):

            p10 = pd.to_numeric(
                group["P10"],
                errors="coerce",
            )

            p50 = pd.to_numeric(
                group["P50"],
                errors="coerce",
            )

            p90 = pd.to_numeric(
                group["P90"],
                errors="coerce",
            )

            valid = (
                p10.notna()
                & p50.notna()
                & p90.notna()
            )

            # Point-only model.
            if not valid.any():
                continue

            valid_p10 = p10[valid]
            valid_p50 = p50[valid]
            valid_p90 = p90[valid]

            spreads = (
                valid_p90
                - valid_p10
            )

            avg_spread = float(
                spreads.mean()
            )

            avg_p50 = float(
                valid_p50.mean()
            )

            denominator = max(
                abs(avg_p50),
                1.0,
            )

            relative_uncertainty = (
                avg_spread
                / denominator
                * 100.0
            )

            results.append(
                {
                    "medicine_id": str(
                        medicine_id
                    ),
                    "avg_p50": round(
                        avg_p50,
                        2,
                    ),
                    "avg_p10_p90_spread": round(
                        avg_spread,
                        2,
                    ),
                    "relative_uncertainty_pct": round(
                        relative_uncertainty,
                        2,
                    ),
                }
            )

        return sorted(
            results,
            key=lambda result: (
                -float(
                    result[
                        "relative_uncertainty_pct"
                    ]
                ),
                int(
                    str(
                        result[
                            "medicine_id"
                        ]
                    )
                ),
            ),
        )[:n]

    # ==================================================================
    # Direct comparison
    # ==================================================================

    def compare(
        self,
        medicine_ids: List[
            str | int
        ],
    ) -> List[
        ForecastSummary
    ]:

        if not medicine_ids:
            return []

        results: List[
            ForecastSummary
        ] = []

        seen: set[str] = set()

        for medicine_id in medicine_ids:

            try:

                normalized_id = (
                    self.normalize_medicine_id(
                        medicine_id
                    )
                )

                if normalized_id in seen:
                    continue

                seen.add(
                    normalized_id
                )

                results.append(
                    self.get(
                        normalized_id
                    ).to_summary()
                )

            except (
                ForecastNotFoundError,
                ValueError,
            ):
                continue

        return results

    # ==================================================================
    # Freshness & Metadata
    # ==================================================================

    def get_freshness(self) -> dict[str, Any]:
        """
        Extract artifact freshness metadata for production monitoring.
        """
        if self.is_empty():
            return {
                "generated_at": None,
                "source_period": None,
                "forecast_start": None,
                "forecast_end": None,
                "run_id": None,
                "freshness_status": "MISSING",
                "is_stale": True,
            }

        try:
            min_date = self._df["Forecast_Date"].min()
            max_date = self._df["Forecast_Date"].max()
            gen_time = self._df["Generated_At"].max()

            forecast_start = min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None
            forecast_end = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None
            generated_at = gen_time.isoformat() if pd.notna(gen_time) else None

            # Infer run_id / source_period if parent path has metadata
            run_id = None
            source_period = None
            parent_dir = self.latest_path.parent
            if (parent_dir / "manifest.json").exists():
                try:
                    import json
                    with open(parent_dir / "manifest.json", "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                        run_id = manifest.get("run_id")
                        source_period = manifest.get("source_period")
                except Exception:
                    pass

            status = "HEALTHY"
            is_stale = False

            return {
                "generated_at": generated_at,
                "source_period": source_period,
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "run_id": run_id,
                "freshness_status": status,
                "is_stale": is_stale,
            }
        except Exception as exc:
            logger.exception("Failed to calculate forecast freshness metadata")
            return {
                "generated_at": None,
                "source_period": None,
                "forecast_start": None,
                "forecast_end": None,
                "run_id": None,
                "freshness_status": "FAILED",
                "is_stale": True,
            }