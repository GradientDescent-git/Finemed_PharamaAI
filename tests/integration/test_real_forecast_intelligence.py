from __future__ import annotations

import math

import pandas as pd
import pytest

from finemed_ai.forecast_intelligence.query_service import (
    ForecastQueryService,
)
from finemed_ai.forecast_intelligence.repository import (
    ForecastRepository,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def repository() -> ForecastRepository:
    return ForecastRepository()


@pytest.fixture(scope="module")
def service(
    repository: ForecastRepository,
) -> ForecastQueryService:
    return ForecastQueryService(
        repository=repository,
    )


@pytest.fixture(scope="module")
def artifacts(
    repository: ForecastRepository,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    forecast, routing, medicines = (
        repository.load_all()
    )

    return (
        forecast,
        routing,
        medicines,
    )


@pytest.fixture(scope="module")
def real_medicine(
    service: ForecastQueryService,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> dict[str, str]:

    forecast, _, _ = artifacts

    medicine_ids = (
        forecast["Medicine_ID"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    assert medicine_ids, (
        "Production forecast artifact contains "
        "no medicine IDs."
    )

    for medicine_id in medicine_ids:

        resolved = service.resolve_medicine(
            medicine_id
        )

        if (
            resolved["resolved"]
            and resolved["medicine_name"]
        ):
            return {
                "medicine_id": str(
                    resolved["medicine_id"]
                ),
                "medicine_name": str(
                    resolved["medicine_name"]
                ),
            }

    pytest.fail(
        "Could not find a forecast medicine that can "
        "be resolved through MedicineResolver."
    )


# ============================================================================
# Artifact loading
# ============================================================================


def test_real_repository_loads_all_artifacts(
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:

    forecast, routing, medicines = artifacts

    assert not forecast.empty
    assert not routing.empty
    assert not medicines.empty


def test_real_forecast_artifact_has_required_data(
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:

    forecast, _, _ = artifacts

    required_columns = {
        "Medicine_ID",
        "Forecast_Date",
        "Predicted_Demand",
        "Selected_Model",
    }

    missing = (
        required_columns
        - set(forecast.columns)
    )

    assert not missing, (
        f"Forecast artifact missing columns: "
        f"{sorted(missing)}"
    )

    assert (
        forecast["Medicine_ID"]
        .notna()
        .all()
    )


# ============================================================================
# Medicine resolution
# ============================================================================


def test_resolve_real_medicine_by_id(
    service: ForecastQueryService,
    real_medicine: dict[str, str],
) -> None:

    result = service.resolve_medicine(
        real_medicine["medicine_id"]
    )

    assert result["resolved"] is True

    assert (
        str(result["medicine_id"])
        == real_medicine["medicine_id"]
    )


def test_resolve_real_medicine_by_name(
    service: ForecastQueryService,
    real_medicine: dict[str, str],
) -> None:

    result = service.resolve_medicine(
        real_medicine["medicine_name"]
    )

    assert result["resolved"] is True

    assert (
        str(result["medicine_id"])
        == real_medicine["medicine_id"]
    )


# ============================================================================
# Real forecast retrieval
# ============================================================================


def test_real_forecast_matches_artifact(
    service: ForecastQueryService,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
    real_medicine: dict[str, str],
) -> None:

    forecast, _, _ = artifacts

    medicine_id = (
        real_medicine["medicine_id"]
    )

    result = service.get_forecast(
        medicine_id
    )

    assert result["found"] is True

    raw = forecast[
        forecast["Medicine_ID"]
        .astype(str)
        == medicine_id
    ].copy()

    raw["Forecast_Date"] = (
        pd.to_datetime(
            raw["Forecast_Date"],
            errors="raise",
        )
    )

    expected_total = float(
        raw["Predicted_Demand"].sum()
    )

    expected_days = int(
        raw["Forecast_Date"]
        .nunique()
    )

    assert result["forecast_days"] == (
        expected_days
    )

    assert math.isclose(
        result["total_predicted_demand"],
        expected_total,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )

    assert result["forecast_start"] == (
        raw["Forecast_Date"]
        .min()
        .date()
        .isoformat()
    )

    assert result["forecast_end"] == (
        raw["Forecast_Date"]
        .max()
        .date()
        .isoformat()
    )


def test_real_forecast_model_matches_artifact(
    service: ForecastQueryService,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
    real_medicine: dict[str, str],
) -> None:

    forecast, _, _ = artifacts

    medicine_id = (
        real_medicine["medicine_id"]
    )

    result = service.get_forecast(
        medicine_id
    )

    raw_models = (
        forecast.loc[
            forecast["Medicine_ID"]
            .astype(str)
            == medicine_id,
            "Selected_Model",
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    assert len(raw_models) == 1

    assert result["selected_model"] == (
        raw_models[0]
    )


# ============================================================================
# Forecast range
# ============================================================================


def test_real_forecast_range_matches_artifact(
    service: ForecastQueryService,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
    real_medicine: dict[str, str],
) -> None:

    forecast, _, _ = artifacts

    medicine_id = (
        real_medicine["medicine_id"]
    )

    result = service.get_forecast_range(
        medicine_id
    )

    assert result["found"] is True

    raw = forecast[
        forecast["Medicine_ID"]
        .astype(str)
        == medicine_id
    ].copy()

    for column, key in (
        ("P10", "total_p10"),
        ("P50", "total_p50"),
        ("P90", "total_p90"),
    ):

        values = (
            pd.to_numeric(
                raw[column],
                errors="coerce",
            )
            .dropna()
        )

        if values.empty:
            assert result[key] is None

        else:
            assert result[key] is not None

            assert math.isclose(
                float(result[key]),
                float(values.sum()),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )


# ============================================================================
# Routing / model information
# ============================================================================


def test_real_model_info_matches_routing_artifact(
    service: ForecastQueryService,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
    real_medicine: dict[str, str],
) -> None:

    _, routing, _ = artifacts

    medicine_id = (
        real_medicine["medicine_id"]
    )

    raw = routing[
        routing["Medicine_ID"]
        .astype(str)
        == medicine_id
    ].copy()

    if raw.empty:
        pytest.skip(
            "Selected medicine has no routing record."
        )

    result = service.get_model_info(
        medicine_id
    )

    assert result["found"] is True

    expected_model = str(
        raw.iloc[0]["Selected_Model"]
    ).strip()

    assert result["selected_model"] == (
        expected_model
    )


# ============================================================================
# Unknown medicine safety
# ============================================================================


def test_unknown_medicine_is_safe(
    service: ForecastQueryService,
) -> None:

    result = service.get_forecast(
        "THIS_MEDICINE_DOES_NOT_EXIST_987654321"
    )

    assert result["found"] is False


# ============================================================================
# Real ranking validation
# ============================================================================


def test_top_demand_ranking_matches_artifact(
    service: ForecastQueryService,
    artifacts: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:

    forecast, _, _ = artifacts

    expected = (
        forecast
        .groupby(
            forecast["Medicine_ID"].astype(str)
        )["Predicted_Demand"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_expected_id = str(
        expected.index[0]
    )

    ranking = service.get_top_demand(
        n=5
    )

    assert ranking

    assert (
        str(ranking[0]["medicine_id"])
        == top_expected_id
    )