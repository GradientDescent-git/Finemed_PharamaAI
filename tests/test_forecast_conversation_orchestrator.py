from __future__ import annotations

from typing import Any

import pytest

from finemed_ai.forecast_intelligence.action_router import (
    ForecastAction,
)
from finemed_ai.forecast_intelligence.conversation_context import (
    ForecastConversationContext,
)
from finemed_ai.forecast_intelligence.conversation_orchestrator import (
    ForecastConversationOrchestrator,
)


# ======================================================================
# Test doubles
# ======================================================================


class FakeActionRouter:
    """
    Deterministic test router.

    Maps exact questions to ForecastAction objects so orchestrator tests
    remain focused on orchestration rather than intent-classification
    behavior.
    """

    def __init__(
        self,
        actions: dict[str, ForecastAction],
    ) -> None:
        self.actions = actions
        self.calls: list[dict[str, Any]] = []

    def route(
        self,
        question: str,
        conversation_context: str | None = None,
    ) -> ForecastAction:
        self.calls.append(
            {
                "question": question,
                "conversation_context": conversation_context,
            }
        )

        return self.actions.get(
            question,
            ForecastAction(
                action="insufficient_information",
                confidence=1.0,
                source="test",
            ),
        )


class FakeQueryService:
    """
    Deterministic forecasting data source for orchestrator tests.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _medicine_result(
        self,
        query: str,
    ) -> dict[str, Any]:
        normalized = query.strip().lower()

        if normalized in {
            "otacare",
            "ota001",
        }:
            return {
                "found": True,
                "query": query,
                "medicine_id": "OTA001",
                "medicine_name": "Otacare",
            }

        return {
            "found": False,
            "reason": "medicine_not_resolved",
            "query": query,
        }

    def get_forecast(
        self,
        query: str,
    ) -> dict[str, Any]:
        self.calls.append(("get_forecast", query))

        result = self._medicine_result(query)

        if not result["found"]:
            return result

        return {
            **result,
            "forecast_start": "2026-05-01",
            "forecast_end": "2026-05-31",
            "forecast_days": 31,
            "total_predicted_demand": 310.0,
            "average_daily_demand": 10.0,
            "selected_model": "tsb",
            "eligibility_status": "ACTIVE",
            "forecast_status": "FORECASTED",
        }

    def get_forecast_range(
        self,
        query: str,
    ) -> dict[str, Any]:
        self.calls.append(("get_forecast_range", query))

        result = self._medicine_result(query)

        if not result["found"]:
            return result

        return {
            **result,
            "forecast_start": "2026-05-01",
            "forecast_end": "2026-05-31",
            "p10_available": True,
            "p50_available": True,
            "p90_available": True,
            "total_p10": 250.0,
            "total_p50": 310.0,
            "total_p90": 380.0,
            "selected_model": "tsb",
        }

    def get_model_info(
        self,
        query: str,
    ) -> dict[str, Any]:
        self.calls.append(("get_model_info", query))

        result = self._medicine_result(query)

        if not result["found"]:
            return result

        return {
            **result,
            "selected_model": "tsb",
            "validation_windows": 4,
            "chronos_absolute_error": 100.0,
            "tsb_absolute_error": 80.0,
            "validation_advantage_pct": 20.0,
            "routing_reason": (
                "chronos_validation_advantage_below_30pct_threshold"
            ),
        }

    def get_routing_explanation(
        self,
        query: str,
    ) -> dict[str, Any]:
        self.calls.append(
            ("get_routing_explanation", query)
        )

        result = self._medicine_result(query)

        if not result["found"]:
            return result

        return {
            **result,
            "selected_model": "tsb",
            "validation_windows": 4,
            "chronos_absolute_error": 100.0,
            "tsb_absolute_error": 80.0,
            "validation_advantage_pct": 20.0,
            "routing_reason": (
                "chronos_validation_advantage_below_30pct_threshold"
            ),
        }

    def get_ranking(
        self,
        top_n: int,
        ascending: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "get_ranking",
                {
                    "top_n": top_n,
                    "ascending": ascending,
                },
            )
        )

        results = [
            {
                "medicine_id": "OTA001",
                "medicine_name": "Otacare",
                "total_predicted_demand": 310.0,
            },
            {
                "medicine_id": "ABC001",
                "medicine_name": "ExampleMed",
                "total_predicted_demand": 250.0,
            },
        ]

        return {
            "found": True,
            "ranking_type": (
                "lowest_predicted_demand"
                if ascending
                else "highest_predicted_demand"
            ),
            "results": results[:top_n],
        }

    def get_status(
        self,
        status: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            ("get_status", status)
        )

        return {
            "found": True,
            "requested_status": status,
            "medicine_count": 2,
            "record_count": 2,
            "eligibility_counts": {
                "ACTIVE": 2,
            },
            "forecast_status_counts": {
                "FORECASTED": 2,
            },
        }

    def get_forecast_summary(
        self,
    ) -> dict[str, Any]:
        self.calls.append(
            ("get_forecast_summary", None)
        )

        return {
            "found": True,
            "medicine_count": 2,
            "forecast_days": 31,
            "forecast_start": "2026-05-01",
            "forecast_end": "2026-05-31",
            "total_predicted_demand": 560.0,
            "average_daily_predicted_demand": (
                18.064516
            ),
            "model_distribution": {
                "tsb": 2,
            },
            "eligibility_distribution": {
                "ACTIVE": 2,
            },
        }


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def query_service() -> FakeQueryService:
    return FakeQueryService()


@pytest.fixture
def context() -> ForecastConversationContext:
    return ForecastConversationContext()


def build_orchestrator(
    actions: dict[str, ForecastAction],
    query_service: FakeQueryService,
    context: ForecastConversationContext | None = None,
) -> ForecastConversationOrchestrator:

    return ForecastConversationOrchestrator(
        action_router=FakeActionRouter(actions),
        query_service=query_service,
        context=context,
    )


# ======================================================================
# Direct medicine queries
# ======================================================================


def test_direct_forecast_query_resolves_medicine_and_stores_context(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "What is the forecast for Otacare?": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    response = orchestrator.ask(
        "What is the forecast for Otacare?"
    )

    assert response["action"] == "forecast"
    assert response["resolved_medicine"] == "Otacare"

    assert response["data"]["found"] is True
    assert response["data"]["medicine_id"] == "OTA001"

    assert context.has_medicine_context() is True
    assert context.medicine_id == "OTA001"
    assert context.medicine_name == "Otacare"


# ======================================================================
# Follow-up conversation
# ======================================================================


def test_follow_up_model_question_uses_previous_medicine_context(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Forecast Otacare": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
            "Which model predicted it?": ForecastAction(
                action="model_info",
                medicine_query=None,
                needs_context=True,
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    orchestrator.ask("Forecast Otacare")

    response = orchestrator.ask(
        "Which model predicted it?"
    )

    assert response["action"] == "model_info"
    assert response["data"]["found"] is True
    assert response["data"]["medicine_name"] == "Otacare"

    assert (
        "Otacare was forecast using tsb."
        in response["answer"]
    )

    assert (
        "get_model_info",
        "Otacare",
    ) in query_service.calls


def test_follow_up_routing_explanation_uses_context(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Forecast Otacare": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
            "Why was that model selected?": ForecastAction(
                action="routing_explanation",
                medicine_query=None,
                needs_context=True,
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    orchestrator.ask("Forecast Otacare")

    response = orchestrator.ask(
        "Why was that model selected?"
    )

    assert response["action"] == "routing_explanation"
    assert response["data"]["found"] is True

    assert "TSB was selected" in response["answer"]

    assert (
        "get_routing_explanation",
        "Otacare",
    ) in query_service.calls


def test_follow_up_forecast_range_uses_context(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Forecast Otacare": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
            "What about the P90?": ForecastAction(
                action="forecast_range",
                medicine_query=None,
                needs_context=True,
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    orchestrator.ask("Forecast Otacare")

    response = orchestrator.ask(
        "What about the P90?"
    )

    assert response["action"] == "forecast_range"
    assert response["data"]["found"] is True
    assert response["data"]["medicine_name"] == "Otacare"

    assert "P90: 380.00" in response["answer"]

    assert (
        "get_forecast_range",
        "Otacare",
    ) in query_service.calls


# ======================================================================
# Context preservation
# ======================================================================


def test_greeting_does_not_destroy_medicine_context(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Forecast Otacare": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
            "Hello": ForecastAction(
                action="greeting",
                confidence=1.0,
                source="test",
            ),
            "What about the P90?": ForecastAction(
                action="forecast_range",
                medicine_query=None,
                needs_context=True,
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    orchestrator.ask("Forecast Otacare")
    orchestrator.ask("Hello")

    response = orchestrator.ask(
        "What about the P90?"
    )

    assert response["data"]["found"] is True
    assert response["data"]["medicine_name"] == "Otacare"

    assert context.medicine_name == "Otacare"


def test_failed_medicine_lookup_does_not_destroy_existing_context(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Forecast Otacare": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
            "Forecast UnknownMedicineXYZ": ForecastAction(
                action="forecast",
                medicine_query="UnknownMedicineXYZ",
                confidence=0.95,
                source="test",
            ),
            "Which model predicted it?": ForecastAction(
                action="model_info",
                medicine_query=None,
                needs_context=True,
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    first_response = orchestrator.ask(
        "Forecast Otacare"
    )

    assert first_response["data"]["found"] is True
    assert context.medicine_name == "Otacare"

    failed_response = orchestrator.ask(
        "Forecast UnknownMedicineXYZ"
    )

    assert failed_response["data"]["found"] is False

    assert context.medicine_name == "Otacare"

    follow_up_response = orchestrator.ask(
        "Which model predicted it?"
    )

    assert follow_up_response["data"]["found"] is True

    assert (
        follow_up_response["data"]["medicine_name"]
        == "Otacare"
    )


# ======================================================================
# Missing context
# ======================================================================


def test_follow_up_without_previous_medicine_returns_safe_response(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Which model predicted it?": ForecastAction(
                action="model_info",
                medicine_query=None,
                needs_context=True,
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask(
        "Which model predicted it?"
    )

    assert response["data"]["found"] is False

    assert (
        response["data"]["reason"]
        == "medicine_not_specified"
    )

    assert "medicine name or medicine code" in (
        response["answer"].lower()
    )


# ======================================================================
# Context clear
# ======================================================================


def test_clear_context_removes_previous_medicine(
    query_service: FakeQueryService,
    context: ForecastConversationContext,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Forecast Otacare": ForecastAction(
                action="forecast",
                medicine_query="Otacare",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
        context,
    )

    orchestrator.ask("Forecast Otacare")

    assert context.has_medicine_context() is True

    orchestrator.clear_context()

    assert context.has_medicine_context() is False
    assert context.medicine_id is None
    assert context.medicine_name is None
    assert context.last_action is None


# ======================================================================
# Ranking
# ======================================================================


def test_highest_ranking_dispatches_descending(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Top 10 medicines": ForecastAction(
                action="ranking",
                ranking_limit=10,
                ranking_direction="highest",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask(
        "Top 10 medicines"
    )

    assert response["data"]["found"] is True

    assert (
        "Medicines with the highest predicted demand:"
        in response["answer"]
    )

    assert (
        "get_ranking",
        {
            "top_n": 10,
            "ascending": False,
        },
    ) in query_service.calls


def test_lowest_ranking_dispatches_ascending(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Bottom 5 medicines": ForecastAction(
                action="ranking",
                ranking_limit=5,
                ranking_direction="lowest",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask(
        "Bottom 5 medicines"
    )

    assert response["data"]["found"] is True

    assert (
        "Medicines with the lowest predicted demand:"
        in response["answer"]
    )

    assert (
        "get_ranking",
        {
            "top_n": 5,
            "ascending": True,
        },
    ) in query_service.calls


def test_ranking_defaults_to_ten(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Show highest demand medicines": ForecastAction(
                action="ranking",
                ranking_limit=None,
                ranking_direction="highest",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
    )

    orchestrator.ask(
        "Show highest demand medicines"
    )

    assert (
        "get_ranking",
        {
            "top_n": 10,
            "ascending": False,
        },
    ) in query_service.calls


# ======================================================================
# Status queries
# ======================================================================


def test_status_query_dispatches_requested_status(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Show dormant medicines": ForecastAction(
                action="status_query",
                status="DORMANT",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask(
        "Show dormant medicines"
    )

    assert response["data"]["found"] is True

    assert (
        "get_status",
        "DORMANT",
    ) in query_service.calls


# ======================================================================
# Forecast summary
# ======================================================================


def test_forecast_summary_dispatches_correct_service(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Give me the forecast summary": ForecastAction(
                action="forecast_summary",
                confidence=0.95,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask(
        "Give me the forecast summary"
    )

    assert response["data"]["found"] is True

    assert (
        "get_forecast_summary",
        None,
    ) in query_service.calls

    assert "covers 2 medicines" in (
        response["answer"]
    )


# ======================================================================
# Greeting, help, and out-of-scope
# ======================================================================


def test_greeting_returns_grounded_response(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Hello": ForecastAction(
                action="greeting",
                confidence=1.0,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask("Hello")

    assert response["action"] == "greeting"

    assert response["data"]["found"] is True

    assert "Demand Forecasting Assistant" in (
        response["answer"]
    )


def test_help_returns_supported_capabilities(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "Help": ForecastAction(
                action="help",
                confidence=1.0,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask("Help")

    assert response["action"] == "help"

    assert "forecast for Otacare" in (
        response["answer"]
    )


def test_out_of_scope_question_returns_safe_response(
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {
            "What is the weather today?": ForecastAction(
                action="out_of_scope",
                confidence=1.0,
                source="test",
            ),
        },
        query_service,
    )

    response = orchestrator.ask(
        "What is the weather today?"
    )

    assert response["action"] == "out_of_scope"

    assert response["data"]["found"] is False

    assert "demand forecasting" in (
        response["answer"].lower()
    )


# ======================================================================
# Empty input
# ======================================================================


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_empty_question_returns_safe_response(
    question: str,
    query_service: FakeQueryService,
) -> None:

    orchestrator = build_orchestrator(
        {},
        query_service,
    )

    response = orchestrator.ask(question)

    assert (
        response["action"]
        == "insufficient_information"
    )

    assert response["data"]["found"] is False