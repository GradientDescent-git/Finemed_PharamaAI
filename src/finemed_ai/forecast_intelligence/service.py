"""
Central orchestration service for Finemed Forecast Intelligence.

This module provides the deterministic entry point for employee questions
related to demand forecasting.

The service:
    1. Detects the question intent.
    2. Extracts a medicine query when required.
    3. Dispatches the request to the appropriate deterministic service.
    4. Applies explainability where appropriate.
    5. Returns a standardized structured response.

The future LLM layer should use this service as the trusted forecasting backend.
"""

from __future__ import annotations

import re
from typing import Any

from finemed_ai.forecast_intelligence.intent_router import (
    ForecastIntent,
    IntentRouter,
)
from finemed_ai.forecast_intelligence.query_service import (
    ForecastQueryService,
)
from finemed_ai.forecast_intelligence.routing_explainable import (
    RoutingExplainabilityService,
)


class ForecastIntelligenceService:
    """
    Central deterministic orchestration layer for forecast intelligence.
    """

    def __init__(
        self,
        query_service: ForecastQueryService | None = None,
        intent_router: IntentRouter | None = None,
        routing_explainer: RoutingExplainabilityService | None = None,
    ) -> None:
        self.query_service = query_service or ForecastQueryService()
        self.intent_router = intent_router or IntentRouter()
        self.routing_explainer = (
            routing_explainer or RoutingExplainabilityService()
        )

    def ask(self, question: str) -> dict[str, Any]:
        """
        Process a natural-language forecast intelligence question.
        """

        question = str(question).strip()

        if not question:
            return self._response(
                question="",
                intent=ForecastIntent.UNKNOWN,
                confidence=0.0,
                answer={
                    "found": False,
                    "message": "Please provide a forecast-related question.",
                },
            )

        intent_result = self.intent_router.route(question)

        answer = self._dispatch(
            question=question,
            intent=intent_result.intent,
        )

        return self._response(
            question=question,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            answer=answer,
            matched_patterns=list(intent_result.matched_patterns),
        )

    def _dispatch(
        self,
        question: str,
        intent: ForecastIntent,
    ) -> dict[str, Any]:
        """
        Dispatch the request to the appropriate deterministic service.
        """

        if intent == ForecastIntent.FORECAST:
            medicine_query = self._extract_medicine_query(question)
            return self.query_service.get_forecast(medicine_query)

        if intent == ForecastIntent.FORECAST_RANGE:
            medicine_query = self._extract_medicine_query(question)
            return self.query_service.get_forecast_range(medicine_query)

        if intent == ForecastIntent.MODEL_INFO:
            medicine_query = self._extract_medicine_query(question)
            return self.query_service.get_model_info(medicine_query)

        if intent == ForecastIntent.ROUTING_EXPLANATION:
            medicine_query = self._extract_medicine_query(question)

            routing_data = (
                self.query_service.get_routing_explanation(
                    medicine_query
                )
            )

            if not routing_data.get("found", False):
                return routing_data

            return self.routing_explainer.explain(routing_data)

        if intent == ForecastIntent.RANKING:
            top_n = self._extract_limit(question)
            ascending = self._is_bottom_query(question)

            return self.query_service.get_ranking(
                top_n=top_n,
                ascending=ascending,
            )

        if intent == ForecastIntent.STATUS_QUERY:
            status = self._extract_status(question)

            return self.query_service.get_status(
                status=status
            )

        if intent == ForecastIntent.FORECAST_SUMMARY:
            return self.query_service.get_forecast_summary()

        return {
            "found": False,
            "message": (
                "I could not determine the forecast question type. "
                "Try asking about a medicine forecast, forecast range, "
                "selected model, routing reason, ranking, medicine status, "
                "or the overall forecast summary."
            ),
        }

    @staticmethod
    def _extract_medicine_query(question: str) -> str:
        """
        Extract the likely medicine reference from a natural-language question.

        Examples:
            What is the forecast for Otacare?
                -> Otacare

            Which model predicted Otacare?
                -> Otacare

            Why was TSB selected for Otacare?
                -> Otacare

            Show P90 forecast of PROZON-S INJECTION
                -> PROZON-S INJECTION
        """

        text = question.strip()

        patterns = [
            r"\bforecast\s+(?:for|of)\s+(.+?)(?:\?|$)",
            r"\b(?:for|of)\s+(.+?)(?:\?|$)",
            r"\bpredicted\s+(.+?)(?:\?|$)",
            r"\bselected\s+(?:for|to)\s+(.+?)(?:\?|$)",
            r"\bmodel\s+(?:for|of)\s+(.+?)(?:\?|$)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                candidate = match.group(1).strip()

                candidate = re.sub(
                    r"\b(?:medicine|product|drug)\b",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                if candidate:
                    return candidate

        cleaned = text

        prefixes = [
            r"^what\s+is\s+the\s+",
            r"^what\s+was\s+the\s+",
            r"^show\s+me\s+the\s+",
            r"^show\s+the\s+",
            r"^give\s+me\s+the\s+",
            r"^which\s+model\s+",
            r"^why\s+was\s+",
            r"^why\s+is\s+",
            r"^tell\s+me\s+the\s+",
        ]

        for prefix in prefixes:
            cleaned = re.sub(
                prefix,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        keywords = [
            "forecast",
            "prediction",
            "predicted",
            "model",
            "selected",
            "selection",
            "routing",
            "reason",
            "why",
            "was",
            "is",
            "the",
            "for",
            "of",
            "p10",
            "p50",
            "p90",
            "range",
        ]

        for keyword in keywords:
            cleaned = re.sub(
                rf"\b{re.escape(keyword)}\b",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip(" ?.,:;!-")

        return cleaned

    @staticmethod
    def _extract_limit(question: str) -> int:
        """
        Extract ranking size.

        Examples:
            top 5 medicines -> 5
            bottom 10 medicines -> 10

        Defaults to 10.
        """

        match = re.search(
            r"\b(?:top|bottom|highest|lowest)\s+(\d+)\b",
            question.lower(),
        )

        if match:
            value = int(match.group(1))
            return max(1, min(value, 100))

        return 10

    @staticmethod
    def _is_bottom_query(question: str) -> bool:
        """
        Determine whether ranking should be ascending.
        """

        normalized = question.lower()

        bottom_terms = (
            "bottom",
            "lowest",
            "least",
            "smallest",
        )

        return any(
            term in normalized
            for term in bottom_terms
        )

    @staticmethod
    def _extract_status(question: str) -> str:
        """
        Infer the requested medicine eligibility/status.
        """

        normalized = question.lower()

        if "not forecasted" in normalized:
            return "NOT_FORECASTED"

        if "forecasted stale" in normalized:
            return "FORECASTED_STALE"

        if "dormant" in normalized:
            return "DORMANT"

        if "stale" in normalized:
            return "STALE"

        if "active" in normalized:
            return "ACTIVE"

        return "DORMANT"

    @staticmethod
    def _response(
        question: str,
        intent: ForecastIntent,
        confidence: float,
        answer: dict[str, Any],
        matched_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Standardize all responses.
        """

        return {
            "question": question,
            "intent": intent.value,
            "confidence": confidence,
            "matched_patterns": matched_patterns or [],
            "answer": answer,
        }