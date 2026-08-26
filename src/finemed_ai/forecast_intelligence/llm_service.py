"""
LLM orchestration layer for production demand forecasting intelligence.

The deterministic ForecastIntelligenceService is the authoritative source
of forecasting facts. This module is responsible only for converting verified
structured results into clear employee-friendly natural language.

The LLM must never invent, modify, calculate, or infer forecast values that
are not present in the deterministic result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from finemed_ai.forecast_intelligence.service import (
    ForecastIntelligenceService,
)

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """
    Protocol for an LLM provider.

    This keeps the orchestration layer independent from a specific provider.
    OpenAI, Gemini, Anthropic, or a local model can implement this interface.
    """

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from the configured LLM provider."""


@dataclass(frozen=True)
class LLMAnswer:
    """
    Final response produced by the LLM orchestration layer.

    Attributes
    ----------
    question:
        Original employee question.

    answer:
        Employee-friendly natural-language answer.

    deterministic_result:
        Verified structured result produced by ForecastIntelligenceService.

    llm_used:
        Whether an LLM was successfully used to generate the final answer.

    fallback_used:
        Whether the deterministic fallback formatter was used.

    """

    question: str
    answer: str
    deterministic_result: dict[str, Any]
    llm_used: bool
    fallback_used: bool


class ForecastLLMService:
    """
    Grounded LLM orchestration service.

    Flow
    ----
    Employee Question
        ↓
    ForecastIntelligenceService
        ↓
    Verified structured facts
        ↓
    LLM grounding prompt
        ↓
    Response validation
        ↓
    Employee-friendly answer

    The deterministic result remains the source of truth.
    """

    SYSTEM_PROMPT = """
You are Finemed Forecast Intelligence, an assistant for employees asking
questions about pharmaceutical demand forecasts.

Your job is to explain ONLY the verified forecasting information supplied
to you.

STRICT RULES:

1. Treat the supplied VERIFIED DATA as the only source of forecasting facts.
2. Never invent a forecast number.
3. Never modify a forecast number.
4. Never calculate or estimate a new forecast value unless that exact value
   is already present in VERIFIED DATA.
5. Never invent P10, P50, or P90 values.
6. If uncertainty values are unavailable, explicitly say they are unavailable.
7. Never invent a model-selection or routing reason.
8. Never claim a medicine was resolved if VERIFIED DATA says otherwise.
9. Clearly distinguish ACTIVE, STALE, and DORMANT statuses.
10. If the requested information is unavailable, explain that clearly.
11. Do not mention internal implementation details unless the verified data
    explicitly contains them and they are relevant to the employee's question.
12. Do not add assumptions, external knowledge, medical advice, or business
    recommendations that are not supported by VERIFIED DATA.
13. Answer concisely, clearly, and professionally.
14. Preserve the meaning and numerical precision of VERIFIED DATA.
15. If the deterministic result indicates an error or unresolved query,
    explain the limitation instead of guessing.

The deterministic forecasting system is authoritative.
You are only responsible for communicating its verified results.
""".strip()

    def __init__(
        self,
        intelligence_service: ForecastIntelligenceService | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the service.

        Parameters
        ----------
        intelligence_service:
            Deterministic forecasting intelligence service.

        llm_client:
            Optional provider implementing the LLMClient protocol.

            If None, the service remains fully functional using the
            deterministic fallback formatter.
        """

        self.intelligence_service = (
            intelligence_service
            or ForecastIntelligenceService()
        )

        self.llm_client = llm_client

    def ask(
        self,
        question: str,
    ) -> LLMAnswer:
        """
        Answer an employee forecasting question.

        The deterministic intelligence layer is always executed first.

        If an LLM provider is available, verified facts are passed to the
        provider with strict grounding instructions.

        If the provider is unavailable or fails, the system safely falls back
        to deterministic formatting.
        """

        self._validate_question(question)

        deterministic_result = self.intelligence_service.ask(question)

        fallback_answer = self._format_deterministic_answer(
            deterministic_result
        )

        if self.llm_client is None:
            return LLMAnswer(
                question=question,
                answer=fallback_answer,
                deterministic_result=deterministic_result,
                llm_used=False,
                fallback_used=True,
            )

        try:
            llm_answer = self._generate_grounded_answer(
                question=question,
                deterministic_result=deterministic_result,
            )

            validated_answer = self._validate_llm_answer(
                llm_answer=llm_answer,
                fallback_answer=fallback_answer,
            )

            return LLMAnswer(
                question=question,
                answer=validated_answer,
                deterministic_result=deterministic_result,
                llm_used=True,
                fallback_used=False,
            )

        except Exception:
            logger.exception(
                "LLM generation failed. Falling back to deterministic answer."
            )

            return LLMAnswer(
                question=question,
                answer=fallback_answer,
                deterministic_result=deterministic_result,
                llm_used=False,
                fallback_used=True,
            )

    def _generate_grounded_answer(
        self,
        *,
        question: str,
        deterministic_result: dict[str, Any],
    ) -> str:
        """Generate an answer using only verified deterministic facts."""

        verified_data = json.dumps(
            deterministic_result,
            ensure_ascii=False,
            default=str,
            indent=2,
        )

        user_prompt = f"""
EMPLOYEE QUESTION:

{question}

VERIFIED DATA:

{verified_data}

Write the employee-facing answer using only VERIFIED DATA.
Do not introduce facts that are absent from the data.
""".strip()

        response = self.llm_client.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if not isinstance(response, str):
            raise TypeError(
                "LLM provider must return a string response."
            )

        response = response.strip()

        if not response:
            raise ValueError(
                "LLM provider returned an empty response."
            )

        return response

    @staticmethod
    def _validate_llm_answer(
        *,
        llm_answer: str,
        fallback_answer: str,
    ) -> str:
        """
        Perform basic response validation.

        The deterministic fallback remains available if the generated
        response is empty or structurally invalid.
        """

        if not isinstance(llm_answer, str):
            return fallback_answer

        cleaned = llm_answer.strip()

        if not cleaned:
            return fallback_answer

        return cleaned

    @staticmethod
    def _validate_question(
        question: str,
    ) -> None:
        """Validate employee input."""

        if not isinstance(question, str):
            raise TypeError(
                "question must be a string"
            )

        if not question.strip():
            raise ValueError(
                "question must not be empty"
            )

    def _format_deterministic_answer(
        self,
        result: dict[str, Any],
    ) -> str:
        """
        Convert deterministic results into a safe fallback answer.

        This fallback ensures that the product continues to function even
        when the external LLM provider is unavailable.
        """

        answer = result.get("answer", {})

        if not isinstance(answer, dict):
            return (
                "I could not retrieve a valid forecasting result "
                "for this question."
            )

        if result.get("intent") == "unknown":
            return answer.get(
                "message",
                (
                    "I could not determine the type of forecasting "
                    "question."
                ),
            )

        if not answer.get("found", True):
            return self._format_not_found(answer)

        intent = result.get("intent")

        if intent == "forecast":
            return self._format_forecast(answer)

        if intent == "forecast_range":
            return self._format_forecast_range(answer)

        if intent == "model_info":
            return self._format_model_info(answer)

        if intent == "routing_explanation":
            return self._format_routing_explanation(answer)

        if intent == "ranking":
            return self._format_ranking(answer)

        if intent == "status_query":
            return self._format_status(answer)

        if intent == "forecast_summary":
            return self._format_summary(answer)

        return (
            "The forecasting system returned verified data, but this "
            "question type does not yet have a dedicated response formatter."
        )

    @staticmethod
    def _format_not_found(
        answer: dict[str, Any],
    ) -> str:
        """Format unresolved or unavailable medicine results."""

        reason = answer.get("reason")

        if reason == "medicine_not_resolved":
            return (
                "I could not identify the medicine from your question. "
                "Please provide the medicine name or medicine code."
            )

        if reason == "no_forecast_records_available":
            return (
                "The medicine was identified, but no forecast records "
                "are currently available."
            )

        if reason == "no_routing_record_available":
            return (
                "The medicine was identified, but no model-routing "
                "information is currently available."
            )

        return (
            "The requested forecasting information is currently "
            "unavailable."
        )

    @staticmethod
    def _format_forecast(
        answer: dict[str, Any],
    ) -> str:
        """Format a medicine forecast answer."""

        return (
            f"Forecast for {answer['medicine_name']} "
            f"from {answer['forecast_start']} to "
            f"{answer['forecast_end']}: "
            f"{answer['total_predicted_demand']:.2f} total predicted demand "
            f"over {answer['forecast_days']} days "
            f"({answer['average_daily_demand']:.2f} average per day). "
            f"The selected model is {answer['selected_model']}. "
            f"Eligibility status: {answer['eligibility_status']}. "
            f"Forecast status: {answer['forecast_status']}."
        )

    @staticmethod
    def _format_forecast_range(
        answer: dict[str, Any],
    ) -> str:
        """Format forecast uncertainty information."""

        medicine_name = answer["medicine_name"]

        available_parts = []

        if answer.get("p10_available"):
            available_parts.append(
                f"P10 total: {answer['total_p10']:.2f}"
            )

        if answer.get("p50_available"):
            available_parts.append(
                f"P50 total: {answer['total_p50']:.2f}"
            )

        if answer.get("p90_available"):
            available_parts.append(
                f"P90 total: {answer['total_p90']:.2f}"
            )

        if not available_parts:
            return (
                f"No P10, P50, or P90 forecast range values are available "
                f"for {medicine_name}. "
                f"The selected model is {answer['selected_model']}."
            )

        return (
            f"Forecast range information for {medicine_name}: "
            + "; ".join(available_parts)
            + "."
        )

    @staticmethod
    def _format_model_info(
        answer: dict[str, Any],
    ) -> str:
        """Format model-selection information."""

        return (
            f"{answer['medicine_name']} was forecast using "
            f"{answer['selected_model']}. "
            f"Across {answer['validation_windows']} validation windows, "
            f"Chronos-2 absolute error was "
            f"{answer['chronos_absolute_error']:.2f} and TSB absolute "
            f"error was {answer['tsb_absolute_error']:.2f}. "
            f"Validation advantage was "
            f"{answer['validation_advantage_pct']:.2f}%. "
            f"Routing reason: {answer['routing_reason']}."
        )

    @staticmethod
    def _format_routing_explanation(
        answer: dict[str, Any],
    ) -> str:
        """Format routing explanation."""

        reason = answer.get(
            "reason_explanation"
        ) or answer.get(
            "routing_reason",
            "No detailed routing reason is available.",
        )

        model_explanation = answer.get("model_explanation")

        if model_explanation:
            return (
                f"{answer['medicine_name']}: {model_explanation}"
            )

        return (
            f"{answer['medicine_name']} was assigned to "
            f"{answer['selected_model']}. {reason}"
        )

    @staticmethod
    def _format_ranking(
        answer: dict[str, Any],
    ) -> str:
        """Format medicine rankings."""

        results = answer.get("results", [])

        if not results:
            return (
                "No medicines are available for this ranking."
            )

        ranking_type = answer.get("ranking_type")

        title = (
            "Lowest predicted demand medicines"
            if ranking_type == "lowest_predicted_demand"
            else "Highest predicted demand medicines"
        )

        lines = [title + ":"]

        for index, item in enumerate(results, start=1):
            lines.append(
                f"{index}. {item['medicine_name']} — "
                f"{item['total_predicted_demand']:.2f} total predicted demand"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_status(
        answer: dict[str, Any],
    ) -> str:
        """Format medicine status information."""

        requested_status = (
            answer.get("requested_status")
            or "all statuses"
        )

        return (
            f"Status query for {requested_status}: "
            f"{answer['medicine_count']} medicines and "
            f"{answer['record_count']} forecast records. "
            f"Eligibility distribution: "
            f"{answer['eligibility_counts']}. "
            f"Forecast status distribution: "
            f"{answer['forecast_status_counts']}."
        )

    @staticmethod
    def _format_summary(
        answer: dict[str, Any],
    ) -> str:
        """Format the overall forecast summary."""

        return (
            f"The current production forecast covers "
            f"{answer['medicine_count']} medicines across "
            f"{answer['forecast_days']} forecast days, from "
            f"{answer['forecast_start']} to {answer['forecast_end']}. "
            f"Total predicted demand is "
            f"{answer['total_predicted_demand']:.2f}, with an average "
            f"daily predicted demand of "
            f"{answer['average_daily_predicted_demand']:.2f}. "
            f"Model distribution: {answer['model_distribution']}. "
            f"Eligibility distribution: "
            f"{answer['eligibility_distribution']}."
        )