"""
Conversation orchestration for Finemed PharmaAI Demand Forecasting.

This module coordinates:

    Employee Question
            ↓
      Action Router
            ↓
  Conversation Context Resolution
            ↓
   Deterministic Query Service
            ↓
 Routing / Analytics / Forecast Data
            ↓
      Natural Language Answer

The orchestrator does not generate or modify forecast values.
All forecasting facts come from the deterministic query layer.
"""

from __future__ import annotations

from typing import Any

from finemed_ai.forecast_intelligence.action_router import (
    ForecastAction,
    ForecastActionRouter,
)
from finemed_ai.forecast_intelligence.conversation_context import (
    ForecastConversationContext,
)
from finemed_ai.forecast_intelligence.query_service import (
    ForecastQueryService,
)
from finemed_ai.forecast_intelligence.routing_explainable import (
    RoutingExplainabilityService,
)


class ForecastConversationOrchestrator:
    """
    Main conversational orchestration layer for demand forecasting.

    Responsibilities
    ----------------
    - Understand the employee question.
    - Route the question to a supported forecasting action.
    - Resolve conversational references using valid conversation context.
    - Query deterministic forecasting services.
    - Preserve valid conversation context.
    - Return a natural-language answer backed by structured data.

    This layer must never invent forecast values or model decisions.
    """

    def __init__(
        self,
        action_router: ForecastActionRouter | None = None,
        query_service: ForecastQueryService | None = None,
        routing_explainer: RoutingExplainabilityService | None = None,
        context: ForecastConversationContext | None = None,
    ) -> None:
        self.action_router = (
            action_router or ForecastActionRouter()
        )

        self.query_service = (
            query_service or ForecastQueryService()
        )

        self.routing_explainer = (
            routing_explainer
            or RoutingExplainabilityService()
        )

        self.context = (
            context or ForecastConversationContext()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Process one employee question.

        Flow
        ----
        1. Validate question.
        2. Route question using available conversation context.
        3. Apply deterministic conversation context where needed.
        4. Execute deterministic query.
        5. Generate grounded natural-language response.
        6. Update conversation context safely.
        """

        question = question.strip()

        if not question:
            return {
                "question": question,
                "action": "insufficient_information",
                "confidence": 1.0,
                "source": "system",
                "resolved_medicine": None,
                "answer": (
                    "Please enter a question about Finemed's "
                    "demand forecasting information."
                ),
                "data": {
                    "found": False,
                    "reason": "empty_question",
                },
            }

        # --------------------------------------------------------------
        # 1. Route the employee question
        #
        # The router receives human-readable conversation context so an
        # LLM router can understand follow-up questions such as:
        #
        # "Which model predicted it?"
        # "Why was that model selected?"
        # "What about the P90?"
        # --------------------------------------------------------------

        action = self.action_router.route(
            question,
            conversation_context=self.context.to_prompt_context(),
        )

        # --------------------------------------------------------------
        # 2. Apply deterministic conversation context
        #
        # This provides a safe fallback even when:
        #
        # - The LLM does not resolve the medicine correctly.
        # - The deterministic fallback extracts "it", "that", or
        #   another conversational reference as a medicine query.
        # - The action explicitly requires previous medicine context.
        #
        # ConversationContext is the single source of truth for
        # applying stored medicine context.
        # --------------------------------------------------------------

        action = self.context.apply_action(action)

        # --------------------------------------------------------------
        # 3. Handle non-data conversational actions
        # --------------------------------------------------------------

        if action.action == "greeting":
            result = self._greeting_result()

        elif action.action == "help":
            result = self._help_result()

        elif action.action == "out_of_scope":
            result = self._out_of_scope_result()

        elif action.action == "insufficient_information":
            result = self._insufficient_information_result()

        # --------------------------------------------------------------
        # 4. Execute deterministic forecasting action
        # --------------------------------------------------------------

        else:
            result = self._dispatch(action)

        # --------------------------------------------------------------
        # 5. Generate grounded answer
        # --------------------------------------------------------------

        answer = self._build_answer(
            action=action,
            result=result,
        )

        # --------------------------------------------------------------
        # 6. Update context safely
        #
        # A failed medicine resolution must NEVER overwrite a valid
        # previously established medicine.
        # --------------------------------------------------------------

        self.context.update_from_result(
            action,
            result,
        )

        return {
            "question": question,
            "action": action.action,
            "confidence": action.confidence,
            "source": action.source,
            "resolved_medicine": result.get("medicine_name"),
            "answer": answer,
            "data": result,
        }

    # ------------------------------------------------------------------
    # Conversation context
    # ------------------------------------------------------------------

    def get_context(self) -> dict[str, Any]:
        """Return the current conversation context."""

        return self.context.to_dict()

    def clear_context(self) -> None:
        """Clear the current conversation context."""

        self.context.clear()

    # ------------------------------------------------------------------
    # Deterministic action dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        action: ForecastAction,
    ) -> dict[str, Any]:
        """
        Dispatch a supported action to the deterministic query layer.
        """

        if action.action == "forecast":
            return self._require_medicine_or_query(
                action,
                self.query_service.get_forecast,
            )

        if action.action == "forecast_range":
            return self._require_medicine_or_query(
                action,
                self.query_service.get_forecast_range,
            )

        if action.action == "model_info":
            return self._require_medicine_or_query(
                action,
                self.query_service.get_model_info,
            )

        if action.action == "routing_explanation":
            result = self._require_medicine_or_query(
                action,
                self.query_service.get_routing_explanation,
            )

            if result.get("found"):
                return self.routing_explainer.explain(result)

            return result

        if action.action == "ranking":

            top_n = (
                action.ranking_limit
                if action.ranking_limit is not None
                else 10
            )

            ascending = (
                action.ranking_direction == "lowest"
            )

            return self.query_service.get_ranking(
                top_n=top_n,
                ascending=ascending,
            )

        if action.action == "status_query":

            return self.query_service.get_status(
                status=action.status,
            )

        if action.action == "forecast_summary":

            return (
                self.query_service
                .get_forecast_summary()
            )

        return self._insufficient_information_result()

    def _require_medicine_or_query(
        self,
        action: ForecastAction,
        query_function: Any,
    ) -> dict[str, Any]:
        """
        Execute a medicine-specific query.

        If no medicine can be determined from either the current
        question or conversation context, return a safe response.
        """

        medicine_query = action.medicine_query

        if not medicine_query:
            return {
                "found": False,
                "reason": "medicine_not_specified",
                "message": (
                    "I need the medicine name or medicine code to "
                    "answer that question."
                ),
            }

        return query_function(medicine_query)

    # ------------------------------------------------------------------
    # Conversational result builders
    # ------------------------------------------------------------------

    @staticmethod
    def _greeting_result() -> dict[str, Any]:
        """Return greeting metadata."""

        return {
            "found": True,
            "type": "greeting",
        }

    @staticmethod
    def _help_result() -> dict[str, Any]:
        """Return help metadata."""

        return {
            "found": True,
            "type": "help",
        }

    @staticmethod
    def _out_of_scope_result() -> dict[str, Any]:
        """Return out-of-scope metadata."""

        return {
            "found": False,
            "reason": "out_of_scope",
        }

    @staticmethod
    def _insufficient_information_result() -> dict[str, Any]:
        """Return insufficient-information metadata."""

        return {
            "found": False,
            "reason": "insufficient_information",
        }

    # ------------------------------------------------------------------
    # Natural language answer generation
    # ------------------------------------------------------------------

    def _build_answer(
        self,
        action: ForecastAction,
        result: dict[str, Any],
    ) -> str:
        """
        Convert deterministic structured results into grounded,
        employee-friendly natural language.

        No forecast values are generated here.
        """

        # --------------------------------------------------------------
        # Greeting
        # --------------------------------------------------------------

        if action.action == "greeting":

            return (
                "Hello! I'm Finemed PharmaAI's Demand Forecasting "
                "Assistant. I can help you explore medicine demand "
                "forecasts, selected models, forecast ranges, "
                "rankings, medicine statuses, and model-selection "
                "decisions."
            )

        # --------------------------------------------------------------
        # Help
        # --------------------------------------------------------------

        if action.action == "help":

            return (
                "I can help you with Finemed's demand forecasting "
                "information. For example:\n\n"
                "• What is the forecast for Otacare?\n"
                "• What is the P90 forecast for Otacare?\n"
                "• Which model predicted Otacare?\n"
                "• Why was TSB selected for Otacare?\n"
                "• Show the top 10 medicines by demand.\n"
                "• Show the bottom 5 medicines by demand.\n"
                "• Show dormant medicines.\n"
                "• Give me the overall forecast summary.\n\n"
                "You can also ask follow-up questions such as "
                "\"Which model predicted it?\" after discussing a "
                "medicine.\n\n"
                "If the available forecasting data does not contain "
                "enough information to answer a question reliably, "
                "I will tell you rather than inventing an answer."
            )

        # --------------------------------------------------------------
        # Out of scope
        # --------------------------------------------------------------

        if action.action == "out_of_scope":

            return (
                "I'm currently focused on Finemed's demand forecasting "
                "information. I can help with medicine forecasts, "
                "forecast ranges, selected models, model-selection "
                "reasons, demand rankings, medicine statuses, and "
                "overall forecast summaries."
            )

        # --------------------------------------------------------------
        # Insufficient information
        # --------------------------------------------------------------

        if action.action == "insufficient_information":

            return (
                "I don't have enough information to answer that "
                "reliably from the currently available demand "
                "forecasting data. I can help with medicine forecasts, "
                "forecast ranges, selected models, routing decisions, "
                "rankings, medicine statuses, and overall forecast "
                "summaries."
            )

        # --------------------------------------------------------------
        # Generic failure
        # --------------------------------------------------------------

        if not result.get("found", False):

            reason = result.get("reason")

            if reason == "medicine_not_resolved":

                return (
                    "I couldn't identify that medicine from the "
                    "available medicine data. Please provide the "
                    "medicine name or medicine code."
                )

            if reason == "medicine_not_specified":

                return (
                    "I need the medicine name or medicine code to "
                    "answer that question. For example, you can ask "
                    "\"What is the forecast for Otacare?\""
                )

            if reason == "no_forecast_records_available":

                return (
                    "I found the medicine, but there are no production "
                    "forecast records currently available for it."
                )

            if reason == "no_routing_record_available":

                return (
                    "I found the medicine, but I don't have enough "
                    "routing information available to explain the "
                    "model selection."
                )

            return (
                "I don't have enough information in the currently "
                "available demand forecasting data to answer that "
                "reliably."
            )

        # --------------------------------------------------------------
        # Forecast
        # --------------------------------------------------------------

        if action.action == "forecast":

            return (
                f"The forecast for {result['medicine_name']} from "
                f"{result['forecast_start']} to "
                f"{result['forecast_end']} is "
                f"{result['total_predicted_demand']:.2f} total "
                f"predicted demand over "
                f"{result['forecast_days']} days. "
                f"That is an average of "
                f"{result['average_daily_demand']:.2f} per day. "
                f"The selected model is "
                f"{result['selected_model']}. "
                f"Eligibility status: "
                f"{result['eligibility_status']}. "
                f"Forecast status: "
                f"{result['forecast_status']}."
            )

        # --------------------------------------------------------------
        # Forecast range
        # --------------------------------------------------------------

        if action.action == "forecast_range":

            medicine_name = result["medicine_name"]

            if not any(
                [
                    result.get("p10_available"),
                    result.get("p50_available"),
                    result.get("p90_available"),
                ]
            ):
                return (
                    f"No P10, P50, or P90 forecast range values are "
                    f"available for {medicine_name}. "
                    f"The selected model is "
                    f"{result.get('selected_model')}."
                )

            parts: list[str] = []

            if result.get("p10_available"):
                parts.append(
                    f"P10: {result['total_p10']:.2f}"
                )

            if result.get("p50_available"):
                parts.append(
                    f"P50: {result['total_p50']:.2f}"
                )

            if result.get("p90_available"):
                parts.append(
                    f"P90: {result['total_p90']:.2f}"
                )

            return (
                f"Forecast range for {medicine_name} from "
                f"{result['forecast_start']} to "
                f"{result['forecast_end']}: "
                + ", ".join(parts)
                + "."
            )

        # --------------------------------------------------------------
        # Model information
        # --------------------------------------------------------------

        if action.action == "model_info":

            answer = (
                f"{result['medicine_name']} was forecast using "
                f"{result['selected_model']}."
            )

            if (
                result.get("validation_windows") is not None
                and result.get("chronos_absolute_error") is not None
                and result.get("tsb_absolute_error") is not None
            ):
                answer += (
                    f" Across {result['validation_windows']} "
                    f"validation windows, Chronos-2 absolute error "
                    f"was {result['chronos_absolute_error']:.2f} "
                    f"and TSB absolute error was "
                    f"{result['tsb_absolute_error']:.2f}."
                )

            if result.get(
                "validation_advantage_pct"
            ) is not None:
                answer += (
                    f" Validation advantage was "
                    f"{result['validation_advantage_pct']:.2f}%."
                )

            if result.get("routing_reason"):
                answer += (
                    f" Routing reason: "
                    f"{result['routing_reason']}."
                )

            return answer

        # --------------------------------------------------------------
        # Routing explanation
        # --------------------------------------------------------------

        if action.action == "routing_explanation":

            model_explanation = result.get(
                "model_explanation"
            )

            if model_explanation:
                return (
                    f"{result.get('medicine_name')}: "
                    f"{model_explanation}"
                )

            reason_explanation = result.get(
                "reason_explanation"
            )

            if reason_explanation:
                return (
                    f"{result.get('medicine_name')}: "
                    f"{reason_explanation}"
                )

            return (
                "The forecasting pipeline selected the configured "
                "model based on the locked validation routing rules."
            )

        # --------------------------------------------------------------
        # Ranking
        # --------------------------------------------------------------

        if action.action == "ranking":

            ranking_type = result.get(
                "ranking_type",
                "highest_predicted_demand",
            )

            if ranking_type == "lowest_predicted_demand":
                heading = (
                    "Medicines with the lowest predicted demand:"
                )
            else:
                heading = (
                    "Medicines with the highest predicted demand:"
                )

            lines = [heading]

            for index, item in enumerate(
                result.get("results", []),
                start=1,
            ):
                medicine_name = (
                    item.get("medicine_name")
                    or item.get("medicine_id")
                )

                lines.append(
                    f"{index}. {medicine_name} — "
                    f"{item['total_predicted_demand']:.2f} "
                    f"total predicted demand"
                )

            if len(lines) == 1:
                return (
                    "I don't have enough ranking data available "
                    "to answer that question."
                )

            return "\n".join(lines)

        # --------------------------------------------------------------
        # Status query
        # --------------------------------------------------------------

        if action.action == "status_query":

            requested_status = (
                result.get("requested_status")
                or "all statuses"
            )

            return (
                f"Status information for {requested_status}: "
                f"{result.get('medicine_count', 0)} medicines and "
                f"{result.get('record_count', 0)} forecast records. "
                f"Eligibility distribution: "
                f"{result.get('eligibility_counts', {})}. "
                f"Forecast status distribution: "
                f"{result.get('forecast_status_counts', {})}."
            )

        # --------------------------------------------------------------
        # Forecast summary
        # --------------------------------------------------------------

        if action.action == "forecast_summary":

            return (
                f"The current production forecast covers "
                f"{result['medicine_count']} medicines across "
                f"{result['forecast_days']} forecast days, from "
                f"{result['forecast_start']} to "
                f"{result['forecast_end']}. "
                f"Total predicted demand is "
                f"{result['total_predicted_demand']:.2f}, "
                f"with an average daily predicted demand of "
                f"{result['average_daily_predicted_demand']:.2f}. "
                f"Model distribution: "
                f"{result.get('model_distribution', {})}. "
                f"Eligibility distribution: "
                f"{result.get('eligibility_distribution', {})}."
            )

        # --------------------------------------------------------------
        # Final safe fallback
        # --------------------------------------------------------------

        return (
            "I don't have enough information in the available demand "
            "forecasting data to answer that reliably."
        )