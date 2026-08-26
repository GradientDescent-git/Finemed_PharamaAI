"""
Conversation state management for Finemed PharmaAI Forecast Intelligence.

The context stores the last successfully resolved medicine so that
follow-up questions such as:

    "Which model predicted it?"
    "Why was that model selected?"
    "What about the P90?"

can be resolved safely.

Non-forecast conversational actions such as greetings, help requests,
or out-of-scope questions must not destroy valid forecasting context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finemed_ai.forecast_intelligence.action_router import ForecastAction


@dataclass
class ForecastConversationContext:
    """
    Stores the active forecasting conversation context.

    Only successfully resolved medicine information is allowed to
    replace the current medicine context.
    """

    medicine_query: str | None = None
    medicine_id: str | None = None
    medicine_name: str | None = None
    last_action: str | None = None

    # ------------------------------------------------------------------
    # Context updates
    # ------------------------------------------------------------------

    def update_from_result(
        self,
        action: ForecastAction,
        result: dict[str, Any],
    ) -> None:
        """
        Update context from a completed query result.

        Important:
        - Greeting/help/out-of-scope actions do not clear medicine context.
        - Failed medicine resolution does not overwrite valid context.
        - A medicine context is replaced only when a medicine was
          successfully resolved.
        - The last successfully resolved medicine remains available
          for follow-up questions.
        """

        if not isinstance(result, dict):
            return

        # Always retain the last action for observability.
        # This does not modify the medicine context.
        self.last_action = action.action

        # Only successful deterministic results may update medicine state.
        if not result.get("found", False):
            return

        medicine_id = result.get("medicine_id")
        medicine_name = result.get("medicine_name")

        # Non-medicine actions such as greeting/help/ranking/summary
        # generally do not contain a medicine identity and therefore
        # must not overwrite an existing medicine context.
        if not medicine_id or not medicine_name:
            return

        resolved_query = result.get("query")

        self.medicine_query = (
            str(resolved_query).strip()
            if resolved_query
            else str(medicine_name).strip()
        )

        self.medicine_id = str(medicine_id).strip()
        self.medicine_name = str(medicine_name).strip()

    # ------------------------------------------------------------------
    # Context application
    # ------------------------------------------------------------------

    def apply_action(
        self,
        action: ForecastAction,
    ) -> ForecastAction:
        """
        Apply the current medicine context to an incoming action.

        Context is used only when:

        1. The action explicitly needs previous context.
        2. The medicine query is missing.
        3. The extracted medicine query is actually a conversational
           reference such as "it", "that", "this", or "same medicine".

        Explicit medicine names always take priority over stored context.
        """

        if not self.has_medicine_context():
            return action

        should_apply_context = (
            action.needs_context
            or not action.medicine_query
            or self._is_context_reference(
                action.medicine_query
            )
        )

        if not should_apply_context:
            return action

        # Only medicine-specific actions can safely receive medicine
        # context. Ranking, status, summary, greeting, etc. should
        # never inherit a previous medicine.
        if not self._action_supports_medicine_context(
            action.action
        ):
            return action

        return ForecastAction(
            action=action.action,
            medicine_query=self.medicine_query,
            ranking_limit=action.ranking_limit,
            ranking_direction=action.ranking_direction,
            status=action.status,
            needs_context=True,
            confidence=action.confidence,
            source=action.source,
        )

    @staticmethod
    def _action_supports_medicine_context(
        action: str,
    ) -> bool:
        """
        Return True only for actions that operate on one medicine.
        """

        return action in {
            "forecast",
            "forecast_range",
            "model_info",
            "routing_explanation",
        }

    @staticmethod
    def _is_context_reference(
        medicine_query: str | None,
    ) -> bool:
        """
        Determine whether a medicine query is actually a conversational
        reference rather than a real medicine name or medicine code.

        Examples:
            it
            its
            that
            this
            them
            about
            that medicine
            this medicine
            same medicine
            that model
            this forecast
        """

        if medicine_query is None:
            return True

        normalized = (
            medicine_query
            .strip()
            .lower()
        )

        context_references = {
            "",
            "it",
            "its",
            "that",
            "this",
            "them",
            "those",
            "these",
            "about",
            "same",
            "same one",
            "the same",
            "the medicine",
            "that medicine",
            "this medicine",
            "same medicine",
            "the product",
            "that product",
            "this product",
            "same product",
            "the model",
            "that model",
            "this model",
            "the forecast",
            "that forecast",
            "this forecast",
            "the previous medicine",
            "previous medicine",
            "previous one",
            "last medicine",
            "last one",
        }

        return normalized in context_references

    # ------------------------------------------------------------------
    # Context state
    # ------------------------------------------------------------------

    def has_medicine_context(self) -> bool:
        """
        Return True when a complete valid medicine context exists.
        """

        return bool(
            self.medicine_query
            and self.medicine_id
            and self.medicine_name
        )

    def clear(self) -> None:
        """
        Clear the entire conversation context.
        """

        self.medicine_query = None
        self.medicine_id = None
        self.medicine_name = None
        self.last_action = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a serializable representation of the context.
        """

        return {
            "medicine_query": self.medicine_query,
            "medicine_id": self.medicine_id,
            "medicine_name": self.medicine_name,
            "last_action": self.last_action,
        }

    # ------------------------------------------------------------------
    # LLM routing context
    # ------------------------------------------------------------------

    def to_prompt_context(self) -> str:
        """
        Return human-readable context for the action-routing LLM.

        Returns a neutral message when no medicine context exists.
        """

        if not self.has_medicine_context():
            return "No previous conversation context."

        return (
            f"Previously discussed medicine: "
            f"{self.medicine_name}\n"
            f"Medicine code: {self.medicine_id}\n"
            f"Previous action: "
            f"{self.last_action or 'unknown'}"
        )