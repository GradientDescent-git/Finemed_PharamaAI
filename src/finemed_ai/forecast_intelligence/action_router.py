"""
LLM-assisted action routing for Finemed PharmaAI Forecast Intelligence.

This module converts a natural-language employee question into a validated,
structured forecasting action.

The router is responsible only for understanding the employee question.
It does not calculate forecasts, select models, or generate business facts.

Safety principles
-----------------
- Only supported actions may be returned.
- All LLM output is validated and normalized.
- Invalid or unavailable LLM responses fall back safely to the
  deterministic IntentRouter.
- No forecast values, rankings, model decisions, or routing evidence
  are generated in this layer.
- Conversation context is applied by ForecastConversationContext,
  not guessed or resolved by this router.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from finemed_ai.forecast_intelligence.intent_router import (
    ForecastIntent,
    IntentRouter,
)
from finemed_ai.forecast_intelligence.prompts import (
    SUPPORTED_ACTIONS,
    build_action_router_prompt,
)


@dataclass(frozen=True)
class ForecastAction:
    """
    Validated structured representation of an employee forecasting question.

    Attributes
    ----------
    action:
        One supported forecasting action.
    medicine_query:
        Medicine name or medicine code when present in the current question.
    ranking_limit:
        Requested number of ranking results.
    ranking_direction:
        "highest" or "lowest" for ranking actions.
    status:
        Requested forecasting or eligibility status.
    needs_context:
        True when the action requires previously established medicine
        context because the current question omits the medicine.
    confidence:
        Router confidence normalized to [0.0, 1.0].
    source:
        "llm", "fallback", or "validation".
    """

    action: str
    medicine_query: str | None = None
    ranking_limit: int | None = None
    ranking_direction: str | None = None
    status: str | None = None
    needs_context: bool = False
    confidence: float = 0.0
    source: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "action": self.action,
            "medicine_query": self.medicine_query,
            "ranking_limit": self.ranking_limit,
            "ranking_direction": self.ranking_direction,
            "status": self.status,
            "needs_context": self.needs_context,
            "confidence": self.confidence,
            "source": self.source,
        }


class ForecastActionRouter:
    """
    Route employee forecasting questions into validated ForecastAction objects.

    Parameters
    ----------
    llm_callable:
        Optional callable that receives a prompt string and returns an LLM
        response containing one JSON object matching the supported action
        contract.

    fallback_router:
        Deterministic IntentRouter used whenever the LLM is unavailable,
        malformed, invalid, or raises an exception.
    """

    MAX_RANKING_LIMIT = 100

    MEDICINE_ACTIONS = frozenset(
        {
            "forecast",
            "forecast_range",
            "model_info",
            "routing_explanation",
        }
    )

    def __init__(
        self,
        llm_callable: Callable[[str], str] | None = None,
        fallback_router: IntentRouter | None = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.fallback_router = (
            fallback_router
            or IntentRouter()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        question: str,
        conversation_context: str | None = None,
    ) -> ForecastAction:
        """
        Convert an employee question into a validated ForecastAction.

        The LLM is attempted first when configured. Any exception,
        malformed response, unsupported action, or validation failure
        falls back safely to deterministic routing.
        """

        normalized_question = (
            self._normalize_question(
                question
            )
        )

        if not normalized_question:
            return ForecastAction(
                action="insufficient_information",
                confidence=1.0,
                source="validation",
            )

        if self.llm_callable is not None:
            try:
                prompt = build_action_router_prompt(
                    question=normalized_question,
                    conversation_context=conversation_context,
                )

                raw_response = self.llm_callable(
                    prompt
                )

                parsed = self._parse_llm_response(
                    raw_response
                )

                if parsed is not None:
                    return self._validate_action(
                        parsed,
                        source="llm",
                    )

            except Exception:
                # Routing must never break the forecasting assistant.
                # The deterministic router remains the safe fallback.
                pass

        return self._fallback_route(
            normalized_question
        )

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(
        raw_response: Any,
    ) -> dict[str, Any] | None:
        """
        Parse an LLM response expected to contain one JSON object.

        Supports:
        - Python dictionaries.
        - Plain JSON.
        - JSON inside markdown fences.
        - JSON surrounded by explanatory text.
        """

        if isinstance(
            raw_response,
            dict,
        ):
            return raw_response

        if not isinstance(
            raw_response,
            str,
        ):
            return None

        text = raw_response.strip()

        if not text:
            return None

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        ).strip()

        try:
            parsed = json.loads(
                text
            )

            return (
                parsed
                if isinstance(
                    parsed,
                    dict,
                )
                else None
            )

        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            return None

        try:
            parsed = json.loads(
                text[start : end + 1]
            )

            return (
                parsed
                if isinstance(
                    parsed,
                    dict,
                )
                else None
            )

        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # LLM output validation
    # ------------------------------------------------------------------

    def _validate_action(
        self,
        data: dict[str, Any],
        source: str,
    ) -> ForecastAction:
        """
        Validate and normalize a structured action.

        Invalid or unsupported fields are removed. Invalid medicine
        actions without either a medicine or explicit context requirement
        fail closed to insufficient_information.
        """

        action = self._normalize_action(
            data.get("action")
        )

        raw_medicine_query = (
            self._normalize_optional_string(
                data.get("medicine_query")
            )
        )

        medicine_query = (
            self._normalize_medicine_query(
                raw_medicine_query
            )
        )

        ranking_limit = (
            self._normalize_ranking_limit(
                data.get("ranking_limit")
            )
        )

        ranking_direction = (
            self._normalize_ranking_direction(
                data.get("ranking_direction")
            )
        )

        status = (
            self._normalize_optional_string(
                data.get("status")
            )
        )

        needs_context = (
            self._normalize_bool(
                data.get(
                    "needs_context",
                    False,
                )
            )
        )

        confidence = (
            self._normalize_confidence(
                data.get(
                    "confidence",
                    0.0,
                )
            )
        )

        # Medicine-specific actions require either:
        #
        # 1. A medicine in the current question.
        # 2. Explicit conversation-context resolution.
        #
        # Otherwise fail closed.
        if (
            action in self.MEDICINE_ACTIONS
            and not medicine_query
            and not needs_context
        ):
            action = (
                "insufficient_information"
            )

        # Ranking defaults.
        if action == "ranking":
            ranking_direction = (
                ranking_direction
                or "highest"
            )

        # Remove irrelevant fields.
        if action != "ranking":
            ranking_limit = None
            ranking_direction = None

        if action != "status_query":
            status = None

        if action not in self.MEDICINE_ACTIONS:
            medicine_query = None
            needs_context = False

        return ForecastAction(
            action=action,
            medicine_query=medicine_query,
            ranking_limit=ranking_limit,
            ranking_direction=ranking_direction,
            status=status,
            needs_context=needs_context,
            confidence=confidence,
            source=source,
        )

    # ------------------------------------------------------------------
    # Value normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_action(
        value: Any,
    ) -> str:
        """Return a supported action or insufficient_information."""

        if not isinstance(
            value,
            str,
        ):
            return (
                "insufficient_information"
            )

        normalized = (
            value.strip().lower()
        )

        if normalized in SUPPORTED_ACTIONS:
            return normalized

        return (
            "insufficient_information"
        )

    @staticmethod
    def _normalize_optional_string(
        value: Any,
    ) -> str | None:
        """Normalize an optional non-empty string."""

        if not isinstance(
            value,
            str,
        ):
            return None

        normalized = " ".join(
            value.strip().split()
        )

        if not normalized:
            return None

        if normalized.lower() in {
            "null",
            "none",
            "n/a",
            "unknown",
        }:
            return None

        return normalized

    @staticmethod
    def _normalize_medicine_query(
        value: Any,
    ) -> str | None:
        """
        Normalize an explicit medicine reference without performing
        medicine resolution.

        Supported explicit code forms include:

        - "0001"
        - "1"
        - "medicine 0001"
        - "medicine id 0001"
        - "medicine code 0001"
        - "medicine number 0001"
        - "product id 0001"
        - "product code 0001"

        The resolver remains the authority for determining whether the
        resulting code/name actually exists and is unambiguous.
        """

        if value is None:
            return None

        normalized = (
            ForecastActionRouter
            ._normalize_optional_string(
                value
            )
        )

        if not normalized:
            return None

        # ---------------------------------------------------------
        # 1. Bare numeric medicine code
        # ---------------------------------------------------------

        if normalized.isdigit():
            return normalized.zfill(4)

        # ---------------------------------------------------------
        # 2. Explicit medicine/product identifier
        # ---------------------------------------------------------

        explicit_code_patterns = (
            r"\b(?:medicine|medicines|product)\s*"
            r"(?:id|code|number)\s*"
            r"[:#-]?\s*(\d{1,6})\b",

            r"\b(?:medicine|medicines|product)\s*"
            r"[:#-]?\s*(\d{1,6})\b",
        )

        for pattern in explicit_code_patterns:
            match = re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1)
                    .zfill(4)
                )

        # ---------------------------------------------------------
        # 3. Normal medicine name
        # ---------------------------------------------------------

        return normalized

    @staticmethod
    def _normalize_bool(
        value: Any,
    ) -> bool:
        """
        Safely normalize common boolean representations.

        Avoids Python's unsafe bool("false") == True behavior.
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):
            normalized = (
                value.strip().lower()
            )

            if normalized in {
                "true",
                "yes",
                "1",
            }:
                return True

            if normalized in {
                "false",
                "no",
                "0",
                "",
            }:
                return False

        if (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        ):
            return bool(value)

        return False

    @classmethod
    def _normalize_ranking_limit(
        cls,
        value: Any,
    ) -> int | None:
        """Normalize ranking limit within production bounds."""

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):
            return None

        try:
            limit = int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

        if (
            limit < 1
            or limit > cls.MAX_RANKING_LIMIT
        ):
            return None

        return limit

    @staticmethod
    def _normalize_ranking_direction(
        value: Any,
    ) -> str | None:
        """Normalize ranking direction."""

        if not isinstance(
            value,
            str,
        ):
            return None

        normalized = (
            value.strip().lower()
        )

        aliases = {
            "highest": "highest",
            "top": "highest",
            "largest": "highest",
            "descending": "highest",
            "lowest": "lowest",
            "bottom": "lowest",
            "least": "lowest",
            "smallest": "lowest",
            "ascending": "lowest",
        }

        return aliases.get(
            normalized
        )

    @staticmethod
    def _normalize_confidence(
        value: Any,
    ) -> float:
        """Normalize confidence into [0.0, 1.0]."""

        try:
            confidence = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if confidence != confidence:
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    def _fallback_route(
        self,
        question: str,
    ) -> ForecastAction:
        """
        Convert deterministic IntentRouter output into ForecastAction.

        This router determines only whether context is required.
        It never resolves the previous medicine itself.

        ForecastConversationContext remains the authoritative layer for
        applying previously established medicine context.
        """

        result = self.fallback_router.route(
            question
        )

        intent_to_action = {
            ForecastIntent.FORECAST:
                "forecast",

            ForecastIntent.FORECAST_RANGE:
                "forecast_range",

            ForecastIntent.MODEL_INFO:
                "model_info",

            ForecastIntent.ROUTING_EXPLANATION:
                "routing_explanation",

            ForecastIntent.RANKING:
                "ranking",

            ForecastIntent.STATUS_QUERY:
                "status_query",

            ForecastIntent.FORECAST_SUMMARY:
                "forecast_summary",
        }

        action = intent_to_action.get(
            result.intent
        )

        if action is None:
            action = (
                self._fallback_special_action(
                    question
                )
            )

        medicine_query = None
        needs_context = False

        if action in self.MEDICINE_ACTIONS:

            # First determine whether this question is referring to
            # the medicine already established in the conversation.
            #
            # This includes:
            # - Which model predicted it?
            # - Why was that model selected?
            # - What is the P90 forecast?
            # - Give me the P10.
            # - What about the forecast range?
            needs_context = (
                self._needs_context(
                    question,
                    action,
                )
            )

            # Only extract a medicine when the current question is
            # expected to explicitly contain one.
            if not needs_context:

                raw_medicine_query = (
                    self._extract_medicine_query(
                        question,
                        result.matched_patterns,
                    )
                )

                medicine_query = (
                    self._normalize_medicine_query(
                        raw_medicine_query
                    )
                )

            # Fail closed when neither an explicit medicine nor a valid
            # contextual reference exists.
            if (
                not medicine_query
                and not needs_context
            ):
                action = (
                    "insufficient_information"
                )

        ranking_limit = None
        ranking_direction = None

        if action == "ranking":

            ranking_limit = (
                self._extract_ranking_limit(
                    question
                )
            )

            ranking_direction = (
                self._extract_ranking_direction(
                    question
                )
            )

        status = None

        if action == "status_query":
            status = self._extract_status(
                question
            )

        return ForecastAction(
            action=action,
            medicine_query=medicine_query,
            ranking_limit=ranking_limit,
            ranking_direction=ranking_direction,
            status=status,
            needs_context=needs_context,
            confidence=self._normalize_confidence(
                result.confidence
            ),
            source="fallback",
        )

    # ------------------------------------------------------------------
    # Special conversational actions
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_special_action(
        question: str,
    ) -> str:
        """Handle greetings, help, and unsupported questions."""

        normalized = (
            question.lower().strip()
        )

        if re.search(
            r"\b("
            r"hi|hello|hey|"
            r"good morning|"
            r"good afternoon|"
            r"good evening"
            r")\b",
            normalized,
        ):
            return "greeting"

        if re.search(
            r"\b("
            r"help|"
            r"what can you do|"
            r"how can you help|"
            r"available information"
            r")\b",
            normalized,
        ):
            return "help"

        return "out_of_scope"

    # ------------------------------------------------------------------
    # Medicine extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_medicine_query(
        question: str,
        matched_patterns: tuple[str, ...]
        | list[str],
    ) -> str | None:
        """
        Extract a probable medicine reference from a deterministic question.

        The returned value is only a candidate. MedicineResolver remains
        the authority for actual medicine resolution.

        Explicit medicine-code references are normalized so phrases such as:

        - "medicine id 0001"
        - "medicine code 0001"
        - "medicine 0001"
        - "product id 0001"

        are passed to MedicineResolver as the canonical numeric code
        "0001".
        """

        del matched_patterns

        if not isinstance(
            question,
            str,
        ):
            return None

        text = question.strip()

        if not text:
            return None

        # ---------------------------------------------------------
        # 1. Explicit medicine/product code references
        # ---------------------------------------------------------

        explicit_code_patterns = (
            r"\b(?:medicine|medicines|product)\s*"
            r"(?:id|code|number)\s*"
            r"[:#-]?\s*(\d{1,6})\b",

            r"\b(?:medicine|medicines|product)\s*"
            r"[:#-]?\s*(\d{1,6})\b",
        )

        for pattern in explicit_code_patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1)
                    .zfill(4)
                )

        # ---------------------------------------------------------
        # 2. Standard medicine/name extraction
        # ---------------------------------------------------------

        patterns = [
            (
                r"\b(?:what is |show |give me )?"
                r"(?:the )?forecast\s+"
                r"(?:for|of)\s+(.+?)(?:\?|$)"
            ),
            (
                r"\bforecast\s+(.+?)(?:\?|$)"
            ),
            (
                r"\b(?:which )?model\s+"
                r"(?:was )?"
                r"(?:used for|for|predicted)\s+"
                r"(.+?)(?:\?|$)"
            ),
            (
                r"\bwhich model predicted\s+"
                r"(.+?)(?:\?|$)"
            ),
            (
                r"\bwhy\s+was\s+.+?"
                r"(?:selected|chosen)"
                r"(?:\s+(?:for|on))?\s+"
                r"(.+?)(?:\?|$)"
            ),
            (
                r"\b(?:what is |show |give me )?"
                r"(?:the )?"
                r"(?:p10|p50|p90)\s+"
                r"(?:forecast\s+)?"
                r"(?:for|of)\s+"
                r"(.+?)(?:\?|$)"
            ),
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = " ".join(
                match.group(1)
                .strip()
                .split()
            )

            if not candidate:
                continue

            # Normalize an explicit medicine code captured by the
            # broader forecast/name regex.
            normalized_candidate = (
                ForecastActionRouter
                ._normalize_medicine_query(
                    candidate
                )
            )

            if normalized_candidate:
                return normalized_candidate

        return None

    # ------------------------------------------------------------------
    # Ranking extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_ranking_limit(
        cls,
        question: str,
    ) -> int | None:
        """Extract a requested ranking count."""

        match = re.search(
            r"\b("
            r"top|bottom|highest|lowest|"
            r"largest|smallest|least"
            r")\s+(\d+)\b",
            question,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        try:
            limit = int(
                match.group(2)
            )

        except ValueError:
            return None

        if (
            limit < 1
            or limit > cls.MAX_RANKING_LIMIT
        ):
            return None

        return limit

    @staticmethod
    def _extract_ranking_direction(
        question: str,
    ) -> str:
        """Determine ranking direction."""

        normalized = (
            question.lower()
        )

        if re.search(
            r"\b("
            r"bottom|lowest|least|smallest"
            r")\b",
            normalized,
        ):
            return "lowest"

        return "highest"

    # ------------------------------------------------------------------
    # Status extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_status(
        question: str,
    ) -> str | None:
        """
        Extract a known status token when explicitly present.

        Status semantics remain controlled by deterministic query-service
        data rather than generated interpretation.
        """

        normalized = question.upper()

        statuses = (
            "FORECASTED_STALE",
            "NOT_FORECASTED",
            "FORECASTED",
            "DORMANT",
            "ACTIVE",
            "STALE",
        )

        for status in statuses:

            if status in normalized:
                return status

        return None

    # ------------------------------------------------------------------
    # Conversation reference detection
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_context(
        question: str,
        action: str,
    ) -> bool:
        """
        Detect medicine-specific follow-up questions that require an
        already established medicine from ForecastConversationContext.

        Explicit references:
        - Which model predicted it?
        - Why was that model selected?
        - What is the forecast for that medicine?

        Implicit range follow-ups:
        - What is the P90 forecast?
        - Give me the P10.
        - Show the P50.
        - What about the forecast range?

        Important:
        A range-only query is treated as contextual only for
        forecast_range. The router does not invent a medicine.
        """

        normalized = (
            question.lower().strip()
        )

        explicit_reference = bool(
            re.search(
                r"\b("
                r"it|its|"
                r"that medicine|"
                r"this medicine|"
                r"that one|"
                r"this one|"
                r"that model|"
                r"this model|"
                r"this forecast|"
                r"that forecast"
                r")\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

        if explicit_reference:
            return True

        if action == "forecast_range":

            implicit_range_reference = bool(
                re.search(
                    r"\b("
                    r"p10|"
                    r"p50|"
                    r"p90|"
                    r"forecast range|"
                    r"confidence interval|"
                    r"lower bound|"
                    r"upper bound|"
                    r"uncertainty"
                    r")\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
            )

            if implicit_range_reference:
                return True

        return False

    # ------------------------------------------------------------------
    # Question normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_question(
        question: Any,
    ) -> str:
        """Normalize and validate the employee question."""

        if not isinstance(
            question,
            str,
        ):
            return ""

        return " ".join(
            question.strip().split()
        )