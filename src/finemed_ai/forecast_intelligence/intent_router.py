"""
Deterministic intent routing for forecast intelligence queries.

This module classifies employee questions into a controlled set of
forecast-domain intents.

The router determines only what the employee is asking for. It does not:

- resolve medicine entities,
- access production forecast artifacts,
- calculate forecast values,
- modify routing decisions.

Medicine resolution and conversational reference handling are performed
later by the conversation orchestration layer.

Examples
--------
"What is the forecast for Otacare?"
    -> FORECAST

"Which model predicted it?"
    -> MODEL_INFO

"Why was that model selected?"
    -> ROUTING_EXPLANATION

"What is the P90 forecast?"
    -> FORECAST_RANGE

"Show the top 5 medicines by demand"
    -> RANKING
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ForecastIntent(str, Enum):
    """Supported employee query intents."""

    FORECAST = "forecast"
    FORECAST_RANGE = "forecast_range"
    FORECAST_SUMMARY = "forecast_summary"
    MODEL_INFO = "model_info"
    ROUTING_EXPLANATION = "routing_explanation"
    RANKING = "ranking"
    STATUS_QUERY = "status_query"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentResult:
    """Normalized deterministic intent classification result."""

    intent: ForecastIntent
    confidence: float
    matched_patterns: tuple[str, ...]


class IntentRouter:
    """
    Deterministic keyword and pattern-based intent router.

    The routing rules are intentionally explicit so that:

    - behavior remains inspectable,
    - tests remain deterministic,
    - conversational follow-ups can be classified safely,
    - downstream execution never depends on uncontrolled LLM output.

    Intent classification is separate from medicine resolution.

    For example:

        "What is the P90 forecast?"

    can correctly resolve to FORECAST_RANGE even though no medicine name
    appears in the current question. The conversation context layer can
    subsequently attach the previously discussed medicine.
    """

    _PATTERNS: dict[ForecastIntent, tuple[str, ...]] = {
        # --------------------------------------------------------------
        # Routing explanation
        #
        # These must have the highest priority because questions such as:
        #
        # "Why was that model selected?"
        #
        # may also contain generic model-related words.
        # --------------------------------------------------------------
        ForecastIntent.ROUTING_EXPLANATION: (
            r"\bwhy\b.*\bwas\b.*\b(selected|chosen)\b",
            r"\bwhy\b.*\b(selected|chosen)\b",
            r"\bwhy\b.*\b(tsb|chronos(?:-?2)?|model)\b",
            r"\bwhy\b.*\broute(?:d|ing)?\b",
            r"\bwhy\b.*\bdecision\b",
            r"\brouting reason\b",
            r"\bselection reason\b",
            r"\bmodel selection reason\b",
            r"\bwhy was\b",
            r"\bexplain\b.*\b(selection|routing|model)\b",
            r"\bhow was\b.*\b(selected|chosen)\b",
            r"\bwhat made\b.*\b(selected|chosen)\b",
        ),

        # --------------------------------------------------------------
        # Model information
        #
        # Explicit questions about which forecasting model was used.
        # --------------------------------------------------------------
        ForecastIntent.MODEL_INFO: (
            r"\bwhich model\b",
            r"\bwhat model\b",
            r"\bselected model\b",
            r"\bprediction model\b",
            r"\bforecasting model\b",
            r"\bmodel used\b",
            r"\bmodel predicted\b",
            r"\bwhich model predicted\b",
            r"\bwhat model predicted\b",
            r"\bwas it predicted by\b",
            r"\bwas that predicted by\b",
            r"\bwas this predicted by\b",
            r"\bwas it tsb\b",
            r"\bwas it chronos(?:-?2)?\b",
        ),

        # --------------------------------------------------------------
        # Forecast range / uncertainty
        #
        # Explicit P10/P50/P90 queries must always classify here,
        # including follow-up questions without a medicine name.
        # --------------------------------------------------------------
        ForecastIntent.FORECAST_RANGE: (
            r"\bp10\b",
            r"\bp50\b",
            r"\bp90\b",
            r"\bpercentile\s*(10|50|90)\b",
            r"\b10th percentile\b",
            r"\b50th percentile\b",
            r"\b90th percentile\b",
            r"\bconfidence interval\b",
            r"\bprediction interval\b",
            r"\bforecast range\b",
            r"\blower bound\b",
            r"\bupper bound\b",
            r"\blower forecast\b",
            r"\bupper forecast\b",
            r"\buncertainty\b",
            r"\bforecast uncertainty\b",
            r"\bdemand range\b",
            r"\brange of the forecast\b",
            r"\bhow certain\b",
        ),

        # --------------------------------------------------------------
        # Ranking
        #
        # Ranking is intentionally detected before generic forecast.
        # --------------------------------------------------------------
        ForecastIntent.RANKING: (
            r"\btop\s+\d+\b",
            r"\bbottom\s+\d+\b",
            r"\btop\b.*\bmedicine",
            r"\bbottom\b.*\bmedicine",
            r"\bhighest\b.*\bdemand",
            r"\blowest\b.*\bdemand",
            r"\bmost\b.*\bdemand",
            r"\bleast\b.*\bdemand",
            r"\blargest\b.*\bdemand",
            r"\bsmallest\b.*\bdemand",
            r"\brank\b",
            r"\branking\b",
            r"\btop medicines\b",
            r"\bbottom medicines\b",
            r"\bhighest demand\b",
            r"\blowest demand\b",
        ),

        # --------------------------------------------------------------
        # Status queries
        #
        # --------------------------------------------------------------
        ForecastIntent.STATUS_QUERY: (
            r"\bdormant\b",
            r"\bactive\b",
            r"\bstale\b",
            r"\beligibility\b",
            r"\bforecast status\b",
            r"\bforecasted\b",
            r"\bnot forecasted\b",
            r"\bstatus\b",
            r"\bhow many\b.*\bactive\b",
            r"\bhow many\b.*\bdormant\b",
            r"\bshow\b.*\bdormant\b",
            r"\bshow\b.*\bactive\b",
            r"\blist\b.*\bdormant\b",
            r"\blist\b.*\bactive\b",
        ),

        # --------------------------------------------------------------
        # Overall forecast summary
        #
        # These patterns should represent a global question rather than
        # a medicine-specific forecast.
        # --------------------------------------------------------------
        ForecastIntent.FORECAST_SUMMARY: (
            r"\bforecast summary\b",
            r"\boverall forecast\b",
            r"\boverall summary\b",
            r"\bdemand summary\b",
            r"\bforecast overview\b",
            r"\bdemand overview\b",
            r"\boverview of\b.*\bforecast",
            r"\bsummary of\b.*\bforecast",
            r"\btotal predicted demand\b",
            r"\btotal forecast\b",
            r"\bcurrent forecast\b.*\bsummary\b",
            r"\bgive me\b.*\bsummary\b",
        ),

        # --------------------------------------------------------------
        # Generic forecast
        #
        # This intentionally has the lowest supported priority.
        # --------------------------------------------------------------
        ForecastIntent.FORECAST: (
            r"\bwhat is\b.*\bforecast\b",
            r"\bforecast\b",
            r"\bpredicted demand\b",
            r"\bprediction\b",
            r"\bdemand for\b",
            r"\bhow much\b.*\bdemand\b",
            r"\bexpected demand\b",
            r"\bnext month\b",
            r"\bnext\s+\d+\s+days?\b",
            r"\bhow much will\b.*\bdemand\b",
            r"\bexpected\b.*\bdemand\b",
        ),
    }

    def route(
        self,
        query: str,
    ) -> IntentResult:
        """
        Classify an employee query into a supported forecast intent.

        Parameters
        ----------
        query:
            Natural-language employee question.

        Returns
        -------
        IntentResult
            Normalized intent, deterministic confidence score, and
            the patterns that matched the selected intent.

        Notes
        -----
        The result only answers:

            "What kind of forecasting question is this?"

        It does not answer:

            "Which medicine does this refer to?"

        That separation is necessary for conversation follow-ups such as:

            User: "What is the forecast for Otacare?"
            User: "What is the P90 forecast?"

        The second question should route to FORECAST_RANGE and the
        conversation context can subsequently resolve "Otacare".
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string, "
                f"got {type(query).__name__}"
            )

        normalized = self._normalize(query)

        if not normalized:
            return IntentResult(
                intent=ForecastIntent.UNKNOWN,
                confidence=0.0,
                matched_patterns=(),
            )

        matches: dict[
            ForecastIntent,
            list[str],
        ] = {}

        for intent, patterns in self._PATTERNS.items():
            intent_matches: list[str] = []

            for pattern in patterns:
                if re.search(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    intent_matches.append(pattern)

            if intent_matches:
                matches[intent] = intent_matches

        if not matches:
            return IntentResult(
                intent=ForecastIntent.UNKNOWN,
                confidence=0.0,
                matched_patterns=(),
            )

        best_intent, matched_patterns = max(
            matches.items(),
            key=lambda item: (
                self._intent_priority(item[0]),
                len(item[1]),
            ),
        )

        confidence = self._calculate_confidence(
            intent=best_intent,
            match_count=len(matched_patterns),
        )

        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            matched_patterns=tuple(matched_patterns),
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        query: str,
    ) -> str:
        """Normalize whitespace and casing."""

        normalized = re.sub(
            r"\s+",
            " ",
            query.strip().lower(),
        )

        return normalized

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        intent: ForecastIntent,
        match_count: int,
    ) -> float:
        """
        Produce a deterministic confidence estimate.

        Confidence represents pattern evidence, not statistical model
        probability.
        """

        if intent == ForecastIntent.UNKNOWN:
            return 0.0

        if match_count <= 0:
            return 0.0

        confidence = 0.70 + (
            0.05 * (match_count - 1)
        )

        return min(
            1.0,
            round(confidence, 2),
        )

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    @staticmethod
    def _intent_priority(
        intent: ForecastIntent,
    ) -> int:
        """
        Resolve overlapping intent matches.

        More specific intents take precedence over generic forecast
        detection.

        Example:

            "What is the P90 forecast?"

        matches both:

            FORECAST_RANGE -> "p90"
            FORECAST       -> "forecast"

        FORECAST_RANGE must win.
        """

        priorities = {
            ForecastIntent.ROUTING_EXPLANATION: 100,
            ForecastIntent.MODEL_INFO: 90,
            ForecastIntent.FORECAST_RANGE: 80,
            ForecastIntent.RANKING: 70,
            ForecastIntent.STATUS_QUERY: 60,
            ForecastIntent.FORECAST_SUMMARY: 50,
            ForecastIntent.FORECAST: 40,
            ForecastIntent.UNKNOWN: 0,
        }

        return priorities[intent]