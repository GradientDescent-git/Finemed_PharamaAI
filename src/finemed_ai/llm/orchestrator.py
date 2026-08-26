from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from finemed_ai.llm.tools import TOOL_SCHEMAS, ForecastTools


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are the FineMed Pharma demand forecasting assistant.

Employees ask questions about medicine demand forecasts, trends, rankings,
uncertainty, and comparisons.

Always use the available tools for any factual forecast information.
Never invent or calculate a forecast number, medicine ID, date, trend,
ranking, or uncertainty value yourself.

Available tools cover:
- single-medicine forecasts
- forecast trends
- forecast summaries
- top-demand medicine rankings
- trend-based medicine filtering
- forecast uncertainty rankings
- comparisons between medicines

Use whichever tool or combination of tools matches the employee's actual
question.

Rules:

- Always call a tool before stating any forecast number, trend, ranking,
  comparison, medicine-specific forecast value, or uncertainty value.

- Never invent a medicine ID, quantity, date, forecast value, ranking,
  trend, or uncertainty range.

- If a tool returns an error, explain the problem plainly. For an unknown
  medicine, ask the employee to check the medicine code. Do not invent data.

- Forecasts are 30-day-ahead demand predictions, not guarantees.

- If someone asks for a purchasing, stocking, or inventory decision, provide
  the forecast-based information and uncertainty range when available.
  Clearly state that the forecast is decision support and not a guarantee.

- For a single medicine, keep the response concrete and include, when
  available:
  Medicine
  Expected demand
  Forecast horizon
  P10
  P50
  P90
  A short interpretation of the uncertainty range

- For ranking questions, present a short ranked list where the available
  tool data supports it.

- If the requested information is not available from the tools, say so
  plainly.

- Keep answers concise and operational. Employees want the result and
  takeaway, not a long explanation of time-series modeling.
"""


class Orchestrator:
    """
    Coordinates Anthropic tool use with the FineMed forecasting tools.

    The orchestrator:
    1. Sends the employee question to Claude.
    2. Executes any requested forecast tools.
    3. Returns tool results to Claude as valid JSON.
    4. Repeats until Claude produces a final answer or the iteration
       safety limit is reached.
    """

    def __init__(
        self,
        tools: ForecastTools,
        model: str = "claude-sonnet-5",
        api_key: Optional[str] = None,
        max_tool_iterations: int = 4,
    ) -> None:
        import anthropic

        if max_tool_iterations < 1:
            raise ValueError(
                "max_tool_iterations must be at least 1."
            )

        self.tools = tools
        self.model = model
        self.max_tool_iterations = max_tool_iterations

        resolved_api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )

        self.client = anthropic.Anthropic(
            api_key=resolved_api_key
        )

    def ask(
        self,
        question: str,
        conversation_history: Optional[
            List[Dict[str, Any]]
        ] = None,
        history: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> str:
        """
        Answer one employee question using the forecast tool-use loop.

        Parameters
        ----------
        question:
            The current employee question.

        conversation_history:
            Optional prior turns in Anthropic message format.

        history:
            Backward-compatible alias for conversation_history.

        Returns
        -------
        str
            Final natural-language answer.
        """

        normalized_question = (
            question.strip()
            if isinstance(question, str)
            else ""
        )

        if not normalized_question:
            return (
                "Please enter a question about a medicine "
                "or demand forecast."
            )

        if (
            conversation_history is not None
            and history is not None
        ):
            logger.warning(
                "Both conversation_history and history were provided. "
                "Using conversation_history."
            )

        prior_history = (
            conversation_history
            if conversation_history is not None
            else history
        )

        messages: List[Dict[str, Any]] = list(
            prior_history or []
        )

        messages.append(
            {
                "role": "user",
                "content": normalized_question,
            }
        )

        for iteration in range(
            self.max_tool_iterations
        ):
            try:
                response = (
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system=SYSTEM_PROMPT,
                        tools=TOOL_SCHEMAS,
                        messages=messages,
                    )
                )

            except Exception as exc:
                logger.exception(
                    "Claude API call failed on iteration %d.",
                    iteration + 1,
                )

                return (
                    "Sorry, I could not reach the AI service right now. "
                    "Please try again shortly. If the problem persists, "
                    "the server configuration or AI service may need "
                    "attention."
                )

            if response.stop_reason != "tool_use":
                return self._extract_text(
                    response
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            tool_results: List[
                Dict[str, Any]
            ] = []

            tool_calls_found = False

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_calls_found = True

                logger.info(
                    "Tool call: %s(%s)",
                    block.name,
                    block.input,
                )

                try:
                    result = self.tools.execute(
                        block.name,
                        block.input,
                    )

                except Exception:
                    logger.exception(
                        "Unexpected failure while executing tool %s.",
                        block.name,
                    )

                    result = {
                        "error": (
                            "The forecast data service encountered "
                            "an unexpected error."
                        )
                    }

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(
                            result,
                            default=str,
                            ensure_ascii=False,
                        ),
                    }
                )

            if not tool_calls_found:
                logger.warning(
                    "Claude returned stop_reason='tool_use' "
                    "but no tool_use blocks were found."
                )

                return self._extract_text(
                    response
                )

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        logger.warning(
            "Hit max_tool_iterations=%d without a final answer.",
            self.max_tool_iterations,
        )

        return (
            "I wasn't able to fully resolve that question. "
            "Please try rephrasing it or ask about one medicine "
            "at a time."
        )

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:
        """
        Extract all text blocks from an Anthropic response.

        Returning all text blocks is safer than returning only the first
        one because a response may contain more than one text block.
        """

        text_parts = [
            block.text
            for block in response.content
            if (
                getattr(
                    block,
                    "type",
                    None,
                )
                == "text"
                and getattr(
                    block,
                    "text",
                    "",
                )
            )
        ]

        if text_parts:
            return "\n".join(
                text_parts
            )

        return (
            "I don't have a response for that question."
        )