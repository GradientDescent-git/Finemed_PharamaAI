from __future__ import annotations
 
import logging
import os
from typing import List, Optional
 
from finemed_ai.llm.tools import TOOL_SCHEMAS, ForecastTools
 
logger = logging.getLogger(__name__)
 
SYSTEM_PROMPT = """\
You are the FineMed Pharma demand forecasting assistant. Employees ask you \
questions about medicine demand forecasts, and you answer using the \
get_forecast, get_trend, and get_summary tools — never guess numbers.
 
Rules:
- Always call a tool before stating any forecast number, trend, or summary.
- If a tool returns an error (e.g. unknown medicine ID), tell the employee \
plainly and suggest they check the medicine code — do not make up a number.
- Forecasts are 30-day-ahead demand predictions from a statistical model, \
not guarantees. If someone asks you to make a purchasing/stocking decision, \
give the forecast-based numbers plus the uncertainty range (P10-P90), and \
note it's an input to their decision, not a substitute for it.
- Keep answers short and concrete — employees want the number and the \
takeaway, not a lecture on time-series modeling.
"""
 
 
class Orchestrator:
    def __init__(
        self,
        tools: ForecastTools,
        model: str = "claude-sonnet-5",
        api_key: Optional[str] = None,
        max_tool_iterations: int = 4,
    ):
        import anthropic  # deferred import
 
        self.tools = tools
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
 
    def ask(self, question: str, conversation_history: Optional[List[dict]] = None) -> str:
        """
        Answer one employee question, running the tool-use loop to
        completion. Returns the final natural-language answer.
 
        conversation_history: optional prior turns in Anthropic message
        format, for multi-turn /chat sessions.
        """
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": question})
 
        for _ in range(self.max_tool_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
 
            if response.stop_reason != "tool_use":
                return self._extract_text(response)
 
            messages.append({"role": "assistant", "content": response.content})
 
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                logger.info("Tool call: %s(%s)", block.name, block.input)
                result = self.tools.execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
 
            messages.append({"role": "user", "content": tool_results})
 
        logger.warning("Hit max_tool_iterations=%d without a final answer", self.max_tool_iterations)
        return (
            "I wasn't able to fully resolve that question — could you rephrase it, "
            "or ask about one medicine at a time?"
        )
 
    @staticmethod
    def _extract_text(response) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text
        return "I don't have a response for that."
 