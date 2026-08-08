from __future__ import annotations
 
from typing import Any, Dict
 
from finemed_ai.forecasting.store import ForecastNotFoundError, ForecastStore
 
# ---------------------------------------------------------------------------
# Anthropic tool schemas (JSON schema, per Claude tool-use spec)
# ---------------------------------------------------------------------------
 
TOOL_SCHEMAS = [
    {
        "name": "get_forecast",
        "description": (
            "Get the raw day-by-day 30-day demand forecast for one medicine, "
            "including P10-P90 uncertainty quantiles. Use this when the "
            "employee asks for specific numbers, a specific date's forecast, "
            "or the full forecast table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_id": {
                    "type": "string",
                    "description": "The medicine code (MDCODE), e.g. '42'.",
                }
            },
            "required": ["medicine_id"],
        },
    },
    {
        "name": "get_trend",
        "description": (
            "Get the demand trend direction (increasing/decreasing/stable/flat) "
            "and percent change for a medicine over its 30-day forecast window. "
            "Use this when the employee asks whether demand is going up, down, "
            "or is expected to change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_id": {"type": "string", "description": "The medicine code (MDCODE)."}
            },
            "required": ["medicine_id"],
        },
    },
    {
        "name": "get_summary",
        "description": (
            "Get a plain-language summary of a medicine's forecast: total "
            "expected demand, average daily demand, peak and trough days. "
            "Use this for general 'how is X looking' or 'summarize the "
            "forecast for X' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_id": {"type": "string", "description": "The medicine code (MDCODE)."}
            },
            "required": ["medicine_id"],
        },
    },
]
 
 
class ForecastTools:
    """Binds tool schemas to a live ForecastStore. One instance per app;
    give it to Orchestrator."""
 
    def __init__(self, store: ForecastStore):
        self.store = store
 
    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call by name. Returns a JSON-serializable dict —
        callers (orchestrator) send this back to Claude as a tool_result."""
        medicine_id = tool_input.get("medicine_id", "")
 
        try:
            if tool_name == "get_forecast":
                result = self.store.get(medicine_id)
                return {
                    "medicine_id": result.medicine_id,
                    "days": [
                        {
                            "date": str(d.forecast_date),
                            "predicted_demand": d.predicted_demand,
                            "p10": d.quantiles.p10,
                            "p90": d.quantiles.p90,
                        }
                        for d in result.days
                    ],
                }
 
            elif tool_name == "get_trend":
                result = self.store.get(medicine_id)
                summary = result.to_summary()
                return {
                    "medicine_id": medicine_id,
                    "trend": summary.trend,
                    "trend_pct_change": summary.trend_pct_change,
                }
 
            elif tool_name == "get_summary":
                result = self.store.get(medicine_id)
                summary = result.to_summary()
                return summary.model_dump(mode="json")
 
            else:
                return {"error": f"Unknown tool: {tool_name}"}
 
        except ForecastNotFoundError:
            return {
                "error": (
                    f"No forecast available for medicine_id={medicine_id}. "
                    f"It may be a new product with insufficient history, or "
                    f"the ID may be wrong."
                )
            }
 