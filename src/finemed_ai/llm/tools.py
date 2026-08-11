from __future__ import annotations
 
from typing import Any, Dict
 
from finemed_ai.demand_forecasting.store import ForecastNotFoundError, ForecastStore
 
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
    {
        "name": "get_top_demand_medicines",
        "description": (
            "Get the medicines with the HIGHEST total predicted demand over "
            "the forecast horizon, ranked highest first. Use this for "
            "'which medicines will have the highest demand', 'top selling "
            "medicines next month', or similar ranking questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many top medicines to return. Default 10."}
            },
        },
    },
    {
        "name": "get_demand_trend_medicines",
        "description": (
            "Get medicines matching a specific trend direction (increasing, "
            "decreasing, or stable demand), ranked by size of change. Use "
            "this for 'which medicines are trending up/down' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trend": {
                    "type": "string",
                    "enum": ["increasing", "decreasing", "stable", "flat"],
                    "description": "Which trend direction to filter for.",
                },
                "n": {"type": "integer", "description": "How many medicines to return. Default 10."},
            },
            "required": ["trend"],
        },
    },
    {
        "name": "get_most_uncertain_medicines",
        "description": (
            "Get the medicines whose forecasts have the WIDEST uncertainty "
            "range (P10-P90 spread relative to the central forecast) -- "
            "these are the forecasts to trust least and may need manual "
            "review. Use this for 'which forecasts are least reliable' or "
            "'which medicines have uncertain demand' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many medicines to return. Default 10."}
            },
        },
    },
    {
        "name": "compare_medicines",
        "description": (
            "Compare the forecasts of two or more specific medicines "
            "side by side. Use this when the employee names multiple "
            "medicine codes and asks to compare them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The medicine codes to compare, e.g. ['0001', '0042'].",
                }
            },
            "required": ["medicine_ids"],
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

            elif tool_name == "get_top_demand_medicines":
                n = tool_input.get("n", 10)
                summaries = self.store.get_top_demand(n=n)
                return {"medicines": [s.model_dump(mode="json") for s in summaries]}

            elif tool_name == "get_demand_trend_medicines":
                trend = tool_input.get("trend", "increasing")
                n = tool_input.get("n", 10)
                summaries = self.store.get_by_trend(trend, n=n)
                return {"medicines": [s.model_dump(mode="json") for s in summaries]}

            elif tool_name == "get_most_uncertain_medicines":
                n = tool_input.get("n", 10)
                return {"medicines": self.store.get_most_uncertain(n=n)}

            elif tool_name == "compare_medicines":
                ids = tool_input.get("medicine_ids", [])
                summaries = self.store.compare(ids)
                return {"medicines": [s.model_dump(mode="json") for s in summaries]}

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
 