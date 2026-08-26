from __future__ import annotations

from typing import Any, Dict

from finemed_ai.demand_forecasting.store import (
    ForecastNotFoundError,
    ForecastStore,
)


# ============================================================================
# Anthropic tool schemas
# ============================================================================

TOOL_SCHEMAS = [
    {
        "name": "get_forecast",
        "description": (
            "Get the raw day-by-day demand forecast for one medicine, "
            "including P10-P90 uncertainty quantiles. Use this when the "
            "employee asks for specific forecast values, a specific date, "
            "or the full forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_id": {
                    "type": "string",
                    "description": (
                        "The medicine code (MDCODE), "
                        "for example '42'."
                    ),
                }
            },
            "required": [
                "medicine_id"
            ],
        },
    },
    {
        "name": "get_trend",
        "description": (
            "Get the demand trend direction and percentage change for "
            "a medicine over its forecast horizon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_id": {
                    "type": "string",
                    "description": (
                        "The medicine code (MDCODE)."
                    ),
                }
            },
            "required": [
                "medicine_id"
            ],
        },
    },
    {
        "name": "get_summary",
        "description": (
            "Get a summary of a medicine's forecast including total "
            "expected demand, average daily demand, peak, trough, "
            "and trend information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_id": {
                    "type": "string",
                    "description": (
                        "The medicine code (MDCODE)."
                    ),
                }
            },
            "required": [
                "medicine_id"
            ],
        },
    },
    {
        "name": "get_top_demand_medicines",
        "description": (
            "Get the medicines with the highest total predicted demand "
            "over the forecast horizon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": (
                        "Number of medicines to return. "
                        "Default is 10."
                    ),
                }
            },
        },
    },
    {
        "name": "get_demand_trend_medicines",
        "description": (
            "Get medicines matching a demand trend direction such as "
            "increasing, decreasing, stable, or flat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trend": {
                    "type": "string",
                    "enum": [
                        "increasing",
                        "decreasing",
                        "stable",
                        "flat",
                    ],
                    "description": (
                        "Trend direction to filter by."
                    ),
                },
                "n": {
                    "type": "integer",
                    "description": (
                        "Number of medicines to return. "
                        "Default is 10."
                    ),
                },
            },
            "required": [
                "trend"
            ],
        },
    },
    {
        "name": "get_most_uncertain_medicines",
        "description": (
            "Get medicines whose forecasts have the highest relative "
            "P10-P90 uncertainty spread."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": (
                        "Number of medicines to return. "
                        "Default is 10."
                    ),
                }
            },
        },
    },
    {
        "name": "compare_medicines",
        "description": (
            "Compare forecasts for two or more specific medicines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "medicine_ids": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": (
                        "Medicine codes to compare."
                    ),
                }
            },
            "required": [
                "medicine_ids"
            ],
        },
    },
]


# ============================================================================
# Forecast tools
# ============================================================================

class ForecastTools:
    """
    Binds LLM tool calls to a live ForecastStore.

    All expected lookup and validation failures are converted into
    JSON-serializable error responses instead of allowing exceptions
    to escape into the orchestrator.
    """

    def __init__(
        self,
        store: ForecastStore,
    ) -> None:

        self.store = store

    def execute(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Dict[str, Any]:

        medicine_id = str(
            tool_input.get(
                "medicine_id",
                "",
            )
        ).strip()

        try:

            # ==========================================================
            # Full forecast
            # ==========================================================

            if tool_name == "get_forecast":

                result = self.store.get(
                    medicine_id
                )

                return {
                    "medicine_id": (
                        result.medicine_id
                    ),
                    "days": [
                        {
                            "date": str(
                                day.forecast_date
                            ),
                            "predicted_demand": (
                                day.predicted_demand
                            ),
                            "p10": (
                                day.quantiles.p10
                            ),
                            "p20": (
                                day.quantiles.p20
                            ),
                            "p30": (
                                day.quantiles.p30
                            ),
                            "p40": (
                                day.quantiles.p40
                            ),
                            "p50": (
                                day.quantiles.p50
                            ),
                            "p60": (
                                day.quantiles.p60
                            ),
                            "p70": (
                                day.quantiles.p70
                            ),
                            "p80": (
                                day.quantiles.p80
                            ),
                            "p90": (
                                day.quantiles.p90
                            ),
                        }
                        for day in result.days
                    ],
                }

            # ==========================================================
            # Trend
            # ==========================================================

            if tool_name == "get_trend":

                result = self.store.get(
                    medicine_id
                )

                summary = result.to_summary()

                return {
                    "medicine_id": (
                        result.medicine_id
                    ),
                    "trend": (
                        summary.trend
                    ),
                    "trend_pct_change": (
                        summary.trend_pct_change
                    ),
                }

            # ==========================================================
            # Summary
            # ==========================================================

            if tool_name == "get_summary":

                result = self.store.get(
                    medicine_id
                )

                summary = result.to_summary()

                return summary.model_dump(
                    mode="json"
                )

            # ==========================================================
            # Top demand
            # ==========================================================

            if (
                tool_name
                == "get_top_demand_medicines"
            ):

                n = int(
                    tool_input.get(
                        "n",
                        10,
                    )
                )

                summaries = (
                    self.store.get_top_demand(
                        n=n
                    )
                )

                return {
                    "medicines": [
                        summary.model_dump(
                            mode="json"
                        )
                        for summary in summaries
                    ]
                }

            # ==========================================================
            # Trend filtering
            # ==========================================================

            if (
                tool_name
                == "get_demand_trend_medicines"
            ):

                trend = str(
                    tool_input.get(
                        "trend",
                        "increasing",
                    )
                ).strip().lower()

                n = int(
                    tool_input.get(
                        "n",
                        10,
                    )
                )

                summaries = (
                    self.store.get_by_trend(
                        trend,
                        n=n,
                    )
                )

                return {
                    "medicines": [
                        summary.model_dump(
                            mode="json"
                        )
                        for summary in summaries
                    ]
                }

            # ==========================================================
            # Uncertainty
            # ==========================================================

            if (
                tool_name
                == "get_most_uncertain_medicines"
            ):

                n = int(
                    tool_input.get(
                        "n",
                        10,
                    )
                )

                medicines = (
                    self.store.get_most_uncertain(
                        n=n
                    )
                )

                return {
                    "medicines": medicines
                }

            # ==========================================================
            # Comparison
            # ==========================================================

            if (
                tool_name
                == "compare_medicines"
            ):

                medicine_ids = (
                    tool_input.get(
                        "medicine_ids",
                        [],
                    )
                )

                if not isinstance(
                    medicine_ids,
                    list,
                ):
                    raise ValueError(
                        "medicine_ids must be a list"
                    )

                medicine_ids = [
                    str(
                        medicine_id
                    ).strip()
                    for medicine_id
                    in medicine_ids
                    if str(
                        medicine_id
                    ).strip()
                ]

                if not medicine_ids:
                    raise ValueError(
                        "At least one medicine ID "
                        "must be provided."
                    )

                summaries = (
                    self.store.compare(
                        medicine_ids
                    )
                )

                return {
                    "medicines": [
                        summary.model_dump(
                            mode="json"
                        )
                        for summary in summaries
                    ]
                }

            # ==========================================================
            # Unknown tool
            # ==========================================================

            return {
                "error": (
                    f"Unknown tool: {tool_name}"
                )
            }

        except ForecastNotFoundError:

            return {
                "error": (
                    f"No forecast available for "
                    f"medicine_id={medicine_id}. "
                    "The medicine ID may be invalid, "
                    "may not exist, or may not yet have "
                    "sufficient forecasting history."
                )
            }

        except ValueError as exc:

            return {
                "error": (
                    "Invalid forecast tool input: "
                    f"{str(exc)}"
                )
            }

        except Exception as exc:

            return {
                "error": (
                    "Forecast tool execution failed: "
                    f"{type(exc).__name__}"
                )
            }