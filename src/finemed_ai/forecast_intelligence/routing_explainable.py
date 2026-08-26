from __future__ import annotations

from typing import Any


class RoutingExplainabilityService:
    """
    Converts deterministic model-routing evidence into structured,
    employee-readable explanations.

    This layer does not perform model selection. It only explains the
    routing decision already produced by the forecasting pipeline.
    """

    CHRONOS_ADVANTAGE_THRESHOLD = 30.0

    def explain(self, routing_data: dict[str, Any]) -> dict[str, Any]:
        """
        Build a structured explanation from routing evidence.

        Expected routing_data keys:
            medicine_id
            medicine_name
            selected_model
            chronos_absolute_error
            tsb_absolute_error
            validation_windows
            validation_advantage_pct
            routing_reason
        """

        if not routing_data:
            return {
                "found": False,
                "explanation": (
                    "No routing information is available for this medicine."
                ),
            }

        selected_model = routing_data.get("selected_model")
        chronos_ae = routing_data.get("chronos_absolute_error")
        tsb_ae = routing_data.get("tsb_absolute_error")
        validation_windows = routing_data.get("validation_windows")
        advantage_pct = routing_data.get("validation_advantage_pct")
        routing_reason = routing_data.get("routing_reason")

        model_explanation = self._build_model_explanation(
            selected_model=selected_model,
            chronos_ae=chronos_ae,
            tsb_ae=tsb_ae,
            validation_windows=validation_windows,
            advantage_pct=advantage_pct,
        )

        reason_explanation = self._translate_reason(
            routing_reason=routing_reason,
            selected_model=selected_model,
        )

        return {
            "found": True,
            "medicine_id": routing_data.get("medicine_id"),
            "medicine_name": routing_data.get("medicine_name"),
            "selected_model": selected_model,
            "validation_windows": validation_windows,
            "chronos_absolute_error": chronos_ae,
            "tsb_absolute_error": tsb_ae,
            "validation_advantage_pct": advantage_pct,
            "routing_reason": routing_reason,
            "reason_explanation": reason_explanation,
            "model_explanation": model_explanation,
        }

    def _build_model_explanation(
        self,
        selected_model: str | None,
        chronos_ae: float | None,
        tsb_ae: float | None,
        validation_windows: int | None,
        advantage_pct: float | None,
    ) -> str:
        """Create a factual explanation of the selected model."""

        windows_text = (
            f"across {validation_windows} locked validation windows"
            if validation_windows
            else "using the available validation evidence"
        )

        if selected_model == "tsb":

            if (
                chronos_ae is not None
                and tsb_ae is not None
                and tsb_ae < chronos_ae
            ):
                comparison = (
                    f"TSB produced lower validation error "
                    f"({tsb_ae:.2f}) than Chronos-2 "
                    f"({chronos_ae:.2f})"
                )
            else:
                comparison = (
                    "Chronos-2 did not demonstrate sufficient validation "
                    "improvement over TSB"
                )

            if advantage_pct is not None:
                threshold_text = (
                    f"Chronos-2 achieved a validation advantage of "
                    f"{advantage_pct:.2f}%, which did not meet the "
                    f"required {self.CHRONOS_ADVANTAGE_THRESHOLD:.0f}% "
                    f"threshold"
                )
            else:
                threshold_text = (
                    "Chronos-2 did not meet the required validation "
                    "threshold for selection"
                )

            return (
                f"TSB was selected because {comparison} {windows_text}. "
                f"{threshold_text}."
            )

        if selected_model == "chronos-2-P50":

            if advantage_pct is not None:
                advantage_text = (
                    f"Chronos-2 achieved a {advantage_pct:.2f}% "
                    f"validation advantage over TSB"
                )
            else:
                advantage_text = (
                    "Chronos-2 demonstrated sufficient validation "
                    "advantage over TSB"
                )

            return (
                f"Chronos-2 P50 was selected because {advantage_text} "
                f"{windows_text}, meeting the required "
                f"{self.CHRONOS_ADVANTAGE_THRESHOLD:.0f}% selection "
                f"threshold."
            )

        return (
            f"The forecasting pipeline selected "
            f"{selected_model or 'the configured model'} based on the "
            f"locked validation routing rules."
        )

    def _translate_reason(
        self,
        routing_reason: str | None,
        selected_model: str | None,
    ) -> str:
        """Translate internal routing rules into readable language."""

        if (
            routing_reason
            == "chronos_validation_advantage_below_30pct_threshold"
        ):
            return (
                "Chronos-2 did not achieve the required 30% validation "
                "advantage over TSB, so TSB was selected."
            )

        if routing_reason:
            return (
                f"The model was selected according to the production "
                f"routing rule: {routing_reason}."
            )

        if selected_model:
            return (
                f"{selected_model} was selected according to the "
                "production validation routing process."
            )

        return "No detailed routing reason is available."