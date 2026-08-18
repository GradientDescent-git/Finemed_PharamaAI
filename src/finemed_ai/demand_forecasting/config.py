from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ForecastConfig:
    """
    Frozen production configuration for demand forecasting.

    These values were selected through the forecasting validation/
    optimization work and must not be changed without re-running
    the evaluation suite.

    Current production configuration:

        Model:
            amazon/chronos-2

        Context:
            730 daily observations

        Horizon:
            30 days

        Point forecast:
            P50

        Bias correction:
            Disabled because validation showed worse WAPE.

    Chronos input schema:

        item_id
        timestamp
        target

    Production output schema:

        Medicine_ID
        Forecast_Date
        Predicted_Demand
    """

    model_id: str = "amazon/chronos-2"

    # ------------------------------------------------------------------
    # Validated production forecasting configuration
    # ------------------------------------------------------------------

    context_length: int = 730
    prediction_length: int = 30

    quantile_levels: Tuple[
        float, ...
    ] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )

    point_quantile: float = 0.5

    # Validation showed that bias correction worsened WAPE.
    apply_bias_correction: bool = False

    # ------------------------------------------------------------------
    # Chronos input schema
    # ------------------------------------------------------------------

    id_column: str = "item_id"
    timestamp_column: str = "timestamp"
    target_column: str = "target"

    # ------------------------------------------------------------------
    # Production output schema
    # ------------------------------------------------------------------

    output_id_column: str = "Medicine_ID"
    output_date_column: str = "Forecast_Date"
    output_point_column: str = "Predicted_Demand"


DEFAULT_CONFIG = ForecastConfig()