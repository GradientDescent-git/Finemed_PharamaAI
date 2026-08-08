from dataclasses import dataclass, field
from typing import Tuple
 
 
@dataclass(frozen=True)
class ForecastConfig:
    model_id: str = "amazon/chronos-2"
 
    # Validated production values — see module docstring for the backtest
    # that produced these. Do not change without re-running the eval suite.
    context_length: int = 730
    prediction_length: int = 30
    quantile_levels: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    point_quantile: float = 0.5          # P50 point forecast
    apply_bias_correction: bool = False  # empirically worsens WAPE, kept off
 
    # Chronos dataframe schema (matches your prepared_history / chronos_df
    # column names throughout the notebook)
    id_column: str = "item_id"
    timestamp_column: str = "timestamp"
    target_column: str = "target"
 
    # Output schema (matches baseline_forecast_df / evaluation_df conventions)
    output_id_column: str = "Medicine_ID"
    output_date_column: str = "Forecast_Date"
    output_point_column: str = "Predicted_Demand"
 
 
DEFAULT_CONFIG = ForecastConfig()
 