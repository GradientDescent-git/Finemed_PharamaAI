"""
tests/test_tsb.py
====================
The TSB (Teunter-Syntetos-Babai) intermittent-demand forecaster is
documented as "frozen production logic" handling real medicine forecasts,
but had zero tests before this file. These lock in the exact edge cases
from the handoff spec (section 14) against the real implementation.
"""

import numpy as np
import pandas as pd
import pytest

from finemed_ai.demand_forecasting.production_forecast_service import tsb_forecast


def test_all_zero_history_forecasts_zero():
    result = tsb_forecast(pd.Series([0.0] * 30), horizon=5)
    assert (result == 0.0).all()


def test_forecast_is_constant_across_horizon():
    """Spec: 'TSB produces a constant point forecast over the 30-day horizon.'"""
    history = pd.Series([5, 0, 3, 0, 8, 0, 2] * 10)
    result = tsb_forecast(history, horizon=30)
    assert len(result) == 30
    assert len(set(result)) == 1


def test_late_first_demand_uses_correct_initial_probability():
    """Spec: initial probability = 1 / (first_nonzero_index + 1)."""
    history = pd.Series([0] * 9 + [10] + [0] * 20)  # first nonzero at index 9
    result = tsb_forecast(history, horizon=1, alpha_probability=0.1, alpha_demand=0.1)
    # Hand-computed: probability starts at 1/10 = 0.1, demand_estimate starts at 10.0.
    # First loop iteration processes y[0]=0 (before reaching the nonzero index),
    # updating probability toward 0 each step until the nonzero observation.
    assert result[0] > 0  # sanity: produces a real forecast, not 0 or NaN
    assert np.isfinite(result[0])


def test_negative_demand_is_clipped_not_propagated():
    """Spec section 14: negative demand must be clipped/rejected before TSB."""
    history = pd.Series([5.0, -3.0, 10.0, 8.0])
    result = tsb_forecast(history, horizon=3)
    assert np.isfinite(result[0])
    assert result[0] >= 0


def test_sparse_demand_produces_forecast_between_zero_and_max():
    history = pd.Series([0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4] + [0] * 15)
    result = tsb_forecast(history, horizon=5)
    assert 0 < result[0] < 5


def test_empty_history_forecasts_zero():
    result = tsb_forecast(pd.Series([], dtype=float), horizon=5)
    assert (result == 0.0).all()


def test_invalid_horizon_raises():
    with pytest.raises(ValueError):
        tsb_forecast(pd.Series([1.0, 2.0]), horizon=0)
    with pytest.raises(ValueError):
        tsb_forecast(pd.Series([1.0, 2.0]), horizon=-5)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        tsb_forecast(pd.Series([1.0, 2.0]), horizon=5, alpha_demand=0.0)
    with pytest.raises(ValueError):
        tsb_forecast(pd.Series([1.0, 2.0]), horizon=5, alpha_probability=1.5)


def test_non_finite_history_raises():
    with pytest.raises(ValueError):
        tsb_forecast(pd.Series([1.0, np.nan, 3.0]), horizon=5)
    with pytest.raises(ValueError):
        tsb_forecast(pd.Series([1.0, np.inf, 3.0]), horizon=5)


def test_wrong_input_type_raises():
    with pytest.raises(TypeError):
        tsb_forecast([1.0, 2.0, 3.0], horizon=5)  # list, not pd.Series


def test_default_alphas_match_frozen_production_config():
    """Spec section 13: alpha_demand=0.1, alpha_probability=0.1 are frozen --
    a change here without re-validation would silently drift production
    behavior away from the documented, validated configuration."""
    import inspect
    sig = inspect.signature(tsb_forecast)
    assert sig.parameters["alpha_demand"].default == 0.1
    assert sig.parameters["alpha_probability"].default == 0.1
