from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.run_monthly_pipeline import main


@patch("scripts.run_monthly_pipeline.MonthlyPipeline")
@patch("scripts.run_monthly_pipeline.Settings")
def test_main_success(
    mock_settings,
    mock_monthly_pipeline,
):
    manifest = MagicMock()
    manifest.run_id = "test_run_001"
    manifest.published = True
    manifest.output_path = "test/forecast.parquet"

    evaluation = MagicMock()
    evaluation.overall_wape_pct = 12.5

    alerts = MagicMock()
    alerts.total_alerts = 2
    alerts.critical_count = 0
    alerts.warning_count = 1

    pipeline_instance = mock_monthly_pipeline.return_value

    pipeline_instance.run.return_value = {
        "manifest": manifest,
        "evaluation": evaluation,
        "alerts": alerts,
    }

    exit_code = main()

    assert exit_code == 0

    mock_settings.assert_called_once()

    mock_monthly_pipeline.assert_called_once_with(
        mock_settings.return_value
    )

    pipeline_instance.run.assert_called_once()


@patch("scripts.run_monthly_pipeline.MonthlyPipeline")
@patch("scripts.run_monthly_pipeline.Settings")
def test_main_returns_one_when_forecast_not_published(
    mock_settings,
    mock_monthly_pipeline,
):
    manifest = MagicMock()
    manifest.run_id = "test_run_002"
    manifest.published = False
    manifest.output_path = "test/forecast.parquet"
    manifest.publish_note = "Publication gate failed."

    pipeline_instance = mock_monthly_pipeline.return_value

    pipeline_instance.run.return_value = {
        "manifest": manifest,
        "evaluation": None,
        "alerts": None,
    }

    exit_code = main()

    assert exit_code == 1

    pipeline_instance.run.assert_called_once()


@patch("scripts.run_monthly_pipeline.MonthlyPipeline")
@patch("scripts.run_monthly_pipeline.Settings")
def test_main_returns_one_when_pipeline_raises(
    mock_settings,
    mock_monthly_pipeline,
):
    pipeline_instance = mock_monthly_pipeline.return_value

    pipeline_instance.run.side_effect = RuntimeError(
        "Unexpected pipeline failure"
    )

    exit_code = main()

    assert exit_code == 1

    pipeline_instance.run.assert_called_once()