from __future__ import annotations
 
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
 
import pandas as pd
 
from finemed_ai.demand_forecasting.config import DEFAULT_CONFIG, ForecastConfig
from finemed_ai.demand_forecasting.predictor_service import PredictorService
from finemed_ai.demand_forecasting.schemas import BatchForecastRunResult
 
logger = logging.getLogger(__name__)
 
 
def _load_and_prepare_daily_demand(
    silver_demand_path: Path,
    id_col: str = "MDCODE",
    date_col: str = "INVDT",
    qty_col: str = "Demand_Qty",
) -> pd.DataFrame:
    """
    Reproduces notebook Module 2 (Daily Timeseries Generation):
    fills a complete daily calendar per medicine so Chronos never sees gaps.
 
    Reads parquet or csv transparently based on suffix.
    """
    if silver_demand_path.suffix == ".parquet":
        raw = pd.read_parquet(silver_demand_path)
    else:
        raw = pd.read_csv(silver_demand_path)
 
    missing = {id_col, date_col, qty_col} - set(raw.columns)
    if missing:
        raise ValueError(
            f"Silver demand source missing columns {missing}. "
            f"Update _load_and_prepare_daily_demand()'s column mapping to match "
            f"your actual silver schema."
        )
 
    base = raw[[id_col, date_col, qty_col]].copy()
    base[date_col] = pd.to_datetime(base[date_col])
    base = base.sort_values([id_col, date_col]).reset_index(drop=True)
 
    daily_frames = []
    for medicine_id, med_df in base.groupby(id_col):
        calendar = pd.DataFrame({
            date_col: pd.date_range(med_df[date_col].min(), med_df[date_col].max(), freq="D")
        })
        calendar[id_col] = medicine_id
 
        merged = calendar.merge(med_df, on=[id_col, date_col], how="left")
        merged[qty_col] = merged[qty_col].fillna(0)
        daily_frames.append(merged)
 
    daily = pd.concat(daily_frames, ignore_index=True)
 
    chronos_df = daily.rename(columns={
        id_col: "item_id",
        date_col: "timestamp",
        qty_col: "target",
    })
    chronos_df["item_id"] = chronos_df["item_id"].astype(str)
    return chronos_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
 
 
def run_monthly_forecast(
    silver_demand_path: Path,
    output_dir: Path,
    config: ForecastConfig = DEFAULT_CONFIG,
    predictor: Optional[PredictorService] = None,
) -> BatchForecastRunResult:
    """
    Full monthly run. Call this from scripts/run_monthly_forecast.py after
    your ETL + silver pipeline has finished writing the latest month's data.
 
    Writes:
        {output_dir}/{run_id}/forecast.parquet   -- flat table, one row per
                                                     medicine-day, matches
                                                     baseline_forecast_df shape
        {output_dir}/{run_id}/manifest.json       -- run metadata
        {output_dir}/latest.parquet               -- symlink-style copy of the
                                                     newest forecast, so the
                                                     API/LLM layer always reads
                                                     a stable path
    """
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    logger.info("Starting forecast run %s", run_id)
 
    history_df = _load_and_prepare_daily_demand(silver_demand_path)
    medicine_ids = sorted(history_df["item_id"].unique())
    logger.info("Loaded daily demand for %d medicines", len(medicine_ids))
 
    service = predictor or PredictorService.get_instance(config)
    results, failed = service.forecast_batch(history_df, item_ids=medicine_ids)
 
    rows = []
    for r in results:
        for d in r.days:
            rows.append({
                "Medicine_ID": r.medicine_id,
                "Forecast_Date": d.forecast_date,
                "Predicted_Demand": d.predicted_demand,
                "P10": d.quantiles.p10, "P20": d.quantiles.p20, "P30": d.quantiles.p30,
                "P40": d.quantiles.p40, "P50": d.quantiles.p50, "P60": d.quantiles.p60,
                "P70": d.quantiles.p70, "P80": d.quantiles.p80, "P90": d.quantiles.p90,
                "Context_Length_Used": r.context_length_used,
                "Model_ID": r.model_id,
                "Generated_At": r.generated_at,
            })
    forecast_df = pd.DataFrame(rows)
 
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = run_dir / "forecast.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
 
    latest_path = output_dir / "latest.parquet"

    # Quality gate (spec section 26 -- "forecast version promotion"): don't
    # blindly replace the production forecast. If too few medicines
    # succeeded, this run is degraded -- it's still published to the
    # versioned run history above for audit purposes, but latest.parquet
    # (what the API/LLM actually serve) keeps the LAST GOOD forecast rather
    # than being replaced with something worse. Threshold is deliberately
    # generous (50%) -- this is a coarse safety net against catastrophic
    # failures (e.g. a broken data feed), not a tight SLA.
    MIN_SUCCESS_RATE = 0.5
    success_rate = len(results) / len(medicine_ids) if medicine_ids else 0.0
    published = success_rate >= MIN_SUCCESS_RATE

    if published:
        forecast_df.to_parquet(latest_path, index=False)
        publish_note = ""
    else:
        publish_note = (
            f"NOT published: only {len(results)}/{len(medicine_ids)} medicines "
            f"succeeded ({success_rate:.0%}, below the {MIN_SUCCESS_RATE:.0%} "
            f"threshold). The previous latest.parquet was retained -- check "
            f"failed_medicine_ids and logs before investigating further."
        )
        logger.error(publish_note)

    completed_at = datetime.now(timezone.utc)
    manifest = BatchForecastRunResult(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        medicines_requested=len(medicine_ids),
        medicines_succeeded=len(results),
        medicines_failed=len(failed),
        failed_medicine_ids=failed,
        output_path=str(forecast_path),
        published=published,
        publish_note=publish_note,
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
 
    logger.info(
        "Forecast run %s complete: %d/%d medicines succeeded (%.1fs)",
        run_id, len(results), len(medicine_ids),
        (completed_at - started_at).total_seconds(),
    )
    if failed:
        logger.warning("Failed medicine_ids (%d): %s", len(failed), failed[:20])
 
    return manifest
 