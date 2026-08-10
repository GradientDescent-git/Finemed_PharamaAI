"""
finemed_ai.demand_forecasting.data_preparation
==================================================
The missing bridge between your ETL warehouse (Postgres) and the
forecasting module (which expects a flat parquet file).

Why this exists
----------------
run_pipeline() takes raw ERP data all the way to Postgres
(warehouse.fact_sales_line, warehouse.fact_sales_invoice, etc.) via
Extract -> Validate -> Warehouse -> Database. But
demand_forecasting/pipeline.py's _load_and_prepare_daily_demand() expects
a flat file at Settings.DEMAND_FILE with columns [MDCODE, INVDT, Demand_Qty].
Nothing in the existing chain produced that file -- this script is that
missing step. Run it AFTER run_pipeline() (or run_etl_pipeline.py) has
loaded warehouse.fact_sales_line and warehouse.fact_sales_invoice into
Postgres, and BEFORE scripts/run_monthly_forecast.py.

What it does
------------
1. Reads warehouse.fact_sales_line (MDCODE, QTY, INVNO, ...) and
   warehouse.fact_sales_invoice (INVNO, INVDT, ...) from Postgres.
2. Joins them on INVNO to attach a real calendar date to each sales line.
3. Aggregates QTY by (MDCODE, INVDT) -- a single invoice date can have
   multiple line items for the same medicine (e.g. split batches), so this
   sums QTY per medicine per day, which is the correct "daily demand"
   definition matching what the notebook's EDA/backtesting used.
4. Writes the result to Settings.DEMAND_FILE
   (data/04_silver/demand_forecasting/daily_demand.parquet).

Usage
-----
    python scripts/prepare_demand_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finemed_ai.config.settings import Settings
from finemed_ai.database.warehouse_reader import read_table
from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_demand_data(schema: str = "warehouse") -> Path:
    logger.info("Reading fact_sales_line and fact_sales_invoice from warehouse...")

    fact_sales_line = read_table("fact_sales_line", schema=schema)
    fact_sales_invoice = read_table("fact_sales_invoice", schema=schema)

    missing_line_cols = {"INVNO", "MDCODE", "QTY", "CANCEL_ID"} - set(fact_sales_line.columns)
    missing_invoice_cols = {"INVNO", "INVDT", "CANCEL_ID"} - set(fact_sales_invoice.columns)
    if missing_line_cols:
        raise ValueError(f"fact_sales_line missing expected columns: {missing_line_cols}")
    if missing_invoice_cols:
        raise ValueError(f"fact_sales_invoice missing expected columns: {missing_invoice_cols}")

    # Exclude cancelled invoices/lines -- cancelled sales are not real demand
    # and would inflate the forecast if included. CANCEL_ID conventions vary
    # by ERP; adjust the filter below if your data uses a different
    # "not cancelled" sentinel than 0/NULL.
    line = fact_sales_line[
        fact_sales_line["CANCEL_ID"].isna() | (fact_sales_line["CANCEL_ID"] == 0)
    ].copy()
    invoice = fact_sales_invoice[
        fact_sales_invoice["CANCEL_ID"].isna() | (fact_sales_invoice["CANCEL_ID"] == 0)
    ].copy()

    logger.info(
        "After excluding cancelled records: %d/%d sales lines, %d/%d invoices",
        len(line), len(fact_sales_line), len(invoice), len(fact_sales_invoice),
    )

    merged = line.merge(
        invoice[["INVNO", "INVDT"]],
        on="INVNO",
        how="inner",
    )

    daily_demand = (
        merged.groupby(["MDCODE", "INVDT"], as_index=False)["QTY"]
        .sum()
        .rename(columns={"QTY": "Demand_Qty"})
        .sort_values(["MDCODE", "INVDT"])
        .reset_index(drop=True)
    )

    output_path = Settings.DEMAND_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_demand.to_parquet(output_path, index=False)

    logger.info(
        "Wrote daily demand: %d rows, %d medicines, date range %s to %s -> %s",
        len(daily_demand),
        daily_demand["MDCODE"].nunique(),
        daily_demand["INVDT"].min(),
        daily_demand["INVDT"].max(),
        output_path,
    )
    return output_path

