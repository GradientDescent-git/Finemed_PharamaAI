from __future__ import annotations

from pathlib import Path

import pandas as pd

from finemed_ai.config.settings import Settings
from finemed_ai.database.warehouse_reader import read_table
from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_demand_data(schema: str = "warehouse") -> Path:
    """
    Build the daily medicine-demand dataset from the warehouse.

    Business key:
        (INVNO, SOURCE_MONTH)

    INVNO is NOT globally unique in the ERP. It is only unique within
    SOURCE_MONTH, so joining on INVNO alone can incorrectly attach a
    sales line to invoices from other source periods.
    """

    logger.info(
        "Reading fact_sales_line and fact_sales_invoice from warehouse..."
    )

    fact_sales_line = read_table(
        "fact_sales_line",
        schema=schema,
    )

    fact_sales_invoice = read_table(
        "fact_sales_invoice",
        schema=schema,
    )

    # ---------------------------------------------------------------
    # Validate source schemas
    # ---------------------------------------------------------------

    required_line_cols = {
        "INVNO",
        "MDCODE",
        "QTY",
        "CANCEL_ID",
        "SOURCE_MONTH",
    }

    required_invoice_cols = {
        "INVNO",
        "INVDT",
        "CANCEL_ID",
        "SOURCE_MONTH",
    }

    missing_line_cols = required_line_cols - set(fact_sales_line.columns)

    missing_invoice_cols = required_invoice_cols - set(
        fact_sales_invoice.columns
    )

    if missing_line_cols:
        raise ValueError(
            f"fact_sales_line missing expected columns: "
            f"{missing_line_cols}"
        )

    if missing_invoice_cols:
        raise ValueError(
            f"fact_sales_invoice missing expected columns: "
            f"{missing_invoice_cols}"
        )

    # ---------------------------------------------------------------
    # Normalize composite-key columns
    # ---------------------------------------------------------------

    for df in (fact_sales_line, fact_sales_invoice):
        df["INVNO"] = df["INVNO"].astype(str).str.strip()
        df["SOURCE_MONTH"] = (
            df["SOURCE_MONTH"]
            .astype(str)
            .str.strip()
        )

    # ---------------------------------------------------------------
    # Remove cancelled records
    # ---------------------------------------------------------------

    line = fact_sales_line[
        fact_sales_line["CANCEL_ID"].isna()
        | (fact_sales_line["CANCEL_ID"] == 0)
    ].copy()

    invoice = fact_sales_invoice[
        fact_sales_invoice["CANCEL_ID"].isna()
        | (fact_sales_invoice["CANCEL_ID"] == 0)
    ].copy()

    logger.info(
        "After excluding cancelled records: "
        "%d/%d sales lines, %d/%d invoices",
        len(line),
        len(fact_sales_line),
        len(invoice),
        len(fact_sales_invoice),
    )

    # ---------------------------------------------------------------
    # Validate invoice business key
    # ---------------------------------------------------------------

    invoice_key = ["INVNO", "SOURCE_MONTH"]

    duplicate_invoice_keys = invoice.duplicated(
        invoice_key,
        keep=False,
    )

    if duplicate_invoice_keys.any():
        duplicates = (
            invoice.loc[
                duplicate_invoice_keys,
                invoice_key,
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "Invoice business key is not unique. "
            f"Found {len(duplicates)} duplicate "
            f"(INVNO, SOURCE_MONTH) keys."
        )

    logger.info(
        "Invoice business-key validation passed: "
        "%d unique (INVNO, SOURCE_MONTH) keys.",
        len(invoice),
    )

    # ---------------------------------------------------------------
    # Join sales lines to invoices
    #
    # IMPORTANT:
    # INVNO alone is NOT a valid key.
    # ---------------------------------------------------------------

    merged = line.merge(
        invoice[
            [
                "INVNO",
                "SOURCE_MONTH",
                "INVDT",
            ]
        ],
        on=[
            "INVNO",
            "SOURCE_MONTH",
        ],
        how="inner",
        validate="many_to_one",
    )

    logger.info(
        "Joined sales lines to invoices: "
        "%d rows from %d active sales lines.",
        len(merged),
        len(line),
    )

    # ---------------------------------------------------------------
    # Detect orphan sales lines
    # ---------------------------------------------------------------

    matched_line_keys = merged[
        ["INVNO", "SOURCE_MONTH"]
    ].drop_duplicates()

    line_keys = line[
        ["INVNO", "SOURCE_MONTH"]
    ].drop_duplicates()

    orphan_keys = (
        line_keys.merge(
            matched_line_keys,
            on=["INVNO", "SOURCE_MONTH"],
            how="left",
            indicator=True,
        )
    )

    orphan_keys = orphan_keys[
        orphan_keys["_merge"] == "left_only"
    ]

    if not orphan_keys.empty:
        logger.warning(
            "Found %d sales invoice keys with no matching invoice header.",
            len(orphan_keys),
        )

    # ---------------------------------------------------------------
    # Normalize dates and quantities
    # ---------------------------------------------------------------

    merged["INVDT"] = pd.to_datetime(
        merged["INVDT"],
        errors="coerce",
    )

    if merged["INVDT"].isna().any():
        bad_dates = int(merged["INVDT"].isna().sum())

        raise ValueError(
            f"Found {bad_dates} sales rows with invalid invoice dates."
        )

    merged["QTY"] = pd.to_numeric(
        merged["QTY"],
        errors="coerce",
    )

    if merged["QTY"].isna().any():
        bad_qty = int(merged["QTY"].isna().sum())

        raise ValueError(
            f"Found {bad_qty} sales rows with invalid QTY values."
        )

    # ---------------------------------------------------------------
    # Aggregate demand
    #
    # Multiple sales lines can belong to the same medicine/invoice
    # date, so aggregate to:
    #
    #     MDCODE + INVDT
    # ---------------------------------------------------------------

    daily_demand = (
        merged
        .groupby(
            ["MDCODE", "INVDT"],
            as_index=False,
        )["QTY"]
        .sum()
        .rename(
            columns={
                "QTY": "Demand_Qty",
            }
        )
        .sort_values(
            ["MDCODE", "INVDT"]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Final data-quality checks
    # ---------------------------------------------------------------

    if daily_demand.empty:
        raise ValueError(
            "Daily demand dataset is empty after preparation."
        )

    duplicate_daily_keys = daily_demand.duplicated(
        ["MDCODE", "INVDT"]
    ).sum()

    if duplicate_daily_keys:
        raise ValueError(
            f"Found {duplicate_daily_keys} duplicate "
            "(MDCODE, INVDT) rows after aggregation."
        )

    if daily_demand["Demand_Qty"].isna().any():
        raise ValueError(
            "Daily demand contains NULL Demand_Qty values."
        )

    if (daily_demand["Demand_Qty"] < 0).any():
        negative_count = int(
            (daily_demand["Demand_Qty"] < 0).sum()
        )

        raise ValueError(
            f"Daily demand contains {negative_count} negative quantities."
        )

    # ---------------------------------------------------------------
    # Write parquet
    # ---------------------------------------------------------------

    output_path = Settings.DEMAND_FILE

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_demand.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Wrote daily demand: "
        "%d rows, %d medicines, date range %s to %s -> %s",
        len(daily_demand),
        daily_demand["MDCODE"].nunique(),
        daily_demand["INVDT"].min(),
        daily_demand["INVDT"].max(),
        output_path,
    )

    return output_path