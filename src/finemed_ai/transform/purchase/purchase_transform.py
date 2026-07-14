from __future__ import annotations

from pathlib import Path

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    load_parquet,
    save_parquet,
    log_step,
    log_dataframe_info,
    validate_dataframe_not_empty,
    trim_whitespace,
    normalize_text,
    fill_missing_text,
    fill_missing_numeric
)

from finemed_ai.transform.common.joins import (
    merge_dimension,
)

logger = get_logger(__name__)

class PurchaseTransformer:
    def __init__(
        self,
        fact_purchase_header_path: Path,
        fact_purchase_line_path: Path,
        dim_supplier_path: Path,
        dim_medicine_path: Path,
        dim_date_path: Path,
    ) -> None:

        self.fact_purchase_header_path = fact_purchase_header_path
        self.fact_purchase_line_path = fact_purchase_line_path

        self.dim_supplier_path = dim_supplier_path
        self.dim_medicine_path = dim_medicine_path
        self.dim_date_path = dim_date_path

        self.purchase_header_df: pd.DataFrame | None = None
        self.purchase_line_df: pd.DataFrame | None = None

        self.supplier_df: pd.DataFrame | None = None
        self.medicine_df: pd.DataFrame | None = None
        self.date_df: pd.DataFrame | None = None

        self.purchase_df: pd.DataFrame | None = None

     # Load Warehouse Tables
     def load_data(self) -> None:

        log_step(logger, "Loading Purchase Warehouse Tables...")

        self.purchase_header_df = load_parquet(
            self.fact_purchase_header_path,
            logger,
        )

        self.purchase_line_df = load_parquet(
            self.fact_purchase_line_path,
            logger,
        )

        self.supplier_df = load_parquet(
            self.dim_supplier_path,
            logger,
        )

        self.medicine_df = load_parquet(
            self.dim_medicine_path,
            logger,
        )

        self.date_df = load_parquet(
            self.dim_date_path,
            logger,
        )

        validate_dataframe_not_empty(
            self.purchase_header_df,
            "Fact Purchase Header",
            logger,
        )

        validate_dataframe_not_empty(
            self.purchase_line_df,
            "Fact Purchase Line",
            logger,
        )

        log_dataframe_info(
            logger,
            self.purchase_header_df,
            "Fact Purchase Header",
        )

        log_dataframe_info(
            logger,
            self.purchase_line_df,
            "Fact Purchase Line",
        )
    # Join Dimension Tables

    def join_dimensions(self) -> None:

        log_step(logger, "Joining Purchase Header and Purchase Line...")

        self.purchase_df = merge_dimension(
            self.purchase_line_df,
            self.purchase_header_df,
            fact_key="Purchase_ID",
            dimension_key="Purchase_ID",
        )

        log_step(logger, "Joining Supplier Dimension...")

        self.purchase_df = merge_dimension(
            self.purchase_df,
            self.supplier_df,
            fact_key="Supplier_ID",
            dimension_key="Supplier_ID",
        )

        log_step(logger, "Joining Medicine Dimension...")

        self.purchase_df = merge_dimension(
            self.purchase_df,
            self.medicine_df,
            fact_key="Medicine_ID",
            dimension_key="Medicine_ID",
        )

        log_step(logger, "Joining Date Dimension...")

        self.purchase_df = merge_dimension(
            self.purchase_df,
            self.date_df,
            fact_key="Date_ID",
            dimension_key="Date_ID",
        )
    # Cleaning
    def clean_data(self) -> None:

    log_step(
        logger,
        "Cleaning Purchase Dataset...",
    )

    validate_dataframe_not_empty(
        self.purchase_df,
    )

    # Trim Whitespace

    trim_whitespace(
        self.purchase_df,
        columns=[
            "Invoice_Number",
            "Supplier_Name",
            "Medicine_Name",
            "Batch_Number",
            "Purchase_Type",
        ],
    )

    # Normalize Text

    normalize_text(
        self.purchase_df,
        columns=[
            "Supplier_Name",
            "Medicine_Name",
            "Purchase_Type",
        ],
    )

    # Fill Missing Text Values
    

    fill_missing_text(
        self.purchase_df,
        columns=[
            "Supplier_Name",
            "Medicine_Name",
            "Purchase_Type",
        ],
    )

    # Fill Missing Numeric Values

    fill_missing_numeric(
        self.purchase_df,
        columns=[
            "Purchase_Quantity",
            "Unit_Cost",
            "Purchase_Amount",
        ],
    )

    
    # Remove Invalid Purchase Records

    self.purchase_df = self.purchase_df[
        (self.purchase_df["Purchase_Quantity"] > 0)
        &
        (self.purchase_df["Unit_Cost"] > 0)
        &
        (self.purchase_df["Purchase_Amount"] > 0)
    ]

    logger.info(
        "Purchase cleaning completed."
    )

    log_dataframe_info(
        logger,
        self.purchase_df,
        "Purchase Silver",
    )

    # Business Transformations
    def business_transformations(self) -> None:

    log_step(
        logger,
        "Creating Purchase Business Columns...",
    )

    validate_dataframe_not_empty(
        self.purchase_df,
    )

    # Purchase Value

    self.purchase_df["Purchase_Value"] = (
        self.purchase_df["Purchase_Quantity"]
        * self.purchase_df["Unit_Cost"]
    )

    # Unit Price Category

    self.purchase_df["Price_Category"] = (
        pd.cut(
            self.purchase_df["Unit_Cost"],
            bins=[
                0,
                100,
                500,
                1000,
                float("inf"),
            ],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH",
                "PREMIUM",
            ],
        )
        .astype(str)
    )

    # Large Purchase Flag

    self.purchase_df["Large_Purchase_Flag"] = (
        self.purchase_df["Purchase_Value"] >= 10000
    )

    # Purchase Size

    self.purchase_df["Purchase_Size"] = (
        pd.cut(
            self.purchase_df["Purchase_Quantity"],
            bins=[
                0,
                10,
                50,
                100,
                float("inf"),
            ],
            labels=[
                "SMALL",
                "MEDIUM",
                "LARGE",
                "BULK",
            ],
        )
        .astype(str)
    )

    logger.info(
        "Purchase business transformations completed."
    )

    log_dataframe_info(
        logger,
        self.purchase_df,
        "Purchase Silver",
    )

    # Save Silver Dataset
    def save(
    self,
    output_path: Path) -> None:

    log_step(
        logger,
        "Saving Purchase Silver Dataset...",
    )

    save_parquet(
        self.purchase_df,
        output_path,
        logger,
    )

    logger.info(
        "Purchase Silver Dataset saved successfully."
    )
    
    # Pipeline

    def run(
    self,
    output_path: Path,
) -> None:

    try:

        self.load_data()

        self.join_dimensions()

        self.clean_data()

        self.business_transformations()

        self.save(output_path)

        logger.info(
            "Purchase Transformation Completed Successfully."
        )

    except Exception:

        logger.exception(
            "Purchase Transformation Failed."
        )

        raise