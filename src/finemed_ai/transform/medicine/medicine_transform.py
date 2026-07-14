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
    log_step,
    trim_whitespace,
    normalize_text,
    fill_missing_text,
    validate_no_duplicate_keys,
    validate_no_nulls
)

logger = get_logger(__name__)


class MedicineTransformer:
    def __init__(
        self,
        medicine_dimension_path: Path,
    ) -> None:

        self.medicine_dimension_path = medicine_dimension_path

        self.medicine_df: pd.DataFrame | None = None

    
    # Load Data
    

    def load_data(self) -> None:

        log_step(logger, "Loading Medicine Dimension...")

        self.medicine_df = load_parquet(
            self.medicine_dimension_path,
            logger,
        )

        validate_dataframe_not_empty(
            self.medicine_df,
            "Medicine Dimension",
            logger,
        )

        log_dataframe_info(
            logger,
            self.medicine_df,
            "Medicine Dimension",
        )

    
    # Cleaning
    

    def clean_data(self) -> None:

    log_step(logger,"Cleaning Medicine Dataset...")

    if self.medicine_df is None:
        raise ValueError("Medicine dataframe is not loaded.")

    # 1. Trim whitespace

    trim_whitespace(
        self.medicine_df,
        columns=[
            "Medicine_Name",
            "Company_Name",
            "Medicine_Type",
            "Medicine_Category",
            "Unit",
        ],
        logger=logger)

    # 2. Normalize text

    normalize_text(
        self.medicine_df,
        columns=[
            "Medicine_Name",
            "Company_Name",
            "Medicine_Type",
            "Medicine_Category",
            "Unit",
        ],
        logger=logger)

    # 3. Fill optional text columns

    fill_missing_text(
        self.medicine_df,
        columns=[
            "Medicine_Type",
            "Medicine_Category",
            "Unit",
        ],
        logger=logger,
        value="UNKNOWN")

    # 4. Validate primary key

    validate_no_duplicate_keys(
        self.medicine_df,
        key_columns=["Medicine_ID"],
        logger=logger,
        df_name="Medicine Dimension")

    # 5. Validate mandatory columns

    validate_no_nulls(
        self.medicine_df,
        columns=[
            "Medicine_ID",
            "Medicine_Name",
        ],
        logger=logger,
        df_name="Medicine Dimension")

    logger.info("Medicine dataset cleaned successfully.")

    
    # Business Transformations
    

    def business_transformations(self) -> None:

    log_step(
        logger,
        "Creating Medicine Business Columns...")

    if self.medicine_df is None:
        raise ValueError(
            "Medicine dataframe is not loaded."
        )

    # Inventory Value
    

    if {
        "Current_Stock",
        "Purchase_Rate",
    }.issubset(self.medicine_df.columns):

        self.medicine_df["Inventory_Value"] = (
            self.medicine_df["Current_Stock"]
            * self.medicine_df["Purchase_Rate"]
        )

    
    # Stock Status

    if {"Current_Stock","Reorder_Level"}.issubset(self.medicine_df.columns):

        self.medicine_df["Stock_Status"] = "IN_STOCK"

        self.medicine_df.loc[
            self.medicine_df["Current_Stock"] <= 0,
            "Stock_Status"] = "OUT_OF_STOCK"

        self.medicine_df.loc[
            (
                self.medicine_df["Current_Stock"] > 0
            )
            &
            (
                self.medicine_df["Current_Stock"]
                <= self.medicine_df["Reorder_Level"]
            ),
            "Stock_Status",
        ] = "LOW_STOCK"

    # Reorder Flag

    if {
        "Current_Stock",
        "Reorder_Level",
    }.issubset(self.medicine_df.columns):

        self.medicine_df["Reorder_Flag"] = (
            self.medicine_df["Current_Stock"]
            <= self.medicine_df["Reorder_Level"]
        )

    
    # Expiry Flag

    if "Expiry_Date" in self.medicine_df.columns:

        self.medicine_df["Expired_Flag"] = (
            self.medicine_df["Expiry_Date"]
            < pd.Timestamp.today().normalize()
        )

    logger.info(
        "Medicine business transformations completed."
    )

    
    # Save Silver Dataset
    

    def save(
    self,
    output_path: Path) -> None:

    log_step(
        logger,
        "Saving Medicine Silver Dataset...",
    )

    save_parquet(
        self.medicine_df,
        output_path,
        logger,
    )

    logger.info(
        "Medicine Silver Dataset saved successfully."
    )
    
    # Pipeline

    def run(
    self,
    output_path: Path) -> None:

    try:

        self.load_data()

        self.clean_data()

        self.business_transformations()

        self.save(output_path)

        logger.info(
            "Medicine Transformation Completed Successfully."
        )

    except Exception:

        logger.exception(
            "Medicine Transformation Failed."
        )

        raise