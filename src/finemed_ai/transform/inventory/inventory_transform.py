from __future__ import annotations

from pathlib import Path

import pandas as pd

from finemed_ai.database.warehouse_reader import read_table
from finemed_ai.transform.common.helper_functions import (
    get_logger,
    save_parquet,
    log_step,
    log_dataframe_info,
    validate_dataframe_not_empty,
    trim_whitespace,
    normalize_text,
    fill_missing_numeric,
    fill_missing_text,
    fill_missing_boolean,
    round_numeric_columns,
    convert_to_datetime)

from finemed_ai.transform.common.joins import (
    merge_dimension,
)

logger = get_logger(__name__)


class InventoryTransformer:

    def __init__(self) -> None:
        
        self.inventory_df: pd.DataFrame | None = None
        self.medicine_df: pd.DataFrame | None = None
        self.date_df: pd.DataFrame | None = None

    # Load Warehouse Tables

    def load_data(self) -> None:

        log_step(
            logger,
            "Loading Inventory Warehouse Tables...",
        )

        self.inventory_df = read_table(table_name="fact_purchase_line")
        
        self.medicine_df = read_table(table_name="dim_product")
        self.date_df = read_table(table_name="dim_date")


        validate_dataframe_not_empty(
            self.inventory_df,
            logger,
            "Inventory Fact",
        )

        validate_dataframe_not_empty(
            self.medicine_df,
            logger,
            "Medicine Dimension",
        )

        validate_dataframe_not_empty(
            self.date_df,
            logger,
            "Date Dimension",
        )

        log_dataframe_info(
            logger,
            self.inventory_df,
            "Inventory Fact",
        )

        log_dataframe_info(
            logger,
            self.medicine_df,
            "Medicine Dimension",
        )

        log_dataframe_info(
            logger,
            self.date_df,
            "Date Dimension",
        )

    # Join Dimension Tables

    def join_dimensions(self) -> None:

        if self.inventory_df is None:
            raise ValueError("Inventory Fact not loaded.")

        if self.medicine_df is None:
            raise ValueError("Medicine Dimension not loaded.")

        if self.date_df is None:
            raise ValueError("Date Dimension not loaded.")

        log_step(
            logger,
            "Joining Medicine Dimension...",
        )

        self.inventory_df = merge_dimension(
            self.inventory_df,
            self.medicine_df,
            fact_key="MDCODE",
            dimension_key="MDCODE")

        log_step(
            logger,
            "Joining Date Dimension...",
        )

        self.inventory_df = merge_dimension(
            self.inventory_df,
            self.date_df,
            fact_key="EXP",
            dimension_key="full_date")

        logger.info(
            "Dimension joins completed successfully."
        )

    # Cleaning
    def clean_data(self) -> None:

        log_step(
            logger,
            "Cleaning Inventory Dataset...",
        )

        if self.inventory_df is None:
            raise ValueError("Inventory dataframe is not loaded.")

        # 1. Trim whitespace

        trim_whitespace(
            self.inventory_df,
            columns=[
                "MDNAME",
                "SUPCODE",
                "BAT" ],
            logger=logger)

        # 2. Normalize text

        normalize_text(
            self.inventory_df,
            columns=[
                "MDNAME",
                "SUPCODE",
            ],
            logger=logger,
        )

        # 3. Convert dates

        convert_to_datetime(
            self.inventory_df,
            column="EXP",
            logger=logger,
        )

        # 4. Fill missing text

        fill_missing_text(
            self.inventory_df,
            columns=[
                "SUPCODE",
            ],
            logger=logger,
        )

        # 5. Fill missing numeric

        fill_missing_numeric(
            self.inventory_df,
            columns=[
                "SOH",
                "PRATE",
                "MRP",
            ],
            logger=logger,
        )

        # 6. Remove duplicates

        before = len(self.inventory_df)

        self.inventory_df = (
            self.inventory_df.drop_duplicates().reset_index(drop=True)
        )

        removed = before - len(self.inventory_df)

        logger.info(
            "Duplicate rows removed: %d",
            removed,
        )

        # 7. Final validation

        validate_dataframe_not_empty(
            self.inventory_df,
            logger,
            "Inventory Dataset",
        )

        log_dataframe_info(
            logger,
            self.inventory_df,
            "Inventory Dataset",
        )

        logger.info(
            "Inventory cleaning completed successfully."
        )

    # Business Transformations

    def business_transformations(self) -> None:

        if self.inventory_df is None:
            raise ValueError("Inventory dataframe is not loaded.")

        log_step(
            logger,
            "Creating Inventory Business Columns...",
        )

        # Inventory Value

        if {"SOH", "PRATE"}.issubset(self.inventory_df.columns):

            self.inventory_df["Inventory_Value"] = (
                self.inventory_df["SOH"]
                * self.inventory_df["PRATE"]
            )

        # Expiry Date

        if "EXP" in self.inventory_df.columns:
            self.inventory_df["Expiry_Date"] = self.inventory_df["EXP"]

        # Days Until Expiry

        if "Expiry_Date" in self.inventory_df.columns:

            today = pd.Timestamp.today().normalize()

            self.inventory_df["Days_To_Expiry"] = (
                self.inventory_df["Expiry_Date"] - today
            ).dt.days

        # Expiry Bucket

        if "Days_To_Expiry" in self.inventory_df.columns:

            self.inventory_df["Expiry_Bucket"] = pd.cut(
                self.inventory_df["Days_To_Expiry"],
                bins=[
                    -99999,
                    0,
                    30,
                    90,
                    180,
                    365,
                    99999,
                ],
                labels=[
                    "Expired",
                    "0-30 Days",
                    "31-90 Days",
                    "91-180 Days",
                    "181-365 Days",
                    ">365 Days",
                ],
            )

        # Stock Status

        if "SOH" in self.inventory_df.columns:

            self.inventory_df["Stock_Status"] = "In Stock"

            self.inventory_df.loc[
                self.inventory_df["SOH"] <= 0,
                "Stock_Status",
            ] = "Out Of Stock"

            self.inventory_df.loc[
                (self.inventory_df["SOH"] > 0)
                & (self.inventory_df["SOH"] <= 10),
                "Stock_Status",
            ] = "Low Stock"

        # Reorder Flag

        if "SOH" in self.inventory_df.columns:

            self.inventory_df["Reorder_Flag"] = (
                self.inventory_df["SOH"] <= 10
            )

        logger.info(
            "Inventory business transformations completed."
        )

    # Save Silver Dataset

    def save(
        self,
        output_path: Path,
    ) -> None:

        log_step(
            logger,
            "Saving Inventory Silver Dataset...",
        )

        validate_dataframe_not_empty(
            self.inventory_df,
            logger,
            "Inventory Silver Dataset",
        )

        save_parquet(
            self.inventory_df,
            output_path,
            logger,
        )

        logger.info(
            "Inventory Silver Dataset saved successfully."
        )

        logger.info(
            "Silver dataset saved to %s",
            output_path,
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
                "Inventory Transformation Completed Successfully. Output: %s",
                output_path,
            )

        except Exception:

            logger.exception(
                "Inventory Transformation Failed."
            )

            raise