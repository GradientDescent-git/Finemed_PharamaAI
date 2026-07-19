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
)

logger = get_logger(__name__)

class SupplierTransformer:
    def __init__(
        self,
        supplier_dimension_path: Path,
    ) -> None:
        self.supplier_dimension_path = supplier_dimension_path

        self.supplier_df: pd.DataFrame | None = None

    # Load Data

    def load_data(self) -> None:
        log_step(logger, "Loading Supplier Dimension...")

        self.supplier_df = load_parquet(
            self.supplier_dimension_path,
            logger,
        )

        validate_dataframe_not_empty(
            self.supplier_df,
            "Supplier Dimension",
            logger,
        )

        log_dataframe_info(
            logger,
            self.supplier_df,
            "Supplier Dimension",
        )

    # Cleaning

    def clean_data(self) -> None:
        log_step(
            logger,
            "Cleaning Supplier Dataset...",
        )

        # Trim whitespace

        text_columns = [
            "SUPNAME",
            "SUPCODE",
            "RNAME",
            "RADD1",
            "RADD2",
            "RADD3",
            "RADD4",
            "RPHON",
        ]

        existing_text_columns = [
            column
            for column in text_columns
            if column in self.supplier_df.columns
        ]

        if existing_text_columns:
            self.supplier_df = trim_whitespace(
                self.supplier_df,
                existing_text_columns,
                logger,
            )

        # Normalize text

        if existing_text_columns:
            self.supplier_df = normalize_text(
                self.supplier_df,
                existing_text_columns,
                logger,
            )

        # Fill missing text values

        if existing_text_columns:
            self.supplier_df = fill_missing_text(
                self.supplier_df,
                existing_text_columns,
                logger,
                value="UNKNOWN",
            )

        # Fill missing numeric values

        numeric_columns = [
            column
            for column in self.supplier_df.columns
            if pd.api.types.is_numeric_dtype(
                self.supplier_df[column]
            )
        ]

        if numeric_columns:
            self.supplier_df = fill_missing_numeric(
                self.supplier_df,
                numeric_columns,
                logger,
                value=0,
            )

        # Remove duplicate suppliers

        before = len(self.supplier_df)

        self.supplier_df = (
            self.supplier_df
            .drop_duplicates(
                subset=["SUPNO"],
            )
            .reset_index(drop=True)
        )

        removed = before - len(self.supplier_df)

        logger.info(
            "Removed %d duplicate suppliers.",
            removed,
        )

        # Validate supplier key

        validate_no_duplicate_keys(
            self.supplier_df,
            ["SUPNO"],
            logger,
            "Supplier Dimension",
        )

        logger.info(
            "Supplier cleaning completed.",
        )

    # Business Transformations

    def business_transformations(self) -> None:
        log_step(
            logger,
            "Creating Supplier Business Columns...",
        )

        # Total Purchase Value

        if {
            "YPURVAL",
            "MPURVAL",
        }.issubset(self.supplier_df.columns):

            self.supplier_df["Total_Purchase_Value"] = (
                self.supplier_df["YPURVAL"]
                + self.supplier_df["MPURVAL"]
            )

        # Total Sales Value

        if {
            "YRSAL",
            "MRSAL",
        }.issubset(self.supplier_df.columns):

            self.supplier_df["Total_Sales_Value"] = (
                self.supplier_df["YRSAL"]
                + self.supplier_df["MRSAL"]
            )

        # Total Returns

        if {
            "YPURRET",
            "YRRET",
        }.issubset(self.supplier_df.columns):

            self.supplier_df["Total_Return_Value"] = (
                self.supplier_df["YPURRET"]
                + self.supplier_df["YRRET"]
            )

        # Active Supplier Flag

        if "Total_Purchase_Value" in self.supplier_df.columns:

            self.supplier_df["Is_Active_Supplier"] = (
                self.supplier_df["Total_Purchase_Value"] > 0
            )

        # High Value Supplier

        if "Total_Purchase_Value" in self.supplier_df.columns:

            threshold = (
                self.supplier_df["Total_Purchase_Value"]
                .quantile(0.90)
            )

            self.supplier_df["High_Value_Supplier"] = (
                self.supplier_df["Total_Purchase_Value"]
                >= threshold
            )

        # Supplier Category

        if "Total_Purchase_Value" in self.supplier_df.columns:

            self.supplier_df["Supplier_Category"] = pd.cut(
                self.supplier_df["Total_Purchase_Value"],
                bins=[
                    -1,
                    0,
                    10000,
                    100000,
                    float("inf"),
                ],
                labels=[
                    "INACTIVE",
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                ],
            )

        logger.info(
            "Supplier business transformations completed.",
        )

    # Save Silver Dataset

    def save(
        self,
        output_path: Path,
    ) -> None:
        log_step(
            logger,
            "Saving Supplier Silver Dataset...",
        )

        save_parquet(
            self.supplier_df,
            output_path,
            logger,
        )

        logger.info(
            "Supplier Silver Dataset saved successfully."
        )

    # Pipeline

    def run(
        self,
        output_path: Path,
    ) -> None:
        try:
            self.load_data()

            self.clean_data()

            self.business_transformations()

            self.save(output_path)

            logger.info(
                "Supplier Transformation Completed Successfully."
            )

        except Exception:
            logger.exception(
                "Supplier Transformation Failed."
            )
            raise