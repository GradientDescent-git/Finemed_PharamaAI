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
    validate_dataframe_not_empty,
    trim_whitespace,
    normalize_text,
    fill_missing_numeric,
    fill_missing_text,
)

from finemed_ai.transform.common.joins import (
    join_sales_with_customer,
    join_sales_with_date,
    join_sales_with_medicine,
)

logger = get_logger(__name__)

class SalesTransformer:
    """Sales Silver Layer Transformer."""

    def __init__(
        self,
        fact_sales_path: Path,
        dim_customer_path: Path,
        dim_date_path: Path,
        dim_medicine_path: Path,
    ) -> None:
        self.fact_sales_path = fact_sales_path
        self.dim_customer_path = dim_customer_path
        self.dim_date_path = dim_date_path
        self.dim_medicine_path = dim_medicine_path

        self.sales_df: pd.DataFrame | None = None
        self.customer_df: pd.DataFrame | None = None
        self.date_df: pd.DataFrame | None = None
        self.medicine_df: pd.DataFrame | None = None

    # Load Warehouse Tables

    def load_data(self) -> None:
        log_step(
            logger,
            "Loading Sales Warehouse Tables...",
        )

        self.sales_df = load_parquet(
            self.fact_sales_path,
            logger,
        )

        self.customer_df = load_parquet(
            self.dim_customer_path,
            logger,
        )

        self.date_df = load_parquet(
            self.dim_date_path,
            logger,
        )

        self.medicine_df = load_parquet(
            self.dim_medicine_path,
            logger,
        )

        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Fact Sales",
        )

        log_dataframe_info(
            logger,
            self.sales_df,
            "Fact Sales",
        )

    # Join Dimension Tables

    def join_dimensions(self) -> None:
        log_step(
            logger,
            "Joining Customer Dimension...",
        )

        self.sales_df = join_sales_with_customer(
            self.sales_df,
            self.customer_df,
        )

        log_dataframe_info(
            logger,
            self.sales_df,
            "Sales + Customer",
        )

        log_step(
            logger,
            "Joining Date Dimension...",
        )

        self.sales_df = join_sales_with_date(
            self.sales_df,
            self.date_df,
        )

        log_dataframe_info(
            logger,
            self.sales_df,
            "Sales + Date",
        )

        log_step(
            logger,
            "Joining Medicine Dimension...",
        )

        self.sales_df = join_sales_with_medicine(
            self.sales_df,
            self.medicine_df,
        )

        log_dataframe_info(
            logger,
            self.sales_df,
            "Sales + Medicine",
        )

        logger.info(
            "All dimension tables joined successfully."
        )

    # Cleaning

    def clean_data(self) -> None:
        log_step(
            logger,
            "Cleaning Sales Dataset...",
        )

        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Dataset",
        )

        # Trim whitespace from text columns

        text_columns = [
            column
            for column in self.sales_df.select_dtypes(
                include="object"
            ).columns
        ]

        if text_columns:
            self.sales_df = trim_whitespace(
                self.sales_df,
                text_columns,
                logger,
            )

            self.sales_df = normalize_text(
                self.sales_df,
                text_columns,
                logger,
            )

        # Fill missing numeric values

        numeric_columns = [
            column
            for column in self.sales_df.select_dtypes(
                include="number"
            ).columns
        ]

        if numeric_columns:
            self.sales_df = fill_missing_numeric(
                self.sales_df,
                numeric_columns,
                logger,
            )

        # Fill missing text values

        if text_columns:
            self.sales_df = fill_missing_text(
                self.sales_df,
                text_columns,
                logger,
            )

        # Remove duplicate records

        duplicate_count = self.sales_df.duplicated().sum()

        if duplicate_count > 0:
            logger.info(
                "Removing %d duplicate records.",
                duplicate_count,
            )

            self.sales_df = (
                self.sales_df
                .drop_duplicates()
                .reset_index(drop=True)
            )

        logger.info(
            "Sales cleaning completed."
        )

    # Business Transformations

    def business_transformations(self) -> None:
        log_step(
            logger,
            "Creating Sales Business Columns...",
        )

        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Dataset",
        )

        # Discount Amount

        if {
            "Gross_Amount",
            "Net_Amount",
        }.issubset(self.sales_df.columns):

            self.sales_df["Discount_Amount"] = (
                self.sales_df["Gross_Amount"]
                - self.sales_df["Net_Amount"]
            )

        # Profit

        if {
            "Net_Amount",
            "Cost_Amount",
        }.issubset(self.sales_df.columns):

            self.sales_df["Profit"] = (
                self.sales_df["Net_Amount"]
                - self.sales_df["Cost_Amount"]
            )

        # High Value Order Flag

        if "Net_Amount" in self.sales_df.columns:
            average_order = (
                self.sales_df["Net_Amount"]
                .mean()
            )

            self.sales_df["High_Value_Order_Flag"] = (
                self.sales_df["Net_Amount"]
                >= average_order
            )

        # Return Flag

        if "Quantity" in self.sales_df.columns:
            self.sales_df["Return_Flag"] = (
                self.sales_df["Quantity"] < 0
            )

        logger.info(
            "Sales business transformations completed."
        )

    # Save Silver Dataset

    def save(
        self,
        output_path: Path,
    ) -> None:
        log_step(
            logger,
            "Saving Sales Silver Dataset...",
        )

        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Silver Dataset",
        )

        save_parquet(
            self.sales_df,
            output_path,
            logger,
        )

        logger.info(
            "Sales Silver Dataset saved successfully."
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
                "Sales Transformation Completed Successfully."
            )

        except Exception:
            logger.exception(
                "Sales Transformation Failed."
            )
            raise