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

from finemed_ai.transform.common.joins import (
    join_sales_with_customer,
    join_sales_with_date,
    join_sales_with_medicine,
)

logger = get_logger(__name__)


class SalesTransformer:
    """
    Sales Silver Layer Transformer.

    Responsibilities
    ----------------
    1. Load warehouse tables
    2. Join dimensions
    3. Clean data
    4. Apply business transformations
    5. Save Silver dataset
    """

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

    # ---------------------------------------------------------
    # Load Warehouse Tables
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Join Dimension Tables
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------

    def clean_data(self) -> None:

        log_step(
            logger,
            "Cleaning Sales Dataset...",
        )

        # Future Cleaning Rules
        # ---------------------
        # Trim whitespace
        # Fill missing values
        # Normalize text
        # Remove duplicates
        # Validate IDs
        # Remove invalid records

        logger.info(
            "Sales cleaning completed."
        )

    # ---------------------------------------------------------
    # Business Transformations
    # ---------------------------------------------------------

    def business_transformations(self) -> None:

        log_step(
            logger,
            "Creating Sales Business Columns...",
        )

        # Future Business KPIs
        # --------------------
        # Total Sales
        # Discount Amount
        # Net Sales
        # Profit
        # Sales Category
        # Return Flag
        # High Value Order Flag

        logger.info(
            "Sales business transformations completed."
        )

    # ---------------------------------------------------------
    # Save Silver Dataset
    # ---------------------------------------------------------

    def save(
        self,
        output_path: Path,
    ) -> None:

        log_step(
            logger,
            "Saving Sales Silver Dataset...",
        )

        save_parquet(
            self.sales_df,
            output_path,
            logger,
        )

        logger.info(
            "Sales Silver Dataset saved successfully."
        )

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

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