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

        log_step(logger, "Cleaning Purchase Dataset...")

    # Business Transformations
    def business_transformations(self) -> None:

        log_step(logger, "Creating Purchase Business Columns...")

    # Feature Engineering
    def feature_engineering(self) -> None:

        log_step(logger, "Creating Purchase ML Features...")

    # Save Silver Dataset
    def save(
        self,
        output_path: Path,
    ) -> None:

        log_step(logger, "Saving Purchase Silver Dataset...")

        save_parquet(
            self.purchase_df,
            output_path,
            logger,
        )

    # Pipeline
    def run(self,output_path: Path) -> None:

        self.load_data()

        self.join_dimensions()

        self.clean_data()

        self.business_transformations()

        self.feature_engineering()

        self.save(output_path)

