from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    load_parquet,
    save_parquet,
    log_step,
    log_dataframe_info,
    validate_dataframe_not_empty
)

from finemed_ai.transform.common.joins import (
    join_sales_with_customer,
    join_sales_with_date,
    join_purchase_with_medicine
)

logger = get_logger(__name__)


class SalesTransformer:    
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

        log_step(logger, "Loading warehouse tables...")

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
            "Fact Sales",
            logger,
        )

        log_dataframe_info(
            logger,
            self.sales_df,
            "Fact Sales",
        )

#Join dimensions table

    def join_dimensions(self) -> None:

        log_step(logger, "Joining Sales with Customer Dimension...")

        self.sales_df = join_sales_with_customer(
            self.sales_df,
            self.customer_df,
        )

        log_step(logger, "Joining Sales with Date Dimension...")

        self.sales_df = join_sales_with_date(
            self.sales_df,
            self.date_df,
        )

        log_step(logger, "Joining Sales with Medicine Dimension...")

        self.sales_df = join_purchase_with_medicine(
            self.sales_df,
            self.medicine_df,
        )

   
    # Cleaning
   

    def clean_data(self) -> None:
        log_step(logger, "Cleaning Sales Dataset...")

   
    # Business Transformations
   

    def business_transformations(self) -> None:
        log_step(logger, "Creating Business Columns...")

  
    # Feature Engineering
   

    def feature_engineering(self) -> None:
        
        log_step(logger, "Creating ML Features...")

    
# Save Silver Dataset
    def save(self, output_path: Path) -> None:
        log_step(logger, "Saving Sales Silver Dataset...")
        save_parquet(
            self.sales_df,
            output_path,
            logger,
        )


# Pipeline
    
    def run(self, output_path: Path) -> None:

        self.load_data()

        self.join_dimensions()

        self.clean_data()

        self.business_transformations()

        self.feature_engineering()

        self.save(output_path)

    
    def join_sales_with_medicine(sales_df: pd.DataFrame,medicine_df: pd.DataFrame) -> pd.DataFrame:
        return merge_dimension(sales_df,medicine_df,fact_key="Medicine_ID",dimension_key="Medicine_ID")