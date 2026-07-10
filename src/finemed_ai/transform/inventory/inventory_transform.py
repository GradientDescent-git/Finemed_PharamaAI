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

class InventoryTransformer:
    def __init__(
        self,
        inventory_fact_path: Path,
        medicine_dimension_path: Path,
        date_dimension_path: Path,
    ) -> None:

        self.inventory_fact_path = inventory_fact_path

        self.medicine_dimension_path = medicine_dimension_path
        self.date_dimension_path = date_dimension_path

        self.inventory_df: pd.DataFrame | None = None

        self.medicine_df: pd.DataFrame | None = None
        self.date_df: pd.DataFrame | None = None

    # Load Warehouse Tables
    def load_data(self) -> None:

        log_step(logger, "Loading Inventory Warehouse Tables...")

        self.inventory_df = load_parquet(
            self.inventory_fact_path,
            logger,
        )

        self.medicine_df = load_parquet(
            self.medicine_dimension_path,
            logger,
        )

        self.date_df = load_parquet(
            self.date_dimension_path,
            logger,
        )

        validate_dataframe_not_empty(
            self.inventory_df,
            "Inventory Fact",
            logger,
        )

        log_dataframe_info(
            logger,
            self.inventory_df,
            "Inventory Fact",
        )

    # Join Dimension Tables
    def join_dimensions(self) -> None:

        log_step(logger, "Joining Medicine Dimension...")

        self.inventory_df = merge_dimension(
            self.inventory_df,
            self.medicine_df,
            fact_key="Medicine_ID",
            dimension_key="Medicine_ID",
        )

        log_step(logger, "Joining Date Dimension...")

        self.inventory_df = merge_dimension(
            self.inventory_df,
            self.date_df,
            fact_key="Date_ID",
            dimension_key="Date_ID",
        )
    
    # Cleaning
    def clean_data(self) -> None:

        log_step(logger, "Cleaning Inventory Dataset...")

    # Business Transformations
    def business_transformations(self) -> None:

        log_step(logger, "Creating Inventory Business Columns...")

    # Feature Engineering
    def feature_engineering(self) -> None:

        log_step(logger, "Creating Inventory ML Features...")

    # Save Silver Dataset
    def save(
        self,
        output_path: Path,
    ) -> None:

        log_step(logger, "Saving Inventory Silver Dataset...")

        save_parquet(
            self.inventory_df,
            output_path,
            logger,
        )

    # Pipeline
    def run(
        self,
        output_path: Path,
    ) -> None:

        self.load_data()

        self.join_dimensions()

        self.clean_data()

        self.business_transformations()

        self.feature_engineering()

        self.save(output_path)