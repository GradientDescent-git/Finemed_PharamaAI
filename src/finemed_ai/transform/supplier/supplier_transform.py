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

        log_step(logger, "Cleaning Supplier Dataset...")

    def business_transformations(self) -> None:

        log_step(logger, "Creating Supplier Business Columns...")

    # Feature Engineering
    def feature_engineering(self) -> None:

        log_step(logger, "Creating Supplier Features...")
    
    # Save Silver Dataset
    def save(
        self,
        output_path: Path,
    ) -> None:

        log_step(logger, "Saving Supplier Silver Dataset...")

        save_parquet(
            self.supplier_df,
            output_path,
            logger,
        )
    
    # Pipeline
    def run(
        self,
        output_path: Path,
    ) -> None:

        self.load_data()

        self.clean_data()

        self.business_transformations()

        self.feature_engineering()

        self.save(output_path)