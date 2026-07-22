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
    fill_missing_text,
)

logger = get_logger(__name__)

class SupplierTransformer:
    def __init__(self):
        self.supplier_df = None

    # Load Data

    def load_data(self) -> None:
        log_step(
            logger,
            "Loading Supplier Dimension...")
        
        self.supplier_df = read_table(
            table_name="dim_supplier")
        
        validate_dataframe_not_empty(
            self.supplier_df,
            logger,
            "Supplier Dimension")
        
        log_dataframe_info(
            logger,
            self.supplier_df,
            "Supplier Dimension")

    # Cleaning

    def clean_data(self) -> None:
        log_step(
            logger,
            "Cleaning Supplier Dataset...")
        
        if self.supplier_df is None:
            raise ValueError("Supplier dataframe is not loaded.")
        
        validate_dataframe_not_empty(
            self.supplier_df,
            logger,
            "Supplier Dataset")
        
        # Trim Whitespace
        trim_whitespace(
            self.supplier_df,
            columns=[
                "SUPNAME",
                "SUPCODE",
                "RNAME",
                "RADD1",
                "RADD2",
                "RADD3",
                "RADD4",
                "RPHON"],logger=logger)
        
        # Normalize Text
        normalize_text(
            self.supplier_df,
            columns=[
                "SUPNAME",
                "SUPCODE",
                "RNAME"],logger=logger)
        
        # Fill Missing Text
        fill_missing_text(
            self.supplier_df,
            columns=[
                "SUPNAME",
                "SUPCODE",
                "RNAME",
                "RADD1",
                "RADD2",
                "RADD3",
                "RADD4",
                "RPHON",
                ],logger=logger)
        
        # Remove Duplicate Suppliers
        before = len(self.supplier_df)
        
        self.supplier_df = (self.supplier_df.drop_duplicates(subset=["SUPNO"]).reset_index(drop=True))
        
        removed = before - len(self.supplier_df)
        logger.info("Duplicate supplier rows removed: %d",removed)
        
        validate_dataframe_not_empty(
            self.supplier_df,
            logger,
            "Supplier Silver")
        
        log_dataframe_info(
            logger,
            self.supplier_df,
            "Supplier Silver")
        logger.info(
            "Supplier cleaning completed successfully.")
        
    #Business Transformation

    def business_transformations(self) -> None:
        if self.supplier_df is None:
            raise ValueError("Supplier dataframe is not loaded.")
        log_step(
            logger,
            "Creating Supplier Business Columns...")
        
        # Supplier Display Name
        if {"SUPCODE", "SUPNAME"}.issubset(self.supplier_df.columns):
            self.supplier_df["Supplier_Display"] = (self.supplier_df["SUPCODE"].astype(str)
                                                    + " - "
                                                    + self.supplier_df["SUPNAME"].astype(str))
        
        # Contact Availability Flag
        if "RPHON" in self.supplier_df.columns:
            self.supplier_df["Has_Contact"] = (
                self.supplier_df["RPHON"].fillna("").astype(str).str.strip().ne(""))
            
        # Full Supplier Address
        address_columns = [
            "RADD1",
            "RADD2",
            "RADD3",
            "RADD4"]
        
        existing_columns = [
            col for col in address_columns
            if col in self.supplier_df.columns]
        
        if existing_columns:
            self.supplier_df["Supplier_Address"] = (
                self.supplier_df[existing_columns].fillna("").astype(str).agg(", ".join, axis=1).str.replace(r"(,\s*)+", ", ", regex=True).str.strip(", "))
        
        logger.info(
            "Supplier business transformations completed successfully.")
        
        log_dataframe_info(
            logger,
            self.supplier_df,
            "Supplier Silver")
        
    # Save Silver Dataset
    def save(
            self,
            output_path: Path) -> None:
        log_step(
            logger,
            "Saving Supplier Silver Dataset...")
        
        validate_dataframe_not_empty(
            self.supplier_df,
            logger,
            "Supplier Silver Dataset")
        
        save_parquet(
            self.supplier_df,
            output_path,
            logger)
        
        logger.info(
            "Supplier Silver Dataset saved successfully.")

    # Pipeline
    def run(self,output_path: Path) -> None:
        try:
            log_step(
                logger,
                "Starting Supplier Silver Transformation..."
                )
            
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