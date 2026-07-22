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
    validate_no_duplicate_keys,
    validate_no_nulls
)

logger = get_logger(__name__)


class MedicineTransformer:
    def __init__(self) -> None:
        
        self.medicine_df: pd.DataFrame | None = None
    
    # Load Data

    def load_data(self) -> None:
        log_step(logger, "Loading Medicine Dimension...")

        self.medicine_df = read_table(
            table_name="dim_product")
        
        validate_dataframe_not_empty(
            self.medicine_df,
            logger,
            "Medicine Dimension")

        log_dataframe_info(
            logger,
            self.medicine_df,
            "Medicine Dimension",
        )

    # Cleaning

    def clean_data(self) -> None:
        log_step(logger, "Cleaning Medicine Dataset...")

        if self.medicine_df is None:
            raise ValueError("Medicine dataframe is not loaded.")

        # 1. Trim whitespace

        trim_whitespace(
            self.medicine_df,
            columns=[
                "MDNAME",
                "SUPCODE",
                "PACKG",
                "DETAIL",
                "UQC"],
            logger=logger,
        )

        # 2. Normalize text

        normalize_text(
            self.medicine_df,
            columns=[
                "MDNAME",
                "SUPCODE",
                "PACKG",
                "DETAIL",
                "UQC"],
            logger=logger,
        )

        # 3. Fill optional text columns

        fill_missing_text(
            self.medicine_df,
            columns=[
                "PACKG",
                "DETAIL",
                "UQC"],
            logger=logger,
            value="UNKNOWN",
        )

        # 4. Validate primary key

        validate_no_duplicate_keys(
            self.medicine_df,
            key_columns=["MDCODE"],
            logger=logger,
            df_name="Medicine Dimension",
        )

        # 5. Validate mandatory columns

        validate_no_nulls(
            self.medicine_df,
            columns=[
                "MDCODE",
                "MDNAME",
            ],
            logger=logger,
            df_name="Medicine Dimension",
        )

        logger.info("Medicine dataset cleaned successfully.")

    # Business Transformations

    def business_transformations(self) -> None:
        if self.medicine_df is None:
            raise ValueError("Medicine dataframe is not loaded.")
        
        log_step(
            logger,
            "Creating Medicine Business Columns...")
        
        # Convert date columns
        if "NEWDT" in self.medicine_df.columns:
            self.medicine_df["NEWDT"] = pd.to_datetime(
                self.medicine_df["NEWDT"],
                errors="coerce")
            
        if "SMDT" in self.medicine_df.columns:
            self.medicine_df["SMDT"] = pd.to_datetime(
                self.medicine_df["SMDT"],
                errors="coerce")
            
        # Product Display Name
        if {"MDNAME", "PACKG"}.issubset(self.medicine_df.columns):
            self.medicine_df["Product_Display_Name"] = (
                self.medicine_df["MDNAME"].fillna("")
                + " "
                + self.medicine_df["PACKG"].fillna("")).str.strip()
            
        # Product Age
        if "NEWDT" in self.medicine_df.columns:
            today = pd.Timestamp.today().normalize()
            self.medicine_df["Product_Age_Days"] = (
                today - self.medicine_df["NEWDT"]).dt.days
            
        # Product Status
        if "SMDT" in self.medicine_df.columns:
            self.medicine_df["Product_Status"] = (
                self.medicine_df["SMDT"].notna().map({
                    True: "ACTIVE",
                    False: "INACTIVE"}))
            
        # Supplier Available
        if "SUPCODE" in self.medicine_df.columns:
            self.medicine_df["Supplier_Available"] = (
                self.medicine_df["SUPCODE"].fillna("").ne(""))
            
        # HSN Available
        if "HSN" in self.medicine_df.columns:
            self.medicine_df["HSN_Available"] = (
                self.medicine_df["HSN"].notna())
            
        logger.info(
            "Medicine business transformations completed successfully.")

    # Save Silver Dataset

    def save(
        self,
        output_path: Path,
    ) -> None:
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
        output_path: Path,
    ) -> None:
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