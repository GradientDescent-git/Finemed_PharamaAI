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
    fill_missing_numeric
)

from finemed_ai.transform.common.joins import (
    merge_dimension,
)

logger = get_logger(__name__)

class PurchaseTransformer:
    def __init__(self):
        self.purchase_header_df: pd.DataFrame | None = None
        self.purchase_line_df: pd.DataFrame | None = None
        self.supplier_df: pd.DataFrame | None = None
        self.medicine_df: pd.DataFrame | None = None
        self.date_df: pd.DataFrame | None = None
        self.purchase_df: pd.DataFrame | None = None

    # Load Warehouse Tables

    def load_data(self) -> None:
        log_step(logger, "Loading Purchase Warehouse Tables...")

        self.purchase_header_df = read_table(
            table_name="fact_purchase_header")

        self.purchase_line_df = read_table(
            table_name="fact_purchase_line")

        self.supplier_df = read_table(
            table_name="dim_supplier")

        self.medicine_df = read_table(
            table_name= "dim_product")

        self.date_df = read_table(
            table_name="dim_date")

        validate_dataframe_not_empty(
            self.purchase_header_df,
            logger,
            "Fact Purchase Header",
        )

        validate_dataframe_not_empty(
            self.purchase_line_df,
            logger,
            "Fact Purchase Line",
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
        if self.purchase_header_df is None:
            raise ValueError("Purchase Header not loaded.")
        
        if self.purchase_line_df is None:
            raise ValueError("Purchase Line not loaded.")
        
        if self.supplier_df is None:
            raise ValueError("Supplier Dimension not loaded.")
        
        if self.medicine_df is None:
            raise ValueError("Medicine Dimension not loaded.")
        
        if self.date_df is None:
            raise ValueError("Date Dimension not loaded.")
        
        # Remove ETL metadata from dimensions
        self.supplier_df = self.supplier_df.drop(
            columns=["SOURCE_MONTH"],
            errors="ignore")
        
        self.medicine_df = self.medicine_df.drop(
            columns=["SOURCE_MONTH"],
            errors="ignore")
        
        self.date_df = self.date_df.drop(
            columns=["SOURCE_MONTH"],
            errors="ignore")
        
        log_step(logger, "Joining Purchase Header and Purchase Line...")
        
        self.purchase_df = merge_dimension(
            self.purchase_line_df,
            self.purchase_header_df,
            fact_key="PINVNO",
            dimension_key="PINVNO")
        
        # Remove duplicate metadata columns created by first merge
        self.purchase_df = self.purchase_df.drop(
            columns=self.purchase_df.filter(like="SOURCE_MONTH").columns,
            errors="ignore")
        
        log_step(logger, "Joining Supplier Dimension...")
        
        self.purchase_df = merge_dimension(
            self.purchase_df,
            self.supplier_df,
            fact_key="SUPNO",
            dimension_key="SUPNO")
        
        log_step(logger, "Joining Medicine Dimension...")
        
        self.purchase_df = merge_dimension(
            self.purchase_df,
            self.medicine_df,
            fact_key="MDCODE",
            dimension_key="MDCODE")
        
        log_step(logger, "Joining Date Dimension...")
        
        self.purchase_df = merge_dimension(
            self.purchase_df,
            self.date_df,
            fact_key="PINVDT",   # use PINVDT unless your warehouse has PDATE
            dimension_key="full_date")
        
        logger.info("Purchase dimension joins completed successfully.")
        
        validate_dataframe_not_empty(
            self.purchase_df,
            logger,
            "Purchase Dataset")
    # Cleaning

    def clean_data(self) -> None:
        log_step(logger,"Cleaning Purchase Dataset...")
        
        if self.purchase_df is None:
            raise ValueError("Purchase dataframe is not loaded.")
        
        validate_dataframe_not_empty(self.purchase_df,logger,"Purchase Dataset")
        
        # Trim Whitespace
        trim_whitespace(self.purchase_df,
                        columns=[
                            "PINVNO",
                            "SUPCODE",
                            "MDNAME",
                            "BAT"],logger=logger)
        
        # Normalize Text
        
        normalize_text(self.purchase_df,
                       columns=[
                           "SUPCODE",
                           "MDNAME"],logger=logger)
        
        # Fill Missing Text
        fill_missing_text(self.purchase_df,
                          columns=[
                              "SUPCODE",
                              "BAT"],logger=logger)
        
        # Fill Missing Numeric
        fill_missing_numeric(self.purchase_df,
                             columns=[
                                 "QTY",
                                 "PRATE",
                                 "SRATE",
                                 "MRP",
                                 "SOH"],logger=logger)
        
        # Remove Invalid Purchase Records
        self.purchase_df = self.purchase_df[(self.purchase_df["QTY"] > 0) & (self.purchase_df["PRATE"] > 0)].reset_index(drop=True)
        
        # Remove Duplicate Rows
        before = len(self.purchase_df)
        
        self.purchase_df = (self.purchase_df.drop_duplicates().reset_index(drop=True))
        
        removed = before - len(self.purchase_df)
        
        logger.info("Duplicate rows removed: %d",removed)
        
        # Final Validation
        validate_dataframe_not_empty(self.purchase_df,logger,"Purchase Silver")
        log_dataframe_info(logger,self.purchase_df,"Purchase Silver")
        
        logger.info("Purchase cleaning completed successfully.")


    # Business Transformations

    def business_transformations(self) -> None:
        if self.purchase_df is None:
            raise ValueError("Purchase dataframe is not loaded.")
        
        log_step(logger,"Creating Purchase Business Columns...")
        
        # Purchase Value
        if {"QTY", "PRATE"}.issubset(self.purchase_df.columns):
            self.purchase_df["Purchase_Value"] = (self.purchase_df["QTY"] * self.purchase_df["PRATE"] )
            
        # Selling Margin Per Unit
        if {"SRATE", "PRATE"}.issubset(self.purchase_df.columns):
            self.purchase_df["Margin_Per_Unit"] = (self.purchase_df["SRATE"] - self.purchase_df["PRATE"])
            
        # Margin Percentage
        if {"Margin_Per_Unit", "PRATE"}.issubset(self.purchase_df.columns):
            self.purchase_df["Margin_Percentage"] = ((self.purchase_df["Margin_Per_Unit"] / self.purchase_df["PRATE"]) * 100).round(2)
            
        # Purchase Price Category
        if "PRATE" in self.purchase_df.columns:
            self.purchase_df["Price_Category"] = pd.cut(self.purchase_df["PRATE"],bins=[0,100,500,1000,float("inf"),],
                                                        labels=[
                                                            "LOW",
                                                            "MEDIUM",
                                                            "HIGH",
                                                            "PREMIUM"])
        
        # Purchase Size 
        if "QTY" in self.purchase_df.columns:
            self.purchase_df["Purchase_Size"] = pd.cut(self.purchase_df["QTY"],
                                                       bins=[0,10,50,100,float("inf")],
                                                       labels=[
                                                           "SMALL",
                                                           "MEDIUM",
                                                           "LARGE",
                                                           "BULK"])
        
        # Large Purchase Flag
        if "Purchase_Value" in self.purchase_df.columns:
            self.purchase_df["Large_Purchase_Flag"] = (self.purchase_df["Purchase_Value"] >= 10000)
            
        # Expired Batch Flag
        if "EXP" in self.purchase_df.columns:
            self.purchase_df["EXP"] = pd.to_datetime(
                self.purchase_df["EXP"],
                errors="coerce")
            
            today = pd.Timestamp.today().normalize()
            self.purchase_df["Expired_Batch"] = (
                self.purchase_df["EXP"] < today)
            
        logger.info("Purchase business transformations completed successfully.")
        
        log_dataframe_info(logger,
                           self.purchase_df,
                           "Purchase Silver")
        
    #save 
    def save(
    self,
    output_path: Path) -> None:
        log_step(logger,"Saving Purchase Silver Dataset...")
        validate_dataframe_not_empty(self.purchase_df,logger,"Purchase Silver Dataset",)
        save_parquet(self.purchase_df,output_path,logger)
        logger.info("Purchase Silver Dataset saved successfully.")


    # Pipeline

    def run(
    self,
    output_path: Path,) -> None:
        
        try:
            log_step(logger,"Starting Purchase Silver Transformation...")
            
            self.load_data()
            self.join_dimensions()
            self.clean_data()
            self.business_transformations()
            self.save(output_path)
            
            logger.info("Purchase Transformation Completed Successfully. "
                        "Output saved to %s",
                        output_path)
            
        except Exception:
            logger.exception("Purchase Transformation Failed.")
            raise