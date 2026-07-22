from __future__ import annotations

from pathlib import Path

import pandas as pd

from finemed_ai.transform.common.joins import merge_dimension

from finemed_ai.database.warehouse_reader import read_table
from finemed_ai.transform.common.helper_functions import (
    get_logger,
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
    join_sales_invoice_line,
    join_sales_with_salesperson,
    join_sales_with_date,
    join_sales_with_product,
    join_sales_with_tax)

logger = get_logger(__name__)

class SalesTransformer:
    """Sales Silver Layer Transformer."""

    def __init__(self):
        self.sales_invoice_df = None
        self.sales_line_df = None
        self.salesperson_df = None
        self.product_df = None
        self.date_df = None
        self.sales_df = None   # Final merged dataframe

    # Load Warehouse Tables

    # Load Warehouse Tables
    def load_data(self) -> None:
        log_step(logger,"Loading Sales Warehouse Tables...")
        
        # Fact Tables
        self.sales_invoice_df = read_table(table_name="fact_sales_invoice")
        self.sales_line_df = read_table(table_name="fact_sales_line")
        
        # Dimension Tables
        self.salesperson_df = read_table(table_name="dim_salesperson")
        
        self.product_df = read_table(table_name="dim_product")
        
        self.date_df = read_table(table_name="dim_date")
        
        # Validate Fact Tables
        validate_dataframe_not_empty(
            self.sales_invoice_df,
            logger,
            "Fact Sales Invoice")
        
        validate_dataframe_not_empty(
            self.sales_line_df,
            logger,
            "Fact Sales Line")
        
        # Log Information
        log_dataframe_info(
            logger,
            self.sales_invoice_df,
            "Fact Sales Invoice")
        
        log_dataframe_info(
            logger,
            self.sales_line_df,
            "Fact Sales Line")
        
        logger.info("Sales warehouse tables loaded successfully.")

    # Join Dimension Tables
    
    def join_dimensions(self) -> None:
        if self.sales_invoice_df is None:
            raise ValueError("Sales Invoice not loaded.")
        
        if self.sales_line_df is None:
            raise ValueError("Sales Line not loaded.")
        
        if self.salesperson_df is None:
            raise ValueError("Salesperson Dimension not loaded.")
        
        if self.product_df is None:
            raise ValueError("Product Dimension not loaded.")
        
        if self.date_df is None:
            raise ValueError("Date Dimension not loaded.")
        
        # Join Invoice + Line
        
        log_step(logger,"Joining Sales Invoice and Sales Line...")
        
        self.sales_df = merge_dimension(
            self.sales_line_df,
            self.sales_invoice_df,
            fact_key="INVNO",
            dimension_key="INVNO")
        
        # Join Salesperson
        log_step(
            logger,
            "Joining Salesperson Dimension...")
        
        self.sales_df = join_sales_with_salesperson(
        self.sales_df,
        self.salesperson_df)
        
        # Join Product
        log_step(
            logger,
            "Joining Product Dimension...")
        
        self.sales_df = join_sales_with_product(
        self.sales_df,
        self.product_df)
        
        # Join Date
        log_step(
            logger,
            "Joining Date Dimension...")
        
        self.sales_df = join_sales_with_date(
        self.sales_df,
        self.date_df)
        
        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Dataset")
        
        log_dataframe_info(
            logger,
            self.sales_df,
            "Sales Silver")
        
        logger.info("Sales dimension joins completed successfully.")

    # Cleaning
    
    def clean_data(self) -> None:
        log_step(
            logger,
            "Cleaning Sales Dataset...")
        
        if self.sales_df is None:
            raise ValueError("Sales dataframe is not loaded.")
        
        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Dataset")
        
        # Trim Whitespace
        
        trim_whitespace(
            self.sales_df,
            columns=[
                "INVNO",
                "BATCH",
                "REMARK"],logger=logger)
        
        # Normalize Text
        normalize_text(
            self.sales_df,
            columns=[
                "CCODE",
                "ACODE",
                "SCODE",
                "MDCODE",
                "SUPNO",
                "TCODE",],logger=logger)
        
        # Fill Missing Text
        fill_missing_text(
            self.sales_df,
            columns=[
                "BATCH",
                "REMARK",
                ],logger=logger)
        
        # Fill Missing Numeric
        fill_missing_numeric(
            self.sales_df,
            columns=[
                "QTY",
                "FQTY",
                "RATE",
                "SERATE",
                "ACRATE",
                "PRATE",
                "MRP",
                "TOT_AMT",
                "DISC_PER",
                "FDISC_AMT",
                "ST_AMT",
                "NET_AMT"],logger=logger)
        
        # Remove Invalid Sales Records
        self.sales_df = self.sales_df[(self.sales_df["QTY"] > 0) & (self.sales_df["NET_AMT"] > 0)].reset_index(drop=True)
        
        # Remove Duplicate Rows
        before = len(self.sales_df)
        self.sales_df = (self.sales_df.drop_duplicates().reset_index(drop=True))
        
        removed = before - len(self.sales_df)
        
        logger.info("Duplicate rows removed: %d",removed)
        
        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Silver")
        
        log_dataframe_info(
            logger,
            self.sales_df,
            "Sales Silver")
        
        logger.info(
            "Sales cleaning completed successfully.")
        
    #Business_transformations

    def business_transformations(self) -> None:
        if self.sales_df is None:
            raise ValueError("Sales dataframe is not loaded.")
        log_step(
            logger,
            "Creating Sales Business Columns...")
        
        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Dataset")
        
        # Discount Percentage
        
        if {
            "TOT_AMT",
            "NET_AMT"}.issubset(self.sales_df.columns):
            self.sales_df["Discount_Amount"] = (self.sales_df["TOT_AMT"] - self.sales_df["NET_AMT"])
            
        
        # Profit Per Unit
        if {
            "RATE",
            "PRATE"}.issubset(self.sales_df.columns):
            
            self.sales_df["Profit_Per_Unit"] = (self.sales_df["RATE"] - self.sales_df["PRATE"])
            
        # Total Profit
        if {
            "Profit_Per_Unit",
            "QTY"}.issubset(self.sales_df.columns):
            
            self.sales_df["Total_Profit"] = (self.sales_df["Profit_Per_Unit"]* self.sales_df["QTY"])
            
        # Profit Percentage
        if {
            "Profit_Per_Unit",
            "PRATE"}.issubset(self.sales_df.columns):
            self.sales_df["Profit_Percentage"] = ((self.sales_df["Profit_Per_Unit"] / self.sales_df["PRATE"]) * 100).round(2)
            
        
        # High Value Order
        if "NET_AMT" in self.sales_df.columns:
            self.sales_df["High_Value_Order_Flag"] = (self.sales_df["NET_AMT"] >= 10000)
            
        # Price Category
        if "RATE" in self.sales_df.columns:
            self.sales_df["Price_Category"] = pd.cut(
                self.sales_df["RATE"],
                bins=[
                    0,
                    100,
                    500,
                    1000,
                    float("inf")],
                    labels=[
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "PREMIUM"])
        # Sales Size
        if "QTY" in self.sales_df.columns:
            self.sales_df["Sales_Size"] = pd.cut(
                self.sales_df["QTY"],
                bins=[
                    0,
                    10,
                    50,
                    100,
                    float("inf")],
                    labels=[
                        "SMALL",
                        "MEDIUM",
                        "LARGE",
                        "BULK"])
            
        # Expired Medicine Sold
        if "EXP" in self.sales_df.columns:
            self.sales_df["EXP"] = pd.to_datetime(
                self.sales_df["EXP"],
                errors="coerce")
            
            today = pd.Timestamp.today().normalize()
            self.sales_df["Expired_Medicine_Sold"] = (
                self.sales_df["EXP"] < today)
            
            logger.info("Sales business transformations completed successfully.")
            log_dataframe_info(
                logger,
                self.sales_df,
                "Sales Silver")
    
    #Save
    def save(
    self,
    output_path: Path) -> None:
        log_step(
            logger,
            "Saving Sales Silver Dataset...")
        
        validate_dataframe_not_empty(
            self.sales_df,
            logger,
            "Sales Silver Dataset")
        
        save_parquet(
            self.sales_df,
            output_path,
            logger)
        
        logger.info(
            "Sales Silver Dataset saved successfully.")