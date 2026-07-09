from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
)

def merge_dimension(fact_df: pd.DataFrame,dimension_df: pd.DataFrame,fact_key: str,dimension_key: str,how: str = "left") -> pd.DataFrame:
    validate_columns_exist(fact_df,[fact_key])
    validate_columns_exist(dimension_df,[dimension_key])
    return fact_df.merge(dimension_df,left_on=fact_key,right_on=dimension_key,how=how)

def join_sales_with_customer(sales_df: pd.DataFrame,customer_df: pd.DataFrame) -> pd.DataFrame:
    return merge_dimension(sales_df,customer_df,fact_key="Customer_ID",dimension_key="Customer_ID")

def join_sales_with_date(sales_df: pd.DataFrame,date_df: pd.DataFrame) -> pd.DataFrame:
    return merge_dimension(sales_df,date_df,fact_key="Date_ID",dimension_key="Date_ID")

def join_purchase_with_supplier(purchase_df: pd.DataFrame,supplier_df: pd.DataFrame) -> pd.DataFrame:
    return merge_dimension(purchase_df,supplier_df,fact_key="Supplier_ID",dimension_key="Supplier_ID")

def join_purchase_with_medicine(purchase_df: pd.DataFrame,medicine_df: pd.DataFrame) -> pd.DataFrame:
    return merge_dimension(purchase_df,medicine_df,fact_key="Medicine_ID",dimension_key="Medicine_ID")

def join_inventory_with_medicine(inventory_df: pd.DataFrame,medicine_df: pd.DataFrame) -> pd.DataFrame:
    return merge_dimension(inventory_df,medicine_df,fact_key="Medicine_ID",dimension_key="Medicine_ID")

def join_multiple_tables(base_df: pd.DataFrame,joins: list[tuple]) -> pd.DataFrame:
    result = base_df.copy()

    for dataframe, left_key, right_key, how in joins:

        result = merge_dimension(
            result,
            dataframe,
            left_key,
            right_key,
            how,
        )

    return result