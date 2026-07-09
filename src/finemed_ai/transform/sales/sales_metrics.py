from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
    safe_divide,
)

#Revenue metrics

def calculate_total_sales(sales_df: pd.DataFrame,amount_column: str,) -> float:
    validate_columns_exist(
        sales_df,
        [amount_column])
    return float(sales_df[amount_column].sum())

def calculate_daily_sales(sales_df: pd.DataFrame,date_column: str,amount_column: str) -> pd.DataFrame:
    validate_columns_exist(
        sales_df,
        [date_column, amount_column])
    
    return (
        sales_df
        .groupby(date_column)[amount_column]
        .sum()
        .reset_index(name="Daily_Sales")
    )

def calculate_monthly_sales(
    sales_df: pd.DataFrame,
    month_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        sales_df,
        [month_column, amount_column],
    )

    return (
        sales_df
        .groupby(month_column)[amount_column]
        .sum()
        .reset_index(name="Monthly_Sales")
    )

def calculate_yearly_sales(
    sales_df: pd.DataFrame,
    year_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        sales_df,
        [year_column, amount_column],
    )

    return (
        sales_df
        .groupby(year_column)[amount_column]
        .sum()
        .reset_index(name="Yearly_Sales")
    )

#Customer metrics

def calculate_customer_sales(
    sales_df: pd.DataFrame,
    customer_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        sales_df,
        [customer_column, amount_column],
    )

    return (
        sales_df
        .groupby(customer_column)[amount_column]
        .sum()
        .reset_index(name="Customer_Sales")
    )

def calculate_average_customer_order(
    sales_df: pd.DataFrame,
    customer_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        sales_df,
        [customer_column, amount_column],
    )

    return (
        sales_df
        .groupby(customer_column)[amount_column]
        .mean()
        .reset_index(name="Average_Order")
    )

#Product metrics

def calculate_product_sales(
    sales_df: pd.DataFrame,
    medicine_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        sales_df,
        [medicine_column, amount_column],
    )

    return (
        sales_df
        .groupby(medicine_column)[amount_column]
        .sum()
        .reset_index(name="Product_Sales")
    )

def calculate_top_selling_products(
    sales_df: pd.DataFrame,
    medicine_column: str,
    quantity_column: str,
    top_n: int = 10,
) -> pd.DataFrame:

    validate_columns_exist(
        sales_df,
        [medicine_column, quantity_column],
    )

    return (
        sales_df
        .groupby(medicine_column)[quantity_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="Quantity_Sold")
    )

#Order metrics

def calculate_total_orders(
    sales_df: pd.DataFrame,
    invoice_column: str,
) -> int:

    validate_columns_exist(
        sales_df,
        [invoice_column],
    )

    return int(
        sales_df[invoice_column].nunique()
    )

def calculate_average_order_value(
    total_sales: float,
    total_orders: int,
) -> float:

    return round(
        safe_divide(
            total_sales,
            total_orders,
        ),
        2,
    )

#Growth metrics

def calculate_sales_growth(
    current_sales: float,
    previous_sales: float,
) -> float:

    difference = current_sales - previous_sales

    return round(
        safe_divide(
            difference * 100,
            previous_sales,
        ),
        2,
    )

#Return rate

def calculate_return_rate(
    returned_orders: int,
    total_orders: int,
) -> float:

    return round(
        safe_divide(
            returned_orders * 100,
            total_orders,
        ),
        2,
    )

