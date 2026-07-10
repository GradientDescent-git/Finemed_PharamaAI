from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
    safe_divide,
)

# Purchase Value Metrics
def calculate_total_purchase(
    purchase_df: pd.DataFrame,
    amount_column: str,
) -> float:

    validate_columns_exist(
        purchase_df,
        [amount_column],
    )

    return float(
        purchase_df[amount_column].sum()
    )


def calculate_daily_purchase(
    purchase_df: pd.DataFrame,
    date_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [date_column, amount_column],
    )

    return (
        purchase_df
        .groupby(date_column)[amount_column]
        .sum()
        .reset_index(name="Daily_Purchase")
    )


def calculate_monthly_purchase(
    purchase_df: pd.DataFrame,
    month_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [month_column, amount_column],
    )

    return (
        purchase_df
        .groupby(month_column)[amount_column]
        .sum()
        .reset_index(name="Monthly_Purchase")
    )


def calculate_yearly_purchase(
    purchase_df: pd.DataFrame,
    year_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [year_column, amount_column],
    )

    return (
        purchase_df
        .groupby(year_column)[amount_column]
        .sum()
        .reset_index(name="Yearly_Purchase")
    )

   # Supplier Metrics
def calculate_supplier_purchase(
    purchase_df: pd.DataFrame,
    supplier_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [supplier_column, amount_column],
    )

    return (
        purchase_df
        .groupby(supplier_column)[amount_column]
        .sum()
        .reset_index(name="Supplier_Purchase")
    )


def calculate_average_supplier_order(
    purchase_df: pd.DataFrame,
    supplier_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [supplier_column, amount_column],
    )

    return (
        purchase_df
        .groupby(supplier_column)[amount_column]
        .mean()
        .reset_index(name="Average_Order")
    )
    
    # Medicine Purchase Metrics
def calculate_medicine_purchase(
    purchase_df: pd.DataFrame,
    medicine_column: str,
    amount_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [medicine_column, amount_column],
    )

    return (
        purchase_df
        .groupby(medicine_column)[amount_column]
        .sum()
        .reset_index(name="Medicine_Purchase")
    )


def calculate_top_purchased_medicines(
    purchase_df: pd.DataFrame,
    medicine_column: str,
    quantity_column: str,
    top_n: int = 10,
) -> pd.DataFrame:

    validate_columns_exist(
        purchase_df,
        [medicine_column, quantity_column],
    )

    return (
        purchase_df
        .groupby(medicine_column)[quantity_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="Quantity_Purchased")
    )

# Purchase Order Metrics
def calculate_total_purchase_orders(
    purchase_df: pd.DataFrame,
    invoice_column: str,
) -> int:

    validate_columns_exist(
        purchase_df,
        [invoice_column],
    )

    return int(
        purchase_df[invoice_column].nunique()
    )


def calculate_average_purchase_order_value(
    total_purchase: float,
    total_orders: int,
) -> float:

    return round(
        safe_divide(
            total_purchase,
            total_orders,
        ),
        2,
    )

# Procurement Growth Metrics
def calculate_purchase_growth(
    current_purchase: float,
    previous_purchase: float,
) -> float:

    difference = current_purchase - previous_purchase

    return round(
        safe_divide(
            difference * 100,
            previous_purchase,
        ),
        2,
    )

# Procurement Cost Metrics
def calculate_average_unit_cost(
    purchase_df: pd.DataFrame,
    quantity_column: str,
    amount_column: str,
) -> float:

    validate_columns_exist(
        purchase_df,
        [quantity_column, amount_column],
    )

    total_quantity = purchase_df[quantity_column].sum()
    total_amount = purchase_df[amount_column].sum()

    return round(
        safe_divide(
            total_amount,
            total_quantity,
        ),
        2,
    )