from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
    safe_divide,
)

# ==========================================================
# Revenue Metrics
# ==========================================================

def calculate_total_sales(
    sales_df: pd.DataFrame,
    amount_column: str,
) -> float:
    """
    Calculate total sales amount.
    """

    validate_columns_exist(
        sales_df,
        [amount_column],
    )

    return float(
        sales_df[amount_column].sum()
    )


def calculate_daily_sales(
    sales_df: pd.DataFrame,
    date_column: str,
    amount_column: str,
) -> pd.DataFrame:
    """
    Calculate daily sales.
    """

    validate_columns_exist(
        sales_df,
        [date_column, amount_column],
    )

    return (
        sales_df
        .groupby(date_column, as_index=False)[amount_column]
        .sum()
        .rename(
            columns={
                amount_column: "Daily_Sales",
            }
        )
    )


def calculate_monthly_sales(
    sales_df: pd.DataFrame,
    month_column: str,
    amount_column: str,
) -> pd.DataFrame:
    """
    Calculate monthly sales.
    """

    validate_columns_exist(
        sales_df,
        [month_column, amount_column],
    )

    return (
        sales_df
        .groupby(month_column, as_index=False)[amount_column]
        .sum()
        .rename(
            columns={
                amount_column: "Monthly_Sales",
            }
        )
    )


def calculate_yearly_sales(
    sales_df: pd.DataFrame,
    year_column: str,
    amount_column: str,
) -> pd.DataFrame:
    """
    Calculate yearly sales.
    """

    validate_columns_exist(
        sales_df,
        [year_column, amount_column],
    )

    return (
        sales_df
        .groupby(year_column, as_index=False)[amount_column]
        .sum()
        .rename(
            columns={
                amount_column: "Yearly_Sales",
            }
        )
    )


# ==========================================================
# Customer Metrics
# ==========================================================

def calculate_customer_sales(
    sales_df: pd.DataFrame,
    customer_column: str,
    amount_column: str,
) -> pd.DataFrame:
    """
    Calculate total sales by customer.
    """

    validate_columns_exist(
        sales_df,
        [customer_column, amount_column],
    )

    return (
        sales_df
        .groupby(customer_column, as_index=False)[amount_column]
        .sum()
        .rename(
            columns={
                amount_column: "Customer_Sales",
            }
        )
    )


def calculate_average_customer_order(
    sales_df: pd.DataFrame,
    customer_column: str,
    amount_column: str,
) -> pd.DataFrame:
    """
    Calculate average order value per customer.
    """

    validate_columns_exist(
        sales_df,
        [customer_column, amount_column],
    )

    return (
        sales_df
        .groupby(customer_column, as_index=False)[amount_column]
        .mean()
        .rename(
            columns={
                amount_column: "Average_Order_Value",
            }
        )
    )


# ==========================================================
# Product Metrics
# ==========================================================

def calculate_product_sales(
    sales_df: pd.DataFrame,
    medicine_column: str,
    amount_column: str,
) -> pd.DataFrame:
    """
    Calculate sales by product.
    """

    validate_columns_exist(
        sales_df,
        [medicine_column, amount_column],
    )

    return (
        sales_df
        .groupby(medicine_column, as_index=False)[amount_column]
        .sum()
        .rename(
            columns={
                amount_column: "Product_Sales",
            }
        )
    )


def calculate_top_selling_products(
    sales_df: pd.DataFrame,
    medicine_column: str,
    quantity_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Calculate top selling products.
    """

    validate_columns_exist(
        sales_df,
        [medicine_column, quantity_column],
    )

    return (
        sales_df
        .groupby(medicine_column, as_index=False)[quantity_column]
        .sum()
        .sort_values(
            by=quantity_column,
            ascending=False,
        )
        .head(top_n)
        .rename(
            columns={
                quantity_column: "Quantity_Sold",
            }
        )
    )


# ==========================================================
# Order Metrics
# ==========================================================

def calculate_total_orders(
    sales_df: pd.DataFrame,
    invoice_column: str,
) -> int:
    """
    Calculate total unique orders.
    """

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
    """
    Calculate average order value.
    """

    return round(
        safe_divide(
            total_sales,
            total_orders,
        ),
        2,
    )


# ==========================================================
# Growth Metrics
# ==========================================================

def calculate_sales_growth(
    current_sales: float,
    previous_sales: float,
) -> float:
    """
    Calculate sales growth percentage.
    """

    return round(
        safe_divide(
            (current_sales - previous_sales) * 100,
            previous_sales,
        ),
        2,
    )


# ==========================================================
# Return Metrics
# ==========================================================

def calculate_return_rate(
    returned_orders: int,
    total_orders: int,
) -> float:
    """
    Calculate return rate percentage.
    """

    return round(
        safe_divide(
            returned_orders * 100,
            total_orders,
        ),
        2,
    )


# ==========================================================
# Business Metrics
# ==========================================================

def calculate_repeat_customer_rate(
    repeat_customers: int,
    total_customers: int,
) -> float:
    """
    Calculate repeat customer percentage.
    """

    return round(
        safe_divide(
            repeat_customers * 100,
            total_customers,
        ),
        2,
    )


def calculate_average_items_per_order(
    total_quantity: float,
    total_orders: int,
) -> float:
    """
    Calculate average items sold per order.
    """

    return round(
        safe_divide(
            total_quantity,
            total_orders,
        ),
        2,
    )


def calculate_product_contribution(
    product_sales: float,
    total_sales: float,
) -> float:
    """
    Calculate product contribution to total sales.
    """

    return round(
        safe_divide(
            product_sales * 100,
            total_sales,
        ),
        2,
    )


def calculate_customer_contribution(
    customer_sales: float,
    total_sales: float,
) -> float:
    """
    Calculate customer contribution to total sales.
    """

    return round(
        safe_divide(
            customer_sales * 100,
            total_sales,
        ),
        2,
    )


def calculate_sales_per_customer(
    total_sales: float,
    total_customers: int,
) -> float:
    """
    Calculate average sales per customer.
    """

    return round(
        safe_divide(
            total_sales,
            total_customers,
        ),
        2,
    )


def calculate_sales_per_day(
    total_sales: float,
    total_days: int,
) -> float:
    """
    Calculate average daily sales.
    """

    return round(
        safe_divide(
            total_sales,
            total_days,
        ),
        2,
    )