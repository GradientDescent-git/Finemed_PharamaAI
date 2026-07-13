"""
Business KPI utility functions.

This module provides reusable business metric calculations used by
the Analytics layer, dashboards, feature engineering, and ML pipelines.
"""

from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    safe_divide,
    validate_columns_exist,
)


# ------------------------------------------------------------------
# Revenue Metrics
# ------------------------------------------------------------------

def calculate_total_sales(
    sales_df: pd.DataFrame,
    amount_column: str,
) -> float:
    """
    Calculate total sales revenue.
    """
    validate_columns_exist(sales_df, [amount_column])

    return float(
        sales_df[amount_column].sum()
    )


def calculate_average_sale(
    sales_df: pd.DataFrame,
    amount_column: str,
) -> float:
    """
    Calculate average sales value.
    """
    validate_columns_exist(sales_df, [amount_column])

    return float(
        sales_df[amount_column].mean()
    )


def calculate_total_purchase(
    purchase_df: pd.DataFrame,
    amount_column: str,
) -> float:
    """
    Calculate total purchase amount.
    """
    validate_columns_exist(purchase_df, [amount_column])

    return float(
        purchase_df[amount_column].sum()
    )


def calculate_average_purchase(
    purchase_df: pd.DataFrame,
    amount_column: str,
) -> float:
    """
    Calculate average purchase value.
    """
    validate_columns_exist(purchase_df, [amount_column])

    return float(
        purchase_df[amount_column].mean()
    )


def calculate_unique_customers(
    sales_df: pd.DataFrame,
    customer_column: str,
) -> int:
    """
    Count unique customers.
    """
    validate_columns_exist(
        sales_df,
        [customer_column],
    )

    return int(
        sales_df[customer_column].nunique()
    )


def calculate_unique_suppliers(
    purchase_df: pd.DataFrame,
    supplier_column: str,
) -> int:
    """
    Count unique suppliers.
    """
    validate_columns_exist(
        purchase_df,
        [supplier_column],
    )

    return int(
        purchase_df[supplier_column].nunique()
    )


# ------------------------------------------------------------------
# Financial KPIs
# ------------------------------------------------------------------

def calculate_profit(
    revenue: float,
    cost: float,
) -> float:
    """
    Calculate absolute profit.
    """
    return float(revenue - cost)


def calculate_profit_margin(
    revenue: float,
    cost: float,
) -> float:
    """
    Calculate profit margin percentage.
    """
    profit = calculate_profit(
        revenue,
        cost,
    )

    return round(
        safe_divide(
            profit * 100,
            revenue,
        ),
        2,
    )


def calculate_percentage_change(
    current_value: float,
    previous_value: float,
) -> float:
    """
    Calculate percentage change.
    """
    difference = current_value - previous_value

    return round(
        safe_divide(
            difference * 100,
            previous_value,
        ),
        2,
    )


def calculate_growth_rate(
    current_value: float,
    previous_value: float,
) -> float:
    """
    Calculate growth rate percentage.
    """
    return calculate_percentage_change(
        current_value,
        previous_value,
    )


# ------------------------------------------------------------------
# Inventory KPIs
# ------------------------------------------------------------------

def calculate_inventory_turnover(
    cost_of_goods_sold: float,
    average_inventory: float,
) -> float:
    """
    Calculate inventory turnover.
    """
    return round(
        safe_divide(
            cost_of_goods_sold,
            average_inventory,
        ),
        2,
    )


def calculate_days_of_inventory(
    inventory_turnover: float,
) -> float:
    """
    Calculate Days Inventory Outstanding (DIO).
    """
    return round(
        safe_divide(
            365,
            inventory_turnover,
        ),
        2,
    )


def calculate_stock_availability(
    inventory_df: pd.DataFrame,
    stock_column: str,
) -> float:
    """
    Calculate total available stock.
    """
    validate_columns_exist(
        inventory_df,
        [stock_column],
    )

    return float(
        inventory_df[stock_column].sum()
    )


def calculate_inventory_value(
    inventory_df: pd.DataFrame,
    quantity_column: str,
    unit_cost_column: str,
) -> float:
    """
    Calculate total inventory value.
    """
    validate_columns_exist(
        inventory_df,
        [
            quantity_column,
            unit_cost_column,
        ],
    )

    return float(
        (
            inventory_df[quantity_column]
            * inventory_df[unit_cost_column]
        ).sum()
    )


def calculate_stock_utilization(
    sold_quantity: float,
    purchased_quantity: float,
) -> float:
    """
    Calculate stock utilization percentage.
    """
    return round(
        safe_divide(
            sold_quantity * 100,
            purchased_quantity,
        ),
        2,
    )


def calculate_stock_out_rate(
    stock_out_count: int,
    total_orders: int,
) -> float:
    """
    Calculate stock-out rate percentage.
    """
    return round(
        safe_divide(
            stock_out_count * 100,
            total_orders,
        ),
        2,
    )