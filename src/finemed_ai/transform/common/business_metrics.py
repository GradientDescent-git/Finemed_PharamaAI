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

def calculate_total_sales(sales_df: pd.DataFrame, amount_column: str) -> float:
    """Calculate total sales revenue."""
    validate_columns_exist(sales_df, [amount_column])
    return float(sales_df[amount_column].sum())


def calculate_average_sale(sales_df: pd.DataFrame, amount_column: str) -> float:
    """Calculate average sales value."""
    validate_columns_exist(sales_df, [amount_column])
    return float(sales_df[amount_column].mean())


def calculate_total_purchase(purchase_df: pd.DataFrame, amount_column: str) -> float:
    """Calculate total purchase amount."""
    validate_columns_exist(purchase_df, [amount_column])
    return float(purchase_df[amount_column].sum())


def calculate_average_purchase(purchase_df: pd.DataFrame, amount_column: str) -> float:
    """Calculate average purchase value."""
    validate_columns_exist(purchase_df, [amount_column])
    return float(purchase_df[amount_column].mean())


def calculate_unique_customers(sales_df: pd.DataFrame, customer_column: str) -> int:
    """Count unique customers."""
    validate_columns_exist(sales_df, [customer_column])
    return int(sales_df[customer_column].nunique())


def calculate_unique_suppliers(purchase_df: pd.DataFrame, supplier_column: str) -> int:
    """Count unique suppliers."""
    validate_columns_exist(purchase_df, [supplier_column])
    return int(purchase_df[supplier_column].nunique())


# ------------------------------------------------------------------
# Profitability & Growth Metrics
# ------------------------------------------------------------------

def calculate_profit(revenue: float, cost: float) -> float:
    """Calculate gross profit."""
    return revenue - cost


def calculate_profit_margin(profit: float, revenue: float) -> float:
    """Calculate profit margin percentage."""
    return safe_divide(profit * 100.0, revenue, fill_value=0.0)


def calculate_percentage_change(current_value: float, previous_value: float) -> float:
    """Calculate percentage growth between current and previous values."""
    if previous_value == 0.0:
        return 0.0
    return ((current_value - previous_value) / abs(previous_value)) * 100.0


def calculate_growth_rate(current_value: float, previous_value: float) -> float:
    """Alias for calculate_percentage_change."""
    return calculate_percentage_change(current_value, previous_value)


# ------------------------------------------------------------------
# Inventory Control Metrics
# ------------------------------------------------------------------

def calculate_inventory_turnover(cogs: float, average_inventory: float) -> float:
    """Calculate inventory turnover ratio."""
    return safe_divide(cogs, average_inventory, fill_value=0.0)


def calculate_days_inventory_outstanding(inventory_turnover: float, days_in_period: int = 365) -> float:
    """Calculate Days Inventory Outstanding (DIO)."""
    return safe_divide(float(days_in_period), inventory_turnover, fill_value=0.0)