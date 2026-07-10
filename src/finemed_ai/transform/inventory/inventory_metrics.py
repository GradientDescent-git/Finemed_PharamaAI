from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
    safe_divide,
)
# Stock Quantity Metrics
def calculate_total_stock(
    inventory_df: pd.DataFrame,
    quantity_column: str,
) -> float:

    validate_columns_exist(
        inventory_df,
        [quantity_column],
    )

    return float(
        inventory_df[quantity_column].sum()
    )

def calculate_average_stock(
    inventory_df: pd.DataFrame,
    quantity_column: str,
) -> float:

    validate_columns_exist(
        inventory_df,
        [quantity_column],
    )

    return float(
        inventory_df[quantity_column].mean()
    )

# Inventory Value Metrics
def calculate_inventory_value(
    inventory_df: pd.DataFrame,
    quantity_column: str,
    unit_cost_column: str,
) -> float:

    validate_columns_exist(
        inventory_df,
        [quantity_column, unit_cost_column],
    )

    return float(
        (
            inventory_df[quantity_column]
            * inventory_df[unit_cost_column]
        ).sum()
    )


def calculate_average_inventory_value(
    inventory_df: pd.DataFrame,
    quantity_column: str,
    unit_cost_column: str,
) -> float:

    validate_columns_exist(
        inventory_df,
        [quantity_column, unit_cost_column],
    )

    inventory_value = (
        inventory_df[quantity_column]
        * inventory_df[unit_cost_column]
    )

    return float(
        inventory_value.mean()
    )

# Medicine Metrics
def calculate_stock_by_medicine(
    inventory_df: pd.DataFrame,
    medicine_column: str,
    quantity_column: str,
) -> pd.DataFrame:

    validate_columns_exist(
        inventory_df,
        [medicine_column, quantity_column],
    )

    return (
        inventory_df
        .groupby(medicine_column)[quantity_column]
        .sum()
        .reset_index(name="Current_Stock")
    )


def calculate_top_stocked_medicines(
    inventory_df: pd.DataFrame,
    medicine_column: str,
    quantity_column: str,
    top_n: int = 10,
) -> pd.DataFrame:

    validate_columns_exist(
        inventory_df,
        [medicine_column, quantity_column],
    )

    return (
        inventory_df
        .groupby(medicine_column)[quantity_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="Stock")
    )

# Inventory Turnover Metrics
def calculate_inventory_turnover(
    cost_of_goods_sold: float,
    average_inventory: float,
) -> float:

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

    return round(
        safe_divide(
            365,
            inventory_turnover,
        ),
        2,
    )

# Stock Utilization Metrics
def calculate_stock_utilization(
    sold_quantity: float,
    available_quantity: float,
) -> float:

    return round(
        safe_divide(
            sold_quantity * 100,
            available_quantity,
        ),
        2,
    )


def calculate_stock_availability(
    available_quantity: float,
    required_quantity: float,
) -> float:

    return round(
        safe_divide(
            available_quantity * 100,
            required_quantity,
        ),
        2,
    )

# Reorder Metrics
def calculate_reorder_rate(
    reorder_items: int,
    total_items: int,
) -> float:

    return round(
        safe_divide(
            reorder_items * 100,
            total_items,
        ),
        2,
    )

# Dead Stock Metrics
def calculate_dead_stock_rate(
    dead_stock_items: int,
    total_items: int,
) -> float:

    return round(
        safe_divide(
            dead_stock_items * 100,
            total_items,
        ),
        2,
    )
# Low Stock Metrics
def calculate_low_stock_rate(
    low_stock_items: int,
    total_items: int,
) -> float:

    return round(
        safe_divide(
            low_stock_items * 100,
            total_items,
        ),
        2,
    )