from __future__ import annotations

import pandas as pd

from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
    safe_divide,
)
# Stock Quantity Metrics

def calculate_total_stock(
    inventory_df: pd.DataFrame,
    quantity_column: str) -> float:
    
    """ Calculate the total stock quantity. """

    logger.info("Calculating Total Stock")

    try:
        validate_columns_exist(
            inventory_df,
            [quantity_column],
            logger=logger,
            df_name="Inventory Data",
        )

        total_stock = float(
            inventory_df[quantity_column].sum()
        )

        logger.info(
            "Total Stock calculated successfully | value=%s",
            total_stock,
        )

        return total_stock

    except Exception as error:
        logger.exception(
            "Failed while calculating Total Stock."
        )

        raise RuntimeError(
            "Total Stock calculation failed."
        ) from error


def calculate_average_stock(
    inventory_df: pd.DataFrame,
    quantity_column: str,
) -> float:
    """ Calculate average stock quantity. """

    logger.info("Calculating Average Stock")

    try:
        validate_columns_exist(
            inventory_df,
            [quantity_column],
            logger=logger,
            df_name="Inventory Data",
        )

        average_stock = float(
            inventory_df[quantity_column].mean()
        )

        logger.info(
            "Average Stock calculated successfully | value=%s",
            average_stock,
        )

        return average_stock

    except Exception as error:
        logger.exception(
            "Failed while calculating Average Stock."
        )

        raise RuntimeError(
            "Average Stock calculation failed."
        ) from error

# Inventory Turnover Metrics

def calculate_inventory_turnover(
    cost_of_goods_sold: float,
    average_inventory: float) -> float:
    """ Calculate inventory turnover ratio.  """

    logger.info("Calculating Inventory Turnover")

    try:
        turnover = round(
            safe_divide(
                cost_of_goods_sold,
                average_inventory,
            ),
            2,
        )

        logger.info(
            "Inventory Turnover calculated successfully | value=%s",
            turnover,
        )

        return turnover

    except Exception as error:
        logger.exception(
            "Failed while calculating Inventory Turnover."
        )

        raise RuntimeError(
            "Inventory Turnover calculation failed."
        ) from error


def calculate_days_of_inventory(
    inventory_turnover: float) -> float:
    """ Calculate Days of Inventory. """

    logger.info("Calculating Days of Inventory")

    try:
        days = round(
            safe_divide(
                365,
                inventory_turnover,
            ),
            2,
        )

        logger.info(
            "Days of Inventory calculated successfully | value=%s",
            days,
        )

        return days

    except Exception as error:
        logger.exception(
            "Failed while calculating Days of Inventory."
        )

        raise RuntimeError(
            "Days of Inventory calculation failed."
        ) from error


# Stock Utilization Metrics

def calculate_stock_utilization(
    sold_quantity: float,
    available_quantity: float) -> float:
    """ Calculate stock utilization percentage.  """

    logger.info("Calculating Stock Utilization")

    try:
        utilization = round(
            safe_divide(
                sold_quantity * 100,
                available_quantity,
            ),
            2,
        )

        logger.info(
            "Stock Utilization calculated successfully | value=%s",
            utilization,
        )

        return utilization

    except Exception as error:
        logger.exception(
            "Failed while calculating Stock Utilization."
        )

        raise RuntimeError(
            "Stock Utilization calculation failed."
        ) from error


def calculate_stock_availability(
    available_quantity: float,
    required_quantity: float) -> float:
    """ Calculate stock availability percentage. """

    logger.info("Calculating Stock Availability")

    try:
        availability = round(
            safe_divide(
                available_quantity * 100,
                required_quantity,
            ),
            2,
        )

        logger.info(
            "Stock Availability calculated successfully | value=%s",
            availability,
        )

        return availability

    except Exception as error:
        logger.exception(
            "Failed while calculating Stock Availability."
        )

        raise RuntimeError(
            "Stock Availability calculation failed."
        ) from error

# Reorder Metrics

def calculate_reorder_rate(
    reorder_items: int,
    total_items: int) -> float:
    """ Calculate reorder rate. """

    logger.info("Calculating Reorder Rate")

    try:
        reorder_rate = round(
            safe_divide(
                reorder_items * 100,
                total_items,
            ),
            2,
        )

        logger.info(
            "Reorder Rate calculated successfully | value=%s",
            reorder_rate,
        )

        return reorder_rate

    except Exception as error:
        logger.exception(
            "Failed while calculating Reorder Rate."
        )

        raise RuntimeError(
            "Reorder Rate calculation failed."
        ) from error


# Dead Stock Metrics

def calculate_dead_stock_rate(
    dead_stock_items: int,
    total_items: int) -> float:
    """ Calculate dead stock rate. """

    logger.info("Calculating Dead Stock Rate")

    try:
        dead_stock_rate = round(
            safe_divide(
                dead_stock_items * 100,
                total_items,
            ),
            2,
        )

        logger.info(
            "Dead Stock Rate calculated successfully | value=%s",
            dead_stock_rate,
        )

        return dead_stock_rate

    except Exception as error:
        logger.exception(
            "Failed while calculating Dead Stock Rate."
        )

        raise RuntimeError(
            "Dead Stock Rate calculation failed."
        ) from error


# Low Stock Metrics

def calculate_low_stock_rate(
    low_stock_items: int,
    total_items: int) -> float:
    """ Calculate low stock rate. """

    logger.info("Calculating Low Stock Rate")

    try:
        low_stock_rate = round(
            safe_divide(
                low_stock_items * 100,
                total_items,
            ),
            2,
        )

        logger.info(
            "Low Stock Rate calculated successfully | value=%s",
            low_stock_rate,
        )

        return low_stock_rate

    except Exception as error:
        logger.exception(
            "Failed while calculating Low Stock Rate."
        )

        raise RuntimeError(
            "Low Stock Rate calculation failed."
        ) from error