from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger
from finemed_ai.transform.common.helper_functions import (
    validate_columns_exist,
)

logger = get_logger(__name__)


# Generic Merge Function


def merge_dimension(fact_df: pd.DataFrame,dimension_df: pd.DataFrame,fact_key: str,dimension_key: str,how: str = "left") -> pd.DataFrame:
    logger.info(
        "Joining fact (%d rows) with dimension (%d rows) on %s -> %s",
        len(fact_df),
        len(dimension_df),
        fact_key,
        dimension_key)
    try:
        validate_columns_exist(
            fact_df,
            [fact_key],
            logger=logger,
            df_name="Fact Table")
        
        validate_columns_exist(
            dimension_df,
            [dimension_key],
            logger=logger,
            df_name="Dimension Table")
        
        # Remove duplicate columns from dimension except join key
        duplicate_columns = [
            col
            for col in dimension_df.columns
            if col in fact_df.columns and col != dimension_key]
        
        dimension_df = dimension_df.drop(
            columns=duplicate_columns,
            errors="ignore")
        
        merged_df = fact_df.merge(
            dimension_df,
            left_on=fact_key,
            right_on=dimension_key,
            how=how)
        
        if fact_key != dimension_key:
            merged_df = merged_df.drop(
                columns=[dimension_key],
                errors="ignore")
            
        logger.info(
            "Join completed successfully | rows=%d | columns=%d",
            len(merged_df),
            len(merged_df.columns))
            
        return merged_df
        
    except Exception:
        logger.exception(
            "Failed while joining %s -> %s",
            fact_key,
            dimension_key)
        raise


# SALES WRAPPERS


def join_sales_invoice_line(
    sales_line_df: pd.DataFrame,
    sales_invoice_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        sales_line_df,
        sales_invoice_df,
        fact_key="INVNO",
        dimension_key="INVNO",
    )


def join_sales_with_salesperson(
    sales_df: pd.DataFrame,
    salesperson_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        sales_df,
        salesperson_df,
        fact_key="SCODE",
        dimension_key="SCODE",
    )


def join_sales_with_date(
    sales_df: pd.DataFrame,
    date_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        sales_df,
        date_df,
        fact_key="INVDT",
        dimension_key="full_date",
    )


def join_sales_with_product(
    sales_df: pd.DataFrame,
    medicine_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        sales_df,
        medicine_df,
        fact_key="MDCODE",
        dimension_key="MDCODE",
    )


def join_sales_with_tax(
    sales_df: pd.DataFrame,
    tax_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        sales_df,
        tax_df,
        fact_key="TCODE",
        dimension_key="TCODE",
    )


# PURCHASE WRAPPERS


def join_purchase_header_line(
    purchase_line_df: pd.DataFrame,
    purchase_header_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        purchase_line_df,
        purchase_header_df,
        fact_key="PINVNO",
        dimension_key="PINVNO",
    )


def join_purchase_with_supplier(
    purchase_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        purchase_df,
        supplier_df,
        fact_key="SUPNO",
        dimension_key="SUPNO",
    )


def join_purchase_with_medicine(
    purchase_df: pd.DataFrame,
    medicine_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        purchase_df,
        medicine_df,
        fact_key="MDCODE",
        dimension_key="MDCODE",
    )


def join_purchase_with_tax(
    purchase_df: pd.DataFrame,
    tax_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        purchase_df,
        tax_df,
        fact_key="TCODE",
        dimension_key="TCODE",
    )


def join_purchase_with_date(
    purchase_df: pd.DataFrame,
    date_df: pd.DataFrame,
) -> pd.DataFrame:
    return merge_dimension(
        purchase_df,
        date_df,
        fact_key="PINVDT",
        dimension_key="full_date",
    )


# MULTIPLE JOINS


def join_multiple_tables(
    base_df: pd.DataFrame,
    joins: list[tuple[pd.DataFrame, str, str, str]],
) -> pd.DataFrame:
    """
    Sequentially performs multiple joins.

    joins = [
        (dimension_df, fact_key, dimension_key, how),
        ...
    ]
    """

    logger.info(
        "Starting multiple joins | total joins=%d",
        len(joins),
    )

    result_df = base_df.copy()

    try:
        for dataframe, left_key, right_key, how in joins:

            result_df = merge_dimension(
                result_df,
                dataframe,
                left_key,
                right_key,
                how,
            )

        logger.info(
            "All joins completed successfully | rows=%d | columns=%d",
            len(result_df),
            len(result_df.columns),
        )

        return result_df

    except Exception:
        logger.exception(
            "Failed while performing multiple joins."
        )
        raise