from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

def log_step(logger: logging.Logger, message: str) -> None:
    logger.info(message)

def log_dataframe_info(logger: logging.Logger,dataframe: pd.DataFrame,dataframe_name: str) -> None:
    rows, columns = dataframe.shape
    memory_mb = dataframe.memory_usage(deep=True).sum() / (1024**2)

    logger.info("=" * 60)
    logger.info("DataFrame : %s", dataframe_name)
    logger.info("Rows      : %d", rows)
    logger.info("Columns   : %d", columns)
    logger.info("Memory    : %.2f MB", memory_mb)
    logger.info("=" * 60)

# File I/O Utilities
def load_parquet(file_path: Path,logger: logging.Logger) -> pd.DataFrame:
    if not file_path.exists():
        logger.error("Parquet file not found: %s", file_path)
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    logger.info("Loading parquet file: %s", file_path)

    try:
        dataframe = pd.read_parquet(file_path)
    except Exception:
        logger.exception("Failed to load parquet file: %s", file_path)
        raise

    rows, columns = dataframe.shape

    logger.info(
        "Loaded %d rows and %d columns.",
        rows,
        columns,
    )

    if dataframe.empty:
        logger.warning("Loaded an empty DataFrame from %s", file_path)

    return dataframe

def load_csv(file_path: Path,logger: logging.Logger,**kwargs) -> pd.DataFrame:
    if not file_path.exists():
        logger.error("CSV file not found: %s", file_path)
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.info("Loading CSV file: %s", file_path)

    try:
        dataframe = pd.read_csv(file_path, **kwargs)
    except Exception:
        logger.exception("Failed to load CSV file: %s", file_path)
        raise

    rows, columns = dataframe.shape

    logger.info(
        "Loaded %d rows and %d columns.",
        rows,
        columns,
    )

    if dataframe.empty:
        logger.warning("Loaded an empty DataFrame from %s", file_path)

    return dataframe

def save_parquet(dataframe: pd.DataFrame,file_path: Path,logger: logging.Logger,index: bool = False) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Saving parquet file: %s", file_path)

    try:
        dataframe.to_parquet(
            file_path,
            index=index,
        )
    except Exception:
        logger.exception("Failed to save parquet file: %s", file_path)
        raise

    rows, columns = dataframe.shape

    logger.info(
        "Saved %d rows and %d columns.",
        rows,
        columns,
    )

def save_csv(dataframe: pd.DataFrame,file_path: Path,logger: logging.Logger,index: bool = False) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Saving CSV file: %s", file_path)

    try:
        dataframe.to_csv(
            file_path,
            index=index,
        )
    except Exception:
        logger.exception("Failed to save CSV file: %s", file_path)
        raise

    rows, columns = dataframe.shape

    logger.info(
        "Saved %d rows and %d columns.",
        rows,
        columns,
    )

# Validation Utilities
def validate_dataframe_not_empty(df: pd.DataFrame,logger: logging.Logger,df_name: str = "DataFrame") -> None:
    if df.empty:
        logger.error("%s is empty.", df_name)
        raise ValueError(f"{df_name} is empty.")

    logger.info(
        "%s validation passed (rows=%d).",
        df_name,
        len(df),
    )

def validate_columns_exist(df: pd.DataFrame,required_columns: list[str],logger: logging.Logger,df_name: str = "DataFrame") -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        logger.error(
            "%s missing columns: %s",
            df_name,
            missing,
        )
        raise ValueError(
            f"{df_name} missing columns: {missing}"
        )

    logger.info(
        "%s contains all required columns.",
        df_name,
    )

def validate_no_duplicate_keys(df: pd.DataFrame,key_columns: list[str],logger: logging.Logger,df_name: str = "DataFrame") -> None:
    validate_columns_exist(
        df,
        key_columns,
        logger=logger,
        df_name=df_name,
    )

    duplicate_count = df.duplicated(
        subset=key_columns
    ).sum()

    if duplicate_count > 0:
        logger.error(
            "%s contains %d duplicate records using keys %s.",
            df_name,
            duplicate_count,
            key_columns,
        )
        raise ValueError(
            f"{df_name} contains duplicate records."
        )

    logger.info(
        "%s duplicate validation passed.",
        df_name,
    )

def validate_no_nulls(df: pd.DataFrame,columns: list[str],logger: logging.Logger,df_name: str = "DataFrame") -> None:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
        df_name=df_name,
    )

    for column in columns:
        null_count = df[column].isna().sum()

        if null_count > 0:
            logger.error(
                "%s.%s contains %d null values.",
                df_name,
                column,
                null_count,
            )
            raise ValueError(
                f"{column} contains null values."
            )

    logger.info(
        "%s null validation passed.",
        df_name,
    )

# Date Utilities
def convert_to_datetime(df: pd.DataFrame,column: str,logger: logging.Logger,errors: str = "coerce") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [column],
        logger=logger,
    )

    df[column] = pd.to_datetime(
        df[column],
        errors=errors,
    )

    return df

def extract_year(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Year") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = df[date_column].dt.year

    return df

def extract_month(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Month") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = df[date_column].dt.month

    return df

def extract_month_name(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Month_Name") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = df[date_column].dt.month_name()

    return df

def extract_quarter(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Quarter") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = df[date_column].dt.quarter

    return df

def extract_week(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Week") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = (
        df[date_column]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )

    return df

def extract_day(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Day") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = df[date_column].dt.day

    return df

def extract_day_name(df: pd.DataFrame,date_column: str,logger: logging.Logger,output_column: str = "Day_Name") -> pd.DataFrame:
    validate_columns_exist(
        df,
        [date_column],
        logger=logger,
    )

    df[output_column] = df[date_column].dt.day_name()

    return df

# Missing Value Utilities
def fill_missing_numeric(df: pd.DataFrame,columns: list[str],logger: logging.Logger,value: float | int = 0) -> pd.DataFrame:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
    )

    df[columns] = df[columns].fillna(value)

    return df

def fill_missing_text(df: pd.DataFrame,columns: list[str],logger: logging.Logger,value: str = "UNKNOWN") -> pd.DataFrame:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
    )

    df[columns] = df[columns].fillna(value)

    return df

def fill_missing_boolean(df: pd.DataFrame,columns: list[str],logger: logging.Logger,value: bool = False) -> pd.DataFrame:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
    )

    df[columns] = df[columns].fillna(value)

    return df

# Text Utilities

def trim_whitespace(df: pd.DataFrame,columns: list[str],logger: logging.Logger) -> pd.DataFrame:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
    )

    for column in columns:
        df[column] = df[column].astype(str).str.strip()

    return df

def normalize_text(df: pd.DataFrame,columns: list[str],logger: logging.Logger) -> pd.DataFrame:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
    )

    for column in columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    return df

# Numeric Utilities
def safe_divide(numerator: float | int,denominator: float | int,default: float = 0.0) -> float:
    if pd.isna(denominator) or denominator == 0:
        return default

    return numerator / denominator

def round_numeric_columns(df: pd.DataFrame,columns: list[str],logger: logging.Logger,decimals: int = 2) -> pd.DataFrame:
    validate_columns_exist(
        df,
        columns,
        logger=logger,
    )

    df[columns] = df[columns].round(decimals)

    return df