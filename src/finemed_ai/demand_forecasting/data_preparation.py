
from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging
import pandas as pd


logger = logging.getLogger(__name__)

class DemandDataPreparation:
    def __init__(
        self,
        sales_path: Path,
        output_directory: Path,
        train_end_date: str,
        validation_end_date: str,
    ):
        self.sales_path = Path(sales_path)

        self.output_directory = Path(output_directory)

        self.train_end_date = pd.Timestamp(train_end_date)

        self.validation_end_date = pd.Timestamp(validation_end_date)

        self.sales_df: Optional[pd.DataFrame] = None

        self.demand_df: Optional[pd.DataFrame] = None

        self.daily_demand_df: Optional[pd.DataFrame] = None

        self.daily_train: Optional[pd.DataFrame] = None

        self.daily_validation: Optional[pd.DataFrame] = None

        self.daily_test: Optional[pd.DataFrame] = None

        self.chronos_train: Optional[pd.DataFrame] = None

        self.chronos_validation: Optional[pd.DataFrame] = None

        self.chronos_test: Optional[pd.DataFrame] = None

        logger.info("DemandDataPreparation Initialized")

    def load_sales(self) -> pd.DataFrame:
        logger.info("Loading Silver Sales Dataset")

        if not self.sales_path.exists():
            raise FileNotFoundError(
                f"Sales dataset not found : {self.sales_path}"
            )

        self.sales_df = pd.read_parquet(self.sales_path)

        logger.info(
            "Sales Loaded | Rows=%s | Medicines=%s",
            len(self.sales_df),
            self.sales_df["MDCODE"].nunique(),
        )

        return self.sales_df

    def create_demand_dataset(self) -> pd.DataFrame:

        logger.info("Creating Demand Dataset")

        if self.sales_df is None:
            raise ValueError(
                "Sales dataset not loaded."
            )

        columns = [
            "MDCODE",
            "MDNAME",
            "PACKG",
            "DETAIL",

            "INVDT",

            "day",
            "day_name",
            "day_of_week",
            "week",
            "month",
            "month_name",
            "quarter",
            "year",
            "is_weekend",
            "financial_year",
            "financial_month",

            "RATE",
            "PRATE",
            "MRP",

            "QTY",
        ]

        demand_df = self.sales_df[columns].copy()

        demand_df = demand_df.rename(
            columns={
                "QTY": "Demand_Qty",
                "RATE": "Selling_Rate",
                "PRATE": "Purchase_Rate",
                "MRP": "Maximum_Retail_Price",
            }
        )

        demand_df = (
            demand_df
            .groupby(
                ["MDCODE", "INVDT"],
                as_index=False
            )
            .agg(
                {
                    "Demand_Qty": "sum",

                    "Selling_Rate": "mean",
                    "Purchase_Rate": "mean",
                    "Maximum_Retail_Price": "mean",

                    "MDNAME": "first",
                    "PACKG": "first",
                    "DETAIL": "first",

                    "day": "first",
                    "day_name": "first",
                    "day_of_week": "first",
                    "week": "first",
                    "month": "first",
                    "month_name": "first",
                    "quarter": "first",
                    "year": "first",
                    "is_weekend": "first",
                    "financial_year": "first",
                    "financial_month": "first",
                }
            )
        )

        demand_df["INVDT"] = pd.to_datetime(
            demand_df["INVDT"]
        )

        demand_df = (
            demand_df
            .sort_values(
                ["MDCODE", "INVDT"]
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Demand Dataset Created | Rows=%s",
            len(demand_df),
        )

        self.demand_df = demand_df

        return demand_df

    def create_daily_time_series(self) -> pd.DataFrame:

        logger.info("Creating Daily Time Series")

    if self.demand_df is None:
        raise ValueError(
            "Demand dataset not created. Run create_demand_dataset() first."
        )

    daily_series = []

    medicine_ids = sorted(
        self.demand_df["MDCODE"].unique()
    )

    logger.info(
        "Medicines Found : %s",
        len(medicine_ids)
    )

    for medicine in medicine_ids:

        medicine_df = (
            self.demand_df[
                self.demand_df["MDCODE"] == medicine
            ]
            .copy()
        )

        calendar = pd.DataFrame(
            {
                "INVDT": pd.date_range(
                    start=medicine_df["INVDT"].min(),
                    end=medicine_df["INVDT"].max(),
                    freq="D",
                )
            }
        )

        calendar["MDCODE"] = medicine

        merged = calendar.merge(
            medicine_df,
            on=["MDCODE", "INVDT"],
            how="left",
        )

        # Fill Demand

        merged["Demand_Qty"] = (
            merged["Demand_Qty"]
            .fillna(0)
            .astype(float)
        )

        # Forward-fill product information

        product_columns = [
            "MDNAME",
            "PACKG",
            "DETAIL",
            "Selling_Rate",
            "Purchase_Rate",
            "Maximum_Retail_Price",
        ]

        for column in product_columns:
            if column in merged.columns:
                merged[column] = (
                    merged[column]
                    .ffill()
                    .bfill()
                )

        # Calendar Columns

        merged["day"] = merged["INVDT"].dt.day
        merged["day_name"] = merged["INVDT"].dt.day_name()
        merged["day_of_week"] = merged["INVDT"].dt.dayofweek
        merged["week"] = (
            merged["INVDT"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        merged["month"] = merged["INVDT"].dt.month
        merged["month_name"] = merged["INVDT"].dt.month_name()
        merged["quarter"] = merged["INVDT"].dt.quarter
        merged["year"] = merged["INVDT"].dt.year

        merged["is_weekend"] = (
            merged["day_of_week"] >= 5
        )

        merged["financial_year"] = merged["year"]

        merged["financial_month"] = (
            (merged["month"] + 8) % 12
        ) + 1

        daily_series.append(merged)

    daily_demand_df = (
        pd.concat(
            daily_series,
            ignore_index=True,
        )
        .sort_values(
            ["MDCODE", "INVDT"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Daily Time Series Created | Rows=%s | Medicines=%s",
        len(daily_demand_df),
        daily_demand_df["MDCODE"].nunique(),
    )

    self.daily_demand_df = daily_demand_df

    return daily_demand_df


    def split_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Creating Train / Validation / Test Split")

    if self.daily_demand_df is None:
        raise ValueError(
            "Daily demand dataframe not created."
        )

    daily_train = (
        self.daily_demand_df[
            self.daily_demand_df["INVDT"] <= self.train_end_date
        ]
        .copy()
    )

    daily_validation = (
        self.daily_demand_df[
            (self.daily_demand_df["INVDT"] > self.train_end_date)
            &
            (self.daily_demand_df["INVDT"] <= self.validation_end_date)
        ]
        .copy()
    )

    daily_test = (
        self.daily_demand_df[
            self.daily_demand_df["INVDT"] > self.validation_end_date
        ]
        .copy()
    )

    logger.info(
        "Train Rows=%s | Medicines=%s",
        len(daily_train),
        daily_train["MDCODE"].nunique(),
    )

    logger.info(
        "Validation Rows=%s | Medicines=%s",
        len(daily_validation),
        daily_validation["MDCODE"].nunique(),
    )

    logger.info(
        "Test Rows=%s | Medicines=%s",
        len(daily_test),
        daily_test["MDCODE"].nunique(),
    )

    self.daily_train = daily_train
    self.daily_validation = daily_validation
    self.daily_test = daily_test

    return (
        daily_train,
        daily_validation,
        daily_test,
    )


    def build_chronos_train(self) -> pd.DataFrame:
        logger.info("Preparing Chronos Training Dataset")

    if self.daily_train is None:
        raise ValueError(
            "Daily training dataset not found. Run split_dataset() first."
        )

    chronos_train = (
        self.daily_train.rename(
            columns={
                "MDCODE": "item_id",
                "INVDT": "timestamp",
                "Demand_Qty": "target",
            }
        )[
            [
                "item_id",
                "timestamp",
                "target",
            ]
        ]
        .copy()
    )

    chronos_train["item_id"] = (
        chronos_train["item_id"]
        .astype(str)
    )

    chronos_train["timestamp"] = pd.to_datetime(
        chronos_train["timestamp"]
    )

    chronos_train["target"] = (
        chronos_train["target"]
        .astype(float)
    )

    chronos_train = (
        chronos_train
        .sort_values(
            ["item_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Chronos Train Dataset Created | Rows=%s | Medicines=%s",
        len(chronos_train),
        chronos_train["item_id"].nunique(),
    )

    self.chronos_train = chronos_train

    return chronos_train


    def build_chronos_validation(self) -> pd.DataFrame:
        logger.info("Preparing Chronos Validation Dataset")

    if self.daily_validation is None:
        raise ValueError(
            "Daily validation dataset not found. Run split_dataset() first."
        )

    chronos_validation = (
        self.daily_validation.rename(
            columns={
                "MDCODE": "item_id",
                "INVDT": "timestamp",
                "Demand_Qty": "target",
            }
        )[
            [
                "item_id",
                "timestamp",
                "target",
            ]
        ]
        .copy()
    )

    chronos_validation["item_id"] = (
        chronos_validation["item_id"]
        .astype(str)
    )

    chronos_validation["timestamp"] = pd.to_datetime(
        chronos_validation["timestamp"]
    )

    chronos_validation["target"] = (
        chronos_validation["target"]
        .astype(float)
    )

    chronos_validation = (
        chronos_validation
        .sort_values(
            ["item_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Chronos Validation Dataset Created | Rows=%s | Medicines=%s",
        len(chronos_validation),
        chronos_validation["item_id"].nunique(),
    )

    self.chronos_validation = chronos_validation

    return chronos_validation

    def build_chronos_test(self) -> pd.DataFrame:
        logger.info("Preparing Chronos Test Dataset")

    if self.daily_test is None:
        raise ValueError(
            "Daily test dataset not found. Run split_dataset() first."
        )

    chronos_test = (
        self.daily_test.rename(
            columns={
                "MDCODE": "item_id",
                "INVDT": "timestamp",
                "Demand_Qty": "target",
            }
        )[
            [
                "item_id",
                "timestamp",
                "target",
            ]
        ]
        .copy()
    )

    chronos_test["item_id"] = (
        chronos_test["item_id"]
        .astype(str)
    )

    chronos_test["timestamp"] = pd.to_datetime(
        chronos_test["timestamp"]
    )

    chronos_test["target"] = (
        chronos_test["target"]
        .astype(float)
    )

    chronos_test = (
        chronos_test
        .sort_values(
            ["item_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Chronos Test Dataset Created | Rows=%s | Medicines=%s",
        len(chronos_test),
        chronos_test["item_id"].nunique(),
    )

    self.chronos_test = chronos_test

    return chronos_test

    def save_outputs(self) -> None:
        logger.info("Saving Prepared Datasets")

    self.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = {
        "daily_demand.parquet": self.daily_demand_df,
        "chronos_train.parquet": self.chronos_train,
        "chronos_validation.parquet": self.chronos_validation,
        "chronos_test.parquet": self.chronos_test,
    }

    for filename, dataframe in datasets.items():

        if dataframe is None:
            raise ValueError(
                f"{filename} has not been created."
            )

        output_path = self.output_directory / filename

        dataframe.to_parquet(
            output_path,
            index=False,
        )

        logger.info(
            "Saved : %s",
            output_path,
        )

    logger.info("All datasets saved successfully.")

    def run(self) -> None:
        logger.info("=" * 80)
    logger.info("Demand Data Preparation Started")
    logger.info("=" * 80)

    self.load_sales()

    self.create_demand_dataset()

    self.create_daily_time_series()

    self.split_dataset()

    self.build_chronos_train()

    self.build_chronos_validation()

    self.build_chronos_test()

    self.save_outputs()

    logger.info("=" * 80)
    logger.info("Demand Data Preparation Completed Successfully")
    logger.info("=" * 80)

