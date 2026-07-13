from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def _get_financial_year(date: pd.Timestamp) -> str:
    """ Return the financial year for a given date. """

    year = date.year

    if date.month >= 4:
        return f"FY {year}-{str(year + 1)[-2:]}"

    return f"FY {year - 1}-{str(year)[-2:]}"


def _get_financial_month(date: pd.Timestamp) -> int:
    """ Return the financial month number. """

    if date.month >= 4:
        return date.month - 3

    return date.month + 9


def build_dim_date(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """ Build the Date Dimension. """

    logger.info(
        "Building Date Dimension | start=%s | end=%s",
        start_date,
        end_date,
    )

    try:

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if start > end:
            raise ValueError(
                "start_date must be earlier than or equal to end_date."
            )

        dates = pd.date_range(
            start=start,
            end=end,
            freq="D",
        )

        dim_date = pd.DataFrame(
            {
                "full_date": dates,
            }
        )

        dim_date["date_key"] = (
            dim_date["full_date"]
            .dt.strftime("%Y%m%d")
            .astype(int)
        )

        dim_date["day"] = dim_date["full_date"].dt.day

        dim_date["day_name"] = (
            dim_date["full_date"]
            .dt.day_name()
        )

        dim_date["day_of_week"] = (
            dim_date["full_date"]
            .dt.dayofweek + 1
        )

        dim_date["week"] = (
            dim_date["full_date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        dim_date["month"] = (
            dim_date["full_date"]
            .dt.month
        )

        dim_date["month_name"] = (
            dim_date["full_date"]
            .dt.month_name()
        )

        dim_date["quarter"] = (
            dim_date["full_date"]
            .dt.quarter
        )

        dim_date["year"] = (
            dim_date["full_date"]
            .dt.year
        )

        dim_date["is_weekend"] = (
            dim_date["full_date"]
            .dt.dayofweek
            .isin([5, 6])
        )

        dim_date["is_month_start"] = (
            dim_date["full_date"]
            .dt.is_month_start
        )

        dim_date["is_month_end"] = (
            dim_date["full_date"]
            .dt.is_month_end
        )

        dim_date["is_quarter_end"] = (
            dim_date["full_date"]
            .dt.is_quarter_end
        )

        dim_date["is_year_end"] = (
            dim_date["full_date"]
            .dt.is_year_end
        )

        dim_date["financial_year"] = (
            dim_date["full_date"]
            .apply(_get_financial_year)
        )

        dim_date["financial_month"] = (
            dim_date["full_date"]
            .apply(_get_financial_month)
        )

        dim_date["month_year"] = (
            dim_date["full_date"]
            .dt.strftime("%b-%Y")
        )

        dim_date["full_date"] = (
            dim_date["full_date"]
            .dt.date
        )

        logger.info(
            "Date Dimension built successfully | rows=%s",
            len(dim_date),
        )

        return dim_date[
            [
                "date_key",
                "full_date",
                "day",
                "day_name",
                "day_of_week",
                "week",
                "month",
                "month_name",
                "month_year",
                "quarter",
                "year",
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "is_quarter_end",
                "is_year_end",
                "financial_year",
                "financial_month",
            ]
        ]

    except Exception as error:

        logger.exception(
            "Failed while building Date Dimension."
        )

        raise RuntimeError(
            "Date Dimension build failed."
        ) from error