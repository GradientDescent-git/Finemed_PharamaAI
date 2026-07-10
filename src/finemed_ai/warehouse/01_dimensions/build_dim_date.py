from __future__ import annotations

import pandas as pd


def _get_financial_year(date: pd.Timestamp) -> str:
    year = date.year

    if date.month >= 4:
        return f"FY {year}-{str(year + 1)[-2:]}"

    return f"FY {year - 1}-{str(year)[-2:]}"

def _get_financial_month(date: pd.Timestamp) -> int:
    if date.month >= 4:
        return date.month - 3

    return date.month + 9

def build_dim_date(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if start > end:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    dates = pd.date_range(start=start, end=end, freq="D")

    dim_date = pd.DataFrame({"full_date": dates})

    dim_date["date_key"] = dim_date["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["day"] = dim_date["full_date"].dt.day
    dim_date["day_name"] = dim_date["full_date"].dt.day_name()
    dim_date["day_of_week"] = dim_date["full_date"].dt.dayofweek + 1
    dim_date["week"] = dim_date["full_date"].dt.isocalendar().week.astype(int)
    dim_date["month"] = dim_date["full_date"].dt.month
    dim_date["month_name"] = dim_date["full_date"].dt.month_name()
    dim_date["quarter"] = dim_date["full_date"].dt.quarter
    dim_date["year"] = dim_date["full_date"].dt.year
    dim_date["is_weekend"] = dim_date["full_date"].dt.dayofweek.isin([5, 6])
    dim_date["financial_year"] = dim_date["full_date"].apply(_get_financial_year)
    dim_date["financial_month"] = dim_date["full_date"].apply(_get_financial_month)

    dim_date["full_date"] = dim_date["full_date"].dt.date

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
            "quarter",
            "year",
            "is_weekend",
            "financial_year",
            "financial_month",
        ]
    ]