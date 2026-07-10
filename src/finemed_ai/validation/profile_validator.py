from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)


def profile_dataframe(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """
    Create a column-level profile for one DataFrame.
    """

    profile_rows = []
    total_rows = len(df)

    for column in df.columns:
        series = df[column]

        row = {
            "table": table_name,
            "column": column,
            "dtype": str(series.dtype),
            "total_rows": total_rows,
            "null_count": int(series.isna().sum()),
            "null_percent": round(series.isna().mean() * 100, 2)
            if total_rows > 0
            else 0,
            "non_null_count": int(series.notna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "unique_percent": round((series.nunique(dropna=True) / total_rows) * 100, 2)
            if total_rows > 0
            else 0,
            "is_constant": series.nunique(dropna=True) <= 1,
            "top_values": series.value_counts(dropna=False).head(5).to_dict(),
        }

        if pd.api.types.is_numeric_dtype(series):
            row["min"] = series.min()
            row["max"] = series.max()
            row["mean"] = round(series.mean(), 2)
            row["median"] = series.median()
        else:
            row["min"] = None
            row["max"] = None
            row["mean"] = None
            row["median"] = None

        profile_rows.append(row)

    return pd.DataFrame(profile_rows)


def profile_tables(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Profile all extracted tables and combine the reports.
    """

    logger.info("Starting table profiling")

    reports = []

    for table_name, df in tables.items():
        logger.info("Profiling table: %s", table_name)
        reports.append(profile_dataframe(df, table_name))

    if not reports:
        return pd.DataFrame()

    final_report = pd.concat(reports, ignore_index=True)

    logger.info(
        "Table profiling completed | profiled_columns=%s",
        len(final_report),
    )

    return final_report