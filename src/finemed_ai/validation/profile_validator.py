from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def profile_dataframe(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """ Generate a column-level profiling report for a single DataFrame. """

    logger.info(
        "Profiling dataframe: %s",
        table_name,
    )

    if df.empty:
        logger.warning(
            "Table %s is empty. Returning empty profile.",
            table_name,
        )
        return pd.DataFrame()

    profile_rows: list[dict] = []

    total_rows = len(df)

    try:

        for column in df.columns:

            series = df[column]

            row = {
                "table": table_name,
                "column": column,
                "dtype": str(series.dtype),
                "total_rows": total_rows,
                "null_count": int(series.isna().sum()),
                "null_percent": (
                    round(series.isna().mean() * 100, 2)
                    if total_rows > 0
                    else 0
                ),
                "non_null_count": int(series.notna().sum()),
                "unique_count": int(
                    series.nunique(dropna=True)
                ),
                "unique_percent": (
                    round(
                        (
                            series.nunique(dropna=True)
                            / total_rows
                        )
                        * 100,
                        2,
                    )
                    if total_rows > 0
                    else 0
                ),
                "is_constant": (
                    series.nunique(dropna=True) <= 1
                ),
                "top_values": (
                    series.value_counts(dropna=False)
                    .head(5)
                    .to_dict()
                ),
            }

            if pd.api.types.is_numeric_dtype(series):

                row["min"] = series.min()
                row["max"] = series.max()
                row["mean"] = round(
                    series.mean(),
                    2,
                )
                row["median"] = series.median()

            else:

                row["min"] = None
                row["max"] = None
                row["mean"] = None
                row["median"] = None

            profile_rows.append(row)

        logger.info(
            "Completed dataframe profiling: %s",
            table_name,
        )

        return pd.DataFrame(profile_rows)

    except Exception as error:

        logger.exception(
            "Profiling failed for table: %s",
            table_name,
        )

        raise RuntimeError(
            f"Profiling failed for table: {table_name}"
        ) from error


def profile_tables(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Generate profiling reports for all extracted ERP tables. """

    logger.info("Starting table profiling")

    reports: list[pd.DataFrame] = []

    for table_name, df in tables.items():

        reports.append(
            profile_dataframe(
                df,
                table_name,
            )
        )

    if not reports:

        logger.warning(
            "No profiling reports generated."
        )

        return pd.DataFrame()

    final_report = pd.concat(
        reports,
        ignore_index=True,
    )

    logger.info(
        "Table profiling completed | tables=%s | profiled_columns=%s",
        len(tables),
        len(final_report),
    )

    return final_report