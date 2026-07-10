from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)


def validate_datatype_consistency(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    report_rows = []

    logger.info("Starting datatype consistency validation")

    for table_name, df in tables.items():
        if "SOURCE_MONTH" not in df.columns:
            raise ValueError(
                f"SOURCE_MONTH column missing in {table_name}. "
                "Extraction layer must add SOURCE_MONTH before validation."
            )

        for column in df.columns:
            if column == "SOURCE_MONTH":
                continue

            dtype_by_month = (
                df.groupby("SOURCE_MONTH")[column]
                .apply(lambda series: str(series.dtype))
                .reset_index(name="dtype")
            )

            dtypes_found = sorted(dtype_by_month["dtype"].dropna().unique().tolist())
            months_affected = sorted(dtype_by_month["SOURCE_MONTH"].unique().tolist())

            if len(dtypes_found) > 1:
                report_rows.append(
                    {
                        "table": table_name,
                        "column": column,
                        "dtype_count": len(dtypes_found),
                        "dtypes_found": dtypes_found,
                        "months_affected": months_affected,
                    }
                )

    report = pd.DataFrame(report_rows)

    logger.info(
        "Datatype validation completed | datatype_issues=%s",
        len(report),
    )

    return report