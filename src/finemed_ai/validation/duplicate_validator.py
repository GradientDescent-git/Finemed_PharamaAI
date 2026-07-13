from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def validate_duplicates(
    tables: dict[str, pd.DataFrame],
    duplicate_keys: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """  Validate duplicate records in ERP tables based on configured business keys   """

    logger.info("Starting duplicate validation")

    report_rows: list[dict] = []
    duplicate_samples: dict[str, pd.DataFrame] = {}

    for table_name, keys in duplicate_keys.items():

        try:

            logger.info(
                "Validating duplicates for table: %s",
                table_name,
            )

            if table_name not in tables:
                logger.warning(
                    "Table not found: %s",
                    table_name,
                )
                continue

            df = tables[table_name]

            if df.empty:
                logger.warning(
                    "Table %s is empty. Skipping duplicate validation.",
                    table_name,
                )
                continue

            missing_keys = [
                key
                for key in keys
                if key not in df.columns
            ]

            if missing_keys:

                report_rows.append(
                    {
                        "table": table_name,
                        "duplicate_key": keys,
                        "total_rows": len(df),
                        "duplicate_rows": None,
                        "duplicate_percent": None,
                        "status": "FAILED",
                        "message": (
                            f"Missing key columns: {missing_keys}"
                        ),
                    }
                )

                logger.warning(
                    "Missing duplicate keys in %s : %s",
                    table_name,
                    missing_keys,
                )

                continue

            duplicate_mask = df.duplicated(
                subset=keys,
                keep=False,
            )

            duplicate_count = int(
                duplicate_mask.sum()
            )

            duplicate_percent = (
                round(
                    (duplicate_count / len(df)) * 100,
                    2,
                )
                if len(df) > 0
                else 0
            )

            report_rows.append(
                {
                    "table": table_name,
                    "duplicate_key": keys,
                    "total_rows": len(df),
                    "duplicate_rows": duplicate_count,
                    "duplicate_percent": duplicate_percent,
                    "status": (
                        "PASSED"
                        if duplicate_count == 0
                        else "WARNING"
                    ),
                    "message": (
                        "No duplicates found"
                        if duplicate_count == 0
                        else "Duplicate records found"
                    ),
                }
            )

            if duplicate_count > 0:

                duplicate_samples[table_name] = (
                    df.loc[duplicate_mask]
                    .sort_values(keys)
                    .head(50)
                )

            logger.info(
                "Completed duplicate validation for table: %s",
                table_name,
            )

        except Exception as error:

            logger.exception(
                "Duplicate validation failed for table: %s",
                table_name,
            )

            raise RuntimeError(
                f"Duplicate validation failed for table: {table_name}"
            ) from error

    report = pd.DataFrame(report_rows)

    logger.info(
        "Duplicate validation completed | tables_checked=%s | tables_with_duplicates=%s",
        len(report),
        (
            report["duplicate_rows"]
            .fillna(0)
            .gt(0)
            .sum()
            if not report.empty
            else 0
        ),
    )

    return report, duplicate_samples