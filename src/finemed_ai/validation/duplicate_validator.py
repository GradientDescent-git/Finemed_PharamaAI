from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)


def validate_duplicates(tables: dict[str, pd.DataFrame], duplicate_keys: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    report_rows = []
    duplicate_samples = {}

    logger.info("Starting duplicate validation")

    for table_name, keys in duplicate_keys.items():
        if table_name not in tables:
            logger.warning("Table not found for duplicate validation: %s", table_name)
            continue

        df = tables[table_name]

        missing_keys = [key for key in keys if key not in df.columns]

        if missing_keys:
            report_rows.append(
                {
                    "table": table_name,
                    "duplicate_key": keys,
                    "total_rows": len(df),
                    "duplicate_rows": None,
                    "duplicate_percent": None,
                    "status": "FAILED",
                    "message": f"Missing key columns: {missing_keys}",
                }
            )
            continue

        duplicate_mask = df.duplicated(subset=keys, keep=False)
        duplicate_count = int(duplicate_mask.sum())

        report_rows.append(
            {
                "table": table_name,
                "duplicate_key": keys,
                "total_rows": len(df),
                "duplicate_rows": duplicate_count,
                "duplicate_percent": round((duplicate_count / len(df)) * 100, 2)
                if len(df) > 0
                else 0,
                "status": "PASSED" if duplicate_count == 0 else "WARNING",
                "message": "No duplicates found"
                if duplicate_count == 0
                else "Duplicate records found",
            }
        )

        if duplicate_count > 0:
            duplicate_samples[table_name] = (
                df.loc[duplicate_mask]
                .sort_values(keys)
                .head(50)
            )

    report = pd.DataFrame(report_rows)

    logger.info(
        "Duplicate validation completed | tables_checked=%s | tables_with_duplicates=%s",
        len(report),
        (report["duplicate_rows"].fillna(0) > 0).sum() if not report.empty else 0,
    )

    return report, duplicate_samples

