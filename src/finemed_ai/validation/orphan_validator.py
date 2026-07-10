from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)


def validate_orphans(tables: dict[str, pd.DataFrame],orphan_checks: list[dict]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    report_rows = []
    orphan_samples = {}

    logger.info("Starting orphan validation")

    for check in orphan_checks:
        check_name = check["check_name"]
        parent_table = check["parent_table"]
        child_table = check["child_table"]
        parent_key = check["parent_key"]
        child_key = check["child_key"]

        if parent_table not in tables or child_table not in tables:
            report_rows.append(
                {
                    "check_name": check_name,
                    "parent_table": parent_table,
                    "child_table": child_table,
                    "parent_key": parent_key,
                    "child_key": child_key,
                    "child_rows": None,
                    "orphan_rows": None,
                    "orphan_percent": None,
                    "status": "FAILED",
                    "message": "Parent or child table missing",
                }
            )
            continue

        parent_df = tables[parent_table]
        child_df = tables[child_table]

        if parent_key not in parent_df.columns or child_key not in child_df.columns:
            report_rows.append(
                {
                    "check_name": check_name,
                    "parent_table": parent_table,
                    "child_table": child_table,
                    "parent_key": parent_key,
                    "child_key": child_key,
                    "child_rows": len(child_df),
                    "orphan_rows": None,
                    "orphan_percent": None,
                    "status": "FAILED",
                    "message": "Parent key or child key missing",
                }
            )
            continue

        parent_keys = set(parent_df[parent_key].dropna().unique())

        orphan_mask = ~child_df[child_key].isin(parent_keys)
        orphan_count = int(orphan_mask.sum())

        report_rows.append(
            {
                "check_name": check_name,
                "parent_table": parent_table,
                "child_table": child_table,
                "parent_key": parent_key,
                "child_key": child_key,
                "child_rows": len(child_df),
                "orphan_rows": orphan_count,
                "orphan_percent": round((orphan_count / len(child_df)) * 100, 2)
                if len(child_df) > 0
                else 0,
                "status": "PASSED" if orphan_count == 0 else "WARNING",
                "message": "No orphan records found"
                if orphan_count == 0
                else "Orphan records found",
            }
        )

        if orphan_count > 0:
            orphan_samples[check_name] = child_df.loc[orphan_mask].head(100)

    report = pd.DataFrame(report_rows)

    logger.info(
        "Orphan validation completed | checks=%s | warnings=%s",
        len(report),
        (report["status"] == "WARNING").sum() if not report.empty else 0,
    )

    return report, orphan_samples