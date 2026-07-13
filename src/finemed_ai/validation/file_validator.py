from __future__ import annotations

from pathlib import Path

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def validate_required_files(
    raw_data_dir: Path,
    required_files: list[str],
) -> pd.DataFrame:
    """Validate the presence and integrity of required ERP DAT files across all monthly folders."""

    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_data_dir}"
        )

    month_folders = sorted(
        folder
        for folder in raw_data_dir.iterdir()
        if folder.is_dir()
    )

    if not month_folders:
        raise ValueError(
            f"No month folders found in raw data directory: {raw_data_dir}"
        )

    logger.info(
        "Starting file validation | total_months=%s",
        len(month_folders),
    )

    report_rows: list[dict] = []

    for month_folder in month_folders:

        try:

            logger.info(
                "Validating files for month: %s",
                month_folder.name,
            )

            for file_name in required_files:

                file_path = month_folder / file_name

                exists = file_path.exists()

                size_bytes = (
                    file_path.stat().st_size
                    if exists
                    else 0
                )

                report_rows.append(
                    {
                        "month": month_folder.name,
                        "file": file_name,
                        "exists": exists,
                        "size_bytes": size_bytes,
                        "is_empty": (
                            exists
                            and size_bytes == 0
                        ),
                    }
                )

            logger.info(
                "Completed file validation for month: %s",
                month_folder.name,
            )

        except Exception as error:

            logger.exception(
                "File validation failed for month: %s",
                month_folder.name,
            )

            raise RuntimeError(
                f"File validation failed for month: {month_folder.name}"
            ) from error

    report = pd.DataFrame(report_rows)

    logger.info(
        "File validation completed | months=%s | checks=%s | missing=%s | empty=%s",
        len(month_folders),
        len(report),
        (~report["exists"]).sum(),
        report["is_empty"].sum(),
    )

    return report