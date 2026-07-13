from __future__ import annotations

from pathlib import Path

import pandas as pd
from dbfread import DBF

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema_consistency(
    raw_data_dir: Path,
    required_files: list[str],
) -> pd.DataFrame:
    """ Validate schema consistency of ERP DAT files across all monthly folders. """

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
        "Starting schema consistency validation"
    )

    report_rows: list[dict] = []

    for table_name in required_files:

        logger.info(
            "Validating schema: %s",
            table_name,
        )

        reference_columns = None
        reference_month = None

        try:

            for month_folder in month_folders:

                file_path = month_folder / table_name

                if not file_path.exists():

                    report_rows.append(
                        {
                            "table": table_name,
                            "month": month_folder.name,
                            "column_count": None,
                            "matches_reference": False,
                            "reference_month": reference_month,
                            "missing_columns": None,
                            "extra_columns": None,
                            "error": (
                                f"File not found: {file_path}"
                            ),
                        }
                    )

                    logger.warning(
                        "Missing file: %s",
                        file_path,
                    )

                    continue

                table = DBF(
                    file_path,
                    load=False,
                )

                columns = list(
                    table.field_names
                )

                if reference_columns is None:

                    reference_columns = columns
                    reference_month = month_folder.name

                missing_columns = sorted(
                    set(reference_columns)
                    - set(columns)
                )

                extra_columns = sorted(
                    set(columns)
                    - set(reference_columns)
                )

                report_rows.append(
                    {
                        "table": table_name,
                        "month": month_folder.name,
                        "column_count": len(columns),
                        "matches_reference": (
                            columns == reference_columns
                        ),
                        "reference_month": reference_month,
                        "missing_columns": missing_columns,
                        "extra_columns": extra_columns,
                        "error": None,
                    }
                )

            logger.info(
                "Completed schema validation: %s",
                table_name,
            )

        except Exception as error:

            logger.exception(
                "Schema validation failed: %s",
                table_name,
            )

            raise RuntimeError(
                f"Schema validation failed for {table_name}"
            ) from error

    report = pd.DataFrame(report_rows)

    logger.info(
        "Schema validation completed | checks=%s | mismatches=%s",
        len(report),
        (
            (~report["matches_reference"]).sum()
            if not report.empty
            else 0
        ),
    )

    return report