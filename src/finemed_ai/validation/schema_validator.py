from pathlib import Path

import pandas as pd
from dbfread import DBF

from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)


def validate_schema_consistency(raw_data_dir: Path, required_files: list[str]) -> pd.DataFrame:
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    month_folders = sorted(
        folder for folder in raw_data_dir.iterdir() if folder.is_dir()
    )

    if not month_folders:
        raise ValueError(f"No month folders found in raw data directory: {raw_data_dir}")

    report_rows = []

    logger.info("Starting schema validation")

    for table_name in required_files:
        reference_columns = None
        reference_month = None

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
                        "error": f"File not found: {file_path}",
                    }
                )
                continue

            try:
                table = DBF(file_path, load=False)
                columns = list(table.field_names)

                if reference_columns is None:
                    reference_columns = columns
                    reference_month = month_folder.name

                missing_columns = sorted(set(reference_columns) - set(columns))
                extra_columns = sorted(set(columns) - set(reference_columns))

                report_rows.append(
                    {
                        "table": table_name,
                        "month": month_folder.name,
                        "column_count": len(columns),
                        "matches_reference": columns == reference_columns,
                        "reference_month": reference_month,
                        "missing_columns": missing_columns,
                        "extra_columns": extra_columns,
                        "error": None,
                    }
                )

            except Exception as error:
                report_rows.append(
                    {
                        "table": table_name,
                        "month": month_folder.name,
                        "column_count": None,
                        "matches_reference": False,
                        "reference_month": reference_month,
                        "missing_columns": None,
                        "extra_columns": None,
                        "error": str(error),
                    }
                )

    report = pd.DataFrame(report_rows)

    logger.info(
        "Schema validation completed | checks=%s | mismatches=%s",
        len(report),
        (~report["matches_reference"]).sum(),
    )

    return report