from __future__ import annotations

import logging

from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.extract.read_dat import read_all_months
from finemed_ai.validation.run_validation import (
    run_all_validations,
)
from finemed_ai.validation.validation_config import (
    REQUIRED_FILES,
)


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    # Extract Layer

    extracted_tables = read_all_months(
        raw_data_dir=RAW_DATA_DIR,
        required_files=REQUIRED_FILES,
    )

    # Validation Layer

    validation_output = run_all_validations(
        extracted_tables,
    )

    validated_tables = validation_output["tables"]
    validation_reports = validation_output["reports"]

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETED")
    print("=" * 80)

    print("\nValidated Tables\n")

    for table_name, dataframe in validated_tables.items():

        print(
            f"{table_name:<20} "
            f"Rows={len(dataframe):>8} "
            f"Cols={len(dataframe.columns):>3}"
        )

    print("\nValidation Reports\n")

    for report_name, report in validation_reports.items():

        print(f"{report_name}")

        if hasattr(report, "shape"):

            print(report.head())

        else:

            print(report)

        print("-" * 80)


if __name__ == "__main__":

    main()