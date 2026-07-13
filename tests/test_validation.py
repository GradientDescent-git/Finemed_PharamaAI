from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.extract.read_dat import read_all_months
from finemed_ai.validation.run_validation import run_all_validations
from finemed_ai.validation.validation_config import REQUIRED_FILES

print("=" * 80)
print("RUNNING EXTRACTION")
print("=" * 80)

tables = read_all_months(
    RAW_DATA_DIR,
    REQUIRED_FILES,
)

print("\nExtraction completed.")

print("=" * 80)
print("RUNNING VALIDATION")
print("=" * 80)

reports = run_all_validations(tables)

print("\nValidation completed successfully.")

print("\nGenerated Reports:\n")

for report_name, report in reports.items():

    print(f"{report_name}")

    if hasattr(report, "shape"):
        print(report.shape)

    else:
        print(type(report))

    print("-" * 50)

print(reports["file_report"].head())

print(reports["schema_report"].head())

print(reports["duplicate_report"])

print(reports["orphan_report"])

print(reports["profile_report"].head())