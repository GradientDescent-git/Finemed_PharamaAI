from pathlib import Path

from finemed_ai.transform.medicine.medicine_transform import (
    MedicineTransformer,
)


def main() -> None:

    transformer = MedicineTransformer()

    print("\nLoading Data...")
    transformer.load_data()

    print("\nCleaning Data...")
    transformer.clean_data()

    print("\nBusiness Transformations...")
    transformer.business_transformations()

    print("\nFirst 5 Rows")
    print(transformer.medicine_df.head())

    print("\nDataFrame Information")
    transformer.medicine_df.info()

    print("\nDescriptive Statistics")
    print(transformer.medicine_df.describe(include="all"))

    print("\nSaving Silver Dataset...")
    transformer.save(
        Path(
            "data/04_silver/medicine/medicine_silver.parquet"
        )
    )

    print("\nMedicine Transformation Test Passed Successfully.")


if __name__ == "__main__":
    main()