from pathlib import Path

from finemed_ai.transform.medicine.medicine_transform import (
    MedicineTransformer,
)


def main() -> None:

    transformer = MedicineTransformer(

        medicine_dimension_path=Path(
            "data/03_warehouse/dimensions/dim_product.parquet"
        ),

    )

    print("\nLoading Data...")
    transformer.load_data()

    print("\nCleaning Data...")
    transformer.clean_data()

    print("\nBusiness Transformations...")
    transformer.business_transformations()

    print("\nSaving Silver Dataset...")
    transformer.save(
        Path(
            "data/04_silver/medicine/medicine_silver.parquet"
        )
    )

    print("\nMedicine Transformation Test Passed Successfully.")


if __name__ == "__main__":
    main()