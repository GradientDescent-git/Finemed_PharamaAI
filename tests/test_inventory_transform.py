from pathlib import Path

from finemed_ai.transform.inventory.inventory_transform import (
    InventoryTransformer,
)


def main() -> None:

    transformer = InventoryTransformer()

    print("\nLoading Data...")
    transformer.load_data()

    print("\nJoining Dimensions...")
    transformer.join_dimensions()

    print("\nCleaning Data...")
    transformer.clean_data()

    print("\nBusiness Transformations...")
    transformer.business_transformations()

    print("\nFirst 5 Rows")
    print(transformer.inventory_df.head())

    print("\nDataFrame Information")
    transformer.inventory_df.info()

    print("\nDescriptive Statistics")
    print(
        transformer.inventory_df.describe(
            include="all"
        )
    )

    print("\nSaving Silver Dataset...")
    transformer.save(
        Path(
            "data/04_silver/inventory/inventory_silver.parquet"
        )
    )

    print(
        "\nInventory Transformation Test Passed Successfully."
    )


if __name__ == "__main__":
    main()