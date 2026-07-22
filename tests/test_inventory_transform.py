from pathlib import Path

from finemed_ai.transform.inventory.inventory_transform import (
    InventoryTransformer,
)


def main() -> None:

    transformer = InventoryTransformer()

    transformer.load_data()

    transformer.join_dimensions()

    transformer.clean_data()

    transformer.business_transformations()

    print("\nFirst 5 Rows")
    print(transformer.inventory_df.head())

    print("\nDataFrame Information")
    transformer.inventory_df.info()

    print("\nDescriptive Statistics")
    print(transformer.inventory_df.describe(include="all"))

    print("\nColumns")
    print(transformer.inventory_df.columns.tolist())


if __name__ == "__main__":
    main()