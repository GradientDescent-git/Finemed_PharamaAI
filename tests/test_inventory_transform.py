from pathlib import Path

from finemed_ai.transform.inventory.inventory_transform import (
    InventoryTransformer,
)


def main():

    transformer = InventoryTransformer(
        inventory_fact_path=Path(
            "data/03_warehouse/facts/fact_inventory.parquet"
        ),
        medicine_dimension_path=Path(
            "data/03_warehouse/dimensions/dim_product.parquet"
        ),
        date_dimension_path=Path(
            "data/03_warehouse/dimensions/dim_date.parquet"
        ),
    )

    transformer.load_data()

    transformer.join_dimensions()

    transformer.clean_data()

    transformer.business_transformations()

    print(transformer.inventory_df.head())

    print(transformer.inventory_df.info())

    print(transformer.inventory_df.describe(include="all"))


if __name__ == "__main__":
    main()