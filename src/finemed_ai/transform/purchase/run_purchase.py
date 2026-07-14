from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.purchase.purchase_transform import (
    PurchaseTransformer,
)

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Starting Purchase Silver Layer Pipeline")
    log_step(logger, "=" * 70)

    
    # Warehouse Inputs
   
    fact_purchase_header_path = Path(
        "data/03_warehouse/facts/fact_purchase_header.parquet"
    )

    fact_purchase_line_path = Path(
        "data/03_warehouse/facts/fact_purchase_line.parquet"
    )

    dim_supplier_path = Path(
        "data/03_warehouse/dimensions/dim_supplier.parquet"
    )

    dim_medicine_path = Path(
        "data/03_warehouse/dimensions/dim_product.parquet"
    )

    dim_date_path = Path(
        "data/03_warehouse/dimensions/dim_date.parquet"
    )

    
    # Silver Output
    
    output_path = Path(
        "data/04_silver/purchase/purchase_silver.parquet"
    )

    
    # Execute Pipeline
    
    transformer = PurchaseTransformer(
        fact_purchase_header_path=fact_purchase_header_path,
        fact_purchase_line_path=fact_purchase_line_path,
        dim_supplier_path=dim_supplier_path,
        dim_medicine_path=dim_medicine_path,
        dim_date_path=dim_date_path,
    )

    transformer.run(output_path)

    log_step(logger, "=" * 70)
    log_step(logger, "Purchase Silver Layer Completed Successfully")
    log_step(logger, "=" * 70)


if __name__ == "__main__":
    main()