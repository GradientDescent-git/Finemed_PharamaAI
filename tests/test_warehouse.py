from __future__ import annotations

from finemed_ai.config.paths import RAW_DATA_DIR

from finemed_ai.extract.read_dat import read_all_months

from finemed_ai.validation.validation_config import REQUIRED_FILES

from finemed_ai.warehouse.dimensions.run_dimensions import (
    run_dimensions,
)

from finemed_ai.warehouse.facts.run_facts import (
    run_facts,
)

print("=" * 80)
print("RUNNING EXTRACTION")
print("=" * 80)

tables = read_all_months(
    RAW_DATA_DIR,
    REQUIRED_FILES,
)

print()

print("=" * 80)
print("BUILDING DIMENSIONS")
print("=" * 80)

dimensions = run_dimensions(
    tables,
)

print()

print("=" * 80)
print("BUILDING FACT TABLES")
print("=" * 80)

facts = run_facts(
    tables,
)

print()

print("=" * 80)
print("DIMENSIONS")
print("=" * 80)

for name, df in dimensions.items():
    print(f"{name:<25} {df.shape}")

print()

print("=" * 80)
print("FACTS")
print("=" * 80)

for name, df in facts.items():
    print(f"{name:<25} {df.shape}")

print()

print("=" * 80)
print("WAREHOUSE BUILT SUCCESSFULLY")
print("=" * 80)