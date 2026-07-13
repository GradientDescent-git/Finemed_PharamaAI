from pathlib import Path

from finemed_ai.extract.read_dat import read_all_months

required_files = [
    "COMPUR.DAT",
    "INVOICE.DAT",
    "INVDET.DAT",
    "MEDIMAST.DAT",
    "PURCHASE.DAT",
    "SUPMAST.DAT",
    "SFILE.DAT",
    "TFILE.DAT"
    ]

raw_data = Path("data/01_raw")

tables = read_all_months(raw_data,required_files)

print("=" * 60)

for name, df in tables.items():
    print(f"{name}")
    print(f"Shape : {df.shape}")
    print(df.head())
    print("-" * 60)

invoice = tables["INVOICE.DAT"]
print(invoice["SOURCE_MONTH"].value_counts().sort_index())