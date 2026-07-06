from __future__ import annotations
from pathlib import Path
from dbfread import DBF
import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)

def read_dat_files(file_path:Path, source_month: str) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info("Reading file: %s", file_path)

    table = DBF(file_path, load =True)
    df = pd.DataFrame(iter(table))

    df["SOURCE_MONTH"] = source_month
    logger.info("Loaded %s | rows = %s | columns = %s",
    file_path.name,
    len(df),
    len(df.columns),
    )
    return df

def read_month_folder(month_folder:Path, required_files: list[str]) -> dict[str, pd.DataFrame]:
    month_folder = Path(month_folder)
    if not month_folder.exists():
        raise FileNotFoundError(f"Month folder not found: {month_folder}")
    source_month = month_folder.name
    logger.info("Reading month folder : %s", source_month)

    month_data = {}

    for file_name in required_files:
        file_path = month_folder / file_name
        month_data[file_name] =  read_dat_files(file_path, source_month)

    logger.info("Completed month folder: %s", source_month)
    return month_data

def read_all_months(raw_data_dir: Path, required_files: list[str]) -> dict[str, pd.DataFrame]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    month_folders = sorted([p for p in raw_data_dir.iterdir() if p.is_dir()])

    if not month_folders:
        raise ValueError(f"No month folders found in raw data directory: {raw_data_dir}")

    logger.info("Total month folders found: %s", len(month_folders))

    combined_tables: dict[str, list[pd.DataFrame]] = {
        file_name: [] for file_name in required_files
    }

    for month_folder in month_folders:
        month_data = read_month_folder(month_folder, required_files)

        for file_name, df in month_data.items():
            combined_tables[file_name].append(df)

    final_tables = {}

    for file_name, dfs in combined_tables.items():
        final_tables[file_name] = pd.concat(dfs, ignore_index=True)

        logger.info(
            "Combined table %s | total_rows = %s, total_columns = %s",
            file_name,
            len(final_tables[file_name]),
            len(final_tables[file_name].columns),
        )

    logger.info("All month folders extracted...")

    return final_tables