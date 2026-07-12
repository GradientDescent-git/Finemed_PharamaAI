from __future__ import annotations
from pathlib import Path
from dbfread import DBF
import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)

def read_dat_files(file_path:Path, source_month: str) -> pd.DataFrame:
    """Read a single DAT file and return it as a pandas DataFrame"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        table = DBF(file_path, load=True)
        df = pd.DataFrame(iter(table))
    
    except Exception as error:
        logger.exception("Failed to read DAT file: %s", file_path)
        raise RuntimeError(f"Unable to read DAT file: {file_path}") from error

    df["SOURCE_MONTH"] = source_month
    logger.info("Loaded %s | rows = %s | columns = %s",
    file_path.name,
    len(df),
    len(df.columns),
    )
    return df

def read_month_folder(month_folder:Path, required_files: list[str]) -> dict[str, pd.DataFrame]:
    """The Function Reads and returns the month Wise Folders in DAT file format"""
    month_folder = Path(month_folder)

    if not month_folder.exists():
        raise FileNotFoundError(
            f"Month folder not found: {month_folder}"
        )

    source_month = month_folder.name

    logger.info(
        "Reading month folder: %s",
        source_month,
    )

    month_data : dict[str, pd.DataFrame] = {}

    try:

        for file_name in required_files:

            file_path = month_folder / file_name

            month_data[file_name] = read_dat_files(
                file_path,
                source_month,
            )

    except Exception as error:

        logger.exception(
            "Failed while processing month folder: %s",
            source_month,
        )

        raise RuntimeError(
            f"Unable to process month folder: {source_month}"
        ) from error

    logger.info(
        "Completed month folder: %s",
        source_month,
    )

    return month_data

def read_all_months(raw_data_dir: Path, required_files: list[str]) -> dict[str, pd.DataFrame]:
    """Read all monthly folders and combine each DAT file into a single DataFrame."""
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

    try:
        for month_folder in month_folders:
            month_data = read_month_folder(month_folder,required_files )
            
            for file_name, df in month_data.items():
                combined_tables[file_name].append(df)
    
    except Exception as error:
        logger.exception( "Failed while reading monthly folders.")
        raise RuntimeError("Monthly extraction pipeline failed.") from error
    
    final_tables: dict[str, pd.DataFrame] = {}

    for file_name, dfs in combined_tables.items():
        try:
            final_tables[file_name] = pd.concat(dfs,ignore_index=True)
        
        except Exception as error:
            logger.exception("Failed while combining table: %s",file_name)
            raise RuntimeError(f"Unable to combine table: {file_name}") from error
        
        logger.info("Combined table %s | total_rows = %s | total_columns = %s",file_name,len(final_tables[file_name]),len(final_tables[file_name].columns))
    logger.info("All month folders extracted...")
    return final_tables