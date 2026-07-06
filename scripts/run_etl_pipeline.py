from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.extract.read_dat import read_all_months
from finemed_ai.utils.logger import get_logger


logger = get_logger(__name__)


def main():
    logger.info("Pipeline started")

    required_files = [
        "INVOICE.DAT",
        "INVDET.DAT",
    ]

    tables = read_all_months(
        raw_data_dir=RAW_DATA_DIR,
        required_files=required_files,
    )

    logger.info("Extract completed")

    for table_name, df in tables.items():
        logger.info("%s shape=%s", table_name, df.shape)
        print(table_name, df.shape)

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()