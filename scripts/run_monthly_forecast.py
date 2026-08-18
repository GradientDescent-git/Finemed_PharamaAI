import argparse
import logging
from finemed_ai.config.settings import Settings
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from finemed_ai.demand_forecasting.pipeline import run_monthly_forecast  # noqa: E402
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_monthly_forecast")
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the monthly Chronos-2 demand forecast.")
    parser.add_argument(
        "--silver-demand",
        type=Path,
        default=Settings.DEMAND_FILE,
        help="Path to the silver-layer daily demand table (parquet or csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/05_forecasts"),
        help="Directory to write versioned forecast runs into.",
    )
    args = parser.parse_args()
 
    if not args.silver_demand.exists():
        logger.error("Silver demand source not found: %s", args.silver_demand)
        return 1
 
    try:
        manifest = run_monthly_forecast(
            forecasting_series_path=args.silver_demand,
            output_dir=args.output)
    except Exception:
        logger.exception("Forecast run failed")
        return 1
 
    logger.info("Run %s written to %s", manifest.run_id, manifest.output_path)
    if manifest.medicines_failed > 0:
        logger.warning(
            "%d/%d medicines failed to forecast this run — see manifest.json",
            manifest.medicines_failed, manifest.medicines_requested,
        )
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())