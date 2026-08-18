import time
import pandas as pd

from finemed_ai.demand_forecasting.config import DEFAULT_CONFIG
from finemed_ai.demand_forecasting.predictor_service import PredictorService


PATH = "data/04_silver/demand_forecasting/chronos_series.parquet"


def main():
    df = pd.read_parquet(PATH)

    df = df.rename(
        columns={
            "MDCODE": "item_id",
            "INVDT": "timestamp",
            "Demand_Qty": "target",
        }
    )

    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    medicine_ids = sorted(df["item_id"].unique())

    print("=" * 70)
    print("CHRONOS-2 BATCH SMOKE TEST")
    print("=" * 70)
    print(f"Medicines: {len(medicine_ids)}")
    print()

    service = PredictorService.get_instance(DEFAULT_CONFIG)

    results = []
    failed = []

    start = time.time()

    for index, medicine_id in enumerate(medicine_ids, start=1):

        print(
            f"[{index:>3}/{len(medicine_ids)}] "
            f"Forecasting {medicine_id}...",
            end=" ",
            flush=True,
        )

        try:
            result = service.forecast_medicine(
                medicine_id,
                df,
            )

            results.append(result)

            print(
                f"OK | "
                f"{len(result.days)} days | "
                f"context={result.context_length_used}"
            )

        except Exception as exc:
            failed.append(
                {
                    "medicine_id": medicine_id,
                    "error": str(exc),
                }
            )

            print(f"FAILED | {exc}")

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("BATCH SMOKE TEST RESULT")
    print("=" * 70)

    print(f"Requested : {len(medicine_ids)}")
    print(f"Succeeded : {len(results)}")
    print(f"Failed    : {len(failed)}")

    success_rate = (
        len(results) / len(medicine_ids)
        if medicine_ids
        else 0
    )

    print(f"Success % : {success_rate:.1%}")
    print(f"Time      : {elapsed:.1f} seconds")

    if results:
        forecast_lengths = [len(r.days) for r in results]

        print()
        print("Forecast lengths:")
        print(
            pd.Series(forecast_lengths)
            .value_counts()
            .sort_index()
            .to_string()
        )

    if failed:
        print()
        print("FAILED MEDICINES")
        print("-" * 70)

        for item in failed:
            print(
                f"{item['medicine_id']}: "
                f"{item['error']}"
            )

    print()
    print("=" * 70)

    if not failed:
        print("ALL MEDICINES PASSED")
    else:
        print("INVESTIGATION REQUIRED")

    print("=" * 70)


if __name__ == "__main__":
    main()