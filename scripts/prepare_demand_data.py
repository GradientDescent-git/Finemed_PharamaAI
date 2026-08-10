"""
scripts/prepare_demand_data.py
=================================
Thin CLI wrapper. The real logic lives in
finemed_ai.demand_forecasting.data_preparation, so both this script and
the API's /admin/upload-monthly-data endpoint call the exact same code --
no duplicated logic to drift out of sync.

Usage:
    python scripts/prepare_demand_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finemed_ai.demand_forecasting.data_preparation import prepare_demand_data

if __name__ == "__main__":
    prepare_demand_data()
