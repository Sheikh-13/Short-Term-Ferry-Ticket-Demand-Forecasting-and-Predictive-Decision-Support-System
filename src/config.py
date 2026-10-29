"""Central configuration for the ferry demand forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "Toronto_Island_Ferry_Tickets.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

# Forecast targets (column in the prepared series -> display name)
TARGETS = {"sales": "Ticket Sales", "redemptions": "Ticket Redemptions"}

# Forecast horizon in minutes -> number of 15-minute steps in the window
HORIZONS = {15: 1, 30: 2, 60: 4, 120: 8}

# Feature engineering
LAGS = [1, 2, 4, 8]          # t-1, t-2, t-4, t-8 (in 15-minute steps)
ROLL_WINDOWS = [4, 96]       # 1 hour and 1 day
MOVING_AVG_WINDOW = 8        # baseline moving average (2 hours)

# Missing-interval handling: gaps shorter than this are filled with 0 (no tickets
# were recorded); gaps this long or longer are masked and excluded.
LONG_GAP_STEPS = 48          # 12 hours

# Chronological split
TRAIN_END = "2023-12-31 23:45"
VAL_END = "2024-12-31 23:45"

# Uncertainty and KPI settings
INTERVAL_ALPHA = 0.10        # 90% prediction intervals
PEAK_QUANTILE = 0.90         # a "peak" window is in the top 10% of training windows
PEAK_MISS_RATIO = 0.75       # a peak is "missed" if predicted < 75% of actual
LEAD_TIME_RECALL = 0.70      # horizon counts as usable if peak recall >= 70%
