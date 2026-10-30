import numpy as np
import pandas as pd

from .config import LONG_GAP_STEPS, RAW_CSV


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    # Read the CSV, parse timestamps, enforce strict chronological order.
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.drop_duplicates("Timestamp").sort_values("Timestamp").set_index("Timestamp")
    return df


def build_series(df: pd.DataFrame | None = None, long_gap_steps: int = LONG_GAP_STEPS) -> pd.DataFrame:
    """
    Return a complete 15-minute series with columns sales, redemptions, masked.
    Intervals missing from the file are treated as 'no tickets recorded' and filled
    with 0, except for very long gaps (>= long_gap_steps), which are masked (NaN)
    so they are never used for training or evaluation.
    """
    df = load_raw() if df is None else df
    full = pd.date_range(df.index.min(), df.index.max(), freq="15min", name="Timestamp")
    out = df[["Sales Count", "Redemption Count"]].reindex(full)
    out.columns = ["sales", "redemptions"]

    missing = out["sales"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    masked = missing & (run_len >= long_gap_steps)

    out.loc[missing & ~masked, ["sales", "redemptions"]] = 0
    out.loc[masked, ["sales", "redemptions"]] = np.nan
    out["masked"] = masked
    return out
