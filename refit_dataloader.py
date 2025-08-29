# refit_dataloader.py
import pandas as pd
import numpy as np

def load_house_csv(
    path,
    appliance_col="Appliance1",
    resample_rule="30s",
    max_rows=None,
    max_gap_factor=2.0,   # allow filling up to 2× original cadence
):
    df = pd.read_csv(path, low_memory=False)

    # --- Parse & index time
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).set_index("Time")

    # --- Keep only needed cols and cast
    cols = ["Aggregate", appliance_col]
    df = df[cols].astype("float32", copy=False)

    # --- Ensure monotonic unique index (important!)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # --- Detect original cadence (robust)
    diffs = df.index.to_series().diff().dropna()
    if len(diffs) == 0:
        raise ValueError("Time index has <2 rows after parsing.")
    orig_dt = diffs.median()
    tgt_dt  = pd.to_timedelta(resample_rule) if resample_rule else orig_dt

    if not resample_rule:
        # No resampling requested
        out = df
    else:
        # How far we are willing to bridge (in target steps)
        max_hold_steps = int(np.ceil((max_gap_factor * orig_dt) / tgt_dt))
        max_hold_steps = max(1, max_hold_steps)

        if tgt_dt >= orig_dt:
            # -------- Downsample (target bin >= raw cadence) --------
            # 1) Aggregate with mean into target bins
            out = df.resample(resample_rule).mean()

            # 2) Bridge only tiny empty bins (caused by jitters) but NOT outages
            #    Allow at most 1 empty bin on either side
            out = out.ffill(limit=1).bfill(limit=1)

            # 3) (Optional) very small interior pinholes → time interpolation inside only
            out = out.interpolate(method="time", limit_area="inside")
        else:
            # -------- Upsample (target bin < raw cadence) -------- THIS IS BAD THOUGH
            # Create target grid
            out = df.resample(resample_rule).asfreq()

            # Step-hold (zero-order hold) for small gaps, never across outages
            out = out.ffill(limit=max_hold_steps).bfill(limit=max_hold_steps)

            # Clean tiny interior NaN islands (caused by leading/trailing edges)
            out = out.interpolate(method="time", limit_area="inside")

    # Optional trim
    if max_rows is not None:
        out = out.iloc[:max_rows]

    return out["Aggregate"].values, out[appliance_col].values, out.index
