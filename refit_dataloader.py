# loader_refit_housecsv.py (updated)
import os
import pandas as pd
import numpy as np
from collections import deque



def load_house_csv(path, appliance_col="Appliance1", resample_rule="1min", max_minutes=None):
    df = pd.read_csv(path, low_memory=False)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).set_index("Time")

    df = df[["Aggregate", appliance_col]].astype("float32")
    df = df.resample(resample_rule).mean().fillna(0.0)

    if max_minutes is not None:
        df = df.iloc[:max_minutes]

    return df["Aggregate"].values, df[appliance_col].values, df.index