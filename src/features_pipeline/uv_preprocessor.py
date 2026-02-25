from config import Config
import pandas as pd 
import json 
import os
import numpy as np

class UvPreprocessor:
    """Preprocess the NOAA EUV dataset."""

    def __init__(self, conf):
        """Store configuration reference."""
        self.conf = conf

    def preprocess_uv(self):
        """Load and preprocess EUV JSON into a cleaned DataFrame."""
        filepath = os.path.join(self.conf.data_dir, "euvs-7-day.json")

        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data)

        # Parse timestamps
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)

        # Expand flag columns
        flags_df = df["flags"].apply(pd.Series)
        df = pd.concat([df.drop(columns=["flags"]), flags_df], axis=1)

        # Mask eclipse/transit samples
        mask = (df["eclipse"] | df["lunar_transit"] | df["geocorona"])
        df.loc[mask, "value"] = np.nan

        # Pivot spectral lines
        df = df.pivot_table(index="time_tag", columns="line", values="value")
        df.columns = [f"euv_{col}" for col in df.columns]

        # Resample to 1‑minute frequency
        df = df.resample('1min').asfreq()

        # Mark missing rows
        df['euv_is_missing'] = df.isna().all(axis=1).astype(int)

        # Select EUV columns
        colonne_euv = [col for col in df.columns if col != 'euv_is_missing']

        # Interpolate short gaps
        df[colonne_euv] = df[colonne_euv].interpolate(method='time', limit=5)

        # Forward-fill long gaps
        df[colonne_euv] = df[colonne_euv].ffill()

        # Backfill leading gaps
        df[colonne_euv] = df[colonne_euv].bfill()

        print("UV preprocessing complete")
        return df
