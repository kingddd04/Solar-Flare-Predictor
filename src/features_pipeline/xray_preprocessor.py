import pandas as pd
import os
import json

class Xray_preprocessor:
    """Preprocess the NOAA X-ray dataset."""

    def __init__(self, conf):
        self.conf = conf

    def preprocess_xray(self):
        filepath = os.path.join(self.conf.data_dir, "xrays-7-day.json")

        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data)

        # 1. Remove empty rows
        df = df.dropna(how="any", subset=["time_tag", "energy"])

        # 2. Convert time_tag
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)

        # 3. Normalize energy labels
        df["energy"] = df["energy"].astype(str).str.strip()
        df["energy"] = df["energy"].str.replace("–", "-", regex=False)
        df["energy"] = df["energy"].str.replace(" ", "", regex=False)

        # 4. Pivot
        df = df.pivot_table(
            index="time_tag",
            columns="energy",
            values="observed_flux"
        )

        # 5. Replace 0 only in X-ray flux columns
        df = df.replace(0, pd.NA)

        # 6. Resample to 1-minute grid
        df = df.resample("1min").asfreq()

        # 7. Missingness flag BEFORE filling
        df["is_missing"] = df.isna().any(axis=1).astype(int)

        # 8. Fill missing values
        df = df.ffill().bfill()
        df = df.infer_objects(copy=False)

        # 9. Rename columns
        df.columns = [f"xray_{col}" for col in df.columns]

        print("X-ray preprocessing complete")
        return df
