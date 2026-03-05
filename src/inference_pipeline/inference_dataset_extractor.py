import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class InferenceDatasetExtractor:
    """Extract the last 180 feature rows for inference."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.DROP_COLS = ["time_tag", "xray_0.1-0.8nm"]
        self.last_date = None

    def create_inference_set(self):
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Convert time_tag to datetime
        self.last_date = pd.to_datetime(df["time_tag"].iloc[-1])

        feature_cols = [c for c in df.columns if c not in self.DROP_COLS]
        X = df[feature_cols].values.astype(np.float32)

        if len(X) < 180:
            raise ValueError(
                f"Dataset too small: requires at least 180 rows, found {len(X)}"
            )

        return X[-180:]

    def get_prediction_date_validity(self):
        return self.last_date + timedelta(minutes=90)
