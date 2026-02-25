"""Extract the latest fixed-size window for live inference."""

import os
import pandas as pd
import numpy as np


class InferenceDatasetExtractor:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.DROP_COLS  = ["time_tag","0.1-0.8nm"]

    def create_inference_set(self):
        """
        Loads the CSV file and returns the last 180 rows of feature data.
        This is the minimal dataset required for inference.

        Returns
        -------
        np.ndarray
            Array of shape (180, n_features) containing the last 180
            feature samples from the dataset.
        """
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Select only feature columns (exclude time_tag and target)
        feature_cols = [c for c in df.columns if c not in self.DROP_COLS]

        X = df[feature_cols].values.astype(np.float32)

        if len(X) < 180:
            raise ValueError(
                f"Dataset too small: requires at least 180 rows, found {len(X)}"
            )

        # Return only the last 180 samples
        return X[-180:]
