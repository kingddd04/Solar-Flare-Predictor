import os
import pandas as pd
import numpy as np


class TrainSetCreator:
    """Create training windows and future targets from the dataset."""

    # Columns to exclude
    DROP_COLS  = ["time_tag", "0.1-0.8nm"]
    TARGET_COL = "0.1-0.8nm"

    # Window and horizon sizes
    WINDOW  = 180
    HORIZON = 90

    def __init__(self, csv_path: str = None):
        """Initialize creator with dataset path."""
        self.csv_path = csv_path
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

    def print_shapes(self):
        """Print dataset shapes."""
        # Print total entries
        print(f"Number of entries: {len(self.x)}")

        # Print shapes
        print(f"Shape x: {self.x.shape}")
        print(f"Shape y: {self.y.shape}")

    def create_train_set(self):
        """Build sliding windows and targets for training."""
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Select feature columns
        feature_cols = [c for c in df.columns if c not in self.DROP_COLS]
        X = df[feature_cols].values.astype(np.float32)     # (N, n_features)
        y = df[self.TARGET_COL].values.astype(np.float32)  # (N,)

        # Number of valid samples
        n_samples = len(df) - self.WINDOW - self.HORIZON

        # Build windows
        for i in range(n_samples):
            x_window = X[i : i + self.WINDOW]               # Input window
            target_idx = i + self.WINDOW + self.HORIZON     # Future target index
            y_val = y[target_idx]                           # Target value

            self.x.append(x_window)
            self.y.append(np.array(y_val))

        # Convert to arrays
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        return self.x, self.y