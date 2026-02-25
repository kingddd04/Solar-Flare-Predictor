import os
import pandas as pd
import numpy as np


class TrainSetCreator:
    """
    Build training feature windows and future targets from the CSV dataset.
    """

    # Define excluded columns
    DROP_COLS  = ["time_tag", "0.1-0.8nm"]
    TARGET_COL = "0.1-0.8nm"

    WINDOW  = 180   # Set input window length
    HORIZON = 90    # Set target horizon

    def __init__(self,  csv_path: str = None):
        self.csv_path = csv_path
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

    def print_shapes(self):
   
        # Print total entries
        print(f"Number of entries: {len(self.x)}")
        
        # Print dataset shapes
        print(f"Shape x: {self.x.shape}")  # Report feature shape
        print(f"Shape y: {self.y.shape}")  # Report target shape





    def create_train_set(self):
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}\n"
            )

        df = pd.read_csv(self.csv_path)

        feature_cols = [c for c in df.columns if c not in self.DROP_COLS]
        X = df[feature_cols].values.astype(np.float32)    # (N, n_features)
        y = df[self.TARGET_COL].values.astype(np.float32) # (N,)

        n_samples = len(df) - self.WINDOW - self.HORIZON

        for i in range(n_samples):

            x_window = X[i : i + self.WINDOW]           
            target_idx = i + self.WINDOW + self.HORIZON   
            y_val      = y[target_idx]                    

            self.x.append(x_window)             
            self.y.append(np.array(y_val))       
        self.x = np.array(self.x)  
        self.y = np.array(self.y) 

        return self.x, self.y
