import os
import pandas as pd
import numpy as np


class TrainSetCreator:
    """
    Carica il CSV sfp_dataset e costruisce le sequenze di training.

    Per ogni campione i:
      - x[i] : finestra delle prime 180 righe (minuti) a partire da i,
                     tutte le colonne feature  →  shape (180, n_features)
      - y[i] : valore della colonna '0.1-0.8nm' al minuto i+90
                     (il target è il valore XRSB a 90 minuti nel futuro)

    Entrambe le liste contengono array con una dimensione batch aggiunta:
      x[i].shape  → (1, 180, n_features)
      y[i].shape  → (1, 1)
    """

    # Colonne da escludere dalle feature
    DROP_COLS  = ["time_tag", "0.1-0.8nm"]
    TARGET_COL = "0.1-0.8nm"

    WINDOW  = 180   # lunghezza sequenza di input (minuti)
    HORIZON = 90    # minuti nel futuro per il target

    def __init__(self,  csv_path: str = None):
        self.csv_path = csv_path
        self.x: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

    def print_shapes(self):
  
        # Shape complessive
        print(f"Numer of entries: {len(self.x)}")
        
        # Shape del primo elemento (tutti gli altri sono identici)
        print(f"Shape x: {self.x.shape}")  # (1, 180, n_features)
        print(f"Shape y: {self.y.shape}")  # (1, 1)





    def create_train_set(self):
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}\n"
            )

        df = pd.read_csv(self.csv_path)

        # Feature matrix: tutte le colonne tranne time_tag ed euv_is_missing
        feature_cols = [c for c in df.columns if c not in self.DROP_COLS]
        X = df[feature_cols].values.astype(np.float32)    # (N, n_features)
        y = df[self.TARGET_COL].values.astype(np.float32) # (N,)

        n_samples = len(df) - self.WINDOW - self.HORIZON + 1

        for i in range(n_samples):
            # Finestra di input: righe [i : i+WINDOW], tutte le feature
            x_window = X[i : i + self.WINDOW]             # (180, n_features)

            # Target: valore 0.1-0.8nm a 90 minuti dall'inizio della finestra
            target_idx = i + self.HORIZON                  # minuto i+90
            y_val      = y[target_idx]                     # scalar

            self.x.append(x_window)              # invece di x_window[np.newaxis, ...]
            self.y.append(np.array(y_val))       # invece di array([[y_val]])
        self.x = np.array(self.x)   # (N, 180, n_features)
        self.y = np.array(self.y)   # (N,)



        
        return self.x, self.y