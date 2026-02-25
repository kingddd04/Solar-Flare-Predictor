import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler


class TrainingPreprocessingManager:
    """
    Gestisce il preprocessing dei dati per il training, evitando il data leakage.
    I scaler vengono fittati SOLO sui dati di training e poi applicati
    sia al training che al validation set.
    """

    def __init__(self, save_dir="saved_scalers"):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def split_data(self, df, target_column, train_ratio=0.8):
        """
        Divide i dati in training e validation rispettando l'ordine temporale.
        Non viene fatto shuffling per preservare la struttura della serie temporale.
        """
        split_index = int(len(df) * train_ratio)

        features = df.drop(columns=[target_column]).values
        target = df[[target_column]].values

        X_train = features[:split_index]
        X_val = features[split_index:]
        y_train = target[:split_index]
        y_val = target[split_index:]

        return X_train, X_val, y_train, y_val

    def fit_and_transform(self, X_train, X_val, y_train, y_val):
        """
        Fitta i scaler SOLO sui dati di training per evitare data leakage,
        poi trasforma sia il training set che il validation set.
        """
        # Fit sui dati di training SOLTANTO
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train)

        # Transform sui dati di validazione usando i scaler già fittati
        X_val_scaled = self.feature_scaler.transform(X_val)
        y_val_scaled = self.target_scaler.transform(y_val)

        return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled

    def save_scalers(self):
        """Salva i scaler su disco per riutilizzarli durante l'inferenza."""
        feature_path = os.path.join(self.save_dir, "feature_scaler.pkl")
        target_path = os.path.join(self.save_dir, "target_scaler.pkl")

        joblib.dump(self.feature_scaler, feature_path)
        joblib.dump(self.target_scaler, target_path)
        print(f"-> Scaler salvati in: {self.save_dir}")

    def load_scalers(self):
        """Carica i scaler salvati dal disco."""
        feature_path = os.path.join(self.save_dir, "feature_scaler.pkl")
        target_path = os.path.join(self.save_dir, "target_scaler.pkl")

        if not os.path.exists(feature_path) or not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Scaler non trovati in {self.save_dir}. "
                "Esegui prima il training per generarli."
            )

        self.feature_scaler = joblib.load(feature_path)
        self.target_scaler = joblib.load(target_path)
        print(f"-> Scaler caricati da: {self.save_dir}")

    def transform_inference_data(self, X_new):
        """
        Trasforma nuovi dati per l'inferenza usando i scaler già fittati.
        NON fitta nuovi scaler — usa quelli salvati dal training.
        """
        return self.feature_scaler.transform(X_new)
