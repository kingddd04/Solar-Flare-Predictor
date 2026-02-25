"""Scale training and inference tensors for the forecasting model."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class TrainingScaler:
    """
    Classe per gestire la normalizzazione dei dati solari.
    Mantiene lo stato degli scaler al suo interno per poterli salvare e riutilizzare.
    """
    
    def __init__(self):
        # Inizializza gli scaler come attributi dell'oggetto
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False  # Flag di sicurezza

    def fit_and_scale_train(self, X_train, y_train, apply_log10_target=True):
        """
        Studia (fit) i dati di training e li scala (transform).
        Da usare SOLO sul Train Set.
        """
        print("-> [Train] Inizio Fit e Scaling...")

        # 1. Target (Logaritmo opzionale)
        if apply_log10_target:
            y_train_proc = np.log10(y_train)
        else:
            y_train_proc = y_train

        y_train_2d = y_train_proc.reshape(-1, 1) if len(y_train_proc.shape) == 1 else y_train_proc

        # 2. Features (Gestione 3D -> 2D)
        orig_x_shape = X_train.shape
        if len(orig_x_shape) == 3:
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        else:
            X_train_2d = X_train

        # 3. Fit & Transform degli attributi di classe
        X_train_scaled_2d = self.x_scaler.fit_transform(X_train_2d)
        y_train_scaled = self.y_scaler.fit_transform(y_train_2d)
        
        self.is_fitted = True

        # 4. Ripristino 3D
        if len(orig_x_shape) == 3:
            X_train_scaled = X_train_scaled_2d.reshape(orig_x_shape)
        else:
            X_train_scaled = X_train_scaled_2d

        return X_train_scaled, y_train_scaled

    def scale_test_or_live(self, X_data, y_data=None, apply_log10_target=True):
        """
        Applica lo scaling a dati nuovi (Test Set o Dati Live) senza reimparare min/max.
        Se y_data non viene passato (es. in inferenza live), scala solo le X.
        """
        if not self.is_fitted:
            raise ValueError("Errore: Gli scaler non sono stati addestrati (chiama prima fit_and_scale_train o load)!")
            
        print("-> [Test/Live] Applicazione Scaling in corso...")
        
        # Features
        orig_x_shape = X_data.shape
        X_2d = X_data.reshape(-1, X_data.shape[-1]) if len(orig_x_shape) == 3 else X_data
        X_scaled_2d = self.x_scaler.transform(X_2d)
        X_scaled = X_scaled_2d.reshape(orig_x_shape) if len(orig_x_shape) == 3 else X_scaled_2d
        
        # Target (se presente, altrimenti in inferenza live restituiamo solo le X)
        if y_data is not None:
            y_proc = np.log10(y_data) if apply_log10_target else y_data
            y_2d = y_proc.reshape(-1, 1) if len(y_proc.shape) == 1 else y_proc
            y_scaled = self.y_scaler.transform(y_2d)
            return X_scaled, y_scaled
            
        return X_scaled

    def decode_prediction(self, scaled_prediction, applied_log10=True):
        """Converte la previsione della rete neurale nel flusso X-Ray reale."""
        if not self.is_fitted:
            raise ValueError("Scaler non inizializzati!")
            
        log_flux = self.y_scaler.inverse_transform(scaled_prediction)
        return (10 ** log_flux) if applied_log10 else log_flux

    def save(self, folder_path):
        """Salva gli attributi scaler su disco."""
        if not self.is_fitted:
            print("Attenzione: Stai salvando degli scaler che non hanno ancora fatto il fit sui dati!")
            
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.x_scaler, os.path.join(folder_path, 'solar_x_scaler.pkl'))
        joblib.dump(self.y_scaler, os.path.join(folder_path, 'solar_y_scaler.pkl'))
        print(f"-> Scaler salvati in: {folder_path}")

    def load(self, folder_path='./modelli_salvati/'):
        """Carica gli scaler dal disco all'interno di questa istanza."""
        path_x = os.path.join(folder_path, 'solar_x_scaler.pkl')
        path_y = os.path.join(folder_path, 'solar_y_scaler.pkl')
        
        if not os.path.exists(path_x) or not os.path.exists(path_y):
            raise FileNotFoundError("File degli scaler non trovati!")
            
        self.x_scaler = joblib.load(path_x)
        self.y_scaler = joblib.load(path_y)
        self.is_fitted = True
        print("-> Scaler caricati in memoria con successo!")
