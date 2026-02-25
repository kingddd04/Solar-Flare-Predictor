import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class TrainingScaler:
    """
    Scale feature and target arrays for training and inference.
    Keep fitted scaler objects for save and load operations.
    """
    
    def __init__(self):
        # Initialize scaler objects
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False  # Track scaler state

    def fit_and_scale_train(self, X_train, y_train, apply_log10_target=True):
        """
        Fit scalers on training data and return scaled arrays.
        """
        print("[Train] Starting fit and scaling")

        # Transform target values
        if apply_log10_target:
            y_train_proc = np.log10(y_train)
        else:
            y_train_proc = y_train

        y_train_2d = y_train_proc.reshape(-1, 1) if len(y_train_proc.shape) == 1 else y_train_proc

        # Flatten feature arrays
        orig_x_shape = X_train.shape
        if len(orig_x_shape) == 3:
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        else:
            X_train_2d = X_train

        # Fit and scale arrays
        X_train_scaled_2d = self.x_scaler.fit_transform(X_train_2d)
        y_train_scaled = self.y_scaler.fit_transform(y_train_2d)
        
        self.is_fitted = True

        # Restore feature shape
        if len(orig_x_shape) == 3:
            X_train_scaled = X_train_scaled_2d.reshape(orig_x_shape)
        else:
            X_train_scaled = X_train_scaled_2d

        return X_train_scaled, y_train_scaled

    def scale_test_or_live(self, X_data, y_data=None, apply_log10_target=True):
        """
        Scale new arrays using previously fitted scalers.
        Return only features when target data is not provided.
        """
        if not self.is_fitted:
            raise ValueError("Errore: Gli scaler non sono stati addestrati (chiama prima fit_and_scale_train o load)!")
            
        print("[Test/Live] Applying scaling")
        
        # Scale feature arrays
        orig_x_shape = X_data.shape
        X_2d = X_data.reshape(-1, X_data.shape[-1]) if len(orig_x_shape) == 3 else X_data
        X_scaled_2d = self.x_scaler.transform(X_2d)
        X_scaled = X_scaled_2d.reshape(orig_x_shape) if len(orig_x_shape) == 3 else X_scaled_2d
        
        # Scale target arrays
        if y_data is not None:
            y_proc = np.log10(y_data) if apply_log10_target else y_data
            y_2d = y_proc.reshape(-1, 1) if len(y_proc.shape) == 1 else y_proc
            y_scaled = self.y_scaler.transform(y_2d)
            return X_scaled, y_scaled
            
        return X_scaled

    def decode_prediction(self, scaled_prediction, applied_log10=True):
        """Convert scaled model output to real X-ray flux."""
        if not self.is_fitted:
            raise ValueError("Scaler non inizializzati!")
            
        log_flux = self.y_scaler.inverse_transform(scaled_prediction)
        return (10 ** log_flux) if applied_log10 else log_flux

    def save(self, folder_path):
        """Save fitted scaler objects to disk."""
        if not self.is_fitted:
            print("Warning: saving scalers before fitting")
            
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.x_scaler, os.path.join(folder_path, 'solar_x_scaler.pkl'))
        joblib.dump(self.y_scaler, os.path.join(folder_path, 'solar_y_scaler.pkl'))
        print(f"Saved scaler files to: {folder_path}")

    def load(self, folder_path='./modelli_salvati/'):
        """Load scaler objects from disk into this instance."""
        path_x = os.path.join(folder_path, 'solar_x_scaler.pkl')
        path_y = os.path.join(folder_path, 'solar_y_scaler.pkl')
        
        if not os.path.exists(path_x) or not os.path.exists(path_y):
            raise FileNotFoundError("File degli scaler non trovati!")
            
        self.x_scaler = joblib.load(path_x)
        self.y_scaler = joblib.load(path_y)
        self.is_fitted = True
        print("Scaler files loaded successfully")
