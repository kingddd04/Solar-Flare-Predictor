import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler


class TrainingScaler:
    """
    Scale feature and target arrays for training and inference.
    Keep fitted scaler objects for save and load operations.
    """
    
    def __init__(self):
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def fit_and_scale_train(self, X_train, y_train, apply_log10_target=True):
        print("[Train] Starting fit and scaling")

        # Fix: avoid log10(0)
        if apply_log10_target:
            eps = 1e-10
            y_train_safe = np.where(y_train <= 0, eps, y_train)
            y_train_proc = np.log10(y_train_safe)
        else:
            y_train_proc = y_train

        y_train_2d = (
            y_train_proc.reshape(-1, 1)
            if len(y_train_proc.shape) == 1
            else y_train_proc
        )

        # Flatten features
        orig_x_shape = X_train.shape
        if len(orig_x_shape) == 3:
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        else:
            X_train_2d = X_train

        # Fit scalers
        X_train_scaled_2d = self.x_scaler.fit_transform(X_train_2d)
        y_train_scaled = self.y_scaler.fit_transform(y_train_2d)

        self.is_fitted = True

        # Restore shape
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
    
    def save(self, folder_path):
        """Save fitted scaler objects to disk."""
        if not self.is_fitted:
            print("Warning: saving scalers before fitting")
            
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.x_scaler, os.path.join(folder_path, 'solar_x_scaler.pkl'))
        joblib.dump(self.y_scaler, os.path.join(folder_path, 'solar_y_scaler.pkl'))
        print(f"Saved scaler files to: {folder_path}")
