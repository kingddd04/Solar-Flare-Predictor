import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class InferenceScaler:
    """Handle feature/target scaling for inference."""

    def __init__(self):
        """Initialize scalers and state."""
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False  # Track scaler state

    def scale_inference_label(self, X_data, y_data=None, apply_log10_target=True):
        """Scale inference features using fitted scalers."""
        if not self.is_fitted:
            raise ValueError(
                "Scalers have not been fitted yet. "
                "Call fit_and_scale_train() or load() before using this method."
            )

        print("[Test/Live] Applying scaling")

        # Save original shape
        original_shape = X_data.shape

        # Flatten 3D input
        if X_data.ndim == 3:
            X_flat = X_data.reshape(-1, X_data.shape[-1])
        else:
            X_flat = X_data

        # Apply scaling
        X_scaled_flat = self.x_scaler.transform(X_flat)

        # Restore original shape
        X_scaled = (
            X_scaled_flat.reshape(original_shape)
            if X_data.ndim == 3
            else X_scaled_flat
        )

        return X_scaled

    def decode_prediction(self, scaled_prediction, applied_log10=True):
        """Inverse-transform model output back to physical units."""
        if not self.is_fitted:
            raise ValueError("Scalers are not initialized or loaded.")

        # Undo scaling
        log_flux = self.y_scaler.inverse_transform(scaled_prediction)

        # Undo log10 if applied
        return 10 ** log_flux if applied_log10 else log_flux

    def load(self, folder_path):
        """Load fitted scalers from disk."""
        path_x = os.path.join(folder_path, "solar_x_scaler.pkl")
        path_y = os.path.join(folder_path, "solar_y_scaler.pkl")

        if not os.path.exists(path_x) or not os.path.exists(path_y):
            raise FileNotFoundError(
                "Scaler files not found. Expected: "
                "'solar_x_scaler.pkl' and 'solar_y_scaler.pkl'."
            )

        # Load scalers
        self.x_scaler = joblib.load(path_x)
        self.y_scaler = joblib.load(path_y)
        self.is_fitted = True

        print("Scaler files loaded successfully")
