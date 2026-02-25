"""Scale inference features and decode model predictions."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class InferenceScaler:
    """
    Utility class for handling feature and target scaling during training,
    testing, and live inference.

    It stores fitted scalers internally so they can be saved, loaded,
    and reused consistently across different stages of the pipeline.
    """

    def __init__(self):
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False  # Safety flag to prevent accidental use


    def scale_inference_label(self, X_data, y_data=None, apply_log10_target=True):
        """
        Scale new data (test set or live inference) using previously fitted scalers.
        The scalers are *not* refitted here — they must already be trained or loaded.

        Parameters
        ----------
        X_data : np.ndarray
            Input features. Can be 2D (samples × features) or 3D
            (samples × timesteps × features).

        y_data : np.ndarray or None
            Optional target values. If None, only X is scaled (useful for live inference).

        apply_log10_target : bool
            If True, applies log10 to the target before scaling.

        Returns
        -------
        X_scaled : np.ndarray
            Scaled features.

        y_scaled : np.ndarray (optional)
            Scaled target values, only if y_data is provided.
        """
        if not self.is_fitted:
            raise ValueError(
                "Scalers have not been fitted yet. "
                "Call fit_and_scale_train() or load() before using this method."
            )

        print("-> [Test/Live] Applying scaling...")

        # --- Scale X -------------------------------------------------------
        original_shape = X_data.shape

        # Flatten only if input is 3D (e.g., sequences)
        if X_data.ndim == 3:
            X_flat = X_data.reshape(-1, X_data.shape[-1])
        else:
            X_flat = X_data

        X_scaled_flat = self.x_scaler.transform(X_flat)

        # Restore original shape if needed
        X_scaled = (
            X_scaled_flat.reshape(original_shape)
            if X_data.ndim == 3
            else X_scaled_flat
        )

        return X_scaled


    def decode_prediction(self, scaled_prediction, applied_log10=True):
        """
        Convert a scaled neural network prediction back to the original flux domain.

        Parameters
        ----------
        scaled_prediction : np.ndarray
            Model output in scaled space.

        applied_log10 : bool
            Whether the original target was log10-transformed.

        Returns
        -------
        np.ndarray
            Decoded prediction in physical units.
        """
        if not self.is_fitted:
            raise ValueError("Scalers are not initialized or loaded.")

        log_flux = self.y_scaler.inverse_transform(scaled_prediction)
        return 10 ** log_flux if applied_log10 else log_flux


    def load(self, folder_path):
        """
        Load previously saved scalers from disk into this instance.

        Parameters
        ----------
        folder_path : str
            Directory containing 'solar_x_scaler.pkl' and 'solar_y_scaler.pkl'.
        """
        path_x = os.path.join(folder_path, "solar_x_scaler.pkl")
        path_y = os.path.join(folder_path, "solar_y_scaler.pkl")

        if not os.path.exists(path_x) or not os.path.exists(path_y):
            raise FileNotFoundError(
                "Scaler files not found. Expected: "
                "'solar_x_scaler.pkl' and 'solar_y_scaler.pkl'."
            )

        self.x_scaler = joblib.load(path_x)
        self.y_scaler = joblib.load(path_y)
        self.is_fitted = True

        print("-> Scalers successfully loaded into memory.")
