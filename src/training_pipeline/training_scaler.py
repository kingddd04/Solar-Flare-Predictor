import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler


class TrainingScaler:
    """
    Robust scaler for features and targets.
    Includes:
    - sanitization of NaN / Inf / extreme values
    - safe log10 transform with epsilon
    - shape‑safe operations for 2D/3D inputs
    """

    def __init__(self, feature_range=(0, 1), eps=1e-10):
        self.x_scaler = MinMaxScaler(feature_range=feature_range)
        self.y_scaler = MinMaxScaler(feature_range=feature_range)
        self.eps = eps
        self.apply_log10_target = True
        self.is_fitted = False

    # ---------------------------------------------------------
    # Sanitization
    # ---------------------------------------------------------
    def _sanitize(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)

    # ---------------------------------------------------------
    # Safe log10
    # ---------------------------------------------------------
    def _safe_log10(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.where(arr <= 0, self.eps, arr)
        return np.log10(arr)

    # ---------------------------------------------------------
    # Shape utilities
    # ---------------------------------------------------------
    def _flatten_if_3d(self, X):
        orig_shape = X.shape
        if len(orig_shape) == 3:
            return X.reshape(-1, orig_shape[-1]), orig_shape
        return X, orig_shape

    def _restore_shape(self, X_flat, orig_shape):
        if len(orig_shape) == 3:
            return X_flat.reshape(orig_shape)
        return X_flat

    # ---------------------------------------------------------
    # Training scaling
    # ---------------------------------------------------------
    def fit_and_scale_train(self, X_train, y_train, apply_log10_target=True):
        print("[Train] Starting fit and scaling")
        self.apply_log10_target = apply_log10_target

        # Sanitize
        X_train = self._sanitize(X_train)
        y_train = self._sanitize(y_train)

        # Log10
        if apply_log10_target:
            y_proc = self._safe_log10(y_train)
        else:
            y_proc = y_train

        # Ensure 2D
        y_2d = y_proc.reshape(-1, 1) if y_proc.ndim == 1 else y_proc

        # Flatten X if needed
        X_flat, orig_shape = self._flatten_if_3d(X_train)

        # Fit scalers
        X_scaled_flat = self.x_scaler.fit_transform(X_flat)
        y_scaled = self.y_scaler.fit_transform(y_2d)

        self.is_fitted = True

        # Restore shape
        X_scaled = self._restore_shape(X_scaled_flat, orig_shape)

        return X_scaled, y_scaled

    # ---------------------------------------------------------
    # Test / live scaling
    # ---------------------------------------------------------
    def scale_test_or_live(self, X_data, y_data=None, apply_log10_target=True):
        print("[Test/Live] Applying scaling")

        # Sanitize
        X_data = self._sanitize(X_data)

        # Flatten X
        X_flat, orig_shape = self._flatten_if_3d(X_data)
        X_scaled_flat = self.x_scaler.transform(X_flat)
        X_scaled = self._restore_shape(X_scaled_flat, orig_shape)

        if y_data is None:
            return X_scaled

        # Sanitize y
        y_data = self._sanitize(y_data)

        # Log10
        if apply_log10_target:
            y_proc = self._safe_log10(y_data)
        else:
            y_proc = y_data

        y_2d = y_proc.reshape(-1, 1) if y_proc.ndim == 1 else y_proc
        y_scaled = self.y_scaler.transform(y_2d)

        return X_scaled, y_scaled

    # ---------------------------------------------------------
    # Save / load
    # ---------------------------------------------------------
    def save(self, folder_path):
        if not self.is_fitted:
            print("Warning: saving scalers before fitting")

        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.x_scaler, os.path.join(folder_path, "solar_x_scaler.pkl"))
        joblib.dump(self.y_scaler, os.path.join(folder_path, "solar_y_scaler.pkl"))
        joblib.dump(
            {
                "eps": self.eps,
                "apply_log10_target": self.apply_log10_target,
            },
            os.path.join(folder_path, "solar_scaler_meta.pkl"),
        )
        print(f"Saved scaler files to: {folder_path}")

    def load(self, folder_path):
        x_path = os.path.join(folder_path, "solar_x_scaler.pkl")
        y_path = os.path.join(folder_path, "solar_y_scaler.pkl")
        meta_path = os.path.join(folder_path, "solar_scaler_meta.pkl")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(
                "Scaler files not found. Expected: "
                "'solar_x_scaler.pkl' and 'solar_y_scaler.pkl'."
            )

        self.x_scaler = joblib.load(x_path)
        self.y_scaler = joblib.load(y_path)
        if os.path.exists(meta_path):
            metadata = joblib.load(meta_path)
            self.eps = metadata.get("eps", self.eps)
            self.apply_log10_target = metadata.get(
                "apply_log10_target",
                self.apply_log10_target,
            )
        self.is_fitted = True
        print(f"Loaded scalers from: {folder_path}")
