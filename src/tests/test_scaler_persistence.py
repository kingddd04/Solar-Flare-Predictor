import os
import sys
import tempfile
import unittest

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "training_pipeline"))
sys.path.insert(0, os.path.join(ROOT_DIR, "inference_pipeline"))

from training_scaler import TrainingScaler
from inference_scaler import InferenceScaler


class TestScalerPersistence(unittest.TestCase):
    def test_inference_uses_saved_log10_setting(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        y = np.array([2.0, 5.0], dtype=np.float64)

        scaler = TrainingScaler()
        scaler.fit_and_scale_train(X, y, apply_log10_target=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler.save(tmpdir)
            inf_scaler = InferenceScaler()
            inf_scaler.load(tmpdir)

            self.assertFalse(inf_scaler.apply_log10_target)
            _, scaled = scaler.scale_test_or_live(
                np.array([[2.0, 3.0]], dtype=np.float64),
                np.array([3.5], dtype=np.float64),
                apply_log10_target=False,
            )
            decoded = inf_scaler.decode_prediction(
                scaled,
                applied_log10=inf_scaler.apply_log10_target,
            )
            self.assertTrue(np.allclose(decoded, np.array([[3.5]]), atol=1e-8))

    def test_inference_scaler_sanitizes_input(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        y_train = np.array([2.0, 5.0], dtype=np.float64)

        scaler = TrainingScaler()
        scaler.fit_and_scale_train(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler.save(tmpdir)
            inf_scaler = InferenceScaler()
            inf_scaler.load(tmpdir)

            X_inference = np.array([[np.nan, np.inf], [1.0, 3.0]], dtype=np.float64)
            X_scaled = inf_scaler.scale_inference_features(X_inference)
            self.assertTrue(np.isfinite(X_scaled).all())


if __name__ == "__main__":
    unittest.main()
