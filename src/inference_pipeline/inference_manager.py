import numpy as np
from config import Config
from .solar_flare_predictor import SolarFlarePredictor
from .inference_scaler import InferenceScaler
from .inference_dataset_extractor import InferenceDatasetExtractor
from .solar_flare_classifier import SolarFlareClassifier


class Inference_Manager:

    def __init__(self):
        # Load configuration
        conf = Config()

        # Resolve paths
        model_dir = conf.model_dir
        model_path = conf.model_path
        dataset_path = conf.dataset_path

        # Load trained model
        predictor = SolarFlarePredictor(model_save_folder=model_path)
        predictor.load()

        # Extract last 180 samples for inference
        extractor = InferenceDatasetExtractor(dataset_path)
        x_input = extractor.create_inference_set()   # shape (180, n_features)

        # Load scalers and scale input
        scaler = InferenceScaler()
        scaler.load(model_dir)
        x_scaled = scaler.scale_inference_features(x_input)

        # Add batch dimension: (1, 180, n_features)
        x_scaled_batched = x_scaled[np.newaxis, ...]

        # Predict scaled output
        y_scaled_pred = predictor.predict_weather(x_scaled_batched)

        # Decode prediction back to real flux
        y_real = scaler.decode_prediction(
            y_scaled_pred,
            applied_log10=scaler.apply_log10_target,
        )

        # Convert flux to NOAA class
        flare_class = SolarFlareClassifier.get_flare_class(y_real)
        self.printResults(flare_class,y_real, extractor)
        

    def printResults(self, flare_class, y_real, extractor):
        print("\n\n")
        print("*"*20)
        print("X ray predicted Flux = ", y_real[0][0], "Solar Class:", flare_class)
        print(SolarFlareClassifier.get_alert_description(flare_class))
        print("Prediction for date: ", extractor.get_prediction_date_validity())
        print("*"*20)


     
