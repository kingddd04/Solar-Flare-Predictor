"""Run end-to-end inference and print the predicted flare class."""

import numpy as np
from config import Config
from .solar_flare_predictor import SolarFlarePredictor
from .inference_scaler import InferenceScaler
from .inference_dataset_extractor import InferenceDatasetExtractor
from .solar_flare_classifier import SolarFlareClassifier

class Inference_Manager:
    def __init__(self):
        conf = Config()
        model_path = conf.model_path
        model_dir = conf.model_dir
        dataset_path = conf.dataset_path
        predictor = SolarFlarePredictor(model_save_folder=model_path)
        predictor.load()

        dataset_extractor = InferenceDatasetExtractor(dataset_path)
        inference_input = dataset_extractor.create_inference_set()

        inference_scaler = InferenceScaler()
        inference_scaler.load(model_dir)
        scaled_x = inference_scaler.scale_inference_label(inference_input)
        batched_scaled_x = scaled_x[np.newaxis, ...]


        predicted_y = predictor.predict_weather(batched_scaled_x)

        decoded_y = inference_scaler.decode_prediction(predicted_y)

        solar_class = SolarFlareClassifier.get_flare_class(decoded_y)

        print("Predicted X-ray flux:", decoded_y, "Solar class:", solar_class)





        



