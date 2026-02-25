import numpy as np
from config import Config
from .solar_flare_predictor import SolarFlarePredictor
from .inference_scaler import InferenceScaler
from .inference_dataset_extractor import InferenceDatasetExtractor
from .solar_flare_classifier import SolarFlareClassifier

class Inference_Manager:
    def __init__(self):
        conf = Config()
        model_p = conf.model_path
        model_d = conf.model_dir
        dataset_path = conf.dataset_path
        sfp = SolarFlarePredictor(model_save_folder=model_p)
        sfp.load()

        datas_extr = InferenceDatasetExtractor(dataset_path)
        x_imput = datas_extr.create_inference_set()

        inf_scaler = InferenceScaler()
        inf_scaler.load(model_d)
        scaled_x = inf_scaler.scale_inference_label(x_imput)
        scaled_x_batched = scaled_x[np.newaxis, ...]


        predicted_y = sfp.predict_weather(scaled_x_batched)

        descaled_y = inf_scaler.decode_prediction(predicted_y)

        solar_class = SolarFlareClassifier.get_flare_class(descaled_y)

        print("Inference result:", descaled_y, "Solar_Class:", solar_class)





        



