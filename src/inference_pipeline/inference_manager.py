from training_pipeline.solar_flare_predictor import SolarFlarePredictor
from training_pipeline.training_preprocessing_manager import TrainingPreprocessingManager


class InferenceManager:
    """
    Gestisce l'inferenza caricando il modello e i scaler salvati durante il training.
    I scaler usati DEVONO essere quelli fittati SOLO sui dati di training
    per evitare data leakage.
    """

    def __init__(self, model_path="best_solar_model.h5", scaler_dir="saved_scalers"):
        self.predictor = SolarFlarePredictor()
        self.predictor.load(model_path)

        self.preprocessing_manager = TrainingPreprocessingManager(save_dir=scaler_dir)
        self.preprocessing_manager.load_scalers()

    def predict(self, X_recent_window):
        """
        Esegue una previsione su dati grezzi (non scalati).
        Applica il feature_scaler e decodifica il risultato col target_scaler.
        Entrambi i scaler sono quelli fittati in fase di training.
        """
        return self.predictor.predict_next_flare(
            X_recent_window,
            feature_scaler=self.preprocessing_manager.feature_scaler,
            target_scaler=self.preprocessing_manager.target_scaler
        )
