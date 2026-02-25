from config import Config
from .train_set_creator import TrainSetCreator
from .train_test_spilt import Train_Test_Split
from .solar_flare_predictor import SolarFlarePredictor
from .training_scaler import TrainingScaler


class Training_Manager:

    def __init__(self):
        # Initialize configuration.
        config = Config()

        # Read paths from configuration.
        model_dir = config.model_dir
        model_path = config.model_path
        dataset_path = config.dataset_path

        # Extract dataset features and labels from CSV.
        train_set_creator = TrainSetCreator(dataset_path)
        x, y = train_set_creator.create_train_set()
        train_set_creator.print_shapes()

        # Split data into training and test sets.
        x_tr, x_te, y_tr, y_te = Train_Test_Split.split_training(x,y)

        # Scale training and test data.
        training_scaler = TrainingScaler()
        x_training_scaled, y_training_scaled = training_scaler.fit_and_scale_train(x_tr, y_tr)
        x_test_fitted, y_test_fitted = training_scaler.scale_test_or_live(x_te, y_te)  
        training_scaler.save(model_dir)    

        # Train the model.
        solar_flare_predictor = SolarFlarePredictor(model_save_folder=model_path)
        solar_flare_predictor.train(x_training_scaled, y_training_scaled,x_test_fitted, y_test_fitted)




        

        

        

        


        
