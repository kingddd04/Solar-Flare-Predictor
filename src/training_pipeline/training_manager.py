"""Coordinate dataset preparation, scaling, and model training."""

from config import Config
from .train_set_creator import TrainSetCreator
from .train_test_spilt import Train_Test_Split
from .solar_flare_predictor import SolarFlarePredictor
from .training_scaler import TrainingScaler


class Training_Manager:

    def __init__(self):
        # Start the configuration class.
        conf = Config()

        # Read required paths from configuration.
        model_dir = conf.model_dir
        model_path = conf.model_path
        dataset_path = conf.dataset_path

        # Load the dataset and build features and target arrays.
        tr_dataset_cr = TrainSetCreator(dataset_path)
        x, y = tr_dataset_cr.create_train_set()
        tr_dataset_cr.print_shapes()

        # Split training and test sets.
        x_tr, x_te, y_tr, y_te = Train_Test_Split.split_training(x,y)

        # Normalize feature and target data.
        training_scaler = TrainingScaler()
        x_trainig_scaled, y_training_scaled = training_scaler.fit_and_scale_train(x_tr, y_tr)
        x_test_fitted, y_test_fitted = training_scaler.scale_test_or_live(x_te, y_te)  
        training_scaler.save(model_dir)    

        # Train and save the model.
        sFP = SolarFlarePredictor(model_save_folder=model_path)
        sFP.train(x_trainig_scaled, y_training_scaled,x_test_fitted, y_test_fitted)




        

        

        

        


        
