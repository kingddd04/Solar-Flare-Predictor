from config import Config
from .train_set_creator import TrainSetCreator
from .train_test_spilt import Train_Test_Split
from .solar_flare_predictor import SolarFlarePredictor
from .training_scaler import TrainingScaler


class Training_Manager:

    def __init__(self):
        # Starting The Configuration class 
        conf = Config()

        # getting of various paths from the config
        model_dir = conf.model_dir
        model_path = conf.model_path
        dataset_path = conf.dataset_path

        # extraction of the dataset by the csv and creation of labels and target values
        tr_dataset_cr = TrainSetCreator(dataset_path)
        x, y = tr_dataset_cr.create_train_set()
        tr_dataset_cr.print_shapes()

        # splitting of the traing and test set
        x_tr, x_te, y_tr, y_te = Train_Test_Split.split_training(x,y)

        # normalization (scaling) of the datas
        training_scaler = TrainingScaler()
        x_trainig_scaled, y_training_scaled = training_scaler.fit_and_scale_train(x_tr, y_tr)
        x_test_fitted, y_test_fitted = training_scaler.scale_test_or_live(x_te, y_te)  
        training_scaler.save(model_dir)    

        # Training of the model
        sFP = SolarFlarePredictor(model_save_folder=model_path)
        sFP.train(x_trainig_scaled, y_training_scaled,x_test_fitted, y_test_fitted)




        

        

        

        


        
