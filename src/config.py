import os

class Config:
    def __init__(self):
        # Go one folder up from the current working directory
        self.root_dir = os.path.dirname(os.getcwd())

        # Build the data directory inside that parent folder
        self.data_dir = os.path.join(self.root_dir, "datas")

        self.model_dir = os.path.join(self.root_dir, "ai_model")

        self.dataset_name = "sfp_dataset.csv"

        self.model_path = os.path.join(self.model_dir ,"sfp_lstm.keras")

        self.dataset_path = os.path.join(self.data_dir,self.dataset_name)

        # URLs for datasets
        self.dataset_urls = [
            "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
            "https://services.swpc.noaa.gov/json/goes/primary/euvs-7-day.json",
            "https://services.swpc.noaa.gov/json/goes/primary/xray-background-7-day.json",
        ]
