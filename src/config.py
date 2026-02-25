import os

class Config:
    """Central configuration for paths and dataset URLs."""

    def __init__(self):
        """Initialize directory paths and dataset settings."""
        # Resolve project root
        self.root_dir = os.path.dirname(os.getcwd())

        # Build data directory path
        self.data_dir = os.path.join(self.root_dir, "datas")

        # Model directory
        self.model_dir = os.path.join(self.root_dir, "ai_model")

        # Dataset filename
        self.dataset_name = "sfp_dataset.csv"

        # Full model path
        self.model_path = os.path.join(self.model_dir, "sfp_lstm.keras")

        # Full dataset path
        self.dataset_path = os.path.join(self.data_dir, self.dataset_name)

        # NOAA dataset URLs
        self.dataset_urls = [
            "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
            "https://services.swpc.noaa.gov/json/goes/primary/euvs-7-day.json",
            "https://services.swpc.noaa.gov/json/goes/primary/xray-background-7-day.json",
        ]
