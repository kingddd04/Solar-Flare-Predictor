import os
import requests
import json
from config import Config

class DatasetDownloader:
    def __init__(self):
        self.conf = Config()
        # Create data directory
        os.makedirs(self.conf.data_dir, exist_ok=True)

    def datasetDownload(self):
        for url in self.conf.dataset_urls :
            filename = url.split("/")[-1]
            response = requests.get(url)
            data = response.json()

            filepath = os.path.join(self.data_dir, filename)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Saved dataset file: {filename} at {filepath}")
            
downloader = DatasetDownloader()
downloader.datasetDownload()
