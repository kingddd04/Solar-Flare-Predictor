import os
import requests
import json
from config import Config

class DatasetDownloader:
    """Download JSON datasets defined in the configuration."""

    def __init__(self):
        """Initialize downloader and ensure data directory exists."""
        self.conf = Config()
        # Ensure data directory exists
        os.makedirs(self.conf.data_dir, exist_ok=True)

    def datasetDownload(self):
        """Download all datasets listed in configuration."""
        # Loop through all dataset URLs
        for url in self.conf.dataset_urls:
            filename = url.split("/")[-1]

            # Download JSON data
            response = requests.get(url)
            data = response.json()

            # Build full save path
            filepath = os.path.join(self.conf.data_dir, filename)

            # Save JSON file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Saved dataset file: {filename} at {filepath}")