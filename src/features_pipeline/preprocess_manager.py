from .xray_preprocessor import Xray_preprocessor
from .uv_preprocessor import UvPreprocessor
from .xraybg_preprocessor import Xraybg_preprocessor
from .dataset_downloader import DatasetDownloader

from config import Config

import os
import pandas as pd


class PreprocesserManager:
    def __init__(self):
        # Load configuration
        self.conf = Config()

        # Download or refresh raw data files
        self.update_files()

        # Run preprocessors and merge results
        merged_df = self.merge_preprocessed()

        # Save or update final dataset
        self.save_and_update_dataset(merged_df)

    def update_files(self):
        # Download raw datasets
        downloader = DatasetDownloader()
        downloader.datasetDownload()

    def merge_preprocessed(self):
        # Initialize preprocessors
        xray_proc = Xray_preprocessor(self.conf)
        uv_proc = UvPreprocessor(self.conf)
        xraybg_proc = Xraybg_preprocessor(self.conf)

        # Run preprocessing steps
        xray_df = xray_proc.preprocess_xray()
        uv_df = uv_proc.preprocess_uv()
        xraybg_df = xraybg_proc.preprocess_xraybg(xray_df)

        # Merge all dataframes on time_tag
        unified_df = (
            xray_df
            .merge(xraybg_df, on="time_tag", how="inner")
            .merge(uv_df, on="time_tag", how="inner")
        )

        return unified_df

    def save_and_update_dataset(self, new_data):
        # Resolve dataset path
        dataset_path = os.path.join(self.conf.data_dir, self.conf.dataset_name)

        if os.path.isfile(dataset_path):
            # Load existing dataset
            print("Dataset found; updating records")
            old_df = pd.read_csv(
                dataset_path,
                index_col="time_tag",
                parse_dates=["time_tag"]
            )

            # Merge old and new data
            combined = pd.concat([old_df, new_data])

            # Remove duplicate timestamps
            combined = combined[~combined.index.duplicated(keep="last")]

            # Sort chronologically
            combined.sort_index(inplace=True)

            # Save updated dataset
            combined.to_csv(dataset_path, index=True)

        else:
            # Create new dataset file
            print("Dataset not found; creating new file")
            new_data.to_csv(dataset_path, index=True)
            print("Dataset creation complete")
