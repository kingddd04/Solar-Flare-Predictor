from .xray_preprocessor import Xray_preprocessor
from .uv_preprocessor import UvPreprocessor
from .xraybg_preprocessor import Xraybg_preprocessor
from .dataset_downloader import DatasetDownloader

from config import Config

import os
import pandas as pd


class PreprocesserManager:
    def __init__(self):
        self.conf = Config()

        self.update_files()

        merged_df = self.merge_preprocessed()

        self.save_and_update_dataset(merged_df)

    def update_files(self):
        downloader = DatasetDownloader()
        downloader.datasetDownload()

    def merge_preprocessed(self):
        xray_proc = Xray_preprocessor(self.conf)
        uv_proc = UvPreprocessor(self.conf)
        xraybg_proc = Xraybg_preprocessor(self.conf)

        xray_df = xray_proc.preprocess_xray()
        uv_df = uv_proc.preprocess_uv()
        xraybg_df = xraybg_proc.preprocess_xraybg(xray_df)

        # Ensure DatetimeIndex everywhere
        xray_df.index = pd.to_datetime(xray_df.index, utc=True)
        uv_df.index = pd.to_datetime(uv_df.index, utc=True)
        xraybg_df.index = pd.to_datetime(xraybg_df.index, utc=True)

        # Merge using index
        unified_df = xray_df.join([xraybg_df, uv_df], how="inner")

        return unified_df

    def save_and_update_dataset(self, new_data):
        dataset_path = os.path.join(self.conf.data_dir, self.conf.dataset_name)

        if os.path.isfile(dataset_path):
            print("Dataset found; updating records")

            old_df = pd.read_csv(
                dataset_path,
                index_col="time_tag",
                parse_dates=["time_tag"]
            )

            combined = pd.concat([old_df, new_data])

            combined = combined[~combined.index.duplicated(keep="last")]

            combined.sort_index(inplace=True)

            combined.to_csv(dataset_path, index=True)

        else:
            print("Dataset not found; creating new file")
            new_data.to_csv(dataset_path, index=True)
            print("Dataset creation complete")

    