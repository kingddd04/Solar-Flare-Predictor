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
        new_datas = self.merge_preprocessed()
        self.save_and_update_dataset(new_datas)

    def update_files(self):
        downloader = DatasetDownloader()
        downloader.datasetDownload()


    def merge_preprocessed(self):    
        xray_Preproc = Xray_preprocessor(self.conf)
        uv_Preproc = UvPreprocessor(self.conf)
        xraybc_Preproc = Xraybg_preprocessor(self.conf)

        xray_7d_df = xray_Preproc.preprocess_xray()
        uv_7d_df = uv_Preproc.preprocess_uv()
        xraybg_7d_df = xraybc_Preproc.preprocess_xraybg(xray_7d_df)

        unified_df = xray_7d_df.merge(xraybg_7d_df, on="time_tag", how="inner").merge(uv_7d_df, on="time_tag", how="inner")

        return unified_df


    def save_and_update_dataset(self, new_datas):
        """
        Save the unified dataset to disk.
        If the dataset already exists, append new rows without duplicates.
        """
        dataset_path = os.path.join(self.conf.data_dir, self.conf.dataset_name)

        # If dataset exists, load it and append new data
        if os.path.isfile(dataset_path):
            print("Dataset found — updating it")

            old_df = pd.read_csv(dataset_path, parse_dates=["time_tag"])

            # Ensure both have time_tag as column
            if old_df.index.name == "time_tag":
                old_df = old_df.reset_index()
            if new_datas.index.name == "time_tag":
                new_datas = new_datas.reset_index()

            combined = pd.concat([old_df, new_datas], ignore_index=True)
            combined = combined.drop_duplicates(subset=["time_tag"], keep="last")
            combined = combined.sort_values("time_tag")

            combined.to_csv(dataset_path, index=False)

        else:
            print("Dataset not found — creating a new one")
            new_datas.to_csv(dataset_path, index=True)
            print("Dataset created successfully")
