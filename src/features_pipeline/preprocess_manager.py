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
        Save the merged dataset to disk.
        Merge with existing data when the dataset file is present.
        """
        dataset_path = os.path.join(self.conf.data_dir, self.conf.dataset_name)

        if os.path.isfile(dataset_path):
            print("Dataset found; updating records")
            old_df = pd.read_csv(dataset_path, index_col="time_tag", parse_dates=["time_tag"])

            combined = old_df.merge(new_datas, on="time_tag", how="outer")

            combined.to_csv(dataset_path, index=True)

        else:
            print("Dataset not found; creating file")
            new_datas.to_csv(dataset_path, index=True)
            print("Dataset creation complete")
