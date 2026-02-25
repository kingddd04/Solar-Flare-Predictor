from config import Config
import pandas as pd
import os
import json

class Xray_preprocessor:
    def __init__(self, conf):
        self.conf = conf
        
    def preprocess_xray(self):
        # Load X-ray JSON data
        filepath = os.path.join(self.conf.data_dir,"xrays-7-day.json")

        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)

        df = df.pivot_table(
            index="time_tag",
            columns="energy",
            values="observed_flux"
        )

        print("X-ray preprocessing complete")

        return df
