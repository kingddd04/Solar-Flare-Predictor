from config import Config
import os
import pandas as pd

conf = Config()

dataset_path = os.path.join(conf.data_dir, conf.dataset_name)

df = pd.read_csv(dataset_path)

print(df.shape)
print(df.isna().sum())

duplicate_dates = df[df["time_tag"].duplicated(keep=False)] 
print(f"\nNumber of duplicated time_tag entries: {duplicate_dates.shape[0]}")