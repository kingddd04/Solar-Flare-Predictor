from config import Config
import os
import pandas as pd

conf = Config()

dataset_path = os.path.join(conf.data_dir, conf.dataset_name)

df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)
print("Missing values per column:\n", df.isna().sum())
