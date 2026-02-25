import pandas as pd
import os 
import json


class Xraybg_preprocessor:
    def __init__(self, conf):
        self.conf = conf

    def preprocess_xraybg(self, df_xray):
        """
        Compute background X-ray flux from minute-level X-ray data
        using NOAA-style rolling operations.
        """
        
        # Select long-channel column
        long_channel_col = "0.1-0.8nm"
        
        # Copy X-ray dataframe
        xray7day = df_xray.copy()
        

        # Apply hourly median filter
        mediana_oraria = xray7day[long_channel_col].rolling(window=60, center=False, min_periods=1).median()

        # Compute 24-hour rolling minimum
        background_flux = mediana_oraria.rolling(window=1440, center=False, min_periods=1).min()

        # Smooth background series
        xray7daybg = background_flux.rolling(window=180, center=False, min_periods=1).mean()
        xray7daybg = xray7daybg.rename("x_ray_bg")

        print("X-ray background preprocessing complete")
        return xray7daybg
