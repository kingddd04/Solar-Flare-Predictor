import pandas as pd
import os 
import json


class Xraybg_preprocessor:
    def __init__(self, conf):
        self.conf = conf

    def preprocess_xraybg(self, df_xray):
        """
        Calcola il Background Flux simulando l'algoritmo ufficiale NOAA,
        direttamente sui dati al minuto. Non necessita di file esterni.
        """
        
        # 1. Identificare la colonna del Canale Lungo
        # Nel tuo df è la seconda colonna, ma per sicurezza prendiamo quella 
        # con la media dei valori più alta (il canale lungo è sempre > del corto)
        long_channel_col = "0.1-0.8nm"
        
        # Copia di sicurezza
        xray7day = df_xray.copy()
        

        # STEP 1 UFFICIALE: Filtro del rumore e micro-brillamenti
        # Mediana mobile su 60 minuti. Usa min_periods=1 per non avere NaN iniziali
        mediana_oraria = xray7day[long_channel_col].rolling(window=60, center=False, min_periods=1).median()

        # STEP 2 UFFICIALE: Ricerca del "Pavimento" (Il livello di quiete magnetica)
        # Minimo mobile sulle 24 ore precedenti (24 ore * 60 minuti = 1440 minuti)
        background_flux = mediana_oraria.rolling(window=1440, center=False, min_periods=1).min()

        # STEP 3 UFFICIALE (Opzionale ma raccomandato): Smoothing
        # Una leggera media mobile di 3 ore (180 min) per smussare gli "scalini" del minimo
        xray7daybg = background_flux.rolling(window=180, center=False, min_periods=1).mean()
        xray7daybg = xray7daybg.rename("x_ray_bg")

        print("->X Ray Background Preproccessed!")
        return xray7daybg