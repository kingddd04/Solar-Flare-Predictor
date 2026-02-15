from config import Config
import pandas as pd 
import json 
import os
import numpy as np

class UvPreprocessor:
    def __init__(self, conf): # Corretto da init a init
        self.conf = conf

    def preprocess_uv(self):
        filepath = os.path.join(self.conf.data_dir, "euvs-7-day.json")

        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        
        # Convertiamo subito in datetime (usando UTC per evitare sfasamenti)
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)

        # Estrapolazione flags
        flags_df = df["flags"].apply(pd.Series) 
        df = pd.concat([df.drop(columns=["flags"]), flags_df], axis=1)

        # 1. Impostiamo a NaN i valori durante le eclissi/transiti PRIMA del pivot.
        # Non facciamo la media qui, la faremo correttamente dopo.
        mask = (df["eclipse"] | df["lunar_transit"] | df["geocorona"])
        df.loc[mask, "value"] = np.nan

        # 2. Pivot: Ora ogni linea (256, 304, etc) ha la sua colonna
        df = df.pivot_table(index="time_tag", columns="line", values="value")
        df.columns = [f"euv_{col}" for col in df.columns] # Prefisso per chiarezza
        
        # 3. ESPLICITARE I BUCHI (Cruciale per i guasti da 40 min)
        # Se mancano 40 minuti, il json salta temporalmente. resample('1min') 
        # forza il dataframe ad avere una riga esatta per ogni minuto, 
        # riempiendo i vuoti fisici con dei NaN evidenti.
        df = df.resample('1min').asfreq()

        # 4. MASCHERA DEI DATI MANCANTI (Per salvare la rete neurale)
        # Se tutte le colonne EUV sono NaN in quel minuto, segniamo 1
        df['euv_is_missing'] = df.isna().all(axis=1).astype(int)

        # Isoliamo le colonne da sistemare (tutte tranne la maschera appena creata)
        colonne_euv = [col for col in df.columns if col != 'euv_is_missing']

        # 5. RIPARAZIONE BUCHI PICCOLI (Calibrazione delle 16:17)
        # Interpola in modo fluido, ma si ferma se il buco è più grande di 5 minuti
        df[colonne_euv] = df[colonne_euv].interpolate(method='time', limit=5)

        # 6. RIPARAZIONE BUCHI GRANDI (Guasto dei 40 minuti)
        # Quello che limit=3 non ha riparato, viene "congelato" col valore precedente.
        # Così non si inventano trend falsi (esplosioni inesistenti) durante il guasto.
        df[colonne_euv] = df[colonne_euv].ffill()

        # Sicurezza sui bordi (se il primissimo dato del dataset era rotto)
        df[colonne_euv] = df[colonne_euv].bfill()

        print("->Uv Preprocessing Complete")
        return df