import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

class SolarFlarePredictor:
    """
    Classe wrapper per il modello LSTM di previsione dei brillamenti solari.
    Gestisce l'intero ciclo di vita del modello: costruzione, training, salvataggio e inferenza.
    """
    
    def __init__(self, window_size=120, n_features=7, learning_rate=0.001):
        """
        Inizializza i parametri base. Il modello non è ancora costruito.
        """
        self.window_size = window_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # Ecco il tuo modello come attributo della classe (inizialmente vuoto)
        self.model = None 

    def build_model(self):
        """Costruisce e compila l'architettura della Rete Neurale."""
        print("-> Costruzione architettura LSTM in corso...")
        
        self.model = Sequential([
            Input(shape=(self.window_size, self.n_features)),
            LSTM(units=64, return_sequences=True, activation='tanh'),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False, activation='tanh'),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print("-> Modello compilato con successo!")

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, save_path='best_solar_model.h5'):
        """Allena il modello e salva i pesi migliori."""
        if self.model is None:
            raise ValueError("Errore: Devi chiamare build_model() prima di fare il training!")
            
        print("\n=== INIZIO ADDESTRAMENTO ===")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True, verbose=1)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            shuffle=False, # Mai mischiare le serie temporali!
            verbose=1
        )
        
        print("=== ADDESTRAMENTO COMPLETATO ===")
        return history

    def save(self, filepath):
        """Salva il modello manualmente."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"-> Modello salvato in: {filepath}")
        else:
            print("Nessun modello da salvare.")

    def load(self, filepath):
        """Carica un modello pre-addestrato dal disco."""
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"-> Modello {filepath} caricato con successo. Pronto per le previsioni!")
            
            # Aggiorniamo le dimensioni lette dal modello caricato
            self.window_size = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
        else:
            raise FileNotFoundError(f"Errore: Il file {filepath} non esiste.")

    def predict_next_flare(self, X_recent_window, target_scaler):
        """
        Fa l'inferenza (previsione) sui nuovi dati e decodifica il risultato.
        X_recent_window: Array Numpy di forma (1, 120, 7) con gli ultimi dati al minuto.
        target_scaler: L'oggetto MinMaxScaler usato durante il preprocessing per la Y.
        """


        if self.model is None:
            raise ValueError("Errore: Carica o addestra un modello prima di prevedere.")
                    
        # 1. La rete neurale fa la previsione (sputerà un numero tra 0 e 1)
        scaled_prediction = self.model.predict(X_recent_window)
                
        # 2. Usiamo lo scaler per riportare il numero alla scala fisica reale (es. 1.5e-5)
        real_xray_flux = target_scaler.inverse_transform(scaled_prediction)
                
        return real_xray_flux[0][0]