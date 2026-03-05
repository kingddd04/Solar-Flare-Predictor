from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os


class SolarFlarePredictor:
    """
    Wrap the LSTM solar flare model.
    Handle model build, train, save, load, and prediction steps.
    """

    def __init__(self, window_size=180, n_features=12, learning_rate=0.001, model_save_folder=None):
        """
        Initialize model settings.
        """
        self.window_size = window_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.model_save_folder = model_save_folder

        # Initialize model attribute
        self.model = None
        self.history = None
        self.build_model()

    def build_model(self):
        """Build and compile the LSTM model."""
        print("Building LSTM model architecture")

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
        print("Model compilation complete")

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
        """Train the model and save the best checkpoint."""

        print("\n=== TRAINING START ===")

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=self.model_save_folder, monitor='val_loss', save_best_only=True, verbose=1)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            shuffle=False,  # Preserve time order
            verbose=1
        )

        # Save history as attribute for later plotting
        self.history = history

        print("=== TRAINING COMPLETE ===")
        return history

    def plot_training_history(self, save_path=None):
        """Plot train and validation loss curves after training."""
        if self.history is None:
            raise ValueError("Nessuna history disponibile. Esegui train() prima.")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.history.history['loss'], label='Train Loss', color='steelblue', linewidth=2)
        ax.plot(self.history.history['val_loss'], label='Validation Loss', color='tomato', linewidth=2)

        ax.set_title('Andamento della Loss durante il Training', fontsize=14)
        ax.set_xlabel('Epoca')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot salvato in: {save_path}")

        plt.show()

    def save(self):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save(self.model_save_folder)
            print(f"Saved model to: {self.model_save_folder}")
        else:
            print("No model available to save")

    def load(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_save_folder):
            self.model = load_model(self.model_save_folder)
            print(f"Loaded model from {self.model_save_folder}")

            # Update input dimensions
            self.window_size = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
        else:
            raise FileNotFoundError(f"Error: the file {self.model_save_folder} does not exists.")

    def predict_next_flare(self, X_recent_window, target_scaler):
        """
        Predict next flare flux from recent data and decode the value.
        """

        # Run model prediction
        scaled_prediction = self.model.predict(X_recent_window)

        # Decode predicted flux
        real_xray_flux = target_scaler.inverse_transform(scaled_prediction)

        return real_xray_flux[0][0]
