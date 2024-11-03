import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the model and encoder
model = load_model("crypto_price_prediction_model.h5")
crypto_encoder = joblib.load("crypto_encoder.pkl")  # Load the fitted encoder
main_scaler = joblib.load("main_scaler.pkl")  # Assuming you saved this scaler as well
close_scaler = joblib.load("close_scaler.pkl")

print("Modelo y codificador cargados y listos para hacer predicciones.")

# Cargar los datos más recientes
latest_data = pd.read_csv("latest_data.csv")

# Seleccionar las columnas relevantes
latest_data = latest_data[["timestamp", "open", "high", "low", "close", "volume", "crypto"]]

# Codificar la columna 'crypto' usando el encoder cargado
crypto_encoded_latest = crypto_encoder.transform(latest_data[["crypto"]])
crypto_encoded_latest_df = pd.DataFrame(crypto_encoded_latest, columns=crypto_encoder.categories_[0])

# Unir la codificación con el DataFrame de los datos recientes
latest_data = pd.concat([latest_data, crypto_encoded_latest_df], axis=1)

# Escalar las características numéricas usando el escalador cargado
latest_data[["open", "high", "low", "close", "volume"]] = main_scaler.transform(latest_data[["open", "high", "low", "close", "volume"]])

# Crear secuencia de los últimos 60 días
sequence_length = 60
X_latest = []

if len(latest_data) >= sequence_length:
    features_latest = latest_data.iloc[-sequence_length:][["open", "high", "low", "close", "volume"] + list(crypto_encoder.categories_[0])].values
    X_latest.append(features_latest)

X_latest = np.array(X_latest)

# Hacer predicción
predicted_scaled_close = model.predict(X_latest)
predicted_close = close_scaler.inverse_transform(predicted_scaled_close)
print("Predicción para el precio de cierre:", predicted_close[0][0])


