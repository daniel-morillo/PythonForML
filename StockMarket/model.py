import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1. Cargar los datos
data = pd.read_csv("crypto_data.csv")

# Seleccionar las columnas relevantes
data = data[["timestamp", "open", "high", "low", "close", "volume", "crypto"]]

# Codificar la columna 'crypto' como una característica categórica
crypto_encoder = OneHotEncoder(sparse_output=False)
crypto_encoded = crypto_encoder.fit_transform(data[["crypto"]])
crypto_encoded_df = pd.DataFrame(crypto_encoded, columns=crypto_encoder.categories_[0])
joblib.dump(crypto_encoder, "crypto_encoder.pkl")

# Unir la codificación con el DataFrame original
data = pd.concat([data, crypto_encoded_df], axis=1)

# 2. Escalar todas las características para el modelo
main_scaler = MinMaxScaler(feature_range=(0, 1))
data[["open", "high", "low", "close", "volume"]] = main_scaler.fit_transform(data[["open", "high", "low", "close", "volume"]])
joblib.dump(crypto_encoder, "main_scaler.pkl")

# Crear un escalador separado solo para el precio de 'close'
close_scaler = MinMaxScaler(feature_range=(0, 1))
data["close_scaled"] = close_scaler.fit_transform(data[["close"]])
joblib.dump(crypto_encoder, "close_scaler.pkl")

# Crear secuencias de datos de entrada (X) y etiquetas (y)
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(data)):
    # Tomar los 60 días anteriores de todas las características (incluyendo las columnas de criptos)
    features = data.loc[i-sequence_length:i-1, ["open", "high", "low", "close", "volume"] + list(crypto_encoder.categories_[0])].values
    X.append(features)
    y.append(data.loc[i, "close_scaled"])  # Etiqueta es el precio de cierre escalado

X, y = np.array(X), np.array(y)

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Salida única para el precio de cierre

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# 4. Evaluar el modelo
predictions = model.predict(X_test)
predictions = close_scaler.inverse_transform(predictions)  # Desescalar predicciones usando solo el escalador de 'close'

# Desescalar y_test
y_test_descaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular el error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test_descaled) ** 2))
print("RMSE:", rmse)
model.save("crypto_price_prediction_model.h5")
print("Modelo guardado exitosamente.")


