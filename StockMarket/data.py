import requests
import pandas as pd

def get_binance_klines(symbol="BTCUSDT", interval="1d", limit=1000):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convertir los datos a DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["crypto"] = symbol  # Añadir columna para identificar la criptomoneda
    return df

def get_multiple_cryptos_data(symbols, interval="1d", limit=1000):
    all_data = pd.DataFrame()
    
    for symbol in symbols:
        df = get_binance_klines(symbol, interval, limit)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    return all_data

# Lista de criptomonedas a obtener datos
crypto_symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]  # Puedes agregar más pares de criptos

# Obtener los datos de múltiples criptomonedas
df_all_cryptos = get_multiple_cryptos_data(crypto_symbols, "1d")

# Guardar los datos combinados en un archivo CSV
df_all_cryptos.to_csv("crypto_data.csv", index=False)

print("Datos guardados en crypto_data.csv")

