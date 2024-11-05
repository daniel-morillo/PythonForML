import requests
import pandas as pd

def get_latest_data(symbol="BTCUSDT", interval="1d", limit=60):
    """
    Fetches the latest data from Binance API for the given symbol.
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", 
                                     "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
                                     "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["crypto"] = symbol  # Add column to identify the cryptocurrency
    return df

# Fetch latest 60 days of data for a list of symbols
crypto_symbols = ["BTCUSDT"]  # Update with desired symbols
latest_data_all = pd.DataFrame()

for symbol in crypto_symbols:
    latest_data = get_latest_data(symbol)
    latest_data_all = pd.concat([latest_data_all, latest_data], ignore_index=True)

# Here, `latest_data_all` contains the latest 60 days of data for all specified cryptocurrencies.
print("Latest data fetched successfully.")
#Saving data in a csv
latest_data_all.to_csv("latest_data.csv", index=False)