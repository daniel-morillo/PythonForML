import yfinance as yf
import pandas as pd

def get_stock_data(stock_name, start_date, end_date):
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    stock_data = stock_data[['Open', 'Close','High','Low','Volume']]

    stock_data['Symbol'] = stock_name
    
    return stock_data

def save_stock_data(stock_symbols: list , start_date, end_date, filename = 'stock_training.csv' ):

    all_data = []

    for symbol in stock_symbols:
        print(f"Collecting data from {symbol}")
        stock_data = get_stock_data(symbol, start_date, end_date)
        all_data.append(stock_data)

    combined_data = pd.concat(all_data)

    combined_data.to_csv(filename, index = True)
    print(f'Data guardada en {filename}')

def main():
    start_date = '2024-09-01'
    end_date = '2024-09-30'
    
    # Lista de símbolos de las acciones que te interesan
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD']
    
    # Guardar los datos en un archivo CSV
    save_stock_data(stock_symbols, start_date, end_date)

main()