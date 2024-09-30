import yfinance as yf 
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_stock_data(stock_name, start_date, end_date):
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    stock_data = stock_data[['Open', 'Close','High','Low','Volume']]

    stock_data['Symbol'] = stock_name
    
    return stock_data

def lookCompaniesSP(url):
    table = pd.read_html(url)
    sp500 = table[0]
    return sp500


def lookCryptoYahoo():
    url = 'https://finance.yahoo.com/markets/crypto/all/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table')
    rows = table.find_all('tr')[1:] #Omitimos primera fila

    crypto = []
    for row in rows:
        columns = row.find_all('td')
        symbol_and_name = columns[0].text
        symbol_and_name = symbol_and_name.split(' ')
        symbol = symbol_and_name[2]
        name = symbol_and_name[3] + ' ' + symbol_and_name[4]
        crypto.append({'Symbol': symbol, 'Name': name})
    print(crypto)
    return pd.DataFrame(crypto)
    
def saveCompaniesCSV(sp, cryptos):
    # Combinar ambos DataFrames
    combined = pd.concat([sp[['Symbol', 'Security']], cryptos], ignore_index=True)
    
    # Guardar en un archivo CSV
    combined.to_csv('sp500_and_cryptos.csv', index=False)
    print("Datos guardados en sp500_and_cryptos.csv")

def main():
    start_date = '2024-09-20'
    end_date = '2024-09-30'
    

    # Obtener datos históricos de una acción específica
    stock_name = 'BTC-USD'
    stock_data = get_stock_data(stock_name, start_date, end_date)
    print(stock_data.head(10))

main()
