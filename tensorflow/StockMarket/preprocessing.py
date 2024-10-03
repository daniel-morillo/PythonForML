import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler

filename = 'stock_training.csv'

df = pd.read_csv(filename)
df['Date'] = pd.to_datetime(df['Date'])

# Normalizar los datos
scaler = MinMaxScaler() #Lo escalamos todo entre 0 y 1

numeric_columns = ['Open', 'Close', 'High', 'Low', 'Volume']

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

df = df.sort_values(by = ['Symbol', 'Date'])

#Calcular medias moviles de 5 y 10 dias
df['MA-5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = 5).mean())
df['MA-10'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = 10).mean())

#calcular RSI
df['Price-Change'] = df.groupby('Symbol')['Close'].diff()

#Separar ganancias y perdidas
df['Gain'] = df['Price-Change'].clip(lower = 0)
df['Loss'] = df['Price-Change'].clip(upper = 0)

DAYS_WINDOW = 14

df['AVG_GAIN'] = df.groupby('Symbol')['Gain'].transform(lambda x: x.rolling(window = DAYS_WINDOW).mean())
df['AVG_LOSS'] = df.groupby('Symbol')['Loss'].transform(lambda x: x.rolling(window = DAYS_WINDOW).mean())

df['RS'] = df['AVG_GAIN'] / df['AVG_LOSS']
df['RSI'] = 100 - (100 / (1 + df['RS']))

#df.drop(columns = ['Price-Change', 'Gain', 'Loss', 'AVG_GAIN', 'AVG_LOSS', 'RS'], inplace = True)

print(df.head(20))


# Guardar los datos normalizados
#df.to_csv('normalized_stock_training.csv', index = False)

