import pandas as pd

def load_data(url):
    data = pd.read_csv(url)
    return data

def main():

    data = load_data('iris.csv')
    print(data.head())
    print(data.describe())

main()