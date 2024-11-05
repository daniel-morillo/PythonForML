import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Defining paramters
S0 = 100 #Initial stock price

mu = 0.05 #Expected return or expected growth rate

sigma = 0.2 #Volatility

T = 1 #Time period (1 year)

dt = 1/252 #Time step (252 trading days in a year)

N = int(T/dt) #Number of time steps

NUM_SIMULATIONS = 1000 #Number of simulations

#Generating random numbers
def generate_random_numbers(seed: int, N: int, num_sims: int) -> np.ndarray:
    np.random.seed(seed) #Setting seed for reproducibility
    return np.random.normal(0, 1, (num_sims, N)) #Generating random numbers using a normal distribution with mean 0 and standard deviation 1

#Generating stock price paths
def generate_stock_price_paths(S0: float, mu: float, sigma: float, N: int, dt: float, random_numbers: np.ndarray, num_sims: int) -> np.ndarray:
    stock_price_paths = np.zeros((num_sims, N+1))
    stock_price_paths[:,0] = S0
    for i in range(1, N + 1):
        stock_price_paths[:,i] = stock_price_paths[:,i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_numbers[:,i-1])
    return stock_price_paths

#Plotting stock price paths
def plot_stock_price_paths(stock_price_paths: np.ndarray) -> None:
    plt.figure(figsize=(10,6))
    for i in range(len(stock_price_paths)):
        plt.plot(stock_price_paths[i])
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Paths')
    plt.show()

def plot_stock_price(S: np.ndarray) -> None:
    plt.figure(figsize=(10,6))
    plt.plot(S)
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Path')
    plt.show()

#Calculating meean and standard deviation of stock price paths
def calculate_mean_and_std(stock_price_paths: np.ndarray) -> pd.DataFrame:
    mean = np.mean(stock_price_paths, axis=0)
    std = np.std(stock_price_paths, axis=0)
    return pd.DataFrame({'Mean': mean, 'Standard Deviation': std})

#Testing
random_numbers = generate_random_numbers(100, N, NUM_SIMULATIONS)
stock_price_paths = generate_stock_price_paths(S0, mu, sigma, N, dt, random_numbers, NUM_SIMULATIONS)
pll = plot_stock_price(stock_price_paths)
mean_and_std = calculate_mean_and_std(stock_price_paths)
print(mean_and_std)