import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Parámetros

S0 = 100 # Precio inicial

mu = 0.05 # Tasa de crecimiento esperada

sigma = 0.2 # Volatilidad

T = 1 # Tiempo en años

dt = 1/252 # Paso de tiempo (suponiendo 252 días hábiles por año)

N = int(T/dt) # Número de pasos

num_sims = 1000 # Número de simulaciones

# Generar números aleatorios

np.random.seed(100)

dW = np.random.normal(size=(num_sims, N))

# Simulación del precio

S = np.zeros((num_sims, N+1))

S[:,0] = S0

for t in range(1, N+1):

    S[:,t] = S[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*dW[:,t-1])

# Visualizar los resultados

plt.plot(S)

plt.xlabel('Tiempo')

plt.ylabel('Precio')

plt.title('Simulación de Monte Carlo para Precios de Acciones')

plt.show()