import math
#Pob Infinita, cola inf, servidor > 1
# Parámetros del problema
lambda_rate = 3 
mu_rate = 1      
cost_service_per_hour = 40  
cost_delay_per_hour = 60  

def calculate_p0(lambda_rate, mu_rate, n):
    rho = lambda_rate / (n * mu_rate)
    sum_terms = sum([(lambda_rate / mu_rate) ** k / math.factorial(k) for k in range(n)])
    last_term = ((lambda_rate / mu_rate) ** n / math.factorial(n)) * (1 / (1 - rho))
    P0 = 1 / (sum_terms + last_term)
    return P0


# Función para calcular Lq 
def calculate_lq(lambda_rate, mu_rate, n):
    P0 = calculate_p0(lambda_rate, mu_rate, n)
    rho = lambda_rate / (n * mu_rate)
    numerator = P0 * ((lambda_rate / mu_rate) ** n) * rho
    denominator = math.factorial(n) * ((1 - rho) ** 2)
    Lq = numerator / denominator
    return Lq

def calculate_l(lambda_rate, mu_rate, n):
    Lq = calculate_lq(lambda_rate, mu_rate, n)
    L = Lq + (lambda_rate / mu_rate)  # Número promedio de clientes en el sistema
    return L



Po = calculate_p0(lambda_rate, mu_rate, 5)
print(f'Po: {Po:.6f}')
Lq = calculate_lq(lambda_rate, mu_rate, 5)
print(f'Lq: {Lq:.6f}')
L = calculate_l(lambda_rate, mu_rate, 5)
print(f'L: {L:.6f}')



