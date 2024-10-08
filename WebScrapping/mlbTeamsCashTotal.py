import requests
from bs4 import BeautifulSoup
import csv

# URL de la página que contiene la tabla
url = 'https://www.spotrac.com/mlb/cash/_/year/2024/sort/cash_total'

# Realiza la petición HTTP
response = requests.get(url)

# Verifica que la solicitud fue exitosa (código 200)
if response.status_code == 200:
    # Parsear el HTML de la página
    soup = BeautifulSoup(response.content, 'html.parser')

    # Encuentra la tabla de interés (puedes buscarla por etiquetas específicas)
    table = soup.find('table')  # Busca la primera tabla en la página

    # Crea listas para almacenar los equipos y sus respectivos salarios totales
    teams = []
    total_cash = []

    # Recorre las filas de la tabla (excluyendo la fila del encabezado)
    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        
        if len(columns) > 1:  # Asegura que hay columnas
            team = columns[1].text.strip()  # Nombre del equipo
            team = team.split('\n')
            team = team[0]
            salary = columns[7].text.strip()  # Salario total (columna "Total Cash")
            
            teams.append(team)
            total_cash.append(salary)

    # Muestra los equipos y salarios extraídos
    for team, cash in zip(teams, total_cash):
        if team == 'Totals':
            print('Salarios Totales: ' + cash)
        elif team == 'Averages':
            print('Salarios Promedio: ' + cash)
        else:
            print(f"Equipo: {team}, Salario Total: {cash}\n")
else:
    print(f"Error al acceder a la página: {response.status_code}")

# Guardar los datos en un archivo CSV
with open('mlb_teams_cash_total.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Equipo', 'Salario Total']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for team, cash in zip(teams, total_cash):
        if team == 'Totals' or team == 'Averages':
            continue
        else:
            writer.writerow({'Equipo': team, 'Salario Total': cash})

print("Datos exportados exitosamente a mlb_teams_cash_total.csv")