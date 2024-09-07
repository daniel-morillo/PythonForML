import requests
from bs4 import BeautifulSoup
import csv

# URL de la página de estadísticas de bateadores de la MLB
url = "https://www.mlb.com/es/stats/batting-average/2024"

# Realizar una solicitud GET a la página
response = requests.get(url)

# Crear el objeto BeautifulSoup para analizar el contenido HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Encontrar la tabla que contiene las estadísticas
table = soup.find('table')

# Crear una lista para almacenar las filas de la tabla
rows = []

# Recorrer todas las filas de la tabla
for row in table.find_all('tr')[1:]:  # Ignoramos el encabezado
    # Buscar tanto 'th' como 'td' ya que los nombres pueden estar en 'th'
    th_cells = row.find_all('th')
    td_cells = row.find_all('td')
    
    # Si las celdas tienen datos
    if len(th_cells) > 0 and len(td_cells) > 0:
        player_data = {
            "jugador": th_cells[0].text.strip(),       # Nombre del jugador
            "equipo": td_cells[0].text.strip(),        # Equipo
            "juegos": td_cells[1].text.strip(),        # Juegos
            "turnos_bateo": td_cells[2].text.strip(),  # Turnos de bateo
            "hits": td_cells[4].text.strip(),          # Hits
            "carreras": td_cells[3].text.strip(),      # Carreras
            "carreras_impulsadas": td_cells[8].text.strip(), # Carre
            "home_runs": td_cells[7].text.strip(),     # Home Runs
            "average": td_cells[13].text.strip(),      # Promedio o Average
            "obp": td_cells[14].text.strip(),          # OBP
            "slg": td_cells[15].text.strip(),          # SLG
            "ops": td_cells[16].text.strip()           # OPS
        }
        rows.append(player_data)

with open('mlb_batting_stats.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["jugador", "equipo", "juegos", "turnos_bateo", "hits", "carreras", "carreras_impulsadas", "home_runs", "average", "obp", "slg", "ops"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("Datos exportados exitosamente a mlb_batting_stats.csv")



