import requests
from bs4 import BeautifulSoup
import csv

# URL base de la página
base_url = 'https://www.spotrac.com/mlb/'

# Lista de equipos y sus URLs individuales
teams_urls = {
    'LAD': 'los-angeles-dodgers/',
    'NYM': 'new-york-mets/',
    'NYY': 'new-york-yankees/',
    'HOU': 'houston-astros/',
    'PHI': 'philadelphia-phillies/',
    'CHC': 'chicago-cubs/',
    'ATL': 'atlanta-braves/',
    'SF': 'san-francisco-giants/',
    'TEX': 'texas-rangers/',
    'TOR': 'toronto-blue-jays/',
    'BOS': 'boston-red-sox/',
    'LAA': 'los-angeles-angels/',
    'STL': 'st-louis-cardinals/',
    'ARI': 'arizona-diamondbacks/',
    'SD': 'san-diego-padres/',
    'COL': 'colorado-rockies/',
    'CHW': 'chicago-white-sox/',
    'SEA': 'seattle-mariners/',
    'MIN': 'minnesota-twins/',
    'KC': 'kansas-city-royals/',
    'WSH': 'washington-nationals/',
    'BAL': 'baltimore-orioles/',
    'MIL': 'milwaukee-brewers/',
    'CLE': 'cleveland-guardians/',
    'CIN': 'cincinnati-reds/',
    'DET': 'detroit-tigers/',
    'MIA': 'miami-marlins/',
    'PIT': 'pittsburgh-pirates/',
    'TB': 'tampa-bay-rays/',
    'OAK': 'oakland-athletics/'
}

# Función para obtener la lista de jugadores y sus contratos desde la página del equipo
def get_players_and_contracts(team_url):
    response = requests.get(base_url + team_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encuentra la tabla con los jugadores y contratos (esto puede variar según la página)
        table = soup.find('table')  # Ajusta según la estructura de la página de los equipos

        players = []
        contracts = []

        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            
            if len(columns) > 1:
                player_name = columns[1].text.strip()  # Nombre del jugador
                player_name = player_name.split('\n')
                player_name = player_name[1]
                contract_value = columns[7].text.strip()  # Valor del contrato
                
                players.append(player_name)
                contracts.append(contract_value)
        
        # Devuelve la lista de jugadores y contratos
        return list(zip(players, contracts))
    else:
        print(f"Error al acceder a la página del equipo: {team_url}")
        return []

# Recorre los equipos y obtiene los jugadores y contratos
#for team, url in teams_urls.items():
#    print(f"Equipo: {team}")
#    players_and_contracts = get_players_and_contracts(url)
#    
#    for player, contract in players_and_contracts:
#        print(f"Jugador: {player}, Contrato: {contract}")

#Guardar los jugadores y los jugadores de los equipos en un CSV
#Usaremos formato: Jugador, Contrato, Equipo
with open('mlb_players_contracts.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Jugador', 'Contrato', 'Equipo']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for team, url in teams_urls.items():
        players_and_contracts = get_players_and_contracts(url)
        for player, contract in players_and_contracts:
            writer.writerow({'Jugador': player, 'Contrato': contract, 'Equipo': team})

print("Datos exportados exitosamente a mlb_players_contracts.csv")
