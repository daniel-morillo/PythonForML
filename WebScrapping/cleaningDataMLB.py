import pandas as pd
import re

# Función para procesar cada nombre
def procesar_nombre(nombre):
    #La cadena tiene 3 partes: nombre, apellido y Rol. Separaremos por espacios
    partes = nombre.split()
    
    #Guardamos nombre y apellido
    nombre = " ".join(partes[:-1])
    
    #Guardamos el Rol del jugador en el campo
    rol = partes[-1]
    return nombre, rol

# Cargar los datos del archivo excel
df = pd.read_excel('mlb_batting_stats.xlsx')

# Procesar los nombres de los jugadores
df['jugador'], df['rol'] = zip(*df['jugador'].map(procesar_nombre))

#Ordenar, Rol va despues de jugador
df = df[['jugador', 'rol', 'equipo', 'juegos', 'turnos_bateo', 'hits', 'carreras', 'carreras_impulsadas', 'home_runs', 'average', 'obp', 'slg', 'ops']]

print(df.head(3))

#Exportar los datos a otro archivo excel
df.to_excel('mlb_batting_stats_cleaned.xlsx', index=False)







