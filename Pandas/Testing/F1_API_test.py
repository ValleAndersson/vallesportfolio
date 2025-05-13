import requests
import pandas as pd

# Steg 1: Hämta data från Ergast API
url = "http://ergast.com/api/f1/current.json?limit=1000"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    # Steg 2: Plocka ut relevanta data. Här tas race-informationen som exempel.
    races = data["MRData"]["RaceTable"]["Races"]
    # Steg 3: Konvertera till en DataFrame
    df_races = pd.DataFrame(races)
    print("Hämtade races:")
    print(df_races.head())
else:
    print("Fel vid hämtning av data:", response.status_code)

# Steg 4: Filtrera ut oönskade kolumner (exempel)
# Om du t.ex. inte vill ha kolumnerna 'url' och 'Circuit'
df_races_clean = df_races.drop(columns=["url", "Circuit"], errors="ignore")
print("Rensad DataFrame:")
print(df_races_clean.head())
