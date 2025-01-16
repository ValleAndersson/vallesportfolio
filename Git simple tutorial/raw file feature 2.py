import pandas as pd

# Re-using the same data from main again.
data = {
    "City": ["Moscow", "London", "Berlin", "Hamburg", "Rome", "Paris", "Munich", "Barcelona", "Wienna", "Budapest"],
    "Population": [12678079, 8961989, 3669491, 1847253, 2785018, 2220445, 1471508, 1620343, 1867582, 1759407],
    "IsCapital": [True, True, True, False, True, True, False, False, True, True],
    "Country": ["Russia", "England", "Germany", "Germany", "Italy", "France", "Germany", "Spain", "Austria", "Hungary"]
}

df = pd.DataFrame(data)

# Adding a filter for large capitals (cities over 3M pop and listed as capitals) as a new feature.
large_capitals = df[(df["IsCapital"] == True) & (df["Population"] > 3000000)]

print(large_capitals)