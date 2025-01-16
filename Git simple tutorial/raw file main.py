import pandas as pd

# Create a DataFrame with cities and population and if they are capitals.
data = {
    "City": ["Moscow", "London", "Berlin", "Hamburg", "Rome", "Paris", "Munich", "Barcelona", "Wienna", "Budapest"],
    "Population": [12678079, 8961989, 3669491, 1847253, 2785018, 2220445, 1471508, 1620343, 1867582, 1759407],
    "IsCapital": [True, True, True, False, True, True, False, False, True, True]
}

df = pd.DataFrame(data)

print(df)