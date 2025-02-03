import streamlit as st
import pandas as pd
import plotly.express as px

# 🔹 Ange din fil-path (exempelfilen som skapades tidigare)
file_path = r"C:\workspace\vallesportfolio\ML\news_data_example.csv"

# 🔹 Läs in CSV-filen
df = pd.read_csv(file_path)

# 🔹 Konvertera datum till datetime-format om det finns en "date"-kolumn
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# 📌 Streamlit UI
st.title("📊 Dashboard för Nyhetsdata")

# 🔍 Sökfunktion
search_query = st.text_input("🔍 Sök i tabellen")
if search_query:
    df = df[df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]

# 📋 Visa filtrerad tabell
st.subheader("📂 Data-tabell")
st.dataframe(df)

# 📊 Statistik och analys om numeriska kolumner finns
if not df.select_dtypes(include=["number"]).empty:
    st.subheader("📈 Statistik & Visualisering")

    # Välj en numerisk kolumn att analysera
    column = st.selectbox("Välj en kolumn för analys", df.select_dtypes(include=["number"]).columns)

    # Visa grundläggande statistik
    st.write(df[column].describe())

    # Histogram över vald kolumn
    fig = px.histogram(df, x=column, title=f"Histogram över {column}")
    st.plotly_chart(fig)

    # Visa korrelationsmatris om fler än en numerisk kolumn finns
    if len(df.select_dtypes(include=["number"]).columns) > 1:
        st.subheader("🔗 Korrelationsmatris")
        st.write(df.corr())

