import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Ange fil-path (använd exempelfilen)
file_path = os.path.join(os.path.dirname(__file__), "news_data_example.csv")
# file_path = r"C:\workspace\vallesportfolio\ML\news_data_example.csv"

# Läs in filen
df = pd.read_csv(file_path)

# Konvertera datum till datetime-format om "date" finns
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Lägg till en dummy-numerisk kolumn om ingen numerisk data hittas
if df.select_dtypes(include=["number"]).empty:
    df["dummy_count"] = range(1, len(df) + 1)  # Skapar en numerisk kolumn

# UI-design
st.title("📊 Dashboard för Nyhetsdata")

# 🔍 Sökfunktion
search_query = st.text_input("🔍 Sök i tabellen")
if search_query:
    df = df[df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]

# 📋 Visa filtrerad tabell
st.subheader("📂 Data-tabell")
st.dataframe(df)

# 📊 Statistik och analys (nu alltid synlig)
st.subheader("📈 Statistik & Visualisering")

# Välj en numerisk kolumn
num_cols = df.select_dtypes(include=["number"]).columns
if len(num_cols) > 0:
    column = st.selectbox("Välj en kolumn för analys", num_cols)

    # Visa statistik
    st.write(df[column].describe())

    # Histogram
    fig = px.histogram(df, x=column, title=f"Histogram över {column}")
    st.plotly_chart(fig)

    # Visa korrelationsmatris om fler än en numerisk kolumn finns
    if len(num_cols) > 1:
        st.subheader("🔗 Korrelationsmatris")
        st.write(df.corr())