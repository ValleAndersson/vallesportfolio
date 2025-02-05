import streamlit as st
import pandas as pd
import os

# Ange fil-path (använd den lösning som fungerade i Streamlit Cloud)
file_path = os.path.join(os.path.dirname(__file__), "data.csv")

# Läs in CSV-filen
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    df = pd.DataFrame()

# Aktivera Streamlits Dark Mode
st.set_page_config(page_title="ML av Grupp 3", layout="wide")

# Banner överst
st.markdown("""
    <h1 style='text-align: center; background-color: #444; color: white; padding: 15px;'>ML av Grupp 3</h1>
    """, unsafe_allow_html=True)

# Skapa två kolumner för layout
sidebar, content = st.columns([1, 3])

# SIDOFÄLT - Filtreringssektion
with sidebar:
    st.header("🔍 Filter")
    category = st.selectbox("Välj kategori", ["Alla", "Politik", "Ekonomi", "Sport"], key="category_filter")
    date_range = st.date_input("Välj datumintervall", [])
    search_query = st.text_input("Sök efter nyckelord", key="search_filter")

# HUVUDSEKTION - 75% med två spalter
with content:
    st.title("📊 Dashboard med Anpassad Layout")

    # Visa data om fil finns
    if not df.empty:
        st.subheader("📄 Dataförhandsvisning")
        st.dataframe(df.head())
    else:
        st.warning("Ingen data hittades. Ladda upp en fil.")

    # Skapa två spalter för analys
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Diagram 1")
        st.write("Plats för graf eller annan analys.")

        st.subheader("📊 Tabell 1")
        st.write("Här kan en tabell visas.")

    with col2:
        st.subheader("📉 Diagram 2")
        st.write("Plats för en andra graf.")

        st.subheader("📋 Tabell 2")
        st.write("Här kan en annan tabell visas.")
