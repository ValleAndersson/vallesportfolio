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

# Anpassad CSS för att styra layout och färger
st.markdown(
    """
    <style>
    .banner {
        width: 100%;
        background-color: #444;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        color: white;
        font-weight: bold;
    }
    .main-container {
        display: flex;
        flex-direction: column;
    }
    .sidebar {
        width: 25%;
        background-color: #333333;
        padding: 20px;
        color: white;
        border-right: 3px solid #555;
        height: 100vh;
        position: fixed;
    }
    .content {
        margin-left: 25%;
        width: 75%;
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stTextInput, .stSelectbox, .stDateInput {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Banner överst
st.markdown('<div class="banner">ML av Grupp 3</div>', unsafe_allow_html=True)

# Skapa huvudlayouten
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# SIDOFÄLT - Filtreringssektion
st.markdown('<div class="sidebar">', unsafe_allow_html=True)
st.sidebar.header("🔍 Filter")
category = st.sidebar.selectbox("Välj kategori", ["Alla", "Politik", "Ekonomi", "Sport"])
date_range = st.sidebar.date_input("Välj datumintervall", [])
search_query = st.sidebar.text_input("Sök efter nyckelord")
st.markdown('</div>', unsafe_allow_html=True)

# HUVUDSEKTION - 75% med två spalter
st.markdown('<div class="content">', unsafe_allow_html=True)
st.title("📊 Dashboard med Anpassad Layout")

# Visa data om fil finns
if not df.empty:
    st.subheader("📄 Dataförhandsvisning")
    st.dataframe(df.head())
else:
    st.warning("Ingen data hittades. Ladda upp en fil.")

# Skapa två spalter
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

st.markdown('</div>', unsafe_allow_html=True)

