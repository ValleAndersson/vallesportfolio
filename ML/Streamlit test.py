import streamlit as st
import pandas as pd

# Ladda in CSV-filen
uploaded_file = st.file_uploader("Ladda upp en CSV-fil", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)  # Visa datan i tabellform
