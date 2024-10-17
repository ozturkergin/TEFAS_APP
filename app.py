import streamlit as st
import pandas as pd
import os

home_page = st.Page(
    "pages/Home.py",
    title="Home",
    icon=":material/account_circle:",
    default=True,
)
analiz_page = st.Page(
    "pages/01_analiz.py",
    title="TEFAS Analiz",
    icon=":material/bar_chart:",
)
islemler_page = st.Page(
    "pages/02_islemler.py",
    title="İşlem Geçmişi",
    icon=":material/add:",
)
portfoy_page = st.Page(
    "pages/02_portfoy.py",
    title="Portföy Analizi",
    icon=":material/data_thresholding:",
)
entegrasyon_page = st.Page(
    "pages/03_entegrasyon.py",
    title="Veri İndir",
    icon=":material/data_check:",
)
fonfavori_page = st.Page(
    "pages/02_fonfavori.py",
    title="Favorileri Yönet",
    icon=":material/book:",
)

# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Analiz": [home_page, analiz_page],
        "Entegrasyon": [entegrasyon_page],
        "Portföy": [islemler_page, portfoy_page, fonfavori_page],
    }
)

st.set_page_config(layout="wide")

if os.path.exists('data/fon_table.csv') :
    if 'df_fon_table' not in st.session_state :
        st.session_state.df_fon_table = pd.read_csv('data/fon_table.csv')

#st.logo("assets/logo.png")
#st.sidebar.markdown("Ergin Öztürk")

# --- RUN NAVIGATION ---
pg.run()