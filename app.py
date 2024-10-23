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

pg = st.navigation(
    {
        "Analiz": [home_page, analiz_page],
        "Entegrasyon": [entegrasyon_page],
        "Portföy": [islemler_page, portfoy_page, fonfavori_page],
    }, 
    position="sidebar"
)

st.set_page_config(layout="wide")

if os.path.exists('data/fon_table.csv') :
    if 'df_fon_table' not in st.session_state :
        st.session_state.df_fon_table = pd.read_csv('data/fon_table.csv')
else: 
    st.page_link(page="pages/03_entegrasyon.py")

if os.path.exists('data/tefas_transformed.csv') :
    if 'df_transformed' not in st.session_state :
        st.session_state.df_transformed = pd.read_csv('data/tefas_transformed.csv')
        st.session_state.df_transformed['date'] = pd.to_datetime(st.session_state.df_transformed['date'])
else: 
    st.page_link(page="pages/03_entegrasyon.py")

if os.path.exists('data/tefas_transformed.csv') and os.path.exists('data/fon_table.csv') :
    df_merged = pd.merge(st.session_state.df_transformed, st.session_state.df_fon_table, on='symbol', how='inner')
    df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')
    if 'df_merged' not in st.session_state :
        st.session_state.df_merged = df_merged

# st.dataframe(st.session_state.df_transformed.head(20))
# st.dataframe(st.session_state.df_merged.head(20))

if os.path.exists('data/myportfolio.csv') :
    if 'myportfolio' not in st.session_state :
        st.session_state.myportfolio = pd.read_csv('data/myportfolio.csv')

pg.run()