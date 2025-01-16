import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")

with open("assets/styles.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

home_page = st.Page(
    "page/Home.py",
    title="Home",
    icon=":material/account_circle:",
    default=True,
)
analiz_page = st.Page(
    "page/01_analiz.py",
    title="TEFAS Analiz",
    icon=":material/bar_chart:",
)
islemler_page = st.Page(
    "page/02_islemler.py",
    title="İşlem Geçmişi",
    icon=":material/add:",
)
portfoy_page = st.Page(
    "page/02_portfoy.py",
    title="Portföy Analizi",
    icon=":material/token:",
)
entegrasyon_page = st.Page(
    "page/03_entegrasyon.py",
    title="Veri İndir",
    icon=":material/data_check:",
)
fonfavori_page = st.Page(
    "page/02_fonfavori.py",
    title="Favorileri Yönet",
    icon=":material/book:",
)
tahmin_page = st.Page(
    "page/04_tahmin.py",
    title="Tahmin",
    icon=":material/data_thresholding:",
)

pg = st.navigation(
    {
        "Analiz": [home_page, analiz_page, tahmin_page],
        "Entegrasyon": [entegrasyon_page],
        "Portföy": [islemler_page, portfoy_page, fonfavori_page],
    }, 
    position="sidebar" , 
)

if os.path.exists('data/fon_table.csv') :
    if not 'df_fon_table' in st.session_state:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table 
        df_fon_table = None
else: 
    st.page_link(page="page/03_entegrasyon.py")

if os.path.exists('data/tefas_transformed.csv') :
    if not 'df_transformed' in st.session_state:
        df_transformed = pd.read_csv('data/tefas_transformed.csv')
        df_transformed['date'] = pd.to_datetime(df_transformed['date'], errors='coerce')
        st.session_state.df_transformed = df_transformed
        df_transformed = None 
else: 
    st.page_link(page="page/03_entegrasyon.py")

if os.path.exists('data/myportfolio.csv') :
    if not 'myportfolio' in st.session_state:
        myportfolio = pd.read_csv('data/myportfolio.csv', parse_dates=['date'])
        myportfolio['quantity'] = pd.to_numeric(myportfolio['quantity'], errors='coerce').fillna(0).astype(int)
        myportfolio = myportfolio[myportfolio.quantity != 0]
        st.session_state.myportfolio = myportfolio

if os.path.exists('data/favourites.csv'):
    if not 'favourites' in st.session_state:
        st.session_state.favourites = pd.read_csv('data/favourites.csv')['symbol'].tolist() 

pg.run()