import streamlit as st
import pandas as pd
import os
from urllib.parse import urlparse

st.set_page_config(layout="wide")

with open("assets/styles.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

home_page = st.Page(
    "page/Home.py",
    title="Anasayfa",
    icon=":material/account_circle:",
    default=True,
)
analiz_page = st.Page(
    "page/01_analiz.py",
    title="TEFAS Analiz",
    icon=":material/move_up:",
)
patterns_page = st.Page(
    "page/05_patterns.py",
    title="Formasyonlar",
    icon=":material/token:",
)
islemler_page = st.Page(
    "page/02_islemler.py",
    title="İşlem Geçmişi",
    icon=":material/add:",
)
portfoy_page = st.Page(
    "page/02_portfoy.py",
    title="Portföy Analizi",
    icon=":material/dataset:",
)
entegrasyon_page = st.Page(
    "page/03_entegrasyon.py",
    title="Veri İndir",
    icon=":material/library_add:",
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
comparison_page = st.Page(
    "page/01_tradingview.py",
    title="Tradingview Lite",
    icon=":material/move_up:",
)
config_page = st.Page(
    "page/07_config.py",
    title="Konfigürasyon",
    icon=":material/settings:",
)

pg = st.navigation(
    {
        "Analiz": [home_page, analiz_page, tahmin_page, patterns_page, comparison_page],
        "Entegrasyon": [entegrasyon_page],
        "Portföy": [islemler_page, portfoy_page],
        "Ayarlar": [config_page, fonfavori_page],
    }, 
    position="sidebar" , 
)

linktointegration = False

if os.path.exists('data/fon_table.csv') :
    if not 'df_fon_table' in st.session_state:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table 
        df_fon_table = None
else: 
    linktointegration = True

if os.path.exists('data/tefas_transformed.csv') :
    if not 'df_transformed' in st.session_state:
        df_transformed = pd.read_csv('data/tefas_transformed.csv')
        df_transformed['date'] = pd.to_datetime(df_transformed['date'], errors='coerce')
        st.session_state.df_transformed = df_transformed
        df_transformed = None 
        # st.write("tefas_transformed app.py içerisinde okundu dosyadan")
else: 
    linktointegration = True

if os.path.exists('data/myportfolio.csv') :
    if not 'myportfolio' in st.session_state:
        myportfolio = pd.read_csv('data/myportfolio.csv', parse_dates=['date'])
        myportfolio['quantity'] = pd.to_numeric(myportfolio['quantity'], errors='coerce').fillna(0).astype(int)
        myportfolio = myportfolio[myportfolio.quantity != 0]
        st.session_state.myportfolio = myportfolio

if os.path.exists('data/favourites.csv'):
    if not 'favourites' in st.session_state:
        st.session_state.favourites = pd.read_csv('data/favourites.csv')['symbol'].tolist() 

if linktointegration:
    st.page_link(page="page/03_entegrasyon.py")

pg.run()