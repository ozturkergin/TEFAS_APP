import requests
import pandas as pd
import math
import time
import streamlit as st
import pandas_ta as ta

from pandas_ta.utils import get_offset, verify_series, signals
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union
from marshmallow import Schema, fields, EXCLUDE, pre_load, post_load
from bs4 import BeautifulSoup

# Special thanks to https://github.com/burakyilmaz321

class InfoSchema(Schema):
    code = fields.String(data_key="FONKODU", allow_none=True)
    date = fields.Date(data_key="TARIH", allow_none=True)
    price = fields.Float(data_key="FIYAT", allow_none=True)
    title = fields.String(data_key="FONUNVAN", allow_none=True)
    market_cap = fields.Float(data_key="PORTFOYBUYUKLUK", allow_none=True)
    number_of_shares = fields.Float(data_key="TEDPAYSAYISI", allow_none=True)
    number_of_investors = fields.Float(data_key="KISISAYISI", allow_none=True)
    FundType = fields.String(data_key="FONUNVANTIP", allow_none=True)  # Fund Type Derived
    UmbrellaFundType = fields.String(data_key="FONUNVANTUR", allow_none=True)  # Umbrella Fund Type Derived
 
    @pre_load
    def pre_load_hook(self, input_data, **kwargs):
        seconds_timestamp = int(input_data["TARIH"]) / 1000
        input_data["TARIH"] = date.fromtimestamp(seconds_timestamp).isoformat()
        return input_data

    @post_load
    def post_load_hool(self, output_data, **kwargs):
        output_data = {f: output_data.setdefault(f) for f in self.fields}
        return output_data

    class Meta:
        unknown = EXCLUDE

class tefas_get:
    root_url = "https://www.tefas.gov.tr"
    info_endpoint = "/api/DB/BindHistoryInfo"
    concurrently = False
    use_Proxy = False
    fon_type = "YAT"
    proxies = None

    @staticmethod
    def get_FundType_combobox_items(url, select_id):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the URL: {response.status_code}")

        soup = BeautifulSoup(response.content, 'html.parser')
        select_element = soup.find('select', id=select_id)

        if not select_element:
            raise Exception(f"Select element with id '{select_id}' not found")

        options = select_element.find_all('option')
        options = list(filter(None, options))

        items = []
        for option in options:
            value = option.get('value')
            items.append(value)

        items.remove('')

        return items
    
    @staticmethod
    def get_UmbrellaFundType_combobox_items(url, select_id):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the URL: {response.status_code}")

        soup = BeautifulSoup(response.content, 'html.parser')
        select_element = soup.find('select', id=select_id)

        if not select_element:
            raise Exception(f"Select element with id '{select_id}' not found")

        options = select_element.find_all('option')
        options = list(filter(None, options))
        
        items = []
        for option in options:
            value = option.get('value')
            text = option.text.strip()
            items.append((value, text))

        items = [item for item in items if item[0] != 'Tümü']

        return items

    def fetch_info(self, FundType, UmbrellaFundType, start_date_initial, end_date_initial):
        counter = 1
        start_date = start_date_initial
        end_date = end_date_initial
        range_date = end_date_initial - start_date_initial
        range_interval = 90
        info_schema = InfoSchema(many=True)
        info_result = pd.DataFrame()

        if range_date.days > range_interval :
            counter = range_date.days / range_interval
            counter = math.ceil(counter)
            end_date = start_date + timedelta(days=range_interval)

        while counter > 0:
            counter -= 1
            lv_post_FundType = ""
            lv_post_UmbrellaFundType = ""

            if FundType != "" :
                lv_post_FundType = FundType
            if UmbrellaFundType != "" :
                lv_post_UmbrellaFundType = UmbrellaFundType[0]

            data = {
                    "fontip": self.fon_type,
                    "bastarih": self._parse_date(start_date),
                    "bittarih": self._parse_date(end_date),
                    "fonunvantip": lv_post_FundType,
                    "sfontur": lv_post_UmbrellaFundType,
                    "fonkod": "",
                  }

            info = self._do_post(data)
            info = info_schema.load(info)
            info = pd.DataFrame(info, columns=info_schema.fields.keys())
            info['FundType'] = ""
            info['UmbrellaFundType'] = ""

            if FundType != "" :
                info['FundType'] = "FundType_" + FundType
            if UmbrellaFundType != "" :
                info['UmbrellaFundType'] = "UmbrellaFundType_" + UmbrellaFundType[1]

            if not info.empty :
                info_result = pd.concat([info_result, info])
                info_result = info_result.reset_index(drop=True)
                info = info.reset_index(drop=True)

            if counter > 0 :
                start_date = end_date + timedelta(days=1)
                end_date = end_date + timedelta(days=range_interval)
                if end_date > end_date_initial :
                    end_date = end_date_initial

        return info_result

    def fetch_info_serial(self, FundTypes, UmbrellaFundTypes, start_date_initial, end_date_initial):
        merged = pd.DataFrame()
        if FundTypes != [""] :
            for FundType in FundTypes:
                time.sleep(2)
                info = self.fetch_info(FundType, "", start_date_initial, end_date_initial)
                if not info.empty :
                    merged = pd.concat([merged, info])
                    print(f"{FundType} - {len(info)} records added total records: {len(merged)} " )
        elif UmbrellaFundTypes != [""] :
            for UmbrellaFundType in UmbrellaFundTypes:
                time.sleep(4)
                info = self.fetch_info("", UmbrellaFundType, start_date_initial, end_date_initial)
                if not info.empty :
                    merged = pd.concat([merged, info])
                    print(f"{UmbrellaFundType} - {len(info)} records added total records: {len(merged)} " )
        else :
            info = self.fetch_info("", "", start_date_initial, end_date_initial)
            if not info.empty :
                merged = pd.concat([merged, info])
                print(f" - {len(info)} records added total records: {len(merged)} " )

        print(f"Data extracted")
        return merged

    def fetch(
        self,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        columns: Optional[List[str]] = None,
        FundType: bool = False,
        UmbrellaFundType: bool = False,
    ):

        start_date_initial = datetime.strptime(start, "%Y-%m-%d")
        end_date_initial = datetime.strptime(end or start, "%Y-%m-%d")

        merged = pd.DataFrame()
        FundTypes = [""]
        UmbrellaFundTypes = [""]
        if FundType :
            FundTypes = self.get_FundType_combobox_items(url="https://www.tefas.gov.tr/TarihselVeriler.aspx", select_id="DropDownListFundTypeExplanationYAT")
        if UmbrellaFundType :
            UmbrellaFundTypes = self.get_UmbrellaFundType_combobox_items(url="https://www.tefas.gov.tr/TarihselVeriler.aspx", select_id="DropDownListUmbrellaFundTypeYAT")

        self.proxies = None
        merged = self.fetch_info_serial(FundTypes, UmbrellaFundTypes, start_date_initial, end_date_initial)
        merged = merged[columns] if columns and not merged.empty else merged
        return merged

    def _do_post(self, data: Dict[str, str]) -> Dict[str, str]:
        timestamp = int(time.time() * 1000)  # Get current timestamp in milliseconds
        headers = {
         "Connection": "keep-alive",
         "Cache-Control": "no-cache",
         "Pragma": "no-cache",
         "X-Requested-With": "XMLHttpRequest",
         "Sec-Fetch-Mode": "cors",
         "Sec-Fetch-Site": "same-origin",
         "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
         "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
         "Accept": "application/json, text/javascript, */*; q=0.01",
         "Origin": "https://www.tefas.gov.tr",
         "Referer": f"https://www.tefas.gov.tr/TarihselVeriler.aspx?timestamp={timestamp}" ,
         }

        response = requests.post(
             url=f"{self.root_url}/{self.info_endpoint}",
             data=data,
             proxies=self.proxies,
             headers=headers,
         )
        # Check the response status code and content
        if response.status_code != 200:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return {}  # Return an empty dictionary if the request failed
        try:
            return response.json().get("data", {})
        except ValueError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Response content: {response.text}")
            return {}

    def _parse_date(self, date: Union[str, datetime]) -> str:
        if isinstance(date, datetime):
            formatted = datetime.strftime(date, "%d.%m.%Y")
        elif isinstance(date, str):
            try:
                parsed = datetime.strptime(date, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(
                    "Date string format is incorrect. " "It should be `YYYY-MM-DD`"
                ) from exc
            else:
                formatted = datetime.strftime(parsed, "%d.%m.%Y")
        else:
            raise ValueError(
                "`date` should be a string like 'YYYY-MM-DD' "
                "or a `datetime.datetime` object."
            )
        return formatted

def calculate_ta(group):
    group_indexed = group.copy()
    group_indexed.set_index('date', inplace=True)
    
    # Include all necessary columns in group_complete
    group_complete = group[['date', 'close', 'market_cap', 'number_of_shares', 'number_of_investors', 'market_cap_per_investors']].copy()
    
    # Create complete date range and forward-fill missing values
    complete_date_range = pd.date_range(start=group_indexed.index.min(), end=group_indexed.index.max(), freq='D')
    group_complete = group_complete.set_index('date').reindex(complete_date_range).ffill().reset_index()
    group_complete.rename(columns={'index': 'date'}, inplace=True)
    
    # Initialize DataFrames with required columns
    group_complete_7d = pd.DataFrame(columns=['date', 'close_7d', 'market_cap_7d', 'number_of_shares_7d', 'number_of_investors_7d', 'market_cap_per_investors_7d'])
    group_complete_1m = pd.DataFrame(columns=['date', 'close_1m', 'market_cap_1m', 'number_of_shares_1m', 'number_of_investors_1m', 'market_cap_per_investors_1m'])
    group_complete_3m = pd.DataFrame(columns=['date', 'close_3m', 'market_cap_3m', 'number_of_shares_3m', 'number_of_investors_3m', 'market_cap_per_investors_3m'])
    group_complete_6m = pd.DataFrame(columns=['date', 'close_6m', 'market_cap_6m', 'number_of_shares_6m', 'number_of_investors_6m', 'market_cap_per_investors_6m'])
    group_complete_1y = pd.DataFrame(columns=['date', 'close_1y', 'market_cap_1y', 'number_of_shares_1y', 'number_of_investors_1y', 'market_cap_per_investors_1y'])
    group_complete_3y = pd.DataFrame(columns=['date', 'close_3y', 'market_cap_3y', 'number_of_shares_3y', 'number_of_investors_3y', 'market_cap_per_investors_3y'])
    
    # Assign values from group_complete to the corresponding columns in group_complete_7d
    group_complete_7d['date'] = group_complete['date'] + pd.DateOffset(days=7)
    group_complete_7d['close_7d'] = group_complete['close']
    group_complete_7d['market_cap_7d'] = group_complete['market_cap']
    group_complete_7d['number_of_shares_7d'] = group_complete['number_of_shares']
    group_complete_7d['number_of_investors_7d'] = group_complete['number_of_investors']
    group_complete_7d['market_cap_per_investors_7d'] = group_complete['market_cap_per_investors']
    
    # Similarly assign values for other time periods
    group_complete_1m['date'] = group_complete['date'] + pd.DateOffset(days=30)
    group_complete_1m['close_1m'] = group_complete['close']
    group_complete_1m['market_cap_1m'] = group_complete['market_cap']
    group_complete_1m['number_of_shares_1m'] = group_complete['number_of_shares']
    group_complete_1m['number_of_investors_1m'] = group_complete['number_of_investors']
    group_complete_1m['market_cap_per_investors_1m'] = group_complete['market_cap_per_investors']
    
    group_complete_3m['date'] = group_complete['date'] + pd.DateOffset(days=90)
    group_complete_3m['close_3m'] = group_complete['close']
    group_complete_3m['market_cap_3m'] = group_complete['market_cap']
    group_complete_3m['number_of_shares_3m'] = group_complete['number_of_shares']
    group_complete_3m['number_of_investors_3m'] = group_complete['number_of_investors']
    group_complete_3m['market_cap_per_investors_3m'] = group_complete['market_cap_per_investors']
    
    group_complete_6m['date'] = group_complete['date'] + pd.DateOffset(days=180)
    group_complete_6m['close_6m'] = group_complete['close']
    group_complete_6m['market_cap_6m'] = group_complete['market_cap']
    group_complete_6m['number_of_shares_6m'] = group_complete['number_of_shares']
    group_complete_6m['number_of_investors_6m'] = group_complete['number_of_investors']
    group_complete_6m['market_cap_per_investors_6m'] = group_complete['market_cap_per_investors']
    
    group_complete_1y['date'] = group_complete['date'] + pd.DateOffset(days=365)
    group_complete_1y['close_1y'] = group_complete['close']
    group_complete_1y['market_cap_1y'] = group_complete['market_cap']
    group_complete_1y['number_of_shares_1y'] = group_complete['number_of_shares']
    group_complete_1y['number_of_investors_1y'] = group_complete['number_of_investors']
    group_complete_1y['market_cap_per_investors_1y'] = group_complete['market_cap_per_investors']
    
    group_complete_3y['date'] = group_complete['date'] + pd.DateOffset(days=1095)
    group_complete_3y['close_3y'] = group_complete['close']
    group_complete_3y['market_cap_3y'] = group_complete['market_cap']
    group_complete_3y['number_of_shares_3y'] = group_complete['number_of_shares']
    group_complete_3y['number_of_investors_3y'] = group_complete['number_of_investors']
    group_complete_3y['market_cap_per_investors_3y'] = group_complete['market_cap_per_investors']
    
    # Perform left joins with group_indexed on the 'date' column
    group_indexed.reset_index(inplace=True)
    group_indexed = group_indexed.merge(group_complete_7d[['date', 'close_7d', 'market_cap_7d', 'number_of_shares_7d', 'number_of_investors_7d', 'market_cap_per_investors_7d']], on='date', how='left')
    group_indexed = group_indexed.merge(group_complete_1m[['date', 'close_1m', 'market_cap_1m', 'number_of_shares_1m', 'number_of_investors_1m', 'market_cap_per_investors_1m']], on='date', how='left')
    group_indexed = group_indexed.merge(group_complete_3m[['date', 'close_3m', 'market_cap_3m', 'number_of_shares_3m', 'number_of_investors_3m', 'market_cap_per_investors_3m']], on='date', how='left')
    group_indexed = group_indexed.merge(group_complete_6m[['date', 'close_6m', 'market_cap_6m', 'number_of_shares_6m', 'number_of_investors_6m', 'market_cap_per_investors_6m']], on='date', how='left')
    group_indexed = group_indexed.merge(group_complete_1y[['date', 'close_1y', 'market_cap_1y', 'number_of_shares_1y', 'number_of_investors_1y', 'market_cap_per_investors_1y']], on='date', how='left')
    group_indexed = group_indexed.merge(group_complete_3y[['date', 'close_3y', 'market_cap_3y', 'number_of_shares_3y', 'number_of_investors_3y', 'market_cap_per_investors_3y']], on='date', how='left')
    
    # Calculate technical indicators
    group_indexed.set_index('date', inplace=True)
    group_indexed["EMA_5"]   = ta.ema(group_indexed['close'], length=5)  # Exponential Moving Average (EMA)
    group_indexed["EMA_10"]  = ta.ema(group_indexed['close'], length=10) 
    group_indexed["EMA_12"]  = ta.ema(group_indexed['close'], length=12) 
    group_indexed["EMA_20"]  = ta.ema(group_indexed['close'], length=20) 
    group_indexed["EMA_26"]  = ta.ema(group_indexed['close'], length=26) 
    group_indexed["EMA_50"]  = ta.ema(group_indexed['close'], length=50) 
    group_indexed["EMA_100"] = ta.ema(group_indexed['close'], length=100)
    group_indexed["EMA_200"] = ta.ema(group_indexed['close'], length=200)
    group_indexed["SMA_5"]   = ta.sma(group_indexed['close'], length=5)  # Simple Moving Average (SMA)
    group_indexed["RSI_14"]  = ta.rsi(group_indexed['close'], length=14) # Relative Strength Index (RSI) with RMA
    group_indexed["MACD"]    = group_indexed["EMA_12"] - group_indexed["EMA_26"] # Moving Average Convergence Divergence (MACD)

    group_indexed.reset_index(inplace=True)
    return group_indexed

st.title("TEFAS Entegrasyon")

tefas = tefas_get()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    start_date = st.date_input("Başlangıç Tarihi", value=date.today() - timedelta(days=30))
with col2:
    end_date = st.date_input("Bitiş Tarihi", value=date.today())
with col3:
    tefas_price = st.checkbox("Fiyatları Çek", key="price_get", value=True, help="Fiyatları Çek")
    tefas_fundtype = st.checkbox("Fon Türlerini Çek", key="fundtype_get", value=True, help="Fon Türlerini Çek")
    calculate_indicators = st.checkbox("Veri Zenginleştir", key="calculate_indicators", value=True, help="Tüm fiyatlarla İndikatörleri Hesapla")

# Button to submit the input
if st.button("Start"):
    if tefas_price: 
        st.warning("TEFAS Fiyatlar Çekiliyor...")
        date_start = start_date.strftime("%Y-%m-%d")
        date_end = end_date.strftime("%Y-%m-%d")

        fetched_data = tefas.fetch(start=date_start, end=date_end, columns=["code", "date", "price", "market_cap", "number_of_shares", "number_of_investors"], FundType=False, UmbrellaFundType=False)
        fetched_data['date'] = pd.to_datetime(fetched_data['date'], errors='coerce')
        fetched_data['price'].astype(float,False)
        fetched_data.rename(columns={'price': 'close'}, inplace=True)
        fetched_data.rename(columns={'code': 'symbol'}, inplace=True)
        fetched_data['market_cap'].astype(float,False)
        fetched_data['number_of_shares'].astype(float,False)
        fetched_data['number_of_investors'].astype(float,False)
        fetched_data['date'] = fetched_data['date'].dt.strftime('%Y-%m-%d')
        fetched_data.dropna()
        
        try:
            existing_data = pd.DataFrame()
            existing_data = pd.read_csv('data/tefas.csv', encoding='utf-8-sig', parse_dates=['date']) 
            existing_data['date'] = existing_data['date'].dt.strftime('%Y-%m-%d')
        except FileNotFoundError:
            fetched_data.to_csv('data/tefas.csv', encoding='utf-8-sig', index=False)

        # If the file exists, merge new data with existing data
        if not existing_data.empty:
            merged_data = pd.concat([existing_data, fetched_data]).drop_duplicates(subset=['symbol', 'date'], keep='last')
            merged_data.to_csv('data/tefas.csv', encoding='utf-8-sig', index=False)

        st.dataframe(fetched_data.head(), hide_index=True)
        st.success("TEFAS Fiyatlar Çekildi")

    if calculate_indicators: 
        st.warning("TEFAS Verisi zenginleştiriliyor...")
        fetched_data=pd.read_csv('data/tefas.csv', encoding='utf-8-sig', parse_dates=['date'])
        fetched_data['close'].astype(float,False)
        fetched_data['year'] = fetched_data['date'].dt.year
        fetched_data['week_no'] = fetched_data['date'].dt.isocalendar().week.astype(str).str.zfill(2)
        fetched_data['year_week'] = fetched_data['year'].astype(str) +'-'+ fetched_data['week_no'].astype(str)
        fetched_data['day_of_week'] = fetched_data['date'].dt.strftime('%A')
        fetched_data['market_cap_per_investors'] = fetched_data['market_cap'] / fetched_data['number_of_investors']
        fetched_data.sort_values(by=['symbol', 'date'], inplace=True)
        fetched_data['open'] = fetched_data.groupby('symbol')['close'].shift(1)
        fetched_data['high'] = fetched_data[['open', 'close']].max(axis=1)
        fetched_data['low'] = fetched_data[['open', 'close']].min(axis=1)
        fetched_data = fetched_data.groupby(['symbol']).apply(calculate_ta)
        fetched_data['date'] = fetched_data['date'].dt.strftime('%Y-%m-%d')
        fetched_data.to_csv('data/tefas_transformed.csv', encoding='utf-8-sig', index=False)
        st.dataframe(fetched_data.head(15), hide_index=True)
        st.success("TEFAS Verisi zenginleştirildi")

    if tefas_fundtype:
        st.warning("TEFAS Fonlarının nitelikleri çekiliyor...")
        start_date_calc = date.today() - timedelta(days=15)
        date_start = start_date_calc.strftime("%Y-%m-%d")
        date_end = date.today().strftime("%Y-%m-%d")

        fetched_data_fundtype = tefas.fetch(start=date_start, end=date_end, columns=["code", "date", "price", "FundType", "title"], FundType=True, UmbrellaFundType=False)
        fetched_data_fundtype.drop_duplicates(subset=['code', 'FundType'], ignore_index=True, inplace=True)
        fon_table_fundtype = fetched_data_fundtype.pivot_table(index=['title', 'code'], columns='FundType', aggfunc='size', fill_value=0)
        fon_table_fundtype.reset_index(inplace=True)
        fon_table_fundtype = fon_table_fundtype.replace(0, False)
        fon_table_fundtype = fon_table_fundtype.replace(1, True)
        fon_table_fundtype.rename(columns={'code': 'symbol'}, inplace=True)
        fon_table_fundtype['symbolwithtitle'] = fon_table_fundtype['symbol'].astype(str) +' - '+ fon_table_fundtype['title'].astype(str)

        try:
            fetched_data_umbrellafundtype = tefas.fetch(start=date_start, end=date_end, columns=["code", "date", "price", "UmbrellaFundType", "title"], FundType=False, UmbrellaFundType=True)
            fetched_data_umbrellafundtype.drop_duplicates(subset=['code', 'UmbrellaFundType'], ignore_index=True, inplace=True)
            fon_table_umbrellafundtype = fetched_data_umbrellafundtype.pivot_table(index=['code'], columns='UmbrellaFundType', aggfunc='size', fill_value=0)
            fon_table_umbrellafundtype.reset_index(inplace=True)
            fon_table_umbrellafundtype = fon_table_umbrellafundtype.replace(0, False)
            fon_table_umbrellafundtype = fon_table_umbrellafundtype.replace(1, True)
            fon_table_umbrellafundtype.rename(columns={'code': 'symbol'}, inplace=True)
        except Exception as e:
            fon_table_umbrellafundtype = pd.DataFrame()
            st.error(f"Error: {e}")

        fon_table = pd.merge(fon_table_fundtype, fon_table_umbrellafundtype, on='symbol', how='left')
        fon_table.to_csv('data/fon_table.csv', encoding='utf-8-sig', index=False)
        st.session_state.df_fon_table = fon_table
        st.dataframe(fon_table.head(), hide_index=True)
        st.success("TEFAS Fonlarının nitelikleri çekildi")
        
    st.success("İşlem başarıyla tamamlandı")
    st.session_state.clear()
    st.cache_data.clear()