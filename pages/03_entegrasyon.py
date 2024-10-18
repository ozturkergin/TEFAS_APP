import requests
import pandas as pd
import math
import concurrent.futures
import time
import streamlit as st
import pandas_ta as ta
import numpy as np

from pandas_ta.utils import get_offset, verify_series, signals
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union
from marshmallow import Schema, fields, EXCLUDE, pre_load, post_load
from bs4 import BeautifulSoup

# Special thanks to https://github.com/burakyilmaz321

class InfoSchema(Schema):
    code = fields.String(data_key="FONKODU", allow_none=True)
    fonunvantip = fields.String(data_key="FONUNVANTIP", allow_none=True)
    date = fields.Date(data_key="TARIH", allow_none=True)
    price = fields.Float(data_key="FIYAT", allow_none=True)
    title = fields.String(data_key="FONUNVAN", allow_none=True)
    market_cap = fields.Float(data_key="PORTFOYBUYUKLUK", allow_none=True)
    number_of_shares = fields.Float(data_key="TEDPAYSAYISI", allow_none=True)
    number_of_investors = fields.Float(data_key="KISISAYISI", allow_none=True)

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
    def get_combobox_items(url, select_id):
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

    def fetch_info(self, fonunvantip, start_date_initial, end_date_initial):
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

            data = {
                    "fontip": self.fon_type,
                    "bastarih": self._parse_date(start_date),
                    "bittarih": self._parse_date(end_date),
                    "fonunvantip": fonunvantip,
                    "fonkod": "",
                  }

            info = self._do_post(data)
            info = info_schema.load(info)
            info = pd.DataFrame(info, columns=info_schema.fields.keys())
            info['fonunvantip'] = ""

            if fonunvantip != "" :
                info['fonunvantip'] = "symbol_" + fonunvantip

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

    def fetch_info_serial(self, fonunvantips, start_date_initial, end_date_initial):
        merged = pd.DataFrame()

        for fonunvantip in fonunvantips:
            info = self.fetch_info(fonunvantip, start_date_initial, end_date_initial)
            if not info.empty :
                merged = pd.concat([merged, info])
                print(f"{fonunvantip} - {len(info)} records added total records: {len(merged)} " )
        print(f"Data extracted")

        return merged

    def fetch_info_concurrently(self, fonunvantips, start_date_initial, end_date_initial):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            merged = pd.DataFrame()
            self.concurrently = True
            futures = {executor.submit(self.fetch_info, fonunvantip, start_date_initial, end_date_initial): fonunvantip for fonunvantip in fonunvantips}

            for future in concurrent.futures.as_completed(futures):
                info = future.result()
                merged = pd.concat([merged, info])
                print(f"{future} - {len(info)} records added total records: {len(merged)} " )

            return merged

    def fetch(
        self,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        columns: Optional[List[str]] = None,
        unvantip: bool = False,
    ):

        start_date_initial = datetime.strptime(start, "%Y-%m-%d")
        end_date_initial = datetime.strptime(end or start, "%Y-%m-%d")

        merged = pd.DataFrame()
        fonunvantips = [""]
        if unvantip :
            fonunvantips = self.get_combobox_items(url="https://www.tefas.gov.tr/TarihselVeriler.aspx", select_id="DropDownListFundTypeExplanationYAT")

        self.proxies = None

        if self.concurrently :
            merged = self.fetch_info_concurrently(fonunvantips, start_date_initial, end_date_initial)
        else :
            merged = self.fetch_info_serial(fonunvantips, start_date_initial, end_date_initial)

        merged = merged[columns] if columns and not merged.empty else merged

        return merged

    # def get_free_proxy(self):
    #     proxy_address = FreeProxy(timeout=1, rand=True, https=True).get()
    #     return proxy_address

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

tefas = tefas_get()

# Title of the app
st.title("TEFAS Entegrasyon")

# Integer input prompt
prompt_number_of_days = st.number_input("Geçmiş Kaç Gün Çekilmeli:", min_value=0, step=1)

# Button to submit the input
if st.button("Start"):
    st.warning("TEFAS Fiyatlar Çekiliyor...")
    time_delta = prompt_number_of_days
    start_date_calc = date.today() - timedelta(days=time_delta)
    date_start = start_date_calc.strftime("%Y-%m-%d")
    date_end = date.today().strftime("%Y-%m-%d")

    fetched_data = tefas.fetch(start=date_start, end=date_end, columns=["code", "date", "price", "market_cap", "number_of_shares", "number_of_investors"], unvantip=False)
    fetched_data['date'] = pd.to_datetime(fetched_data['date'], errors='coerce')
    fetched_data['price'].astype(float,False)
    fetched_data.rename(columns={'price': 'close'}, inplace=True)
    fetched_data.rename(columns={'code': 'symbol'}, inplace=True)
    fetched_data['market_cap'].astype(float,False)
    fetched_data['number_of_shares'].astype(float,False)
    fetched_data['number_of_investors'].astype(float,False)
    fetched_data['market_cap_per_investors'] = fetched_data['market_cap'] / fetched_data['number_of_investors']
    # fetched_data = fetched_data[(fetched_data!=0)&(pd.isnull(fetched_data))]
    fetched_data = fetched_data.sort_values(['symbol', 'date'])
    fetched_data['open'] = fetched_data.groupby('symbol')['close'].shift(1)
    fetched_data['high'] = fetched_data[['open', 'close']].max(axis=1)
    fetched_data['low'] = fetched_data[['open', 'close']].min(axis=1)
    fetched_data['date'] = fetched_data['date'].dt.strftime('%Y-%m-%d')
    fetched_data.dropna()
    
    st.dataframe(fetched_data.head())
    existing_data = pd.DataFrame()

    try:
        existing_data = pd.read_csv('data/tefas.csv', encoding='utf-8-sig')
        existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
        existing_data['date'] = existing_data['date'].dt.strftime('%Y-%m-%d')
    except FileNotFoundError:
        fetched_data.to_csv('data/tefas.csv', encoding='utf-8-sig', index=False)

    # If the file exists, merge new data with existing data
    if not existing_data.empty:
        merged_data = pd.concat([existing_data, fetched_data]).drop_duplicates(subset=['symbol', 'date'], keep='last')
        merged_data.to_csv('data/tefas.csv', encoding='utf-8-sig', index=False)

    st.success("TEFAS Fiyatlar Çekildi")
    st.warning("TEFAS Verisi zenginleştiriliyor...")

    def calculate_ta(group):

        group_indexed = group.copy()
        group_indexed.set_index('date', inplace=True)

        # Simple Moving Average (SMA)
        SMA = ta.sma(group_indexed['close'], length=5)
        group_indexed = pd.concat([group_indexed, SMA], axis=1)

        # Relative Strength Index (RSI)
        RSI = ta.rsi(group_indexed['close'], length=14)
        group_indexed = pd.concat([group_indexed, RSI], axis=1)
        
        #Bollinger Bands (BBands)
        BB = ta.bbands(group_indexed['close'], length=20, std=2, ddof=0, mamode=None, talib=None, offset=None)
        group_indexed = pd.concat([group_indexed, BB], axis=1)

        # group_indexed['close_007d_ago'] = group_indexed['close'].shift(7, freq='D').interpolate(method='linear')
        # group_indexed['close_030d_ago'] = group_indexed['close'].shift(30, freq='D').interpolate(method='linear')
        # group_indexed['close_060d_ago'] = group_indexed['close'].shift(60, freq='D').interpolate(method='linear')
        # group_indexed['close_090d_ago'] = group_indexed['close'].shift(90, freq='D').interpolate(method='linear')
        # group_indexed['close_180d_ago'] = group_indexed['close'].shift(180, freq='D').interpolate(method='linear')
        # group_indexed['close_365d_ago'] = group_indexed['close'].shift(365, freq='D').interpolate(method='linear')
        group_indexed.reset_index(inplace=True)

        return group_indexed

    fetched_data=pd.read_csv('data/tefas.csv')
    fetched_data['close'].astype(float,False)
    fetched_data['date'] = pd.to_datetime(fetched_data['date'], errors='coerce')
    fetched_data['year'] = fetched_data['date'].dt.year
    fetched_data['week_no'] = fetched_data['date'].dt.isocalendar().week.astype(str).str.zfill(2)
    fetched_data['year_week'] = fetched_data['year'].astype(str) +'-'+ fetched_data['week_no'].astype(str)
    fetched_data['day_of_week'] = fetched_data['date'].dt.strftime('%A')
    #idx = fetched_data.groupby(['symbol', 'year_week'])['date'].idxmax()
    #max_prices = fetched_data.loc[idx, ['symbol', 'year_week', 'close']]
    #max_prices = max_prices.rename(columns={'close': 'price_at_week_close'})
    #fetched_data = fetched_data.merge(max_prices, on=['symbol', 'year_week'], how='left')
    #fetched_data['qty'] = 100/fetched_data['close']
    #fetched_data['valuation_at_week_close'] = fetched_data['price_at_week_close'] * fetched_data['qty']
    fetched_data.sort_values(by=['symbol', 'date'], inplace=True)
    fetched_data = fetched_data.groupby(['symbol']).apply(calculate_ta)
    fetched_data['date'] = fetched_data['date'].dt.strftime('%Y-%m-%d')
    fetched_data.to_csv('data/tefas_transformed.csv', encoding='utf-8-sig', index=False)

    st.dataframe(fetched_data.head())
    st.success("TEFAS Verisi zenginleştirildi")
    st.warning("TEFAS Fonlarının nitelikleri çekiliyor...")

    time_delta = 15
    start_date_calc = date.today() - timedelta(days=time_delta)
    date_start = start_date_calc.strftime("%Y-%m-%d")
    date_end = date.today().strftime("%Y-%m-%d")
    fetched_data_agg = tefas.fetch(start=date_start, end=date_end, columns=["code", "date", "price", "fonunvantip", "title"], unvantip=True)
    fetched_data_agg.drop_duplicates(subset=['code', 'fonunvantip'], ignore_index=True, inplace=True)
    fon_table = fetched_data_agg.pivot_table(index=['title', 'code'], columns='fonunvantip', aggfunc='size', fill_value=0)
    fon_table.reset_index(inplace=True)
    fon_table = fon_table.replace(0, False)
    fon_table = fon_table.replace(1, True)
    fon_table.rename(columns={'code': 'symbol'}, inplace=True)
    fon_table['symbolwithtitle'] = fon_table['symbol'].astype(str) +' - '+ fon_table['title'].astype(str)
    fon_table.to_csv('data/fon_table.csv', encoding='utf-8-sig', index=False)

    st.session_state.df_fon_table = fon_table

    st.dataframe(fon_table.head())
    st.success("TEFAS Fonlarının nitelikleri çekildi")
    st.success("İşlem başarıyla tamamlandı")