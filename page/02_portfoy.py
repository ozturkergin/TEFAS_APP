import streamlit as st
import pandas as pd
import os
import concurrent.futures
from datetime import datetime

# Load fon_table.csv if it exists, otherwise warn the user
if os.path.exists('data/fon_table.csv'):
    if 'df_fon_table' in st.session_state:
        df_fon_table = st.session_state.df_fon_table
    else:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table
else:
    df_fon_table = pd.DataFrame()
    st.warning("Entegrasyon çalıştırınız")

# Stop if fon_table is empty
if df_fon_table.empty:
    st.stop()

unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())

# Load tefas_transformed.csv if it exists
if os.path.exists('data/tefas_transformed.csv'):
    if 'df_transformed' in st.session_state:
        df_transformed = st.session_state.df_transformed
    else:
        df_transformed = pd.read_csv('data/tefas_transformed.csv')
        df_transformed['date'] = pd.to_datetime(df_transformed['date'], errors='coerce')
        st.session_state.df_transformed = df_transformed

# Load portfolio data or create an empty DataFrame
def load_portfolio():
    if os.path.exists("data/myportfolio.csv"):
        if 'myportfolio' in st.session_state:
            myportfolio = st.session_state.myportfolio
        else:
            myportfolio = pd.read_csv('data/myportfolio.csv')
            myportfolio['quantity'] = pd.to_numeric(myportfolio['quantity'], errors='coerce').fillna(0).astype(int)
            myportfolio['date'] = pd.to_datetime(myportfolio['date'], errors='coerce')
            st.session_state.myportfolio = myportfolio

        # Merge portfolio with transformed data
        merged_df = pd.merge(myportfolio, df_transformed[['symbol', 'date', 'close']], on=['symbol', 'date'], how='left')
        merged_df.rename(columns={'close': 'price'}, inplace=True)
        return merged_df
    else:
        st.warning("İşlem girdikten sonra portföyünüz oluşacaktır")
        return pd.DataFrame(columns=['symbol', 'date', 'transaction_type', 'quantity', 'price'])

# Load the portfolio data
df_portfolio = load_portfolio()

# Stop if the portfolio is empty
if df_portfolio.empty:
    st.stop()

col3, col2 = st.columns([100, 1])

# Create a summary dataframe
df_summary = pd.DataFrame(columns=['Fon', 'Unvan', 'Miktar', 'Maliyet', 'Gider', 'Fiyat', 'Tutar', 'Δ', 'Başarı Δ', 'Volatilite', 'Sharpe Oranı'])
df_portfolio['date'] = pd.to_datetime(df_portfolio['date'], errors='coerce')
df_portfolio = df_portfolio[df_portfolio['symbol'] != ""].sort_values(by=['symbol', 'date'])

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(daily_returns):
    return daily_returns.mean() / daily_returns.std() * (252 ** 0.5)

# Function to process each symbol
def process_symbol(symbol):
    symbol_data = df_portfolio[df_portfolio['symbol'] == symbol].sort_values('date')
    recent_data = df_transformed[df_transformed['symbol'] == symbol].sort_values('date')

    if recent_data.empty:
        return None

    most_recent_price = recent_data.iloc[-1]['close']
    most_recent_date = recent_data.iloc[-1]['date']

    total_quantity = 0
    total_value = 0
    avg_buy_price = 0
    weighted_daily_gain = 0
    total_days = 0
    quantity_remained = 0

    for i, (idx, row) in enumerate(symbol_data.iterrows()):
        transaction_type = row['transaction_type']
        transaction_date = row['date']
        quantity = row['quantity']
        unit_price = df_transformed.loc[(df_transformed['symbol'] == symbol) & (df_transformed['date'] == transaction_date), 'close']
        symbol_title = df_fon_table.loc[df_fon_table['symbol'] == symbol, 'title']

        if not unit_price.empty:
            unit_price = unit_price.iloc[0]
        else:
            unit_price = 0

        quantity_remained = 0

        if transaction_type == 'buy':
            total_value += quantity * unit_price
            total_quantity += quantity
            avg_buy_price = total_value / total_quantity
            quantity_remained += quantity

            # Check for subsequent sell transactions for the current buy
            for j in range(i + 1, len(symbol_data)):
                next_row = symbol_data.iloc[j]
                if next_row['transaction_type'] == 'sell':
                    sell_quantity = next_row['quantity']
                    sell_date = next_row['date']
                    sell_price = next_row['price']
                    days_held = (sell_date - transaction_date).days
                    if sell_quantity <= quantity_remained:
                        weighted_daily_gain += ((sell_price - unit_price) / unit_price * 100) / days_held * 365 * sell_quantity
                        quantity_remained -= sell_quantity
                        total_days += sell_quantity
                    else:
                        weighted_daily_gain += ((sell_price - unit_price) / unit_price * 100) / days_held * 365 * quantity
                        quantity_remained -= sell_quantity
                        total_days += quantity
                        break

            # Remaining quantity at most recent price
            if quantity_remained > 0:
                days_held = (most_recent_date - transaction_date).days
                weighted_daily_gain += ((most_recent_price - unit_price) / unit_price * 100) / days_held * 365 * quantity_remained
                total_days += quantity_remained

        elif transaction_type == 'sell':
            total_value -= quantity * avg_buy_price
            total_quantity -= quantity
            if total_quantity == 0:
                avg_buy_price = 0
            else:
                avg_buy_price = total_value / total_quantity

    if total_quantity > 0:
        percentage_change = ((most_recent_price - avg_buy_price) / avg_buy_price) * 100
        yearly_gain = weighted_daily_gain / total_days
        daily_returns = recent_data['close'].pct_change()
        volatility = daily_returns.std() * (252 ** 0.5)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)

        return {
            'Fon': symbol,
            'Unvan': symbol_title.iloc[0] if not symbol_title.empty else "",
            'Miktar': total_quantity,
            'Maliyet': avg_buy_price,
            'Gider': round(total_value, 2),
            'Fiyat': most_recent_price,
            'Tutar': round(total_quantity * most_recent_price, 2),
            'Δ': percentage_change,
            'Başarı Δ': round(yearly_gain, 2),
            'Volatilite': volatility,
            'Sharpe Oranı': sharpe_ratio
        }

# Execute the process for each unique symbol in parallel
summary_rows = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in df_portfolio['symbol'].unique()}
    for future in concurrent.futures.as_completed(future_to_symbol):
        result = future.result()
        if result:
            summary_rows.append(result)

# Convert summary rows to DataFrame and display
if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    st.subheader("Portföy Analiz")
    dataframe_height = (len(df_summary) + 1) * 35 + 2

    column_configuration = {
        "Fon": st.column_config.TextColumn("Fon", help="Fon Kodu", width="small"),
        "Unvan": st.column_config.TextColumn("Unvan", help="Fonun Ünvanı", width="large"),
        "Miktar": st.column_config.NumberColumn("Miktar", help="Fon Adedi", width="small", format="%d"),
        "Maliyet": st.column_config.NumberColumn("Maliyet", help="İşlemler sonucu birim maliyeti", width="small", format="%.2f"),
        "Gider": st.column_config.NumberColumn("Gider", help="İşlemler sonucu gider", width="small", format="%.2f"),
        "Fiyat": st.column_config.NumberColumn("Fiyat", help="Güncel Fiyat", width="small", format="%.4f"),
        "Tutar": st.column_config.NumberColumn("Tutar", help="Güncel Tutar", width="small", format="%.2f"),
        "Δ": st.column_config.NumberColumn("Δ", help="Güncel fiyat değişim yüzdesi", width="small", format="%.2f"),
        "Başarı Δ": st.column_config.NumberColumn("Başarı Δ", help="Yıllıklandırılmış işlem getiri yüzdesi", width="small", format="%.2f"),
        "Volatilite": st.column_config.NumberColumn("Volatilite", help="Volatilite", width="small", format="%.4f"),
        "Sharpe Oranı": st.column_config.NumberColumn("Sharpe Oranı", help="Sharpe Oranı", width="small", format="%.4f"),
    }

    st.dataframe(df_summary, hide_index=True, height=dataframe_height, use_container_width=True, column_config=column_configuration)
else:
    st.write("No data to display.")
