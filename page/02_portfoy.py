import streamlit as st
import pandas as pd
import os
import concurrent.futures
import seaborn as sns
import numpy as np

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
df_summary = pd.DataFrame(columns=['Fon', 'Unvan', 'Miktar', 'Maliyet', 'Gider', 'Fiyat', 'Tutar', 'Δ', 'Başarı Δ', 'RSI', 'Volatilite', 'Sharpe Oranı'])
df_portfolio['date'] = pd.to_datetime(df_portfolio['date'], errors='coerce')
df_portfolio = df_portfolio[df_portfolio['symbol'] != ""].sort_values(by=['symbol', 'date'])

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(daily_returns):
    return daily_returns.mean() / daily_returns.std() * (252 ** 0.5)

def color_gradient(val, column_name):
    if pd.isna(val) or np.isinf(val):   # Exclude NaN and inf values
        return ''
    
    ranks = df_summary[column_name].rank(method='min')  # Get the ranks of the values in the specified column
    max_rank = ranks.max()
    current_rank = ranks.loc[df_summary[column_name] == val].values[0] # Get the rank of the current value
    norm_val = (current_rank - 1) / (max_rank - 1)  # Normalize the rank to [0, 1] Subtract 1 to make it 0-indexed
    norm_val = np.clip(norm_val, 0, 1) # Ensure normalization is within [0, 1]
    color = sns.color_palette("RdYlGn", as_cmap=True)(norm_val)
    return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}'

def RSI_gradient(val):
    if pd.isna(val) or np.isinf(val):  # Handle NaN and inf values
        return ''

    if val < 40:  # Values below 40 should be green with a star sign
        norm_val = val / 40  # Normalize in [0, 1] range for green gradient
        color = sns.color_palette("Greens", as_cmap=True)(norm_val)
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}; color: white; font-weight: bold;'
    
    elif val > 70:  # Values above 70 should be red
        norm_val = (val - 40) / 30  # Normalize in [0, 1] range for gradient
        color = sns.color_palette("RdYlGn", as_cmap=True)(1 - norm_val)  # Green to Red gradient
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}; color: white; font-weight: bold;'
    
    else:  # Values between 40 and 70 should transition from green to red
        norm_val = (val - 40) / 30  # Normalize in [0, 1] range for gradient
        color = sns.color_palette("RdYlGn", as_cmap=True)(1 - norm_val)  # Green to Red gradient
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}; color: black; font-weight: normal;'

# Function to process each symbol
def process_symbol(symbol):
    symbol_data = df_portfolio[df_portfolio['symbol'] == symbol].sort_values('date')
    recent_data = df_transformed[df_transformed['symbol'] == symbol].sort_values('date')

    if recent_data.empty:
        return None

    most_recent_price = recent_data.iloc[-1]['close']
    most_recent_date = recent_data.iloc[-1]['date']
    most_recent_rsi = recent_data.iloc[-1]['RSI_14']

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

            for j in range(i + 1, len(symbol_data)): # Check for subsequent sell transactions for the current buy
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

            if quantity_remained > 0: # Remaining quantity at most recent price
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
            'RSI': round(most_recent_rsi, 2),
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
    df_summary = pd.DataFrame(summary_rows).sort_values(by="Tutar", ascending=False)

    st.subheader("Portföy Analiz")
    dataframe_height = (len(df_summary) + 1) * 35 + 2

    column_configuration = {
        "Fon": st.column_config.LinkColumn("Fon", help="Fon Kodu", width="small"),
        "Unvan": st.column_config.TextColumn("Unvan", help="Fonun Ünvanı", width="large"),
        "Miktar": st.column_config.NumberColumn("Miktar", help="Fon Adedi", width="small"),
        "Maliyet": st.column_config.NumberColumn("Maliyet", help="İşlemler sonucu birim maliyeti", width="small"),
        "Gider": st.column_config.NumberColumn("Gider", help="İşlemler sonucu gider", width="small"),
        "Fiyat": st.column_config.NumberColumn("Fiyat", help="Güncel Fiyat", width="small"),
        "Tutar": st.column_config.NumberColumn("Tutar", help="Güncel Tutar", width="small"),
        "Δ": st.column_config.NumberColumn("Δ", help="Güncel fiyat değişim yüzdesi", width="small"),
        "Başarı Δ": st.column_config.NumberColumn("Başarı Δ", help="Yıllıklandırılmış işlem getiri yüzdesi", width="small"),
        "RSI": st.column_config.NumberColumn("RSI", help="RSI 14", width="small"),
        "Volatilite": st.column_config.NumberColumn("Volatilite", help="Volatilite", width="small"),
        "Sharpe Oranı": st.column_config.NumberColumn("Sharpe Oranı", help="Sharpe Oranı", width="small"),
    }

    styled_df = df_summary.style
    styled_df = styled_df.format({f'Gider': '₺ {:,.2f}', 
                                  f'Miktar': '{:,.0f}', 
                                  f'Maliyet': '₺ {:.4f}', 
                                  f'Fiyat': '₺ {:.4f}', 
                                  f'Tutar': '₺ {:,.2f}', 
                                  f'Volatilite': '{:.2f}', 
                                  f'Sharpe Oranı': '{:.2f}', 
                                  f'Δ': '% {:,.2f}', 
                                  f'Başarı Δ': '% {:,.2f}' , 
                                  'RSI': '{:.2f}' })
  
    styled_df = styled_df.map(lambda val: f'<span><a target="_blank" href="https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={val}">{val}</a></span>' , subset=['Fon'])
    styled_df = styled_df.map(lambda val: color_gradient(val, f'Δ') if pd.notnull(val) else '', subset=[f'Δ'])
    styled_df = styled_df.map(lambda val: color_gradient(val, f'Başarı Δ') if pd.notnull(val) else '', subset=[f'Başarı Δ'])
    styled_df = styled_df.map(lambda val: RSI_gradient(val) if pd.notnull(val) else '', subset=['RSI'])

    # st.write(styled_df.to_html(), unsafe_allow_html=True)
    st.dataframe(styled_df, hide_index=True, height=dataframe_height, use_container_width=True, column_config=column_configuration)
else:
    st.write("No data to display.")
