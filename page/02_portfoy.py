import streamlit as st
import pandas as pd
import os
import concurrent.futures
import seaborn as sns
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

# Load tefas_transformed.csv if it exists
if os.path.exists('data/tefas_transformed.csv'):
    if 'df_transformed' in st.session_state:
        df_transformed = st.session_state.df_transformed
    else:
        df_transformed = pd.read_csv('data/tefas_transformed.csv', parse_dates=['date'])
        st.session_state.df_transformed = df_transformed

# Load portfolio data or create an empty DataFrame
def load_portfolio():
    if os.path.exists("data/myportfolio.csv"):
        if 'myportfolio' in st.session_state:
            myportfolio = st.session_state.myportfolio
        else:
            myportfolio = pd.read_csv('data/myportfolio.csv', parse_dates=['date'])
            myportfolio['quantity'] = pd.to_numeric(myportfolio['quantity'], errors='coerce').fillna(0).astype(int)
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

# Create a summary dataframe
df_summary = pd.DataFrame(columns=['Count', 'Date', 'Fon', 'Unvan', 'Miktar', 'Maliyet', 'Gider', 'Fiyat', 'Tutar', 'Gün', 'Δ', 'Başarı Δ', 'RSI'])

df_portfolio['date'] = pd.to_datetime(df_portfolio['date'], errors='coerce')
df_portfolio = df_portfolio[df_portfolio['symbol'] != ""].sort_values(by=['symbol', 'date'])

# Function to calculate Sharpe ratio
# def calculate_sharpe_ratio(daily_returns):
#     mean_return = daily_returns.mean()
#     std_return = daily_returns.std()
#     sharpe_ratio = mean_return / std_return * (252 ** 0.5)
#     return sharpe_ratio

def color_gradient(val, column_name):
    if pd.isna(val) or pd.isnull(val):  # Exclude NaN and inf values
        return ''

    ranks = df_summary[column_name].rank(method='min')  # Get the ranks of the values in the specified column
    max_rank = ranks.max()
    
    try:
        current_rank = ranks[df_summary[column_name] == val].iloc[0]  # Get the rank of the current value
    except IndexError:
        return ''  # Or you could return a default color
    
    norm_val = (current_rank - 1) / (max_rank)  # Normalize the rank to [0, 1] Subtract 1 to make it 0-indexed
    norm_val = max(0, min(1, norm_val))  # Clip to [0, 1] manually
    color = sns.color_palette("RdYlGn", as_cmap=True)(norm_val)
    return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}'

def RSI_gradient(val):
    if pd.isna(val) or pd.isnull(val):  # Handle NaN and inf values
        return ''

    if val < 40:  # Values below 40 should be green with a star sign
        norm_val = 1 - ( (val - 40) / 30 )  # Normalize in [0, 1] range for gradient
        color = sns.color_palette("RdYlGn", as_cmap=True)(norm_val)  # Green to Red gradient
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])};color: white; font-weight: bold;'
    
    elif val > 70:  # Values above 70 should be red
        return 'background-color: darkred; color: white; font-weight: bold;' 
    
    else:  # Values between 40 and 70 should transition from green to red
        norm_val = 1 - ( (val - 40) / 30 )  # Normalize in [0, 1] range for gradient
        color = sns.color_palette("RdYlGn", as_cmap=True)(norm_val)  # Green to Red gradient
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])};'

# Function to process each symbol
def process_symbol(symbol, count):
    count_index = count * -1 
    recent_data = df_transformed[df_transformed['symbol'] == symbol].sort_values('date')

    if recent_data.empty:
        return None

    most_recent_price = recent_data.iloc[count_index]['close']
    most_recent_date = recent_data.iloc[count_index]['date']
    most_recent_rsi = recent_data.iloc[count_index]['RSI_14']

    symbol_data = df_portfolio[(df_portfolio['symbol'] == symbol) & (df_portfolio['date'] <= most_recent_date)].sort_values('date')

    total_quantity = 0
    total_value = 0
    avg_buy_price = 0
    weighted_daily_gain = 0
    total_days = 0
    avg_days = 0
    total_quantity_bought = 0
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
            total_quantity_bought += quantity
            avg_buy_price = total_value / total_quantity
            quantity_remained += quantity
            avg_days += (most_recent_date - transaction_date).days * quantity 

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
            avg_days -= (most_recent_date - transaction_date).days * quantity 
            if total_quantity == 0:
                avg_buy_price = 0
            else:
                avg_buy_price = total_value / total_quantity

    if total_quantity > 0:
        percentage_change = ((most_recent_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price != 0 else 0
        annual_gain = weighted_daily_gain / total_days if total_days != 0 else 0
        # daily_returns = recent_data['close'].pct_change().dropna()
        # volatility = daily_returns.std() * (252 ** 0.5) if not daily_returns.empty else 0
        # sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        avg_days = avg_days / total_quantity_bought if total_quantity_bought != 0 else 0

        return {
            'Count' : count,
            'Date' : most_recent_date,
            'Fon': symbol,
            'Unvan': symbol_title.iloc[0] if not symbol_title.empty else "",
            'Miktar': total_quantity,
            'Maliyet': avg_buy_price,
            'Gider': round(total_value, 2),
            'Fiyat': most_recent_price,
            'Tutar': round(total_quantity * most_recent_price, 2),
            'Gün': avg_days,
            'Δ': percentage_change,
            'Başarı Δ': round(annual_gain, 2),
            'RSI': round(most_recent_rsi, 2),
            # 'Volatilite': volatility,
            # 'Sharpe': sharpe_ratio
        }

# Execute the process for each unique symbol in parallel
summary_rows = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(1, 6):
        future_to_symbol = {executor.submit(process_symbol, symbol, i): symbol for symbol in df_portfolio['symbol'].unique()}
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                summary_rows.append(result)

# Convert summary rows to DataFrame and display
if summary_rows:
    df_summary = pd.DataFrame(summary_rows).sort_values(by="Tutar", ascending=False)

    st.subheader("Portföy Analiz")

    column_configuration = {
        "Count"     : st.column_config.NumberColumn("Count", help="Count", width="small"),
        "Date"      : st.column_config.DateColumn("Date", help="Date", width="small"),
        "Fon"       : st.column_config.LinkColumn("Fon", help="Fon Kodu", width="small"),
        "Unvan"     : st.column_config.TextColumn("Unvan", help="Fonun Ünvanı", width="large"),
        "Miktar"    : st.column_config.NumberColumn("Miktar", help="Fon Adedi", width="small"),
        "Maliyet"   : st.column_config.NumberColumn("Maliyet", help="İşlemler sonucu birim maliyeti", width="small"),
        "Gider"     : st.column_config.NumberColumn("Gider", help="İşlemler sonucu gider", width="small"),
        "Fiyat"     : st.column_config.NumberColumn("Fiyat", help="Güncel Fiyat", width="small"),
        "Tutar"     : st.column_config.NumberColumn("Tutar", help="Güncel Tutar", width="small"),
        "Gün"       : st.column_config.NumberColumn("Gün", help="Gün", width="small"),
        "Δ"         : st.column_config.NumberColumn("Δ", help="Güncel fiyat değişim yüzdesi", width="small"),
        "Başarı Δ"  : st.column_config.NumberColumn("Başarı Δ", help="Yıllıklandırılmış işlem getiri yüzdesi", width="small"),
        "RSI"       : st.column_config.NumberColumn("RSI", help="RSI 14", width="small"),
        # "Volatilite": st.column_config.NumberColumn("Volatilite", help="Volatilite", width="small"),
        # "Sharpe"    : st.column_config.NumberColumn("Sharpe", help="Sharpe Oranı", width="small"),
    }
 
    recent_dates = df_summary['Date'].sort_values(ascending=False).unique()
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col5:
        if len(recent_dates) > 4:
            recent_date = recent_dates[4].strftime('%Y-%m-%d')
            total_portfoy_5 = df_summary.loc[df_summary['Count'] == 5, 'Tutar'].sum()
            st.metric( label=f"{recent_date} Portföy:", value=f"{total_portfoy_5:,.0f} ₺")
    with col4:
        if len(recent_dates) > 3:
            recent_date = recent_dates[3].strftime('%Y-%m-%d')
            total_portfoy_4 = df_summary.loc[df_summary['Count'] == 4, 'Tutar'].sum()
            delta = total_portfoy_4 - total_portfoy_5 
            st.metric( label=f"{recent_date} Portföy:", value=f"{total_portfoy_4:,.0f} ₺", delta=f"{delta:,.0f} ₺" )
    with col3:
        if len(recent_dates) > 2:
            recent_date = recent_dates[2].strftime('%Y-%m-%d')
            total_portfoy_3 = df_summary.loc[df_summary['Count'] == 3, 'Tutar'].sum()
            delta = total_portfoy_3 - total_portfoy_4 
            st.metric( label=f"{recent_date} Portföy:", value=f"{total_portfoy_3:,.0f} ₺", delta=f"{delta:,.0f} ₺" )
    with col2:
        if len(recent_dates) > 1:
            recent_date = recent_dates[1].strftime('%Y-%m-%d')
            total_portfoy_2 = df_summary.loc[df_summary['Count'] == 2, 'Tutar'].sum()
            delta = total_portfoy_2 - total_portfoy_3 
            st.metric( label=f"{recent_date} Portföy:", value=f"{total_portfoy_2:,.0f} ₺", delta=f"{delta:,.0f} ₺" )
    with col1:
        if len(recent_dates) > 0:
            recent_date = recent_dates[0].strftime('%Y-%m-%d')
            total_portfoy_1 = df_summary.loc[df_summary['Count'] == 1, 'Tutar'].sum()
            delta = total_portfoy_1 - total_portfoy_2 
            st.metric( label=f"{recent_date} Portföy:", value=f"{total_portfoy_1:,.0f} ₺", delta=f"{delta:,.0f} ₺" )

    df_summary = df_summary[df_summary['Count'] <= 1]
    df_summary.drop(columns=['Date', 'Count'], inplace=True)

    styled_df = df_summary.style
    styled_df = styled_df.format({f'Gider'       : '₺ {:,.2f}', 
                                  f'Miktar'      : '{:,.0f}', 
                                  f'Maliyet'     : '₺ {:.4f}', 
                                  f'Fiyat'       : '₺ {:.4f}', 
                                  f'Tutar'       : '₺ {:,.2f}', 
                                  f'Gün'         : '{:,.0f}', 
                                #   f'Volatilite'  : '{:.2f}', 
                                #   f'Sharpe Oranı': '{:.2f}', 
                                  f'Δ'           : '% {:,.2f}', 
                                  f'Başarı Δ'    : '% {:,.2f}' , 
                                  f'RSI'         : '{:.2f}' })

    styled_df = styled_df.map(lambda val: f'<span><a target="_blank" href="https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={val}">{val}</a></span>' , subset=['Fon'])
    styled_df = styled_df.map(lambda val: color_gradient(val, f'Δ') if pd.notnull(val) else '', subset=[f'Δ'])
    styled_df = styled_df.map(lambda val: color_gradient(val, f'Başarı Δ') if pd.notnull(val) else '', subset=[f'Başarı Δ'])
    styled_df = styled_df.map(lambda val: RSI_gradient(val) if pd.notnull(val) else '', subset=['RSI'])
    
    dataframe_height = (len(df_summary) + 1) * 35 + 2
    st.dataframe(styled_df, hide_index=True, height=dataframe_height, use_container_width=True, column_config=column_configuration)
else:
    st.write("No data to display.")
