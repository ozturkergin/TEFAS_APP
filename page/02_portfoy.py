import streamlit as st
import pandas as pd
import os
from datetime import datetime

if os.path.exists('data/fon_table.csv') :
    if 'df_fon_table' in st.session_state :
        df_fon_table = st.session_state.df_fon_table 
    else : 
        df_fon_table = pd.read_csv('data/fon_table.csv')
else : 
    df_fon_table = pd.DataFrame()
    st.warning("Entegrasyon çalıştırınız")

if df_fon_table.empty:
    st.stop()

unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())

if os.path.exists('data/tefas_transformed.csv') :
    if 'df_transformed' in st.session_state :
        df_transformed = st.session_state.df_transformed 
    else : 
        df_transformed = pd.read_csv('data/tefas_transformed.csv')
        df_transformed['date'] = pd.to_datetime(df_transformed['date'], errors='coerce')
        st.session_state.df_transformed = df_transformed

# Define a function to load portfolio data or create an empty DataFrame
def load_portfolio():
    if os.path.exists("data/myportfolio.csv"):
        if 'myportfolio' in st.session_state :
            myportfolio = st.session_state.myportfolio 
        else : 
            myportfolio = pd.read_csv('data/myportfolio.csv')
            myportfolio['quantity'] = pd.to_numeric(myportfolio['quantity'], errors='coerce').fillna(0).astype(int)
            myportfolio['date'] = pd.to_datetime(myportfolio['date'], errors='coerce')  # Convert date to datetime

        merged_df = pd.merge(myportfolio, df_transformed[['symbol', 'date', 'close']], on=['symbol', 'date'], how='left') # Merge portfolio with tefas price data on 'symbol' and 'date'
        merged_df.rename(columns={'close': 'price'}, inplace=True)
        return merged_df
    else:
        st.warning("İşlem girdikten sonra portföyünüz oluşacaktır")
        return pd.DataFrame(columns=['symbol', 'date', 'transaction_type', 'quantity', 'price'])

# Load the portfolio data
df_portfolio = load_portfolio()

# Check if the portfolio is empty and set up initial DataFrame
if df_portfolio.empty:
    st.stop()

col3, col2 = st.columns([100, 1])

df_summary = pd.DataFrame(columns=['Fon', 'Unvan', 'Miktar', 'Maliyet', 'Gider', 'Fiyat', 'Tutar', 'Δ', 'Başarı Δ', 'Volatilite', 'Sharpe Oranı']) # Create a summary dataframe
df_portfolio['date'] = pd.to_datetime(df_portfolio['date'], errors='coerce') # Ensure the date column is treated as datetime for the data editor
df_portfolio = df_portfolio[df_portfolio['symbol'] != ""].sort_values(by=['symbol', 'date']) # Sorting transactions by symbol and date

# Function to calculate volatility and Sharpe ratio (simplified)
def calculate_sharpe_ratio(daily_returns):
    return daily_returns.mean() / daily_returns.std() * (252 ** 0.5)  # Annualized Sharpe ratio

# Display the summary
with col3:
    summary_rows = []
    for symbol in df_portfolio['symbol'].unique():
        if symbol == "":
            continue

        symbol_data = df_portfolio[df_portfolio['symbol'] == symbol].sort_values('date')
        
        total_quantity = 0
        total_value = 0
        avg_buy_price = 0
        weighted_daily_gain = 0
        total_days = 0
        quantity_remained = 0

        recent_data = df_transformed[df_transformed['symbol'] == symbol].sort_values('date') # Get the most recent price and date for this symbol
        
        if len(recent_data) == 0:
            st.warning(f"No recent price data found for {symbol}. Skipping...")
            continue

        most_recent_price = recent_data.iloc[-1]['close']
        most_recent_date = recent_data.iloc[-1]['date']

        # Process each transaction for the symbol
        for i, (idx, row) in enumerate(symbol_data.iterrows()):
            transaction_type = row['transaction_type']
            transaction_date = row['date']
            quantity = row['quantity']
            unit_price = df_transformed.loc[(df_transformed['symbol'] == symbol) & (df_transformed['date'] == transaction_date), 'close']
            symbol_title = df_fon_table.loc[(df_fon_table['symbol'] == symbol), 'title']
            
            if not unit_price.empty:
                unit_price = unit_price.iloc[0]
            else:
                st.warning(f"Price not found for {symbol} on {transaction_date}. Skipping...")
                unit_price = 0 

            quantity_remained = 0 

            if transaction_type == 'buy':
                total_value += quantity * unit_price
                total_quantity += quantity
                avg_buy_price = total_value / total_quantity  # Calculate weighted average price

                quantity_remained += quantity  # Initially, all quantity is unsold

                # Check for subsequent sell transactions for the current buy
                for j in range(i + 1, len(symbol_data)):
                    next_row = symbol_data.iloc[j]
                    if next_row['transaction_type'] == 'sell' :
                        # If a sell transaction is found, adjust the remaining quantity
                        sell_quantity = next_row['quantity']
                        sell_date = next_row['date']
                        sell_price = next_row['price']  # Assuming 'price' is the sell price
                        days_held = (sell_date - transaction_date).days
                        if sell_quantity <= quantity_remained : 
                            weighted_daily_gain += ((sell_price - unit_price) / unit_price * 100) / days_held * 365 * sell_quantity
                            quantity_remained -= sell_quantity
                            total_days += sell_quantity
                        else: 
                            weighted_daily_gain += ((sell_price - unit_price) / unit_price * 100) / days_held * 365 * quantity
                            quantity_remained -= sell_quantity
                            total_days += quantity
                            break 

                # If there's any remaining quantity after all sell transactions, value it at the most recent price
                if quantity_remained > 0:
                    days_held = (most_recent_date - transaction_date).days
                    weighted_daily_gain += ((most_recent_price - unit_price) / unit_price * 100) / days_held * 365 * quantity_remained
                    total_days += quantity_remained  # Total days should count for the full quantity, sold or held

            elif transaction_type == 'sell':
                # If selling, adjust the quantity and total value
                total_value -= quantity * avg_buy_price
                total_quantity -= quantity
                if total_quantity == 0:
                    avg_buy_price = 0  # Reset if everything is sold off
                else:
                    avg_buy_price = total_value / total_quantity  # Recalculate weighted average price

        # Check if there are remaining units held after the last transaction
        if total_quantity > 0 :
            percentage_change = ((most_recent_price - avg_buy_price) / avg_buy_price) * 100
            # Annualize the weighted daily gain over the total number of days
            yearly_gain = weighted_daily_gain / total_days

            # Calculate volatility (standard deviation of daily returns) and Sharpe ratio
            daily_returns = df_transformed[df_transformed['symbol'] == symbol].sort_values('date')['close'].pct_change()
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
            sharpe_ratio = calculate_sharpe_ratio(daily_returns)

            summary_rows.append({
                'Fon': symbol,
                'Unvan': symbol_title.iloc[0] if not symbol_title.empty else "",
                'Miktar': total_quantity,
                'Maliyet': avg_buy_price,
                'Gider': round(total_value,2),
                'Fiyat': most_recent_price,
                'Tutar': round(total_quantity * most_recent_price,2),
                'Δ': percentage_change,
                'Başarı Δ': round(yearly_gain,2),
                'Volatilite': volatility,
                'Sharpe Oranı': sharpe_ratio
            })

    # Convert summary rows to DataFrame and display
    if summary_rows:
        df_summary = pd.concat([df_summary, pd.DataFrame(summary_rows)], ignore_index=True)
        st.subheader("Portföy Analiz")
        dataframe_height = (len(df_summary) + 1) * 35 + 2
        st.dataframe(df_summary, hide_index=True, height=dataframe_height, use_container_width=True)
    else:
        st.write("No data to display.")
