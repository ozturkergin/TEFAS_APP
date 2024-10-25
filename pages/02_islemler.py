import streamlit as st
import pandas as pd
import os
from datetime import datetime

if os.path.exists('data/fon_table.csv') :
    if 'df_fon_table' in st.session_state :
        df_fon_table = st.session_state.df_fon_table 
    else : 
        df_fon_table = pd.read_csv('data/fon_table.csv')
else: 
    df_fon_table = pd.DataFrame()
    st.warning("Entegrasyon çalıştırınız")

if df_fon_table.empty:
    st.stop()

if 'prompt_number_of_lines' in st.session_state :
    prompt_number_of_lines = st.session_state.prompt_number_of_lines  
else : 
    prompt_number_of_lines = 10

unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())

if os.path.exists('data/tefas_transformed.csv') :
    if 'df_transformed' in st.session_state :
        df_transformed = st.session_state.df_transformed 
    else : 
        df_transformed = pd.read_csv('data/tefas_transformed.csv')
        df_transformed['date'] = pd.to_datetime(df_transformed['date'])
        st.session_state.df_transformed = df_transformed

# Define a function to load portfolio data or create an empty DataFrame
def load_portfolio():
    if os.path.exists("data/myportfolio.csv"):
        return_df = pd.read_csv("data/myportfolio.csv")

        return_df['quantity'] = pd.to_numeric(return_df['quantity'], errors='coerce').fillna(0).astype(int) # Fill missing values in 'quantity' column with 0 before casting to integer
        return_df['date']     = pd.to_datetime(return_df['date'], errors='coerce')  # Convert date to datetime
        
        # Merge portfolio with tefas price data on 'symbol' and 'date'
        merged_df = pd.merge(return_df, df_transformed[['symbol', 'date', 'close']], on=['symbol', 'date'], how='left')
        merged_df.rename(columns={'close': 'price'}, inplace=True)
        return merged_df
    else:
        # Create an empty DataFrame with predefined columns
        return pd.DataFrame(columns=['symbol', 'date', 'transaction_type', 'quantity', 'price'])

df_portfolio = load_portfolio()
df_portfolio = df_portfolio[df_portfolio.quantity != 0]

if df_portfolio.empty:  # Check if the portfolio is empty and set up initial DataFrame
    df_portfolio = pd.DataFrame({"symbol": [""], "date": [datetime.today().date()], "transaction_type": [""], "quantity": [0], "price": [0],})
else:
    empty_row = pd.DataFrame({"symbol": [""], "date": [""], "transaction_type": [""], "quantity": [0], "price": [0],})
    for _ in range(prompt_number_of_lines): # Add five extra empty lines if the portfolio is not empty
        df_portfolio = pd.concat([df_portfolio, empty_row], ignore_index=True)

# Ensure the date column is treated as datetime for the data editor
df_portfolio['date'] = pd.to_datetime(df_portfolio['date'], errors='coerce')

# Set up column configuration
column_config = {
    "symbol": st.column_config.SelectboxColumn("Fon", help="Stock symbol", options=unique_symbols),
    "date": st.column_config.DateColumn("Tarih", help="Transaction date"),  # Proper date column
    "transaction_type": st.column_config.SelectboxColumn("İşlem", options=["buy", "sell"], help="Select buy or sell"),
    "quantity": st.column_config.NumberColumn("Miktar", help="Number of shares", min_value=1, step=1),
}

col2, col3 = st.columns([1, 2])

with col2:
    st.title("İşlemler")

    # Wrap data editor and save button in a form
    with st.form(key="portfolio_form"):
        # Display data editor within the form
        prompt_number_of_lines = st.number_input("Boş Satır Sayısı:", min_value=0, step=1, value=prompt_number_of_lines)
        st.session_state.prompt_number_of_lines = prompt_number_of_lines

        save_button = st.form_submit_button("Sakla") # Submit button

        dataframe_height = (len(df_portfolio) + 1) * 35 + 2
        edited_df = st.data_editor(df_portfolio, column_config=column_config, hide_index=True, height=dataframe_height, use_container_width=True,)
        edited_df = edited_df[edited_df['symbol'] != ""]
        
        if save_button: # Check if save button is clicked
            # Convert date column back to datetime
            edited_df['date'] = pd.to_datetime(edited_df['date'], errors='coerce')
            columns_to_save = ['symbol', 'date', 'transaction_type', 'quantity']
            filtered_df = edited_df[columns_to_save]
            
            if not edited_df.empty:
                filtered_df.to_csv('data/myportfolio.csv', index=False) # Save to CSV
                st.success("Portfolio saved successfully!")
                st.rerun()
            else:
                st.warning("No valid entries to save.")
