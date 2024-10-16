import streamlit as st
import pandas as pd
import os
from datetime import datetime

#st.set_page_config(page_title="TEFAS Analiz", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

# Load unique symbols from fon_table.csv
df_fon_table = pd.read_csv('data/fon_table.csv')
unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())

# Load tefas_transformed.csv for unit prices
df_tefas = pd.read_csv('data/tefas_transformed.csv')
df_tefas['date'] = pd.to_datetime(df_tefas['date'])

# Define a function to load portfolio data or create an empty DataFrame
def load_portfolio():
    if os.path.exists("data/myportfolio.csv"):
        return_df = pd.read_csv("data/myportfolio.csv")
        
        # Fill missing values in 'quantity' column with 0 before casting to integer
        return_df['quantity'] = pd.to_numeric(return_df['quantity'], errors='coerce').fillna(0).astype(int)
        
        return_df['date'] = pd.to_datetime(return_df['date'], errors='coerce')  # Convert date to datetime
        
        # Merge portfolio with tefas price data on 'symbol' and 'date'
        merged_df = pd.merge(return_df, df_tefas[['symbol', 'date', 'close']],
                             on=['symbol', 'date'], how='left')
        merged_df.rename(columns={'close': 'price'}, inplace=True)
        return merged_df
    else:
        # Create an empty DataFrame with predefined columns
        return pd.DataFrame(columns=['symbol', 'date', 'transaction_type', 'quantity', 'price'])

# Load the portfolio data
if 'df_portfolio' not in st.session_state:
    st.session_state.df_portfolio = load_portfolio()

df_portfolio = st.session_state.df_portfolio

# Check if the portfolio is empty and set up initial DataFrame
if df_portfolio.empty:
    df_portfolio = pd.DataFrame({
        "symbol": [""],  # Empty cell for user input
        "date": [datetime.today().date()],  # Default to today's date
        "transaction_type": [""],  # Empty cell for selection
        "quantity": [0],  # Empty cell for user input
        "price": [0],  # Empty cell for user input
    })
else:
    # Add an extra empty line if the portfolio is not empty
    empty_row = pd.DataFrame({
        "symbol": [""],
        "date": [datetime.today().date()],  # Today's date
        "transaction_type": [""],
        "quantity": [0],
        "price": [0],
    })
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
    
    # Create the data editor with validation
    dataframe_height = (len(df_portfolio) + 1) * 35 + 2
    edited_df = st.data_editor(df_portfolio, column_config=column_config, hide_index=True, height=dataframe_height, use_container_width=True)
    edited_df = edited_df[edited_df['symbol'] != ""]

    # Store the edited DataFrame in session state (instead of refreshing the page)
    st.session_state.edited_df = edited_df

    # Save button to store the data
    if st.button('Sakla'):
        # Convert the date column back to datetime format before saving
        edited_df['date'] = pd.to_datetime(edited_df['date'], errors='coerce')
        columns_to_save = ['symbol', 'date', 'transaction_type', 'quantity']
        filtered_df = edited_df[columns_to_save]
        
        if not edited_df.empty:  # Only save if there are valid entries
            filtered_df.to_csv('data/myportfolio.csv', index=False)
            st.success("Portfolio saved successfully!")
            st.rerun()  # Refresh the page only after saving
        else:
            st.warning("No valid entries to save.")
