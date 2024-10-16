import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Load unique symbols and titles from fon_table.csv
df_fon_table = pd.read_csv('data/fon_table.csv')
unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())
symbol_titles = df_fon_table.set_index('symbol')['title'].to_dict()  # Dictionary for codewithtext

# Function to load/save favorites
def manage_favorites(favorites_file='data/favorites.csv'):
    if os.path.exists(favorites_file):
        favorites = pd.read_csv(favorites_file)['symbol'].tolist()
    else:
        favorites = []
    return favorites

# Initialize session state if not already initialized
if 'multiselect' not in st.session_state:
    st.session_state.multiselect = manage_favorites()  

# Multiselect with unique symbols
selected_symbols = st.multiselect(
    'Select Symbols:',
    unique_symbols,
    default=st.session_state.multiselect,  # Use session state for default
    key='multiselect'
)

# Callback function to save on change
def save_favorites():
    pd.DataFrame({'symbol': st.session_state.multiselect}).to_csv('data/favorites.csv', index=False)

# Register the callback function
st.cache_data(save_favorites, show_spinner=False) 
if st.session_state.multiselect != selected_symbols:  # Check for changes
    st.session_state.multiselect = selected_symbols  # Update session state
    save_favorites()  # Save to CSV
    st.experimental_rerun()  # Rerun to update the app

# Create DataFrame for st.dataframe
selected_df = pd.DataFrame({
    'code': selected_symbols,
    'codewithtext': [f"{code} - {symbol_titles.get(code, '')}" for code in selected_symbols]
})

# Display selected symbols using st.dataframe
st.dataframe(selected_df, use_container_width=True, hide_index=True)
