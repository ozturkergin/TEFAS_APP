import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.title("Home Page")

# Load data and get available dates
if os.path.exists('data/fon_table.csv'):
    if 'df_fon_table' in st.session_state:
        df_fon_table = st.session_state.df_fon_table
    else:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table
else: 
    st.stop()

if os.path.exists('data/tefas_transformed.csv'):
    if 'df_transformed' in st.session_state:
        df_transformed = st.session_state.df_transformed
    else:
        df_transformed = pd.read_csv('data/tefas_transformed.csv', encoding='utf-8-sig', parse_dates=['date'])
        st.session_state.df_transformed = df_transformed
else: 
    st.stop()

# Get all available dates
all_dates = pd.to_datetime(df_transformed['date'].unique())
all_dates = pd.Series(sorted(all_dates))

# Default dates
default_recent_date = all_dates.max()
default_prev_date = all_dates[all_dates < default_recent_date].max()

# User selects dates (widgets OUTSIDE cached function)
st.subheader("Select Dates for Comparison")

col1, col2 = st.columns(2)

with col1:
    selected_recent_date = st.date_input("Date To", value=default_recent_date, min_value=all_dates.min(), max_value=all_dates.max(), key="recent_date" )
with col2:
    prev_date_options = all_dates[all_dates < pd.to_datetime(selected_recent_date)]
    if not prev_date_options.empty:
        selected_prev_date = st.date_input(
            "Date From", value=default_prev_date if default_prev_date < pd.to_datetime(selected_recent_date) else prev_date_options.max(),
            min_value=all_dates.min(), max_value=pd.to_datetime(selected_recent_date) - pd.Timedelta(days=1), key="prev_date"
        )
    else:
        st.warning("No previous date available before the selected recent date.")
        st.stop()

recent_date = pd.to_datetime(selected_recent_date)
prev_date = pd.to_datetime(selected_prev_date)

@st.cache_data
def fetch_todays_data(recent_date, prev_date):
    summary_recent = pd.DataFrame(columns=['Fon Unvan Türü', 'symbol', 'market_cap'])  # Initialize as empty DataFrame

    # Separate data for recent_date and prev_date
    df_transformed_recent = df_transformed[df_transformed['date'] == recent_date]
    df_transformed_recent = pd.merge(df_transformed_recent, df_fon_table, on='symbol', how='inner')
    df_transformed_prev = df_transformed[df_transformed['date'] == prev_date]
    df_transformed_prev = pd.merge(df_transformed_prev, df_fon_table, on='symbol', how='inner')

    # Extract symbol attributes
    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('FundType_')]
    symbol_attributes_list = [col.replace('FundType_', '') for col in symbol_attributes_of_fon_table]
    
    data_fon_turu_summary = []
    for attribute in symbol_attributes_list:  # Calculate totals and deltas for each attribute
        attribute_col = 'FundType_' + attribute
        
        if not attribute_col in df_transformed_recent.columns: 
            continue 
        if not attribute_col in df_transformed_prev.columns: 
            continue

        filtered_recent = df_transformed_recent[df_transformed_recent[attribute_col] == True].copy()
        filtered_recent.loc[:, "Fon Unvan Türü"] = attribute
        filtered_prev = df_transformed_prev[df_transformed_prev[attribute_col] == True]
        amount_t = filtered_recent['market_cap'].sum()
        amount_t_minus_1 = filtered_prev['market_cap'].sum()

        if amount_t_minus_1 == 0:
            continue

        delta = amount_t - amount_t_minus_1
        delta_pct = (amount_t - amount_t_minus_1) / amount_t_minus_1
        indicator = '✅' if delta > 0 else '🔻' if delta < 0 else '➡️'
        recent_date_str = datetime.strftime(recent_date, "%Y-%m-%d")
        prev_date_str = datetime.strftime(prev_date, "%Y-%m-%d")

        data_fon_turu_summary.append({ # Add to summary data
            'Fon Unvan Türü': attribute,
            recent_date_str: round(amount_t, 0),
            prev_date_str: round(amount_t_minus_1, 0),
            'Δ': round(delta, 0),
            'Δ %': round(delta_pct, 5),
            '': indicator
        })

        # Summarize market_cap by Fon Unvan Türü and Fon
        if not summary_recent.empty:
            summary_recent = pd.concat([summary_recent, filtered_recent.groupby(['Fon Unvan Türü', 'symbol'])['market_cap'].sum().reset_index()])
        else:
            summary_recent = filtered_recent.groupby(['Fon Unvan Türü', 'symbol'])['market_cap'].sum().reset_index()

    return summary_recent, data_fon_turu_summary, recent_date, prev_date

summary_recent, data_fon_turu_summary, recent_date, prev_date = fetch_todays_data(recent_date, prev_date)

dataframe_height = (len(data_fon_turu_summary) + 1) * 35 + 2

col1, col2 = st.columns([6, 7])
with col1:
    with st.container():
        df_summary = pd.DataFrame(data_fon_turu_summary)
        st.dataframe(df_summary, hide_index=True, height=dataframe_height)
with col2:
    with st.container():
        treemap_data = pd.DataFrame({
            'names': summary_recent['symbol'],
            'parents': summary_recent['Fon Unvan Türü'],
            'values': summary_recent['market_cap']
        })

        treemap_data = treemap_data[treemap_data['values'] > 0]  # Remove negative values

        fig = px.treemap(
            treemap_data,
            path=['parents', 'names'],  # Path to the names
            values='values',
            color='values',  # Color by the values to show increase or decrease
            color_continuous_scale='RdYlGn',  # Red for negative, green for positive
            # title="Market Cap Treemap",
            height=dataframe_height, )

        st.plotly_chart(fig, use_container_width=True) # Display the treemap next to the DataFrame
