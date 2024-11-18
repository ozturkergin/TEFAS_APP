import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.title("Home Page")

symbol_attributes_df = pd.DataFrame()
df_todays_total = pd.DataFrame()
df_transformed = pd.DataFrame()

@st.cache_data
def fetch_todays_data():
    if os.path.exists('data/fon_table.csv'):
        if 'df_fon_table' in st.session_state:
            df_fon_table = st.session_state.df_fon_table
        else:
            df_fon_table = pd.read_csv('data/fon_table.csv')
            st.session_state.df_fon_table = df_fon_table

    if os.path.exists('data/tefas_transformed.csv'):
        if 'df_transformed' in st.session_state:
            df_transformed = st.session_state.df_transformed
        else:
            df_transformed = pd.read_csv('data/tefas_transformed.csv')
            df_transformed['date'] = pd.to_datetime(df_transformed['date'])
            st.session_state.df_transformed = df_transformed

    # Get the latest and previous dates
    max_date = df_transformed['date'].max()
    prev_date = df_transformed[df_transformed['date'] < max_date]['date'].max()

    # Separate data for max_date and prev_date
    df_transformed_max = df_transformed[df_transformed['date'] == max_date]
    df_transformed_max = pd.merge(df_transformed_max, df_fon_table, on='symbol', how='inner')
    df_transformed_prev = df_transformed[df_transformed['date'] == prev_date]
    df_transformed_prev = pd.merge(df_transformed_prev, df_fon_table, on='symbol', how='inner')

    # Extract symbol attributes
    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('symbol_')]
    symbol_attributes_list = [col.replace('symbol_', '') for col in symbol_attributes_of_fon_table]
    symbol_attributes_df = pd.DataFrame({'Fon Unvan Türü': symbol_attributes_list})
    
    data_summary = []
    for attribute in symbol_attributes_list:  # Calculate totals and deltas for each attribute
        attribute_col = 'symbol_' + attribute
        
        if not attribute_col in df_transformed_max.columns: 
            continue 
        if not attribute_col in df_transformed_prev.columns: 
            continue

        filtered_max = df_transformed_max[df_transformed_max[attribute_col] == True]
        filtered_prev = df_transformed_prev[df_transformed_prev[attribute_col] == True]
        amount_t = filtered_max['market_cap'].sum()
        amount_t_minus_1 = filtered_prev['market_cap'].sum()

        if amount_t_minus_1 == 0:
            continue

        delta = amount_t - amount_t_minus_1
        delta_pct = (amount_t - amount_t_minus_1) / amount_t_minus_1
        indicator = '✅' if delta > 0 else '🔻' if delta < 0 else '➡️'
        max_date_str = datetime.strftime(max_date, "%Y-%m-%d")
        prev_date_str = datetime.strftime(prev_date, "%Y-%m-%d")

        data_summary.append({ # Add to summary data
            'Fon Unvan Türü': attribute,
            max_date_str: round(amount_t, 0),
            prev_date_str: round(amount_t_minus_1, 0),
            'delta': round(delta, 0),
            '%': round(delta_pct, 5),
            '': indicator
        })

    df_summary = pd.DataFrame(data_summary)
    dataframe_height = (len(df_summary) + 1) * 35 + 2
    
    col1, col2 = st.columns([7, 6])
    with col1:
        with st.container():
            st.dataframe(df_summary, hide_index=True, height=dataframe_height)
    with col2:
        with st.container():
            max_date_column = df_summary.columns[1]  # Assumes the max_date market cap column is the second column
            fig = px.treemap(   
                                df_summary, # Treemap visualization
                                path=['Fon Unvan Türü'],
                                values=max_date_column,
                                color='delta',  # Color by the delta to show increase or decrease
                                color_continuous_scale='RdYlGn',  # Red for negative, green for positive
                                title="Market Cap Treemap",
                                height=dataframe_height,
                            )
            st.plotly_chart(fig, use_container_width=True) # Display the treemap next to the DataFrame

    return df_summary, symbol_attributes_df

df_summary, symbol_attributes_df = fetch_todays_data()
