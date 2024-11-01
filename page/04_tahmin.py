import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from prophet import Prophet

# symbol_attributes_df = pd.DataFrame()
set_filtered_symbols = set()

# Turkish sorting function
def turkish_sort(x):
    import locale
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    return locale.strxfrm(x)

@st.cache_data
def fetch_data():
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

    if 'df_merged' in st.session_state:
        df_merged = st.session_state.df_merged
    else:
        df_merged = pd.merge(df_transformed, df_fon_table, on='symbol', how='inner')
        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

    return df_merged, df_fon_table

df_merged, df_fon_table = fetch_data()

unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())

# Multiselect with unique symbols
selected_symbols = st.multiselect(
    'Fon:',
    unique_symbols,
    default=None,
    key='selected_symbols'
)

with st.sidebar:
    with st.container():
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            show_favourites = st.checkbox("Favorilerim", key="Favorilerim")
        with row1_col2:
            show_portfolio = st.checkbox("Portföyüm", key="Portföyüm")
            
        prompt_number_of_days_to_predict = st.number_input("Gelecek Kaç Gözlem Tahminlenmeli:", min_value=0, step=1, value=30)

        set_filtered_symbols.update(selected_symbols)

        if show_favourites:
            if 'favourites' in st.session_state :
                set_filtered_symbols.update(st.session_state.favourites)

        if show_portfolio:
            if 'myportfolio' in st.session_state :
                myportfolio_summarized = ( st.session_state.myportfolio
                .groupby('symbol', as_index=False)                              
                .apply(lambda df: pd.Series({'net_quantity': df.loc[df['transaction_type'] == 'buy', 'quantity'].sum() - df.loc[df['transaction_type'] == 'sell', 'quantity'].sum()}))
                .query('net_quantity != 0') )  # Keep only symbols with non-zero net quantity  
                set_filtered_symbols.update(myportfolio_summarized['symbol'].unique().tolist())

for symbol in set_filtered_symbols:
    data = df_merged[df_merged['symbol'] == symbol]
    title = data.iloc[0]['title']
    data = data[['date', 'close']].copy()
    data.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)

    model = Prophet()    # Prophet model
    model.fit(data)

    future = model.make_future_dataframe(periods=prompt_number_of_days_to_predict)    # Prediction with x days into the future
    forecast = model.predict(future)

    future_x_days = forecast[['ds', 'yhat']].tail(prompt_number_of_days_to_predict)    # Select the last x days of predictions for display
    future_x_days.rename(columns={'ds': 'Date', 'yhat': 'Predicted Close'}, inplace=True)

    last_actual_price = data['y'].iloc[-1]    # Calculate the percentage change between the last actual close and the predicted latest close
    predicted_latest_price = future_x_days['Predicted Close'].iloc[-1]
    percentage_change = ((predicted_latest_price - last_actual_price) / last_actual_price) * 100

    with st.expander(f"{symbol} - {title}", expanded=True):
        fig = go.Figure()        # Create Plotly figure
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Past Close'))        # Add historical data
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Full Prediction', line=dict(color='blue', dash='dash')))        # Add full prediction range in a single color

        # Overlay final x days in a different color
        fig.add_trace(go.Scatter(x=forecast['ds'][-prompt_number_of_days_to_predict:], y=forecast['yhat'][-prompt_number_of_days_to_predict:], mode='lines', name=f'{prompt_number_of_days_to_predict}-Day Future Prediction', line=dict(color='red')))
        
        # Update layout for clarity
        fig.update_layout(
            title=f'Tahmin: {symbol} - {title}',
            xaxis_title='Date',
            yaxis_title='Close Price',
            xaxis=dict(rangeslider=dict(visible=True), type="date")
        )
        col1, col2, col3 = st.columns([2, 2, 8])

        with col1:            # Display percentage change in a card
            st.metric(
                label=f"{prompt_number_of_days_to_predict}-Günlük Tahmin: {symbol}",
                value=f"{percentage_change:.2f}%",
                delta=round(percentage_change, 2),
                delta_color="normal"
            )
        with col2:            # Display the future prediction data in a dataframe
            future_x_days.sort_values(by="Date", ascending=False, inplace=True)
            future_x_days['Date'] = future_x_days['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(future_x_days, use_container_width=True, hide_index=True)
        with col3:            # Display the chart in Streamlit
            st.plotly_chart(fig)

            