import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import os

symbol_attributes_df = pd.DataFrame()
df_combined_symbol_metrics = pd.DataFrame()
df_combined_symbol_history = pd.DataFrame()
set_filtered_symbols = set()

# Turkish sorting function
def turkish_sort(x):
    import locale
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    return locale.strxfrm(x)

@st.cache_data
def fetch_data():
    if os.path.exists('data/fon_table.csv') :
        if 'df_fon_table' in st.session_state :
            df_fon_table = st.session_state.df_fon_table 
        else : 
            df_fon_table = pd.read_csv('data/fon_table.csv')
            st.session_state.df_fon_table = df_fon_table

    if os.path.exists('data/tefas_transformed.csv') :
        if 'df_transformed' in st.session_state :
            df_transformed = st.session_state.df_transformed 
        else : 
            df_transformed = pd.read_csv('data/tefas_transformed.csv')
            df_transformed['date'] = pd.to_datetime(df_transformed['date'])
            st.session_state.df_transformed = df_transformed

    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('symbol_')]
    symbol_attributes_list = np.array([col.replace('symbol_', '') for col in symbol_attributes_of_fon_table])
    symbol_attributes_list = sorted(symbol_attributes_list, key=turkish_sort)
    symbol_attributes_df = pd.DataFrame({'Fon Unvan Türü': symbol_attributes_list})

    if 'df_merged' in st.session_state :
        df_merged = st.session_state.df_merged 
    else : 
        df_merged = pd.merge(df_transformed, df_fon_table, on='symbol', how='inner')
        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

    return df_merged, df_fon_table, symbol_attributes_df

# Date range filtering
def filter_get_min_date(range_type):
    max_date = df_merged['date'].max()
    if range_type == "1m":
        min_date = max_date - pd.DateOffset(days=30)
    elif range_type == "3m":
        min_date = max_date - pd.DateOffset(months=3)
    elif range_type == "6m":
        min_date = max_date - pd.DateOffset(months=6)
    elif range_type == "1y":
        min_date = max_date - pd.DateOffset(years=1)
    elif range_type == "YTD":
        min_date = pd.Timestamp(f"{max_date.year}-01-01")
    else:
        min_date = df_merged['date'].min()
    return min_date

if 'filter_label' not in st.session_state:
    st.session_state.filter_label = "1m"

def color_gradient(val, column_name):
    # Exclude NaN and inf values
    if pd.isna(val) or np.isinf(val):
        return ''
    
    # Get the ranks of the values in the specified column
    ranks = df_combined_symbol_metrics[column_name].rank(method='min')
    max_rank = ranks.max()
    
    # Get the rank of the current value
    current_rank = ranks.loc[df_combined_symbol_metrics[column_name] == val].values[0]
    
    # Normalize the rank to [0, 1]
    norm_val = (current_rank - 1) / (max_rank - 1)  # Subtract 1 to make it 0-indexed
    
    # Ensure normalization is within [0, 1]
    norm_val = np.clip(norm_val, 0, 1)
    
    color = sns.color_palette("RdYlGn", as_cmap=True)(norm_val)
    return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}'

def filter_by_date_range(range_type):
    max_date = df_merged['date'].max()
    if range_type == "1m":
        min_date = max_date - pd.DateOffset(days=30)
    elif range_type == "3m":
        min_date = max_date - pd.DateOffset(days=90)
    elif range_type == "6m":
        min_date = max_date - pd.DateOffset(days=180)
    elif range_type == "1y":
        min_date = max_date - pd.DateOffset(years=1)
    elif range_type == "YTD":
        min_date = pd.Timestamp(f"{max_date.year}-01-01")
    else:
        min_date = df_merged['date'].min()
    return df_merged[(df_merged['date'] >= min_date) & (df_merged['date'] <= max_date)]

# Cumulative change plot function
def plot_cumulative_change(df_filtered, set_filtered_symbols, title=""):
    fig = go.Figure()

    # Loop through each selected symbol in filtered_fons
    for symbol in set_filtered_symbols:
        symbol_data = df_filtered[df_filtered['symbol'] == symbol]
        
        # Calculate percentage change and cumulative change
        symbol_data['price_change_pct'] = symbol_data['close'].pct_change().fillna(0)
        symbol_data['cumulative_change'] = (1 + symbol_data['price_change_pct']).cumprod() - 1
        
        # Add the symbol's cumulative change to the figure
        fig.add_trace(
            go.Scatter(
                x=symbol_data['date'], 
                y=symbol_data['cumulative_change'],
                mode='lines',
                name=symbol
            )
        )

    # Update the layout with rangeslider, rangeselector, and custom plot height
    fig.update_layout(
        title=title,
        height=600,  # Adjust the height here
        xaxis_title="Date",
        yaxis_title="Cumulative Price Change (%)",
        xaxis=dict(
            rangeslider_visible=True
        ),
        yaxis=dict(
            tickformat="%",  # Display y-axis as percentages
        )
    )
    return fig

# Display filter buttons horizontally and update chart
def display_buttons():
    button_labels = ["1m", "3m", "6m", "1y", "YTD", "All"]
    cols = st.columns(len(button_labels))
    for i, label in enumerate(button_labels):
        if cols[i].button(label):
            st.session_state.filter_label = label
            
df_merged, df_fon_table, symbol_attributes_df = fetch_data()

column_configuration_fon = {
    "symbol": st.column_config.TextColumn("Fon", help="Fon 3 haneli kod", width="small"),
    "title" : st.column_config.TextColumn("Unvan", help="Fonun Unvanı", width="large"),
    "1m-F%" : st.column_config.NumberColumn("1m-F%", help="1 aylık fiyat değişimi", width="small", format="%.2d"),
    "1m-YS%": st.column_config.NumberColumn("1m-YS%", help="1 aylık yatırımcı sayısı değişimi", width="small", format="%.2d"),
    "1m-BY%": st.column_config.NumberColumn("1m-BY%", help="1 aylık yatırımcı başına yatırım tutarı değişimi", width="small", format="%.2d"),
}

col2, col3 = st.columns([6, 6])

with st.sidebar:
    with st.container():
        selectable_attributes = st.dataframe(symbol_attributes_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row")
        filtered_attributes   = symbol_attributes_df.loc[selectable_attributes.selection.rows]

with col2:
    with st.container():
        if not filtered_attributes.empty:
            display_buttons() 
            
            for filtered_attribute in filtered_attributes['Fon Unvan Türü']:
                df_filtered_symbols = df_merged[df_merged[f'symbol_{filtered_attribute}'] == True]['symbol'].unique().tolist()
                set_filtered_symbols.update(df_filtered_symbols)

            if set_filtered_symbols:
                for symbol in set_filtered_symbols:
                    lv_time_range = st.session_state.filter_label
                    df_symbol_history = df_merged[(df_merged['symbol'] == symbol) & (df_merged['date'] >= filter_get_min_date(lv_time_range))]
                    df_combined_symbol_history = pd.concat([df_combined_symbol_history, df_symbol_history], ignore_index=True)
                    
                    if not df_symbol_history.empty:
                        df_symbol_history_sorted        = df_symbol_history.sort_values('date')
                        start_close                     = df_symbol_history_sorted.iloc[0]['close']
                        end_close                       = df_symbol_history_sorted.iloc[-1]['close']
                        start_number_of_investors       = df_symbol_history_sorted.iloc[0]['number_of_investors']
                        end_number_of_investors         = df_symbol_history_sorted.iloc[-1]['number_of_investors']
                        start_market_cap_per_investors  = df_symbol_history_sorted.iloc[0]['market_cap_per_investors']
                        end_market_cap_per_investors    = df_symbol_history_sorted.iloc[-1]['market_cap_per_investors']
                        # Check for NaN or inf values
                        if pd.notnull(start_close) and pd.notnull(end_close) and start_close != 0:
                            change_price = (end_close - start_close) / start_close * 100
                        else:
                            change_price = float('nan')  # or handle it as you need, e.g., change_price = 0
    
                        if pd.notnull(start_number_of_investors) and pd.notnull(end_number_of_investors) and start_number_of_investors != 0:
                            change_number_of_investors = (end_number_of_investors - start_number_of_investors) / start_number_of_investors * 100
                        else:
                            change_number_of_investors = float('nan')
    
                        if pd.notnull(start_market_cap_per_investors) and pd.notnull(end_market_cap_per_investors) and start_market_cap_per_investors != 0:
                            change_market_cap_per_investors = (end_market_cap_per_investors - start_market_cap_per_investors) / start_market_cap_per_investors * 100
                        else:
                            change_market_cap_per_investors = float('nan')
                        
                        df_symbol_metrics = pd.DataFrame({
                            'symbol'              : [symbol],
                            'title'               : [df_symbol_history_sorted.iloc[0]['title']],
                            f'{lv_time_range}-F%' : round(change_price, 2),
                            f'{lv_time_range}-YS%': round(change_number_of_investors, 2),
                            f'{lv_time_range}-BY%': round(change_market_cap_per_investors, 2)
                        })
                        df_combined_symbol_metrics = pd.concat([df_combined_symbol_metrics, df_symbol_metrics], ignore_index=True)
                        
                styled_df           = df_combined_symbol_metrics.style.map(lambda val: color_gradient(val, f'{lv_time_range}-F%'), subset=[f'{lv_time_range}-F%'])
                styled_df           = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-YS%'), subset=[f'{lv_time_range}-YS%'])
                styled_df           = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-BY%'), subset=[f'{lv_time_range}-BY%'])
                selectable_symbols  = st.dataframe(styled_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row", column_config=column_configuration_fon)
                
                set_filtered_symbols.clear()
                for selected_symbol in selectable_symbols.selection.rows:
                    symbol_value = df_combined_symbol_metrics.loc[selected_symbol]['symbol']
                    set_filtered_symbols.update([symbol_value]) 
                    
with col3:
    with st.container():
        chart_placeholder = st.empty()
        if set_filtered_symbols :
            chart_placeholder.plotly_chart(plot_cumulative_change(df_combined_symbol_history, set_filtered_symbols))

