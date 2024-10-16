import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#st.set_page_config(page_title="TEFAS Analiz", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

combined_filtered_symbols = pd.DataFrame()
filtered_symbols = pd.DataFrame()
df_merged = pd.DataFrame() 
symbol_attributes_df = pd.DataFrame()
df_fon_table = pd.DataFrame()
combined_symbol_data = pd.DataFrame()

# Turkish sorting function
def turkish_sort(x):
    import locale
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    return locale.strxfrm(x)

@st.cache_data
def fetch_data():
    df_fon_table = pd.read_csv('data/fon_table.csv')
    df_transformed = pd.read_csv('data/tefas_transformed.csv')
    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('symbol_')]
    symbol_attributes_list = np.array([col.replace('symbol_', '') for col in symbol_attributes_of_fon_table])
    symbol_attributes_list = sorted(symbol_attributes_list, key=turkish_sort)
    symbol_attributes_df = pd.DataFrame({'Fon Unvan Türü': symbol_attributes_list})
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
    ranks = combined_symbol_data[column_name].rank(method='min')
    max_rank = ranks.max()
    
    # Get the rank of the current value
    current_rank = ranks.loc[combined_symbol_data[column_name] == val].values[0]
    
    # Normalize the rank to [0, 1]
    norm_val = (current_rank - 1) / (max_rank - 1)  # Subtract 1 to make it 0-indexed
    
    # Ensure normalization is within [0, 1]
    norm_val = np.clip(norm_val, 0, 1)
    
    color = sns.color_palette("RdYlGn", as_cmap=True)(norm_val)
    return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}'

def filter_by_date_range(df, range_type):
    max_date = df['date'].max()
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
        min_date = df['date'].min()
    return df[(df['date'] >= min_date) & (df['date'] <= max_date)]

# Cumulative change plot function
def plot_cumulative_change(df_filtered, filtered_fons, title="Cumulative Price Change"):
    fig = go.Figure()

    # Loop through each selected symbol in filtered_fons
    for symbol in filtered_fons['symbol'].unique():
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
def display_buttons_for_chart(df_filtered, filtered_fons, chart_placeholder):
    button_labels = ["1m.", "3m.", "6m.", "1y.", "YTD.", "All."]
    cols = st.columns(len(button_labels))
    for i, label in enumerate(button_labels):
        if cols[i].button(label):
            filtered_df = filter_by_date_range(df_filtered, label)
            chart_placeholder.plotly_chart(plot_cumulative_change(filtered_df, filtered_fons, title=f"{label} Cumulative Price Change"))

# Display filter buttons horizontally and update chart
def display_buttons_for_fonlist():
    button_labels = ["1m", "3m", "6m", "1y", "YTD", "All"]
    cols = st.columns(8)
    for i, label in enumerate(button_labels):
        if cols[i].button(label):
            st.session_state.filter_label = label

# Fetch the data
df_merged, df_fon_table, symbol_attributes_df = fetch_data()

column_configuration_fon = {
    "symbol": st.column_config.TextColumn("Fon", help="Fon 3 haneli kod", width="short"),
    "title": st.column_config.TextColumn("Unvan", help="Fonun Unvanı", width="medium"),
    "1m-F%": st.column_config.TextColumn("1m-F%", help="1 aylık fiyat değişimi", width="short"),
    "1m-YS%": st.column_config.TextColumn("1m-YS%", help="1 aylık yatırımcı sayısı değişimi", width="short"),
    "1m-BY%": st.column_config.TextColumn("1m-BY%", help="1 aylık yatırımcı başına yatırım tutarı değişimi", width="short"),
}

col2, col3 = st.columns([6, 6])

with st.sidebar:
    with st.container():
        selectable_attributes = st.dataframe(symbol_attributes_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row")
        filtered_attributes = symbol_attributes_df.loc[selectable_attributes.selection.rows]

with col2:
    with st.container():
        if not filtered_attributes.empty:
            display_buttons_for_fonlist()
            combined_filtered_symbols = pd.DataFrame()  # Ensure this is defined
            
            # Collect symbols based on filtered attributes
            for filtered_attribute in filtered_attributes['Fon Unvan Türü']:
                filtered_symbols = df_merged[df_merged[f'symbol_{filtered_attribute}'] == True].reset_index(drop=True)
                if not filtered_symbols.empty:
                    combined_filtered_symbols = pd.concat([combined_filtered_symbols, filtered_symbols], ignore_index=True)
                    
            if not combined_filtered_symbols.empty:
                combined_filtered_symbols_to_list = combined_filtered_symbols[['symbol', 'title']].drop_duplicates().sort_values(by="symbol").reset_index(drop=True)
                combined_symbol_data = pd.DataFrame()  # Define this DataFrame for aggregation
                
                # Calculate 1-month price change percentage
                for symbol in combined_filtered_symbols_to_list['symbol'].unique():
                    symbol_data = df_merged[(df_merged['symbol'] == symbol) & (df_merged['date'] >= filter_get_min_date(st.session_state.filter_label))]
                    
                    if not symbol_data.empty:
                        symbol_data_sorted = symbol_data.sort_values('date')
                        start_close = symbol_data_sorted.iloc[0]['close']
                        end_close = symbol_data_sorted.iloc[-1]['close']
                        start_number_of_investors = symbol_data_sorted.iloc[0]['number_of_investors']
                        end_number_of_investors = symbol_data_sorted.iloc[-1]['number_of_investors']
                        start_market_cap_per_investors = symbol_data_sorted.iloc[0]['market_cap_per_investors']
                        end_market_cap_per_investors = symbol_data_sorted.iloc[-1]['market_cap_per_investors']
                        change_price = (end_close - start_close) / start_close * 100
                        change_number_of_investors = (end_number_of_investors - start_number_of_investors) / start_number_of_investors * 100
                        change_market_cap_per_investors = (end_market_cap_per_investors - start_market_cap_per_investors) / start_market_cap_per_investors * 100
                        
                        # Prepare a DataFrame with the result for each symbol
                        result_df = pd.DataFrame({
                            'symbol': [symbol],
                            'title': [symbol_data_sorted.iloc[0]['title']],
                            f'{st.session_state.filter_label}-F%': round(change_price, 2),
                            f'{st.session_state.filter_label}-YS%': round(change_number_of_investors, 2),
                            f'{st.session_state.filter_label}-BY%': round(change_market_cap_per_investors, 2)
                        })
                        combined_symbol_data = pd.concat([combined_symbol_data, result_df], ignore_index=True)
                
                # Ensure numeric columns
                combined_symbol_data[f'{st.session_state.filter_label}-F%'] = pd.to_numeric(combined_symbol_data[f'{st.session_state.filter_label}-F%'], errors='coerce')
                combined_symbol_data[f'{st.session_state.filter_label}-YS%'] = pd.to_numeric(combined_symbol_data[f'{st.session_state.filter_label}-YS%'], errors='coerce')
                combined_symbol_data[f'{st.session_state.filter_label}-BY%'] = pd.to_numeric(combined_symbol_data[f'{st.session_state.filter_label}-BY%'], errors='coerce')
                
                # Apply styling to DataFrame for each column
                styled_df = combined_symbol_data.style.applymap(lambda val: color_gradient(val, f'{st.session_state.filter_label}-F%'), subset=[f'{st.session_state.filter_label}-F%'])
                styled_df = styled_df.applymap(lambda val: color_gradient(val, f'{st.session_state.filter_label}-YS%'), subset=[f'{st.session_state.filter_label}-YS%'])
                styled_df = styled_df.applymap(lambda val: color_gradient(val, f'{st.session_state.filter_label}-BY%'), subset=[f'{st.session_state.filter_label}-BY%'])
                selectable_symbols = st.dataframe(styled_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row", column_config=column_configuration_fon)
                filtered_symbols = combined_filtered_symbols_to_list.loc[selectable_symbols.selection.rows]

with col3:
    with st.container():
        if not filtered_symbols.empty:
            # Create a placeholder for the chart
            chart_placeholder = st.empty()
            # Display buttons and update chart in the placeholder
            display_buttons_for_chart(df_merged, filtered_symbols, chart_placeholder)
            # Initially render the chart with all data
            chart_placeholder.plotly_chart(plot_cumulative_change(df_merged, filtered_symbols))
