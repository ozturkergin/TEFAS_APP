import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import os
import json
from st_aggrid import AgGrid

symbol_attributes_df = pd.DataFrame()
set_filtered_symbols = set()

def rerun():
    st.rerun()

# Turkish sorting function
def turkish_sort(x):
    import locale
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    return locale.strxfrm(x)

# @st.cache_data
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
            # st.write("tefas_transformed Analiz.py'de okundu session ile")
        else : 
            df_transformed = pd.read_csv('data/tefas_transformed.csv', parse_dates=['date'])
            st.session_state.df_transformed = df_transformed
            # st.write("tefas_transformed Analiz.py'de dosyadan okundu")

    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('FundType_')]
    symbol_attributes_list = [col.replace('FundType_', '') for col in symbol_attributes_of_fon_table]
    symbol_attributes_list = sorted(symbol_attributes_list, key=turkish_sort)
    symbol_attributes_df = pd.DataFrame({'Fon Unvan Türü': symbol_attributes_list})
    recent_date = df_transformed['date'].max()
    df_transformed_recent = df_transformed[(df_transformed['date'] == recent_date)]
    df_merged = pd.merge(df_transformed_recent, df_fon_table, on='symbol', how='inner')
    df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

    return df_merged, df_fon_table, symbol_attributes_df

if 'filter_label' not in st.session_state:
    st.session_state.filter_label = "1m"

def color_gradient(val, column_name):
    if pd.isna(val) or pd.isnull(val):  # Exclude NaN and inf values
        return ''

    ranks = df_symbol_metrics[column_name].rank(method='min')  # Get the ranks of the values in the specified column
    max_rank = ranks.max()
    
    try:
        current_rank = ranks[df_symbol_metrics[column_name] == val].iloc[0]  # Get the rank of the current value
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

def shorten_hyperlink(val):
    try:
        # Create HTML formatted link with JavaScript onclick
        return f'<a href="https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={val}" target="_blank" onclick="alert(\'Clicked on {val}\')">{val}</a>'
    except Exception as e:
        return val

# Load weights from config.json
config_file_path = "config.json"
with open(config_file_path, "r") as file:
    config = json.load(file)

weights = config["weights"]

# Cumulative change plot function
def plot_cumulative_change(df_filtered, set_filtered_symbols, title=""):
    fig = go.Figure()

    for symbol in set_filtered_symbols: # Loop through each selected symbol in filtered_fons
        symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
        symbol_data.loc[:,'price_change_pct'] = symbol_data['close'].pct_change().fillna(0) # Calculate percentage change and cumulative change
        symbol_data.loc[:,'cumulative_change'] = (1 + symbol_data['price_change_pct']).cumprod() - 1
        
        fig.add_trace( # Add the symbol's cumulative change to the figure
            go.Scatter(
                x=symbol_data['date'], 
                y=symbol_data['cumulative_change'],
                mode='lines',
                name=symbol
            )
        )

    fig.update_layout( # Update the layout with rangeslider, rangeselector, and custom plot height
        title=title,
        height=600,  # Adjust the height here
        xaxis_title="Date",
        yaxis_title="Cumulative Price Change (%)",
        xaxis=dict(
            rangeslider_visible=True
        ),
        yaxis=dict(
            tickformat=".2%" ,  # Display y-axis as percentages
        )
    )
    return fig

def display_buttons():
    button_labels = ["7d", "1m", "3m", "6m", "1y", "3y"]
    cols = st.columns(len(button_labels))
    for i, label in enumerate(button_labels):
        if cols[i].button(label):
            st.session_state.filter_label = label
            
df_merged, df_fon_table, symbol_attributes_df = fetch_data()

lv_time_range = st.session_state.filter_label

column_configuration_fon = {
    "symbollink"          : st.column_config.LinkColumn("Link", help="Link", width="small", display_text="🔗"),
    "symbol"              : st.column_config.TextColumn("Fon", help="Fon Kodu", width="small"),
    "title"               : st.column_config.TextColumn("Unvan", help="Fonun Ünvanı", width="large"),
    f'Skor'               : st.column_config.NumberColumn(f'Skor', help="Sıralama", width="small"),
    f'{lv_time_range}-F%' : st.column_config.NumberColumn(f'{lv_time_range}-F%', help="Fiyat değişimi yüzdesi", width="small"),
    f'{lv_time_range}-YS%': st.column_config.NumberColumn(f'{lv_time_range}-YS%', help="Yatırımcı sayısı değişimi yüzdesi", width="small"),
    f'YS' : st.column_config.NumberColumn(f'YS', help="Güncel Yatırımcı sayısı", width="small"),
    f'{lv_time_range}-BY%': st.column_config.NumberColumn(f'{lv_time_range}-BY%', help="Yatırımcı başına yatırım değişimi yüzdesi", width="small"),
    f'BY' : st.column_config.NumberColumn(f'BY', help="Güncel Yatırımcı başına yatırım tutarı", width="small"),
    f'{lv_time_range}-BYΔ': st.column_config.NumberColumn(f'{lv_time_range}-BYΔ', help="Yatırımcı başına yatırım tutarı değişimi", width="small"),
    f'RSI_14'             : st.column_config.NumberColumn(f'RSI_14', help="Güncel RSI", width="small"),
}

col2, col3 = st.columns([10, 6])

with st.sidebar:
    with st.container():
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            show_favourites = st.checkbox("Favorilerim", key="Favorilerim")
        with row1_col2:
            show_portfolio = st.checkbox("Portföyüm", key="Portföyüm")
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            no_show_ozel = st.checkbox("🚷Özel", key="Özel", value=True, help="Özel Fonlar Hariç")
        with row2_col2:
            no_show_serbest = st.checkbox("🚫Serbest", key="Serbest", value=True, help="Serbest Fonlar Hariç")

        selectable_attributes = st.dataframe(symbol_attributes_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row")
        filtered_attributes   = symbol_attributes_df.loc[selectable_attributes.selection.rows]

with col2:
    with st.container():
        display_buttons() 

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

        if not filtered_attributes.empty or set_filtered_symbols:
            df_symbol_history_list = []
            lv_time_range = st.session_state.filter_label

            if not filtered_attributes.empty:    
                for filtered_attribute in filtered_attributes['Fon Unvan Türü']:
                    df_filtered_symbols = df_fon_table[df_fon_table[f'FundType_{filtered_attribute}'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.update(df_filtered_symbols)
                if no_show_ozel: 
                    df_ozel_symbols = df_fon_table[df_fon_table[f'FundType_Özel'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.difference_update(df_ozel_symbols)
                if no_show_serbest: 
                    df_serbest_symbols = df_fon_table[df_fon_table[f'FundType_Serbest'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.difference_update(df_serbest_symbols)
            
            recent_date = df_merged['date'].max()

            df_symbol_history = df_merged[(df_merged['symbol'].isin(set_filtered_symbols))].copy()

            # Calculate price change for each period
            df_symbol_history['7d-F%'] = (df_symbol_history['close'] - df_symbol_history['close_7d']) / df_symbol_history['close_7d'] * 100
            df_symbol_history['1m-F%'] = (df_symbol_history['close'] - df_symbol_history['close_1m']) / df_symbol_history['close_1m'] * 100
            df_symbol_history['3m-F%'] = (df_symbol_history['close'] - df_symbol_history['close_3m']) / df_symbol_history['close_3m'] * 100
            df_symbol_history['6m-F%'] = (df_symbol_history['close'] - df_symbol_history['close_6m']) / df_symbol_history['close_6m'] * 100
            df_symbol_history['1y-F%'] = (df_symbol_history['close'] - df_symbol_history['close_1y']) / df_symbol_history['close_1y'] * 100
            df_symbol_history['3y-F%'] = (df_symbol_history['close'] - df_symbol_history['close_3y']) / df_symbol_history['close_3y'] * 100

            # Assign ranks based on price changes
            df_symbol_history['7d-Rank'] = df_symbol_history['7d-F%'].rank(ascending=False, method='min')
            df_symbol_history['1m-Rank'] = df_symbol_history['1m-F%'].rank(ascending=False, method='min')
            df_symbol_history['3m-Rank'] = df_symbol_history['3m-F%'].rank(ascending=False, method='min')
            df_symbol_history['6m-Rank'] = df_symbol_history['6m-F%'].rank(ascending=False, method='min')
            df_symbol_history['1y-Rank'] = df_symbol_history['1y-F%'].rank(ascending=False, method='min')
            df_symbol_history['3y-Rank'] = df_symbol_history['3y-F%'].rank(ascending=False, method='min')

            df_symbol_metrics = pd.DataFrame()
            df_symbol_metrics["symbol"] = df_symbol_history["symbol"] 
            df_symbol_metrics["title"]  = df_symbol_history["title"] 
            df_symbol_metrics[f'{lv_time_range}-F%'] = ( df_symbol_history[f'close'] - df_symbol_history[f'close_{lv_time_range}'] ) / df_symbol_history[f'close_{lv_time_range}'] * 100
            df_symbol_metrics[f'{lv_time_range}-YS%'] = ( df_symbol_history[f'number_of_investors'] - df_symbol_history[f'number_of_investors_{lv_time_range}'] ) / df_symbol_history[f'number_of_investors_{lv_time_range}'] * 100
            df_symbol_metrics[f'{lv_time_range}-BY%'] = ( df_symbol_history[f'market_cap_per_investors'] - df_symbol_history[f'market_cap_per_investors_{lv_time_range}'] ) / df_symbol_history[f'market_cap_per_investors_{lv_time_range}'] * 100
            df_symbol_metrics[f'YS'] = df_symbol_history[f'number_of_investors'] 
            df_symbol_metrics[f'BY'] = df_symbol_history[f'market_cap_per_investors'] 
            df_symbol_metrics[f'{lv_time_range}-BYΔ'] = ( df_symbol_history[f'market_cap_per_investors'] - df_symbol_history[f'market_cap_per_investors_{lv_time_range}'] ) 
            df_symbol_metrics["RSI_14"] = df_symbol_history[f'RSI_14'] 
            # Calculate the weighted score
            weighted_sum = (
                df_symbol_history['7d-Rank'] * weights['7d'] +
                df_symbol_history['1m-Rank'] * weights['1m'] +
                df_symbol_history['3m-Rank'] * weights['3m'] +
                df_symbol_history['6m-Rank'] * weights['6m'] +
                df_symbol_history['1y-Rank'] * weights['1y'] +
                df_symbol_history['3y-Rank'] * weights['3y']
            )

            # Calculate the sum of weights for non-null values
            weights_sum = (
                df_symbol_history['7d-Rank'].notnull().astype(int) * weights['7d'] +
                df_symbol_history['1m-Rank'].notnull().astype(int) * weights['1m'] +
                df_symbol_history['3m-Rank'].notnull().astype(int) * weights['3m'] +
                df_symbol_history['6m-Rank'].notnull().astype(int) * weights['6m'] +
                df_symbol_history['1y-Rank'].notnull().astype(int) * weights['1y'] +
                df_symbol_history['3y-Rank'].notnull().astype(int) * weights['3y']
            )

            # Calculate the average score for non-null values
            df_symbol_metrics["Skor"] = ( weighted_sum / weights_sum ).rank(ascending=True, method='min')
            # df_symbol_metrics["Skor"] = ( weighted_sum / weights_sum )
            df_symbol_metrics["symbollink"] = "https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod=" + df_symbol_history["symbol"] 
 
            styled_df = df_symbol_metrics.style
            styled_df = styled_df.format({f'{lv_time_range}-F%': '{:.2f}', 
                                          f'{lv_time_range}-YS%': '{:.2f}', 
                                          f'{lv_time_range}-BY%': '{:.2f}', 
                                          f'YS': '{:,.0f}', 
                                          f'Skor': '{:,.0f}', 
                                          f'BY': '₺ {:,.0f}' , 
                                          f'{lv_time_range}-BYΔ': '₺ {:,.0f}' , 
                                          'RSI_14': '{:,.2f}' })
            
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-F%') if pd.notnull(val) else '', subset=[f'{lv_time_range}-F%'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-YS%') if pd.notnull(val) else '', subset=[f'{lv_time_range}-YS%'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-BY%') if pd.notnull(val) else '', subset=[f'{lv_time_range}-BY%'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'YS') if pd.notnull(val) else '', subset=[f'YS'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'BY') if pd.notnull(val) else '', subset=[f'BY'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-BYΔ') if pd.notnull(val) else '', subset=[f'{lv_time_range}-BYΔ'])
            styled_df = styled_df.map(lambda val: RSI_gradient(val) if pd.notnull(val) else '', subset=['RSI_14'])

            if not df_symbol_metrics.empty:
                selectable_symbols = st.dataframe(styled_df, use_container_width=True, hide_index=True, height=800, on_select="rerun", selection_mode="multi-row", column_config=column_configuration_fon)
                set_filtered_symbols.clear()
                for selected_symbol_index in selectable_symbols.selection.rows:
                    selected_symbol = df_symbol_metrics.iloc[selected_symbol_index]['symbol']
                    set_filtered_symbols.add(selected_symbol)
                    
with col3:
    with st.container():
        chart_placeholder = st.empty()
        if set_filtered_symbols :
            chart_placeholder.plotly_chart(plot_cumulative_change(st.session_state.df_transformed, set_filtered_symbols))
