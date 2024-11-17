import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import os

symbol_attributes_df = pd.DataFrame()
df_combined_symbol_history = pd.DataFrame()
set_filtered_symbols = set()

def rerun():
    st.rerun()

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
    symbol_attributes_list = [col.replace('symbol_', '') for col in symbol_attributes_of_fon_table]
    symbol_attributes_list = sorted(symbol_attributes_list, key=turkish_sort)
    symbol_attributes_df = pd.DataFrame({'Fon Unvan Türü': symbol_attributes_list})

    if 'df_merged' in st.session_state :
        df_merged = st.session_state.df_merged 
    else : 
        df_merged = pd.merge(df_transformed, df_fon_table, on='symbol', how='inner')
        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

    return df_merged, df_fon_table, symbol_attributes_df

if 'filter_label' not in st.session_state:
    st.session_state.filter_label = "1m"

# Date range filtering
def filter_get_min_date(range_type):
    max_date = df_merged['date'].max()
    if range_type == "1m":
        min_date = max_date - pd.DateOffset(months=1)
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

def color_gradient(val, column_name):
    if pd.isna(val) or pd.isnull(val):  # Exclude NaN and inf values
        return ''

    ranks = df_combined_symbol_metrics[column_name].rank(method='min')  # Get the ranks of the values in the specified column
    max_rank = ranks.max()
    
    try:
        current_rank = ranks[df_combined_symbol_metrics[column_name] == val].iloc[0]  # Get the rank of the current value
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
        norm_val = val / 40  # Normalize in [0, 1] range for green gradient
        color = sns.color_palette("Greens", as_cmap=True)(norm_val)
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}; color: white; font-weight: bold;'
    
    elif val > 70:  # Values above 70 should be red
        norm_val = (val - 40) / 30  # Normalize in [0, 1] range for gradient
        color = sns.color_palette("RdYlGn", as_cmap=True)(1 - norm_val)  # Green to Red gradient
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}; color: white; font-weight: bold;'
    
    else:  # Values between 40 and 70 should transition from green to red
        norm_val = (val - 40) / 30  # Normalize in [0, 1] range for gradient
        color = sns.color_palette("RdYlGn", as_cmap=True)(1 - norm_val)  # Green to Red gradient
        return f'background-color: rgba{tuple(int(c * 255) for c in color[:3])}; color: black; font-weight: normal;'

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
    button_labels = ["1m", "3m", "6m", "1y", "YTD", "All"]
    cols = st.columns(len(button_labels))
    for i, label in enumerate(button_labels):
        if cols[i].button(label):
            st.session_state.filter_label = label
            
df_merged, df_fon_table, symbol_attributes_df = fetch_data()

lv_time_range = st.session_state.filter_label

column_configuration_fon = {
    "symbol"              : st.column_config.TextColumn("Fon", help="Fon Kodu", width="small"),
    "title"               : st.column_config.TextColumn("Unvan", help="Fonun Ünvanı", width="large"),
    f'{lv_time_range}-F%' : st.column_config.NumberColumn(f'{lv_time_range}-F%', help="Fiyat değişimi yüzdesi", width="small"),
    f'{lv_time_range}-YS%': st.column_config.NumberColumn(f'{lv_time_range}-YS%', help="Yatırımcı sayısı değişimi yüzdesi", width="small"),
    f'{lv_time_range}-YS' : st.column_config.NumberColumn(f'{lv_time_range}-YS', help="Güncel Yatırımcı sayısı", width="small"),
    f'{lv_time_range}-BY%': st.column_config.NumberColumn(f'{lv_time_range}-BY%', help="Yatırımcı başına yatırım değişimi yüzdesi", width="small"),
    f'{lv_time_range}-BY' : st.column_config.NumberColumn(f'{lv_time_range}-BY', help="Güncel Yatırımcı başına yatırım tutarı", width="small"),
    f'{lv_time_range}-BYΔ': st.column_config.NumberColumn(f'{lv_time_range}-BYΔ', help="Yatırımcı başına yatırım tutarı değişimi", width="small"),
    f'RSI-14'             : st.column_config.NumberColumn(f'RSI-14', help="Güncel RSI", width="small"),
}

col2, col3 = st.columns([9, 6])

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
            df_combined_symbol_metrics_list = []
            lv_time_range = st.session_state.filter_label

            if not filtered_attributes.empty:    
                for filtered_attribute in filtered_attributes['Fon Unvan Türü']:
                    df_filtered_symbols = df_merged[df_merged[f'symbol_{filtered_attribute}'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.update(df_filtered_symbols)
                if no_show_ozel: 
                    df_ozel_symbols = df_merged[df_merged[f'symbol_Özel'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.difference_update(df_ozel_symbols)
                if no_show_serbest: 
                    df_serbest_symbols = df_merged[df_merged[f'symbol_Serbest'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.difference_update(df_serbest_symbols)

            for symbol in set_filtered_symbols:
                df_symbol_history = df_merged[(df_merged['symbol'] == symbol) & (df_merged['date'] >= filter_get_min_date(lv_time_range))].copy()
                df_symbol_history_list.append(df_symbol_history)

                if not df_symbol_history.empty:
                    df_symbol_history.sort_values('date', inplace=True, ascending=True)
                    end_close                       = df_symbol_history.iloc[-1]['close']
                    if end_close == 0 : 
                        continue

                    start_close                     = df_symbol_history.iloc[0]['close']
                    if start_close == 0 : 
                        continue
                
                    start_number_of_investors       = df_symbol_history.iloc[0]['number_of_investors']
                    end_number_of_investors         = df_symbol_history.iloc[-1]['number_of_investors']
                    start_market_cap_per_investors  = df_symbol_history.iloc[0]['market_cap_per_investors']
                    end_market_cap_per_investors    = df_symbol_history.iloc[-1]['market_cap_per_investors']
                    rsi_14                          = df_symbol_history.iloc[-1]['RSI_14']

                    if pd.notnull(start_close) and pd.notnull(end_close) and start_close != 0:
                        change_price = (end_close - start_close) / start_close * 100
                    else:
                        change_price = float('nan') 

                    if pd.notnull(start_number_of_investors) and pd.notnull(end_number_of_investors) and start_number_of_investors != 0:
                        change_number_of_investors_percent = (end_number_of_investors - start_number_of_investors) / start_number_of_investors * 100
                        change_number_of_investors = (end_number_of_investors - start_number_of_investors) 
                    else:
                        change_number_of_investors_percent = float('nan')
                        change_number_of_investors = float('nan')

                    if pd.notnull(start_market_cap_per_investors) and pd.notnull(end_market_cap_per_investors) and start_market_cap_per_investors != 0:
                        change_market_cap_per_investors_percent = (end_market_cap_per_investors - start_market_cap_per_investors) / start_market_cap_per_investors * 100
                        change_market_cap_per_investors = (end_market_cap_per_investors - start_market_cap_per_investors)
                    else:
                        change_market_cap_per_investors_percent = float('nan')
                        change_market_cap_per_investors = float('nan')
                    
                    df_symbol_metrics = pd.DataFrame({
                        'symbol'              : [symbol],
                        'title'               : [df_symbol_history.iloc[0]['title']],
                        f'{lv_time_range}-F%' : change_price,
                        f'{lv_time_range}-YS%': change_number_of_investors_percent,
                        f'{lv_time_range}-BY%': change_market_cap_per_investors_percent,
                        f'{lv_time_range}-YS' : end_number_of_investors,
                        f'{lv_time_range}-BY' : end_market_cap_per_investors,
                        f'{lv_time_range}-BYΔ': change_market_cap_per_investors,
                        f'RSI-14'             : rsi_14
                    })
                    df_combined_symbol_metrics_list.append(df_symbol_metrics) 
            
            if df_combined_symbol_metrics_list:  
                df_combined_symbol_metrics = pd.concat(df_combined_symbol_metrics_list, ignore_index=True)
            else:
                df_combined_symbol_metrics = pd.DataFrame()
    
            df_combined_symbol_metrics_list = None

            if df_symbol_history_list:  
                df_combined_symbol_history = pd.concat(df_symbol_history_list, ignore_index=True)
            else:
                df_combined_symbol_history = pd.DataFrame()

            df_symbol_history = None
            df_symbol_history_list = None
            
            styled_df = df_combined_symbol_metrics.style
            styled_df = styled_df.format({f'{lv_time_range}-F%': '{:.2f}', 
                                          f'{lv_time_range}-YS%': '{:.2f}', 
                                          f'{lv_time_range}-BY%': '{:.2f}', 
                                          f'{lv_time_range}-YS': '{:,.0f}', 
                                          f'{lv_time_range}-BY': '₺ {:,.0f}' , 
                                          f'{lv_time_range}-BYΔ': '₺ {:,.0f}' , 
                                          'RSI-14': '{:,.2f}' })
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-F%') if pd.notnull(val) else '', subset=[f'{lv_time_range}-F%'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-YS%') if pd.notnull(val) else '', subset=[f'{lv_time_range}-YS%'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-BY%') if pd.notnull(val) else '', subset=[f'{lv_time_range}-BY%'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-YS') if pd.notnull(val) else '', subset=[f'{lv_time_range}-YS'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-BY') if pd.notnull(val) else '', subset=[f'{lv_time_range}-BY'])
            styled_df = styled_df.map(lambda val: color_gradient(val, f'{lv_time_range}-BYΔ') if pd.notnull(val) else '', subset=[f'{lv_time_range}-BYΔ'])
            styled_df = styled_df.map(lambda val: RSI_gradient(val) if pd.notnull(val) else '', subset=['RSI-14'])

            if not df_combined_symbol_metrics.empty:
                selectable_symbols = st.dataframe(styled_df, use_container_width=True, hide_index=True, height=800, on_select="rerun", selection_mode="multi-row", column_config=column_configuration_fon)
            
                styled_df = None

                set_filtered_symbols.clear()
                for selected_symbol in selectable_symbols.selection.rows:
                    symbol_value = df_combined_symbol_metrics.loc[selected_symbol]['symbol']
                    set_filtered_symbols.update([symbol_value]) 
                    
with col3:
    with st.container():
        chart_placeholder = st.empty()
        if set_filtered_symbols :
            chart_placeholder.plotly_chart(plot_cumulative_change(df_combined_symbol_history, set_filtered_symbols))
