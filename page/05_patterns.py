import streamlit as st
import pandas as pd
import os
import talib as ta

set_filtered_symbols = set()

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
            df_transformed = pd.read_csv('data/tefas_transformed.csv', parse_dates=['date'])
            st.session_state.df_transformed = df_transformed

    if 'df_merged' in st.session_state:
        df_merged = st.session_state.df_merged
    else:
        df_merged = pd.merge(df_transformed, df_fon_table, on='symbol', how='inner')
        df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

    return df_merged, df_fon_table

df_merged, df_fon_table = fetch_data()

# Define the patterns dictionary
patterns = {
'CDL2CROWS':'Two Crows',
'CDL3BLACKCROWS':'Three Black Crows',
'CDL3INSIDE':'Three Inside Up/Down',
'CDL3LINESTRIKE':'Three-Line Strike ',
'CDL3OUTSIDE':'Three Outside Up/Down',
'CDL3STARSINSOUTH':'Three Stars In The South',
'CDL3WHITESOLDIERS':'Three Advancing White Soldiers',
'CDLABANDONEDBABY':'Abandoned Baby',
'CDLADVANCEBLOCK':'Advance Block',
'CDLBELTHOLD':'Belt-hold',
'CDLBREAKAWAY':'Breakaway',
'CDLCLOSINGMARUBOZU':'Closing Marubozu',
'CDLCONCEALBABYSWALL':'Concealing Baby Swallow',
'CDLCOUNTERATTACK':'Counterattack',
'CDLDARKCLOUDCOVER':'Dark Cloud Cover',
'CDLDOJI':'Doji',
'CDLDOJISTAR':'Doji Star',
'CDLDRAGONFLYDOJI':'Dragonfly Doji',
'CDLENGULFING':'Engulfing Pattern',
'CDLEVENINGDOJISTAR':'Evening Doji Star',
'CDLEVENINGSTAR':'Evening Star',
'CDLGAPSIDESIDEWHITE':'Up/Down-gap side-by-side white lines',
'CDLGRAVESTONEDOJI':'Gravestone Doji',
'CDLHAMMER':'Hammer',
'CDLHANGINGMAN':'Hanging Man',
'CDLHARAMI':'Harami Pattern',
'CDLHARAMICROSS':'Harami Cross Pattern',
'CDLHIGHWAVE':'High-Wave Candle',
'CDLHIKKAKE':'Hikkake Pattern',
'CDLHIKKAKEMOD':'Modified Hikkake Pattern',
'CDLHOMINGPIGEON':'Homing Pigeon',
'CDLIDENTICAL3CROWS':'Identical Three Crows',
'CDLINNECK':'In-Neck Pattern',
'CDLINVERTEDHAMMER':'Inverted Hammer',
'CDLKICKING':'Kicking',
'CDLKICKINGBYLENGTH':'Kicking-bull/bear determined by the longer marubozu',
'CDLLADDERBOTTOM':'Ladder Bottom',
'CDLLONGLEGGEDDOJI':'Long Legged Doji',
'CDLLONGLINE':'Long Line Candle',
'CDLMARUBOZU':'Marubozu',
'CDLMATCHINGLOW':'Matching Low',
'CDLMATHOLD':'Mat Hold',
'CDLMORNINGDOJISTAR':'Morning Doji Star',
'CDLMORNINGSTAR':'Morning Star',
'CDLONNECK':'On-Neck Pattern',
'CDLPIERCING':'Piercing Pattern',
'CDLRICKSHAWMAN':'Rickshaw Man',
'CDLRISEFALL3METHODS':'Rising/Falling Three Methods',
'CDLSEPARATINGLINES':'Separating Lines',
'CDLSHOOTINGSTAR':'Shooting Star',
'CDLSHORTLINE':'Short Line Candle',
'CDLSPINNINGTOP':'Spinning Top',
'CDLSTALLEDPATTERN':'Stalled Pattern',
'CDLSTICKSANDWICH':'Stick Sandwich',
'CDLTAKURI':'Takuri (Dragonfly Doji with very long lower shadow)',
'CDLTASUKIGAP':'Tasuki Gap',
'CDLTHRUSTING':'Thrusting Pattern',
'CDLTRISTAR':'Tristar Pattern',
'CDLUNIQUE3RIVER':'Unique 3 River',
'CDLUPSIDEGAP2CROWS':'Upside Gap Two Crows',
'CDLXSIDEGAP3METHODS':'Upside/Downside Gap Three Methods'
}

with st.sidebar:
    with st.container():
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            show_favourites = st.checkbox("Favorilerim", key="Favorilerim")
        with row1_col2:
            show_portfolio = st.checkbox("Portföyüm", key="Portföyüm", value=True)

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

# Convert the patterns dictionary to a DataFrame
pattern_ids = [(pid, desc) for pid, desc in patterns.items() if pid != 'ALL']
columns = []
symbol_data = {}

# Convert set to list and sort
set_filtered_symbols = sorted(list(set_filtered_symbols))

for symbol in set_filtered_symbols:
    data = df_merged[df_merged['symbol'] == symbol]
    title = data.iloc[0]['title']
    col_name = f"{symbol}"
    columns.append(col_name)
    
    # Calculate patterns for this symbol
    symbol_patterns = {}
    for pattern_id, pattern_desc in pattern_ids:
        pattern_function = getattr(ta, pattern_id)
        result = pattern_function(data['open'], data['high'], data['low'], data['close'])
        symbol_patterns[pattern_desc] = result.iloc[-1]  # Get last value
    
    symbol_data[col_name] = symbol_patterns

# Create DataFrame using descriptions as index
df_results = pd.DataFrame.from_dict(symbol_data, orient='columns')
df_results.index = [desc for _, desc in pattern_ids]  # Use descriptions as index

# Style the DataFrame
def style_arrow(val):
    if pd.isna(val):
        return ''
    elif val > 0:
        return '🢅'  # Green up arrow
    elif val < 0:
        return '🢆'  # Red down arrow
    else:
        return ''

def color_arrows(val):
    if pd.isna(val):
        return ''
    elif val > 0:
        return 'color: green'
    elif val < 0:
        return 'color: red'
    else:
        return ''

styled_df = df_results.style.format(style_arrow).map(color_arrows) 

# Initialize column configuration dictionary
column_configuration_fon = {}

# Loop through sorted symbols and create column config
for symbol in sorted(list(set_filtered_symbols)):
    # Get title for the symbol from df_merged
    symbol_title = df_fon_table[df_fon_table['symbol'] == symbol].iloc[0]['title']
    
    # Add column configuration for this symbol
    column_configuration_fon[symbol] = st.column_config.NumberColumn(
        symbol,
        help=symbol_title,
        width="small"
    )

# Display in Streamlit
dataframe_height = (len(df_results) + 1) * 35 + 2
st.dataframe(styled_df, use_container_width=True, column_config=column_configuration_fon, height=800)
