import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime, timedelta

symbol_attributes_df = pd.DataFrame()
# Turkish sorting function
def turkish_sort(x):
    import locale
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    return locale.strxfrm(x)

# Load unique symbols and titles from 'fon_table.csv'
if os.path.exists('data/fon_table.csv'):
    if 'df_fon_table' in st.session_state:
        df_fon_table = st.session_state.df_fon_table
    else:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table

    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('FundType_')]
    symbol_attributes_list = [col.replace('FundType_', '') for col in symbol_attributes_of_fon_table]
    symbol_attributes_list = sorted(symbol_attributes_list, key=turkish_sort)
    symbol_attributes_df = pd.DataFrame({'Fon Unvan Türü': symbol_attributes_list})
else:
    st.error("The file 'fon_table.csv' is missing.")
    st.stop()

set_filtered_symbols = set()

with st.sidebar:
    with st.container():
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            show_favourites = st.checkbox("Favorilerim", key="Favorilerim")
        with row1_col2:
            show_portfolio = st.checkbox("Portföyüm", key="Portföyüm", value=True)

        selectable_attributes = st.dataframe(symbol_attributes_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row")
        filtered_attributes   = symbol_attributes_df.loc[selectable_attributes.selection.rows]

        if not filtered_attributes.empty or set_filtered_symbols:
            df_symbol_history_list = []

            if not filtered_attributes.empty:    
                for filtered_attribute in filtered_attributes['Fon Unvan Türü']:
                    df_filtered_symbols = df_fon_table[df_fon_table[f'FundType_{filtered_attribute}'] == True]['symbol'].unique().tolist()
                    set_filtered_symbols.update(df_filtered_symbols)

        if show_favourites:
            if 'favourites' in st.session_state:
                set_filtered_symbols.update(st.session_state.favourites)

        if show_portfolio:
            if 'myportfolio' in st.session_state:
                myportfolio_summarized = (st.session_state.myportfolio
                                          .groupby('symbol', as_index=False)
                                          .apply(lambda df: pd.Series(
                                              {'net_quantity': df.loc[df['transaction_type'] == 'buy', 'quantity'].sum() - df.loc[
                                                  df['transaction_type'] == 'sell', 'quantity'].sum()}))
                                          .query('net_quantity != 0'))  # Keep only symbols with non-zero net quantity
                set_filtered_symbols.update(myportfolio_summarized['symbol'].unique().tolist())

# Ensure 'df_transformed' exists in session_state
if 'df_transformed' in st.session_state:
    # Filter and prepare data for the selected symbols
    df_raw = st.session_state.df_transformed[(st.session_state.df_transformed['symbol'].isin(set_filtered_symbols))].copy()

    if df_raw.empty:
        st.warning(f"No data available for symbols: {set_filtered_symbols}")
    else:
        # Pivot the data to have symbols as columns and dates as rows
        df_pivot = df_raw.pivot(index='date', columns='symbol', values='close').reset_index()
        df_pivot['date'] = pd.to_datetime(df_pivot['date']).dt.strftime('%Y-%m-%d')  # Format date as YYYY-MM-DD
        df_pivot.sort_values('date', inplace=True)  # Sort by date
        dates = df_pivot['date'].tolist()  # Get the list of dates for the slider
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

        tab1, tab2 = st.tabs(["📈 Karşılaştırma", ""])
        # Add a Streamlit slider for selecting start and end dates using indices
        start_date, end_date = tab1.slider(
            "Select date range",
            min_value=dates[0],  # Default to the earliest date
            max_value=dates[-1],  # Default to the latest date
            value=(dates[0], dates[-1]),  # Default to full range
            format="YYYY-MM-DD",  # Format for display
            key="date_range_slider",
            label_visibility="visible",
        )
        start_date_f = start_date.strftime('%Y-%m-%d')
        end_date_f = end_date.strftime('%Y-%m-%d')
        start_date_f_buffer = start_date - timedelta(days=15)
        start_date_f_buffer_str = start_date_f_buffer.strftime('%Y-%m-%d')
        # Create a complete date range
        date_range = pd.date_range(start=start_date_f_buffer, end=end_date)
        # Prepare data for each symbol for the chart
        symbols_data = {}
        for symbol in set_filtered_symbols:
            df_symbol = df_raw[df_raw['symbol'] == symbol][['date', 'close']].copy()
            df_symbol.rename(columns={'date': 'time'}, inplace=True)
            df_symbol.set_index('time', inplace=True)
            df_symbol = df_symbol.reindex(date_range).ffill().reset_index()
            df_symbol.rename(columns={'index': 'time'}, inplace=True)
            df_symbol['time'] = df_symbol['time'].dt.strftime('%Y-%m-%d') 
            df_symbol = df_symbol[(df_symbol['time'] >= start_date_f) & (df_symbol['time'] <= end_date_f)]

            # Handle missing data at the start of the range
            if df_symbol.empty or df_symbol['time'].min() > start_date_f:
                df_symbol = pd.concat([
                    pd.DataFrame({'time': [start_date_f], 'close': [0]}),  # Add a starting point with 0
                    df_symbol
                ])

            # Calculate cumulative gains
            base_value = df_symbol['close'].iloc[0] if not df_symbol.empty else 0
            df_symbol['cumulative_gain'] = ((df_symbol['close'] - base_value) / base_value) * 100 if base_value != 0 else 0

            symbols_data[symbol] = df_symbol[['time', 'cumulative_gain']].rename(columns={'cumulative_gain': 'value'}).to_dict(orient='records')

        # HTML/JavaScript for Lightweight Charts
        def render_lightweight_chart(symbols_data):
            return f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        #chart-container {{
            display: flex;
            width: 100%;
            height: 600px;
        }}
        #toolbox {{
            width: 200px;
            background: white;
            padding: 10px;
            border-right: 1px solid #e0e0e0;
        }}
        #chart {{
            flex: 1;
            position: relative;
        }}
        .symbol-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            cursor: pointer;
        }}
        .symbol-color {{
            width: 12px;
            height: 12px;
            margin-right: 8px;
            border-radius: 2px;
        }}
        .symbol-text {{
            font-size: 14px;
            font-family: Arial, sans-serif;
        }}
        .hidden {{
            opacity: 0.5;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e0e0e0;
            padding: 8px;
            font-size: 12px;
            font-family: Arial, sans-serif;
            pointer-events: none;
            display: none;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div id="chart-container">
        <div id="toolbox"></div>
        <div id="chart">
            <div id="tooltip" class="tooltip"></div>
        </div>
    </div>
    <script>
        try {{
            const chart = window.LightweightCharts.createChart(
                document.getElementById('chart'),
                {{
                    width: document.getElementById('chart').clientWidth,
                    height: 600,
                    layout: {{
                        background: {{ color: '#ffffff' }},
                        textColor: '#333'
                    }},
                    grid: {{
                        vertLines: {{ color: '#e0e0e0' }},
                        horzLines: {{ color: '#e0e0e0' }}
                    }},
                    crosshair: {{
                        mode: window.LightweightCharts.CrosshairMode.Normal,
                        vertLine: {{
                            width: 1,
                            color: '#758696',
                            style: 0
                        }},
                        horzLine: {{
                            width: 1,
                            color: '#758696',
                            style: 0
                        }}
                    }},
                    timeScale: {{
                        timeVisible: true,
                        secondsVisible: false,
                        borderColor: '#D1D4DC'
                    }}
                }}
            );

            const symbolsData = {json.dumps(symbols_data)};
            const lineSeriesMap = {{}};
            const colors = ['#2962FF', '#FF6D00', '#00C853', '#D50000', '#6200EA'];
            let colorIndex = 0;

            const toolbox = document.getElementById('toolbox');
            const tooltip = document.getElementById('tooltip');

            Object.entries(symbolsData).forEach(([symbol, data]) => {{
                const color = colors[colorIndex % colors.length];
                const series = chart.addLineSeries({{
                    color: color,
                    lineWidth: 2,
                    priceFormat: {{
                        type: 'price',
                        precision: 2,
                        minMove: 0.01,
                    }}
                }});

                series.setData(data);
                lineSeriesMap[symbol] = series;

                const item = document.createElement('div');
                item.className = 'symbol-item';
                item.innerHTML = `
                    <div class="symbol-color" style="background: ${{color}}"></div>
                    <div class="symbol-text">${{symbol}}</div>
                `;
                
                item.addEventListener('click', () => {{
                    const isHidden = item.classList.toggle('hidden');
                    series.applyOptions({{
                        visible: !isHidden
                    }});
                }});

                toolbox.appendChild(item);
                colorIndex++;
            }});

            // Track crosshair movement to display tooltip
            chart.subscribeCrosshairMove((param) => {{
                if (param.point === undefined || !param.time) {{
                    tooltip.style.display = 'none';
                    return;
                }}

                let tooltipContent = '';
                Object.entries(lineSeriesMap).forEach(([symbol, series]) => {{
                    const price = param.seriesPrices.get(series);
                    if (price !== undefined) {{
                        tooltipContent += `<div><strong>${{symbol}}</strong>: ${{price.toFixed(2)}}%</div>`;
                    }}
                }});

                if (tooltipContent) {{
                    tooltip.innerHTML = tooltipContent;
                    tooltip.style.display = 'block';
                    tooltip.style.left = param.point.x + 10 + 'px';
                    tooltip.style.top = param.point.y + 10 + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }});

            chart.timeScale().fitContent();

        }} catch (error) {{
            console.error('Chart error:', error);
        }}
    </script>
</body>
</html>
"""
        # Display the chart
        with tab1:
            chart_html = render_lightweight_chart(symbols_data)
            st.components.v1.html(chart_html, height=650, scrolling=True)
            df_show = df_pivot[(df_pivot['date'] >= start_date_f) & (df_pivot['date'] <= end_date_f)].sort_values(by="date", ascending=False).copy()
            st.dataframe(df_show, hide_index=True, height=600, selection_mode=["multi-row", "multi-column"])

else:
    st.error("No transformed data available in the session state.")