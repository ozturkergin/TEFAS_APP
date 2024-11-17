import streamlit as st
import pandas as pd
import os

# Load unique symbols and titles from 'fon_table.csv'
if os.path.exists('data/fon_table.csv'):
    if 'df_fon_table' in st.session_state:
        df_fon_table = st.session_state.df_fon_table
    else:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table
else:
    st.error("The file 'fon_table.csv' is missing.")
    st.stop()

unique_symbols = sorted(df_fon_table['symbol'].unique().tolist())
symbol_titles = df_fon_table.set_index('symbol')['title'].to_dict()

# Multiselect for selecting symbols
selected_symbols = st.multiselect( 
    '',
    unique_symbols,
    default=unique_symbols[0],  # Default selection
    key='selected_symbols',
    placeholder='Select Symbol:',
)

# Use only the first selected symbol
if selected_symbols:
    selected_symbol = selected_symbols[0]
else:
    st.warning("Please select at least one symbol.")
    st.stop()

# Ensure 'df_transformed' exists in session_state
if 'df_transformed' in st.session_state:
    # Filter and prepare data for the selected symbol
    df_raw = st.session_state.df_transformed[
        st.session_state.df_transformed['symbol'] == selected_symbol
    ].copy()

    if df_raw.empty:
        st.warning(f"No data available for symbol: {selected_symbol}")
    else:
        df = df_raw[['date', 'open', 'high', 'low', 'close']].copy()
        df.rename(columns={'date': 'time'}, inplace=True)
        df['time'] = df['time'].dt.strftime('%Y-%m-%d')  # Format date for the chart

        # Handle missing values
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df = df.dropna(subset=['time', 'open', 'high', 'low', 'close'])

        # Prepare JSON data for the chart
        chart_data = df[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')

        # HTML/JavaScript for Lightweight Charts
        def render_lightweight_chart(data):
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            </head>
            <body>
                <div id="chart" style="width: 100%; height: 500px;"></div>
                <script>
                    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                        width: 1000,
                        height: 800,
                        layout: {{
                            backgroundColor: '#FFFFFF',
                            textColor: '#000000',
                        }},
                        grid: {{
                            vertLines: {{
                                color: '#EEEEEE',
                            }},
                            horzLines: {{
                                color: '#EEEEEE',
                            }},
                        }},
                        crosshair: {{
                            mode: LightweightCharts.CrosshairMode.Normal,
                        }},
                        priceScale: {{
                            borderColor: '#CCCCCC',
                        }},
                        timeScale: {{
                            borderColor: '#CCCCCC',
                        }},
                    }});

                    const candleSeries = chart.addCandlestickSeries();
                    candleSeries.setData({data});

                </script>
            </body>
            </html>
            """

        # Display the chart
        chart_html = render_lightweight_chart(chart_data)
        st.components.v1.html(chart_html, height=800)

else:
    st.error("No transformed data available in the session state.")
