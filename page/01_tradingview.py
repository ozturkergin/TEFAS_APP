import streamlit as st
import pandas as pd
import os
import json  # Ensure JSON module is imported

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

# Selectbox for selecting a single symbol
selected_symbol = st.selectbox("Select Symbol:", unique_symbols)

# Ensure 'df_transformed' exists in session_state
if 'df_transformed' in st.session_state:
    # Filter and prepare data for the selected symbol
    df_raw = st.session_state.df_transformed[
        st.session_state.df_transformed['symbol'] == selected_symbol
    ].copy()

    if df_raw.empty:
        st.warning(f"No data available for symbol: {selected_symbol}")
    else:
        df = df_raw[['date', 'open', 'high', 'low', 'close', 'RSI_14', 'number_of_investors']].copy()
        df.rename(columns={'date': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')  # Format date for the chart

        # Handle missing values
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df = df.dropna(subset=['time', 'open', 'high', 'low', 'close', 'RSI_14', 'number_of_investors'])

        # Prepare JSON data for the chart
        candle_data = df[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
        rsi_data = df[['time', 'RSI_14']].rename(columns={'RSI_14': 'value'}).to_dict(orient='records')
        volume_data = df[['time', 'number_of_investors']].rename(columns={'number_of_investors': 'value'}).to_dict(orient='records')

        # HTML/JavaScript for Lightweight Charts
        def render_lightweight_chart(candle_data, rsi_data, volume_data):
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            </head>
            <body>
                <div id="chart" style="width: 100%; height: 600px;"></div>
                <script>
                    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                        width: 1000,
                        height: 600,
                        layout: {{
                            backgroundColor: '#FFFFFF',
                            textColor: '#000000',
                        }},
                        grid: {{
                            vertLines: {{ color: '#EEEEEE' }},
                            horzLines: {{ color: '#EEEEEE' }},
                        }},
                        crosshair: {{
                            mode: LightweightCharts.CrosshairMode.Normal,
                        }},
                        timeScale: {{
                            borderColor: '#CCCCCC',
                        }},
                    }});

                    // Add candlestick series for price
                    const candleSeries = chart.addCandlestickSeries({{
                        priceScaleId: 'right', // Default scale on the right
                    }});
                    candleSeries.setData({json.dumps(candle_data)});

                    // Add RSI line series
                    const rsiSeries = chart.addLineSeries({{
                        color: 'blue',
                        lineWidth: 2,
                        priceScaleId: 'left', // Separate scale on the left
                    }});
                    rsiSeries.setData({json.dumps(rsi_data)});

                    // Add volume histogram series
                    const volumeSeries = chart.addHistogramSeries({{
                        color: 'rgba(76, 175, 80, 0.5)', // Green for positive values
                        priceScaleId: '', // Auto scales separately
                        scaleMargins: {{
                            top: 0.85,
                            bottom: 0,
                        }},
                    }});
                    volumeSeries.setData({json.dumps(volume_data)});

                    // Configure scales
                    chart.priceScale('right').applyOptions({{
                        scaleMargins: {{
                            top: 0.1,
                            bottom: 0.3,
                        }},
                        borderColor: '#CCCCCC',
                    }});

                    chart.priceScale('left').applyOptions({{
                        scaleMargins: {{
                            top: 0.7,
                            bottom: 0.3,
                        }},
                        borderColor: '#CCCCCC',
                    }});

                    chart.timeScale().fitContent();
                </script>
            </body>
            </html>
            """

        # Display the chart
        chart_html = render_lightweight_chart(candle_data, rsi_data, volume_data)
        col1, col2 = st.columns([9, 6])
        with col1:
            st.components.v1.html(chart_html, height=600)
        with col2:
            df = df.sort_values('time', ascending=False)
            st.dataframe(df, hide_index=True, height=600)

else:
    st.error("No transformed data available in the session state.")
