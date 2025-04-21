import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.title("Similar Period Analysis")

# Add input for days variable outside the cached function
days = st.number_input("Enter the number of days for analysis:", min_value=1, value=7, step=1)
days_forward = st.number_input("Enter the number of days for time range before start of similar period:", min_value=1, value=14, step=1)

@st.cache_data
def find_similar_period(days, days_forward, top_n=3):
    # Load data from session state or CSV
    if 'df_fon_table' in st.session_state:
        df_fon_table = st.session_state.df_fon_table
    else:
        df_fon_table = pd.read_csv('data/fon_table.csv')
        st.session_state.df_fon_table = df_fon_table

    if 'df_transformed' in st.session_state:
        df_transformed = st.session_state.df_transformed
    else:
        df_transformed = pd.read_csv('data/tefas_transformed.csv', encoding='utf-8-sig', parse_dates=['date'])
        st.session_state.df_transformed = df_transformed

    # Get symbol attributes
    symbol_attributes_of_fon_table = [col for col in df_fon_table.columns if col.startswith('FundType_')]
    symbol_attributes_list = [col.replace('FundType_', '') for col in symbol_attributes_of_fon_table]

    # Get the most recent week (last 7 days)
    recent_date = df_transformed['date'].max()

    # Update week_ago calculation
    week_ago = recent_date - timedelta(days=days)
    
    # Filter recent week's data
    recent_data = df_transformed[(df_transformed['date'] >= week_ago) & (df_transformed['date'] <= recent_date)]
    recent_data = pd.merge(recent_data, df_fon_table, on='symbol', how='inner')

    # Calculate recent week's percentage change by Fon Unvan Türü
    recent_summary = []
    for attribute in symbol_attributes_list:
        attribute_col = 'FundType_' + attribute
        if attribute_col not in recent_data.columns:
            continue
            
        filtered_data = recent_data[recent_data[attribute_col] == True]
        if filtered_data.empty:
            continue
            
        start_value = filtered_data[filtered_data['date'] == filtered_data['date'].min()]['market_cap'].sum()
        end_value = filtered_data[filtered_data['date'] == filtered_data['date'].max()]['market_cap'].sum()
        
        if start_value == 0:
            continue
            
        pct_change = (end_value - start_value) / start_value * 100
        
        recent_summary.append({
            'Fon Unvan Türü': attribute,
            'Recent_%_Change': pct_change
        })

    recent_df = pd.DataFrame(recent_summary)
    if recent_df.empty:
        st.warning("No recent data available for comparison.")
        return None

    # Find historical periods with similar changes
    results = []
    min_date = df_transformed['date'].min()
    max_date = recent_date - timedelta(days=days)  # Don't overlap with recent period
    
    # Iterate through historical periods
    current_start = min_date
    while current_start + timedelta(days=days) <= max_date:
        current_end = current_start + timedelta(days=days)
        period_data = df_transformed[(df_transformed['date'] >= current_start) & 
                                  (df_transformed['date'] <= current_end)]
        period_data = pd.merge(period_data, df_fon_table, on='symbol', how='inner')
        
        period_summary = []
        for attribute in symbol_attributes_list:
            attribute_col = 'FundType_' + attribute
            if attribute_col not in period_data.columns:
                continue
                
            filtered_data = period_data[period_data[attribute_col] == True]
            if filtered_data.empty:
                continue
                
            start_value = filtered_data[filtered_data['date'] == filtered_data['date'].min()]['market_cap'].sum()
            end_value = filtered_data[filtered_data['date'] == filtered_data['date'].max()]['market_cap'].sum()
            
            if start_value == 0:
                continue
                
            pct_change = (end_value - start_value) / start_value * 100
            
            period_summary.append({
                'Fon Unvan Türü': attribute,
                'Period_%_Change': pct_change
            })
        
        if period_summary:
            period_df = pd.DataFrame(period_summary)
            # Merge with recent data to compare
            comparison = pd.merge(recent_df, period_df, on='Fon Unvan Türü', how='inner')
            
            if not comparison.empty:
                # Calculate similarity (mean absolute difference)
                comparison['Diff'] = abs(comparison['Recent_%_Change'] - comparison['Period_%_Change'])
                similarity_score = comparison['Diff'].mean()
                
                results.append({
                    'Period_Start': current_start,
                    'Period_End': current_end,
                    'Similarity_Score': similarity_score,
                    'Details': comparison[['Fon Unvan Türü', 'Recent_%_Change', 'Period_%_Change']]
                })
        
        current_start += timedelta(days=1)  # Slide window by 1 day

    if not results:
        st.warning("No similar periods found in historical data.")
        return None

    # Find the most similar period
    results_df = pd.DataFrame([(r['Period_Start'], r['Period_End'], r['Similarity_Score']) 
                             for r in results], 
                            columns=['Period_Start', 'Period_End', 'Similarity_Score'])
    most_similar = results_df.loc[results_df['Similarity_Score'].idxmin()]
    
    # Get details for the most similar period
    most_similar_details = next(r['Details'] for r in results 
                              if r['Period_Start'] == most_similar['Period_Start'] and 
                              r['Period_End'] == most_similar['Period_End'])
    
    # Calculate top n most profitable symbols per category for the `days_forward` period after the historical start date
    historical_start_date = most_similar['Period_Start']
    future_start_date = historical_start_date
    future_end_date = future_start_date + timedelta(days=days_forward)
    
    # Filter data for the `days_forward` period after the historical start date
    future_data = df_transformed[(df_transformed['date'] >= future_start_date) & 
                                 (df_transformed['date'] <= future_end_date)]
    future_data = pd.merge(future_data, df_fon_table, on='symbol', how='inner')
    
    # Calculate profitability for each symbol in each category (future period)
    profitability_data = []
    for attribute in symbol_attributes_list:
        attribute_col = 'FundType_' + attribute
        if attribute_col not in future_data.columns:
            continue
            
        filtered_data = future_data[future_data[attribute_col] == True]
        if filtered_data.empty:
            continue
            
        # Group by symbol and calculate percentage change over the `days_forward` period using close
        symbols = filtered_data['symbol'].unique()
        for symbol in symbols:
            symbol_data = filtered_data[filtered_data['symbol'] == symbol]
            if symbol_data.empty:
                continue

            # Check if 'close' column exists in the symbol data
            if 'close' not in symbol_data.columns:
                st.warning(f"close data is missing for symbol {symbol}. Skipping this symbol.")
                continue

            start_price = symbol_data[symbol_data['date'] == symbol_data['date'].min()]['close'].mean()
            end_price = symbol_data[symbol_data['date'] == symbol_data['date'].max()]['close'].mean()
            
            if start_price > 0 and end_price > 0:
                pct_change = (end_price - start_price) / start_price * 100
                profitability_data.append({
                    'Fon Unvan Türü': attribute,
                    'Symbol': symbol,
                    'Profitability_%_Change': pct_change
                })

    profitability_df = pd.DataFrame(profitability_data)

    # Get top n symbols per category based on profitability
    top_symbols = profitability_df.groupby('Fon Unvan Türü').apply(
        lambda x: x.nlargest(top_n, 'Profitability_%_Change')
    ).reset_index(drop=True)
    
    # Calculate similarity score based on market_cap
    symbol_changes = []
    for _, row in top_symbols.iterrows():
        category = row['Fon Unvan Türü']
        symbol = row['Symbol']
        
        # Historical period change (market_cap)
        historical_symbol_data = future_data[future_data['symbol'] == symbol]
        if not historical_symbol_data.empty:
            start_value_hist = historical_symbol_data[historical_symbol_data['date'] == historical_symbol_data['date'].min()]['market_cap'].sum()
            end_value_hist = historical_symbol_data[historical_symbol_data['date'] == historical_symbol_data['date'].max()]['market_cap'].sum()
            hist_pct_change = (end_value_hist - start_value_hist) / start_value_hist * 100 if start_value_hist != 0 else 0
        else:
            hist_pct_change = 0
        
        # Recent period change (market_cap)
        recent_symbol_data = recent_data[recent_data['symbol'] == symbol]
        if not recent_symbol_data.empty:
            start_value_recent = recent_symbol_data[recent_symbol_data['date'] == recent_symbol_data['date'].min()]['market_cap'].sum()
            end_value_recent = recent_symbol_data[recent_symbol_data['date'] == recent_symbol_data['date'].max()]['market_cap'].sum()
            recent_pct_change = (end_value_recent - start_value_recent) / start_value_recent * 100 if start_value_recent != 0 else 0
        else:
            recent_pct_change = 0
        
        # Calculate similarity score
        if recent_pct_change == 0 and hist_pct_change == 0:
            similarity_score = 100  # Both are 0, so 100% similar
        else:
            max_val = max(abs(recent_pct_change), abs(hist_pct_change))
            if max_val == 0:
                similarity_score = 0
            else:
                similarity_score = 100 * (1 - abs(recent_pct_change - hist_pct_change) / max_val)
        
        symbol_changes.append({
            'Fon Unvan Türü': category,
            'Symbol': symbol,
            'Historical_%_Change': hist_pct_change,
            'Recent_%_Change': recent_pct_change,
            'Similarity_Score': similarity_score
        })

    symbol_changes_df = pd.DataFrame(symbol_changes)

    # Pivot the top symbols into separate columns
    top_symbols['Rank'] = top_symbols.groupby('Fon Unvan Türü').cumcount() + 1
    top_symbols_pivot = top_symbols.pivot(index='Fon Unvan Türü', columns='Rank', values=['Symbol', 'Profitability_%_Change'])
    
    # Flatten the multi-level column names
    top_symbols_pivot.columns = [
        f'Top{rank}_Symbol' if name == 'Symbol' else 
        f'Top{rank}_%_Profitability'
        for name, rank in top_symbols_pivot.columns
    ]
    top_symbols_pivot = top_symbols_pivot.reset_index()
    
    # Merge with comparison details
    most_similar_details = most_similar_details.merge(
        top_symbols_pivot,
        on='Fon Unvan Türü',
        how='left'
    )
    
    # Do NOT fill NaN values - let them remain as NaN to display as empty in Streamlit
    # Format output (round numerical columns, but preserve NaN)
    most_similar_details['Recent_%_Change'] = most_similar_details['Recent_%_Change'].round(2)
    most_similar_details['Period_%_Change'] = most_similar_details['Period_%_Change'].round(2)
    for rank in range(1, top_n + 1):
        most_similar_details[f'Top{rank}_%_Profitability'] = most_similar_details[f'Top{rank}_%_Profitability'].round(2)
    
    return {
        'Period_Start': most_similar['Period_Start'],
        'Period_End': most_similar['Period_End'],
        'Similarity_Score': round(most_similar['Similarity_Score'], 2),
        'Details': most_similar_details
    }

# Execute analysis
result = find_similar_period(days, days_forward)
if result:
    dataframe_height = (len(result['Details']) + 1) * 35 + 2

    st.write(f"**Most Similar Historical Period**")
    st.write(f"**Period**: {result['Period_Start'].strftime('%Y-%m-%d')} to {result['Period_End'].strftime('%Y-%m-%d')}")
    st.write(f"**Similarity Score**: {result['Similarity_Score']}% (lower is more similar)")
    
    st.subheader("Comparison Details")
    st.dataframe(
        result['Details'],
        column_config={
            'Fon Unvan Türü': 'Category',
            'Recent_%_Change': 'Recent Week % Change',
            'Period_%_Change': 'Historical Period % Change',
            'Top1_Symbol': 'Top1',
            'Top1_%_Profitability': 'Top1 % Profitability',
            'Top2_Symbol': 'Top2',
            'Top2_%_Profitability': 'Top2 % Profitability',
            'Top3_Symbol': 'Top3',
            'Top3_%_Profitability': 'Top3 % Profitability'
        },
        height=dataframe_height,
        hide_index=True
    )