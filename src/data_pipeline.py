import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    # normalize the data
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    df['week_of'] = df['shipment_date'] - pd.to_timedelta(df['shipment_date'].dt.dayofweek, unit='D')
    df['route'] = df['origin'] + " to " + df['destination']
    return df

def calculate_baselines(df):
    # 1. Find the weekly cost by sum(freight cost) / sum(tonee * km)
    weekly_df = df.groupby(['route', 'route_type', 'week_of']).apply(
        lambda x: x['freight_cost_inr'].sum() / (x['quantity_tonnes'] * x['distance_km']).sum()
    ).reset_index(name='weekly_cost_per_tonne_km')
    
    # 2. Historical Baseline (vs. own history)
    # Sort so that rolling window work correctly
    weekly_df = weekly_df.sort_values(by=['route', 'week_of'])

    # this simply sees previous 8 weeks for current week to get the average price
    weekly_df['historical_baseline'] = weekly_df.groupby('route')['weekly_cost_per_tonne_km'].transform(
        lambda x: x.shift(1).rolling(window=8, min_periods=1).mean()
    )
    
    # 3. Peer Baseline (vs. similar routes)
    # Calculate total sum and count per route_type per week so that we can find average of that particular weeks count
    peer_stats = weekly_df.groupby(['route_type', 'week_of']).agg(
        total_type_cost=('weekly_cost_per_tonne_km', 'sum'),
        route_count=('route', 'count')
    ).reset_index()
    # merge the above data with our table to get avg price of other routes
    weekly_df = pd.merge(weekly_df, peer_stats, on=['route_type', 'week_of'], how='left')
    
    # Compute peer average excluding the route itself: (Total - Own) / (Count - 1)
    weekly_df['peer_baseline'] = np.where(
        weekly_df['route_count'] > 1,
        (weekly_df['total_type_cost'] - weekly_df['weekly_cost_per_tonne_km']) / (weekly_df['route_count'] - 1),
        np.nan # If it's the only route of its type that week, peer baseline is NaN
    )
    
    # Clean up intermediate columns
    weekly_df = weekly_df.drop(columns=['total_type_cost', 'route_count'])
    
    return weekly_df #outputs a table of 728 route week(7 route * 104 weeks)

if __name__ == "__main__":
    # Test execution
    df_raw = load_and_preprocess('data/shipment_records.csv')
    df_baselines = calculate_baselines(df_raw)
    
    print("\n--- Processed Baselines ---")
    print(df_baselines.head(10))