import pandas as pd
import numpy as np

def engineer_features(df):
    year_col = 'Year' if 'Year' in df.columns else 'year'
    month_col = 'Month' if 'Month' in df.columns else 'month'
    price_col = 'Avg_Price_USD' if 'Avg_Price_USD' in df.columns else 'avg_price_usd'
    range_col = 'Range_km' if 'Range_km' in df.columns else 'range_km'
    target_col = 'Estimated_Deliveries' if 'Estimated_Deliveries' in df.columns else 'estimated_deliveries'
    prod_col = 'Production_Units' if 'Production_Units' in df.columns else 'production_units'
    
    # 1. Date
    df['date'] = pd.to_datetime(dict(year=df[year_col], month=df[month_col], day=1))
    
    # 2. Quarter
    df['quarter'] = df[month_col].apply(lambda m: (m-1)//3 + 1)
    
    # 3. Production efficiency (leaky, will be dropped)
    if target_col in df.columns and prod_col in df.columns:
        df['efficiency'] = df[target_col] / (df[prod_col] + 1)
        
    # 4. Price per km
    if price_col in df.columns and range_col in df.columns:
        df['price_per_km'] = df[price_col] / (df[range_col] + 1)
        
    # 5. Year trend
    df['years_since_2015'] = df[year_col] - 2015
    
    # 6. Is Q4 flag
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    
    # 7. Lag features
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if target_col in df.columns:
        df['lag_1'] = df[target_col].shift(1)
        df['lag_3'] = df[target_col].shift(3)
        df['rolling_mean_3'] = df[target_col].shift(1).rolling(3).mean()
        
    # Drop rows with NaN from lag features
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    new_features = ['date','quarter','efficiency','price_per_km','years_since_2015','is_q4','lag_1','lag_3','rolling_mean_3']
    new_features = [f for f in new_features if f in df.columns]
    
    # Drop leaky feature
    if 'efficiency' in df.columns:
        df.drop(columns=['efficiency'], inplace=True)
        new_features.remove('efficiency')
        
    return df, new_features
