import os
import pandas as pd
import numpy as np

def load_data(filepath='tesla_deliveries_dataset_2015_2025.csv'):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        # Fallback to synthetic if not found
        np.random.seed(42)
        n = 500
        years  = np.random.choice(range(2015, 2026), n)
        months = np.random.choice(range(1, 13), n)
        regions = np.random.choice(['North America','Europe','Asia Pacific','Other'], n, p=[0.4, 0.3, 0.2, 0.1])
        models  = np.random.choice(['Model S','Model 3','Model X','Model Y','Cybertruck'], n, p=[0.1, 0.35, 0.1, 0.35, 0.1])
        prod_units = np.random.randint(5_000, 120_000, n)
        avg_price  = np.random.uniform(35_000, 130_000, n)
        batt_kwh   = np.random.uniform(60, 100, n)
        range_km   = np.random.uniform(300, 620, n)
        co2_saved  = np.random.uniform(500, 15_000, n)
        charging   = np.random.randint(100, 5_000, n)
        
        deliveries = (
            0.88 * prod_units
            + 500 * (years - 2015)
            + np.where(months == 12, 8000, 0)
            + np.random.normal(0, 3000, n)
        ).astype(int).clip(min=0)

        df = pd.DataFrame({
            'year': years, 'month': months,
            'region': regions, 'model': models,
            'production_units': prod_units,
            'avg_price_usd': avg_price.round(2),
            'battery_capacity_kwh': batt_kwh.round(1),
            'range_km': range_km.round(1),
            'co2_saved_tons': co2_saved.round(1),
            'charging_stations': charging,
            'estimated_deliveries': deliveries
        })
        for col in ['avg_price_usd','battery_capacity_kwh','co2_saved_tons']:
            idx = df.sample(frac=0.08, random_state=42).index
            df.loc[idx, col] = np.nan
        df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)
    
    info = {
        'shape': list(df.shape),
        'columns': list(df.columns)
    }
    return df, info
