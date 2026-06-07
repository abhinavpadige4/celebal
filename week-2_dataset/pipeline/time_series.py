import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

def forecast(df, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    target_col = 'Estimated_Deliveries' if 'Estimated_Deliveries' in df.columns else 'estimated_deliveries'
    
    if 'date' not in df.columns or target_col not in df.columns:
        return None, None
        
    ts_df = (
        df.groupby('date')[target_col]
        .sum()
        .reset_index()
        .sort_values('date')
        .copy()
    )
    ts_df.columns = ['date','deliveries']
    
    for lag in [1, 2, 3, 6]:
        ts_df[f'lag_{lag}'] = ts_df['deliveries'].shift(lag)
    ts_df['rolling_mean_3'] = ts_df['deliveries'].shift(1).rolling(3).mean()
    ts_df['rolling_std_3']  = ts_df['deliveries'].shift(1).rolling(3).std()
    ts_df['month']  = ts_df['date'].dt.month
    ts_df['year']   = ts_df['date'].dt.year
    ts_df['is_q4']  = (ts_df['month'] >= 10).astype(int)
    ts_df.dropna(inplace=True)
    
    split_idx = int(len(ts_df) * 0.8)
    ts_features = [c for c in ts_df.columns if c not in ['date','deliveries']]
    
    X_ts_train = ts_df[ts_features].iloc[:split_idx]
    y_ts_train = ts_df['deliveries'].iloc[:split_idx]
    X_ts_test  = ts_df[ts_features].iloc[split_idx:]
    y_ts_test  = ts_df['deliveries'].iloc[split_idx:]
    
    ts_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
    ts_model.fit(X_ts_train, y_ts_train)
    
    ts_preds = ts_model.predict(X_ts_test)
    
    r2 = r2_score(y_ts_test, ts_preds)
    rmse = np.sqrt(mean_squared_error(y_ts_test, ts_preds))
    mae = mean_absolute_error(y_ts_test, ts_preds)
    
    metrics = {
        'R²': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }
    
    test_dates = ts_df['date'].iloc[split_idx:]
    plt.figure(figsize=(12, 5))
    plt.plot(ts_df['date'].iloc[:split_idx], y_ts_train, label='Training Data', color='steelblue', linewidth=1.5)
    plt.plot(test_dates, y_ts_test, label='Actual', color='green', linewidth=2)
    plt.plot(test_dates, ts_preds, label='Forecast', color='crimson', linestyle='--', linewidth=2)
    plt.axvline(ts_df['date'].iloc[split_idx], color='gray', linestyle=':', linewidth=1.5, label='Train/Test Split')
    plt.title('Tesla Deliveries — Time Series Forecast', fontsize=13, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Total Deliveries')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'forecast.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return metrics, plot_path.replace('\\', '/'), ts_model
