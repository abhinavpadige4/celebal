import pandas as pd

def clean_data(df):
    log = []
    
    # 1. Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    log.append(f"Removed duplicates: {initial_shape[0] - df.shape[0]} rows dropped.")
    
    # Check column names (handle both capitalized and lowercased cases)
    year_col = 'Year' if 'Year' in df.columns else 'year'
    month_col = 'Month' if 'Month' in df.columns else 'month'
    price_col = 'Avg_Price_USD' if 'Avg_Price_USD' in df.columns else 'avg_price_usd'

    # 2. Fix data types
    if year_col in df.columns:
        df[year_col] = df[year_col].astype(int)
    if month_col in df.columns:
        df[month_col] = df[month_col].astype(int)
        
    # 3. Impute numeric missing values with median
    num_cols_missing = df.select_dtypes(include='number').columns[
        df.select_dtypes(include='number').isnull().any()
    ]
    for col in num_cols_missing:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        log.append(f"Imputed {col} with median={median:.2f}")
        
    # 4. Outlier detection — IQR on avg_price_usd
    if price_col in df.columns:
        Q1, Q3 = df[price_col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        outliers_count = len(df[(df[price_col] < lower) | (df[price_col] > upper)])
        df[price_col] = df[price_col].clip(lower, upper)
        log.append(f"Capped {outliers_count} outliers in {price_col}")
        
    # 5. Reset index
    df.reset_index(drop=True, inplace=True)
    log.append(f"Final clean shape: {df.shape}")
    
    return df, log
