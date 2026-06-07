import pandas as pd

def understand_data(df):
    miss = df.isnull().sum()
    missing_dict = miss[miss > 0].to_dict()
    
    dtypes_dict = df.dtypes.astype(str).to_dict()
    
    target_stats = df['Estimated_Deliveries'].describe().to_dict() if 'Estimated_Deliveries' in df.columns else df['estimated_deliveries'].describe().to_dict() if 'estimated_deliveries' in df.columns else {}
    
    return {
        'dtypes': dtypes_dict,
        'missing': missing_dict,
        'duplicates': int(df.duplicated().sum()),
        'target_stats': target_stats
    }
