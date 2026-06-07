import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess(df):
    target_col = 'Estimated_Deliveries' if 'Estimated_Deliveries' in df.columns else 'estimated_deliveries'
    
    # Drop target and non-features like 'date' (and 'Date' if present)
    drop_cols = ['date', 'Date', 'Source_Type', 'source_type']
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]
    
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include='number').columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)
    
    metadata = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'train_shape': list(X_train_processed.shape),
        'test_shape': list(X_test_processed.shape)
    }
    
    return preprocessor, X_train_processed, X_test_processed, y_train, y_test, metadata
