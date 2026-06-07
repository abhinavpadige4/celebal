import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def cross_validate_models(X_train_processed, y_train, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
    }
    
    cv_results = {}
    cv_stats = []
    
    for name, model in cv_models.items():
        scores = cross_val_score(model, X_train_processed, y_train, cv=kf, scoring='r2', n_jobs=-1)
        cv_results[name] = scores
        cv_stats.append({
            'Model': name,
            'Mean R²': round(scores.mean(), 4),
            'Std Dev': round(scores.std(), 4)
        })
        
    plt.figure(figsize=(9, 4))
    plt.boxplot(cv_results.values(), labels=cv_results.keys(), patch_artist=True)
    plt.title('Cross Validation R² — All Models')
    plt.ylabel('R²')
    plt.xticks(rotation=15)
    plt.tight_layout()
    cv_plot_path = os.path.join(output_dir, 'cv_results.png')
    plt.savefig(cv_plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return cv_stats, cv_plot_path.replace('\\', '/')
