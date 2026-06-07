import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def tune_xgboost(X_train_processed, y_train, X_test_processed, y_test, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        XGBRegressor(verbosity=0, random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train_processed, y_train)
    
    best_xgb = grid_search.best_estimator_
    y_pred_best = best_xgb.predict(X_test_processed)
    
    r2 = r2_score(y_test, y_pred_best)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
    mae = mean_absolute_error(y_test, y_pred_best)
    
    metrics = {
        'R²': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }
    
    # Actual vs Predicted Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_best, alpha=0.5, c='steelblue', edgecolors='none')
    mn = min(y_test.min(), y_pred_best.min())
    mx = max(y_test.max(), y_pred_best.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Deliveries')
    plt.ylabel('Predicted Deliveries')
    plt.title('Actual vs Predicted — Tuned XGBoost')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return best_xgb, grid_search.best_params_, metrics, plot_path.replace('\\', '/')
