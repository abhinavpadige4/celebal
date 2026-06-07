import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_and_evaluate(X_train_processed, y_train, X_test_processed, y_test):
    def evaluate(name, model, Xtr, ytr, Xte, yte):
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        r2   = r2_score(yte, preds)
        rmse = np.sqrt(mean_squared_error(yte, preds))
        mae  = mean_absolute_error(yte, preds)
        return {'Model': name, 'R²': round(r2,4), 'RMSE': round(rmse,2), 'MAE': round(mae,2)}, model

    results = []
    models = {}

    # Linear Regression
    lr = LinearRegression()
    res, mod = evaluate('Linear Regression', lr, X_train_processed, y_train, X_test_processed, y_test)
    results.append(res); models['Linear Regression'] = mod

    # Ridge
    ridge = Ridge(alpha=10)
    res, mod = evaluate('Ridge Regression', ridge, X_train_processed, y_train, X_test_processed, y_test)
    results.append(res); models['Ridge Regression'] = mod

    # Lasso
    lasso = Lasso(alpha=10)
    res, mod = evaluate('Lasso Regression', lasso, X_train_processed, y_train, X_test_processed, y_test)
    results.append(res); models['Lasso Regression'] = mod

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    res, mod = evaluate('Random Forest', rf, X_train_processed, y_train, X_test_processed, y_test)
    results.append(res); models['Random Forest'] = mod

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    res, mod = evaluate('XGBoost', xgb, X_train_processed, y_train, X_test_processed, y_test)
    results.append(res); models['XGBoost'] = mod

    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    
    return results_df.to_dict('records'), models
