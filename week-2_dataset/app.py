from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np

# Import pipeline modules
from pipeline.data_loader import load_data
from pipeline.data_understanding import understand_data
from pipeline.data_cleaning import clean_data
from pipeline.eda import run_eda
from pipeline.feature_engineering import engineer_features
from pipeline.preprocessing import preprocess
from pipeline.modeling import train_and_evaluate
from pipeline.cross_validation import cross_validate_models
from pipeline.tuning import tune_xgboost
from pipeline.feature_importance import get_feature_importance
from pipeline.shap_analysis import run_shap
from pipeline.time_series import forecast
from pipeline.model_io import save_models

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    results = {}
    
    try:
        # Phase 2: Data Loading
        df, load_info = load_data('tesla_deliveries_dataset_2015_2025.csv')
        results['phase2'] = load_info
        
        # Phase 3: Data Understanding
        results['phase3'] = understand_data(df)
        
        # Phase 4: Data Cleaning
        df, clean_log = clean_data(df)
        results['phase4'] = {'log': clean_log, 'shape': list(df.shape)}
        
        # Phase 5: EDA
        eda_paths = run_eda(df)
        results['phase5'] = {'paths': eda_paths}
        
        # Phase 6: Feature Engineering
        df, new_features = engineer_features(df)
        results['phase6'] = {'new_features': new_features, 'shape': list(df.shape)}
        
        # Phase 7: Preprocessing
        preprocessor, X_train_processed, X_test_processed, y_train, y_test, prep_meta = preprocess(df)
        results['phase7'] = prep_meta
        
        # Phase 8: Modeling
        modeling_res, models = train_and_evaluate(X_train_processed, y_train, X_test_processed, y_test)
        results['phase8'] = modeling_res
        
        # Phase 9: Cross Validation
        cv_stats, cv_plot = cross_validate_models(X_train_processed, y_train)
        results['phase9'] = {'stats': cv_stats, 'plot': cv_plot}
        
        # Phase 10: Tuning
        best_xgb, best_params, tuning_metrics, tuning_plot = tune_xgboost(X_train_processed, y_train, X_test_processed, y_test)
        # Convert np types to native Python types for JSON serialization
        clean_params = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v for k, v in best_params.items()}
        results['phase10'] = {'best_params': clean_params, 'metrics': tuning_metrics, 'plot': tuning_plot}
        
        # Phase 11: Feature Importance
        fi_stats, fi_plot = get_feature_importance(best_xgb, preprocessor, prep_meta['cat_cols'], prep_meta['num_cols'])
        # Convert float32 from XGBoost feature_importances to standard float
        fi_stats = [{'Feature': row['Feature'], 'Importance': float(row['Importance'])} for row in fi_stats]
        results['phase11'] = {'top_features': fi_stats, 'plot': fi_plot}
        
        # Phase 12: SHAP Analysis
        shap_paths = run_shap(best_xgb, X_test_processed)
        results['phase12'] = {'paths': shap_paths}
        
        # Phase 13: Time Series Forecasting
        ts_metrics, ts_plot, ts_model = forecast(df)
        if ts_metrics:
            results['phase13'] = {'metrics': ts_metrics, 'plot': ts_plot}
        else:
            results['phase13'] = {'error': 'Required columns for time series missing.'}
            
        # Phase 14/15: Model Saving
        saved_paths = save_models(preprocessor, best_xgb, ts_model)
        results['phase15'] = {'paths': saved_paths}
        
        return jsonify({'status': 'success', 'data': results})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
