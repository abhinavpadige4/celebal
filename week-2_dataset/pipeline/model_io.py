import joblib
import os

def save_models(preprocessor, best_xgb, ts_model, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    
    prep_path = os.path.join(output_dir, 'preprocessor.pkl')
    xgb_path = os.path.join(output_dir, 'xgboost_model.pkl')
    ts_path = os.path.join(output_dir, 'ts_forecast_model.pkl')
    
    joblib.dump(preprocessor, prep_path)
    joblib.dump(best_xgb, xgb_path)
    if ts_model is not None:
        joblib.dump(ts_model, ts_path)
        
    return [prep_path.replace('\\', '/'), xgb_path.replace('\\', '/'), ts_path.replace('\\', '/')]

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None
