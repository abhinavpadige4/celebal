import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os

def run_shap(best_xgb, X_test_processed, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer(X_test_processed)
    
    paths = []
    
    # Global Feature Importance (Bar)
    plt.figure()
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'shap_bar.png')
    plt.savefig(bar_path, dpi=120, bbox_inches='tight')
    plt.close()
    paths.append(bar_path.replace('\\', '/'))
    
    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, 'shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=120, bbox_inches='tight')
    plt.close()
    paths.append(beeswarm_path.replace('\\', '/'))
    
    # Waterfall
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    waterfall_path = os.path.join(output_dir, 'shap_waterfall.png')
    plt.savefig(waterfall_path, dpi=120, bbox_inches='tight')
    plt.close()
    paths.append(waterfall_path.replace('\\', '/'))
    
    return paths
