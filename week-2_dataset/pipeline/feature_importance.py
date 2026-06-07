import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style='whitegrid', palette='muted')

def get_feature_importance(best_xgb, preprocessor, cat_cols, num_cols, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    ohe_features = list(
        preprocessor.named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(cat_cols)
    )
    all_features = num_cols + ohe_features
    
    importance_df = (
        pd.DataFrame({'Feature': all_features,
                      'Importance': best_xgb.feature_importances_})
        .sort_values('Importance', ascending=False)
        .head(15)
    )
    
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='Blues_r', legend=False)
    plt.title('Top 15 Feature Importances — XGBoost', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return importance_df.head(5).to_dict('records'), plot_path.replace('\\', '/')
