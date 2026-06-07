import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style='whitegrid', palette='muted')

def run_eda(df, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    
    # Check column names
    region_col = 'Region' if 'Region' in df.columns else 'region'
    target_col = 'Estimated_Deliveries' if 'Estimated_Deliveries' in df.columns else 'estimated_deliveries'
    year_col = 'Year' if 'Year' in df.columns else 'year'
    model_col = 'Model' if 'Model' in df.columns else 'model'
    prod_col = 'Production_Units' if 'Production_Units' in df.columns else 'production_units'
    month_col = 'Month' if 'Month' in df.columns else 'month'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Tesla Deliveries — EDA Dashboard', fontsize=16, fontweight='bold')

    # 1. Deliveries by Region
    if region_col in df.columns and target_col in df.columns:
        reg = df.groupby(region_col)[target_col].sum().sort_values(ascending=False)
        axes[0,0].bar(reg.index, reg.values, color=sns.color_palette('muted'))
        axes[0,0].set_title('Total Deliveries by Region')
        axes[0,0].set_ylabel('Deliveries')
        axes[0,0].tick_params(axis='x', rotation=20)

    # 2. Yearly Trend
    if year_col in df.columns and target_col in df.columns:
        yr = df.groupby(year_col)[target_col].sum()
        axes[0,1].plot(yr.index, yr.values, marker='o', color='steelblue', linewidth=2)
        axes[0,1].fill_between(yr.index, yr.values, alpha=0.15, color='steelblue')
        axes[0,1].set_title('Deliveries Growth Over Years')
        axes[0,1].set_ylabel('Total Deliveries')

    # 3. Deliveries by Model
    if model_col in df.columns and target_col in df.columns:
        mod = df.groupby(model_col)[target_col].sum().sort_values(ascending=False)
        axes[0,2].bar(mod.index, mod.values, color=sns.color_palette('Set2'))
        axes[0,2].set_title('Deliveries by Tesla Model')
        axes[0,2].tick_params(axis='x', rotation=25)

    # 4. Production vs Deliveries Scatter
    if prod_col in df.columns and target_col in df.columns:
        axes[1,0].scatter(df[prod_col], df[target_col], alpha=0.4, c='coral', edgecolors='none')
        axes[1,0].set_title('Production vs Deliveries')
        axes[1,0].set_xlabel('Production Units')
        axes[1,0].set_ylabel('Estimated Deliveries')

    # 5. Distribution of Target
    if target_col in df.columns:
        axes[1,1].hist(df[target_col], bins=30, color='mediumseagreen', edgecolor='white')
        axes[1,1].set_title('Distribution of Deliveries')
        axes[1,1].set_xlabel('Deliveries')

    # 6. Monthly Average
    if month_col in df.columns and target_col in df.columns:
        mon = df.groupby(month_col)[target_col].mean()
        axes[1,2].bar(mon.index, mon.values, color='mediumpurple')
        axes[1,2].set_title('Avg Deliveries by Month')
        axes[1,2].set_xlabel('Month')
        axes[1,2].set_xticks(range(1,13))

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, 'eda_dashboard.png')
    plt.savefig(dashboard_path, dpi=120, bbox_inches='tight')
    plt.close()
    paths.append(dashboard_path.replace('\\', '/'))

    # Correlation Heatmap
    plt.figure(figsize=(10, 7))
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(corr_path, dpi=120, bbox_inches='tight')
    plt.close()
    paths.append(corr_path.replace('\\', '/'))
    
    return paths
