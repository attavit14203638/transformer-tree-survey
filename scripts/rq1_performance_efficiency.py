"""RQ1 Figure: Performance-Efficiency Analysis (Composite 2-Panel)

Panel (a): Violin + box plots showing performance gain distribution by architecture category
         Categories: Pure ViT, Hierarchical ViT, CNN-Trans. Hybrid, FM
Panel (b): Scatter plot showing parameters vs performance
         Same categories with 2x2 legend layout

Data Source: cleaned_data/performance_clean.csv + models_clean.csv
Output: /Users/fadil/Desktop/Survey/68f823600ac5436c4d362b39/figures/rq1_performance.jpg
"""

from utils import load_data, save_fig
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# Metrics that should NOT appear on a 0-100% scale (must be whole words or specific patterns)
NON_PERCENTAGE_METRICS_EXACT = ['RMSE', 'MAE', 'MSE', 'PSNR', 'R²', 'RMSE LOG', 'ABS REL']
NON_PERCENTAGE_METRICS_PATTERNS = ['R-SQUARED', 'R SQUARED', 'CORRELATION']


def is_percentage_metric(metric_name):
    """Check if metric should be on a 0-100% scale."""
    if pd.isna(metric_name):
        return True  # Assume percentage if unknown
    
    # Get primary metric (first in pipe-separated list)
    primary = str(metric_name).split('|')[0].upper().strip()
    
    # Check for exact matches (whole word or at boundaries)
    for non_pct in NON_PERCENTAGE_METRICS_EXACT:
        # Check if the metric name IS the non-percentage metric or starts/ends with it
        if primary == non_pct or primary.startswith(non_pct + ' ') or primary.endswith(' ' + non_pct):
            return False
        # Also check for metrics like "RMSE (m)" or "MAE|..."
        if primary.startswith(non_pct + '(') or primary.startswith(non_pct + ':'):
            return False
    
    # Check for pattern matches
    for pattern in NON_PERCENTAGE_METRICS_PATTERNS:
        if pattern in primary:
            return False
    
    # Special case: standalone 'r' for correlation (not inside words)
    import re
    if re.search(r'\br\b', primary, re.IGNORECASE):
        return False
    
    return True


def normalize_metric_value(value, metric_name):
    """Convert decimal values (0-1) to percentage (0-100) for percentage metrics."""
    if pd.isna(value):
        return value
    if not is_percentage_metric(metric_name):
        return None  # Will be filtered out
    if 0 < value <= 1.0:
        return value * 100
    return value


# Set publication-quality style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# Hierarchical ViT patterns for classification
HIERARCHICAL_PATTERNS = ['swin', 'segformer', 'pvt', 'davit', 'twins', 'mvit', 'mit-b', 'hierarchical', 'cvt', 'poolformer']


def classify_architecture(row):
    """Classify model into refined architecture categories."""
    backbone = str(row.get('backbones', '')).lower()
    category = str(row.get('category_clean', ''))
    model_name = str(row.get('model_name', '')).lower()
    
    if category == 'Foundation Model Adaptation':
        return 'Foundation Model'
    if category == 'Vision-Language Model':
        return 'Vision-Language Model'
    if category == 'Other':
        return 'Other'
    
    # Check backbones for hierarchical patterns
    for pattern in HIERARCHICAL_PATTERNS:
        if pattern in backbone or pattern in model_name:
            if category == 'Pure Vision Transformer':
                return 'Hierarchical ViT'
            else:
                return 'CNN-Transformer Hybrid'
    
    if category == 'Pure Vision Transformer':
        return 'Pure ViT'
    
    return category


# Professional color palette
ARCH_COLORS = {
    'Pure ViT': '#2166ac',                    # Deep blue
    'Hierarchical ViT': '#762a83',             # Purple
    'CNN-Transformer Hybrid': '#1a9850',       # Forest green
    'Foundation Model': '#d73027',             # Rich red
    'Vision-Language Model': '#7b3294',        # Deep purple (not used)
    'Other': '#878787'                         # Neutral gray
}


# Short labels for display
ARCH_SHORT = {
    'Pure ViT': 'Pure\nViT',
    'Hierarchical ViT': 'Hierarchical\nViT',
    'CNN-Transformer Hybrid': 'CNN-Trans.\nHybrid',
    'Foundation Model': 'FM',
    'Vision-Language Model': 'VLM',
    'Other': 'Other'
}


# Order for consistent display (user requested order)
ARCH_ORDER = [
    'Pure ViT',
    'Hierarchical ViT',
    'CNN-Transformer Hybrid',
    'Foundation Model'
]


def generate():
    _, models, perf, _, _, _ = load_data()
    if models is None or perf is None:
        print("❌ Could not load data")
        return

    # Apply refined architecture classification
    models['arch_category'] = models.apply(classify_architecture, axis=1)
    
    # Merge models and performance data
    merged = pd.merge(
        perf,
        models[['paper_id', 'model_id', 'arch_category', 'parameter_count_millions']],
        on=['paper_id', 'model_id'],
        how='left'
    )

    # Create figure with two panels (reduced inter-panel spacing)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'wspace': 0.18})
    fig.patch.set_facecolor('white')

    # === Panel (a): Violin + Box plots of performance gain by architecture ===
    gain_data = merged.dropna(subset=['performance_gain_numeric', 'arch_category'])
    gain_data = gain_data[gain_data['arch_category'].isin(ARCH_ORDER)]
    # Keep only gains that are explicitly reported relative to a CNN baseline
    if 'has_cnn_baseline' in gain_data.columns:
        gain_data = gain_data[gain_data['has_cnn_baseline'].astype(str).str.upper() == 'YES']

    # Filter to valid range for visualization
    gain_data_filtered = gain_data[gain_data['performance_gain_numeric'].between(-5, 60)]

    box_order = [cat for cat in ARCH_ORDER if cat in gain_data['arch_category'].unique()]
    colors = [ARCH_COLORS.get(cat, '#878787') for cat in box_order]

    sns.violinplot(
        data=gain_data_filtered,
        x='arch_category',
        y='performance_gain_numeric',
        order=box_order,
        palette=colors,
        ax=ax1,
        inner=None,
        alpha=0.3,
        linewidth=0,
        cut=0
    )

    sns.boxplot(
        data=gain_data_filtered,
        x='arch_category',
        y='performance_gain_numeric',
        order=box_order,
        palette=colors,
        ax=ax1,
        width=0.25,
        boxprops=dict(alpha=0.9),
        whiskerprops=dict(linewidth=1.5),
        medianprops=dict(color='white', linewidth=2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )

    # Add individual points with jitter
    for i, cat in enumerate(box_order):
        cat_points = gain_data_filtered[gain_data_filtered['arch_category'] == cat]['performance_gain_numeric']
        jitter = np.random.normal(0, 0.08, len(cat_points))
        ax1.scatter(i + jitter, cat_points, c='black', alpha=0.25, s=20, zorder=3)

    # Reference line at 0
    ax1.axhline(y=0, color='#404040', linestyle='--', alpha=0.7, linewidth=1.2, zorder=1)
    # Keep the label inside the axes so it doesn't create extra whitespace in the exported figure
    ax1.text(
        0.98, 0.02, 'CNN baseline',
        fontsize=8, color='#404040', style='italic',
        ha='right', va='bottom', transform=ax1.transAxes,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2)
    )

    # Formatting for panel (a)
    ax1.set_xlabel('')
    ax1.set_ylabel('Performance Gain vs CNN Baseline (%)', fontsize=11, fontweight='medium')
    ax1.text(
        0.5, -0.35, '(a) Performance Gain Distribution by Architecture',
        fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes
    )

    ax1.set_xticklabels([ARCH_SHORT[cat] for cat in box_order], fontsize=9)

    # Add statistics annotations
    y_max = gain_data_filtered['performance_gain_numeric'].max()
    for i, cat in enumerate(box_order):
        cat_data = gain_data[gain_data['arch_category'] == cat]['performance_gain_numeric']
        n = len(cat_data)
        median = cat_data.median()
        ax1.annotate(
            f'n={n}\nmed={median:.1f}%',
            xy=(i, y_max + 3),
            ha='center',
            fontsize=8,
            color=ARCH_COLORS.get(cat, '#404040'),
            fontweight='medium',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=ARCH_COLORS.get(cat, '#404040'),
                alpha=0.9,
                linewidth=0.8
            )
        )

    ax1.set_ylim(-5, min(y_max + 12, 58))
    ax1.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # === Panel (b): Scatter plot of params vs performance ===
    scatter_data = merged.dropna(subset=['parameter_count_millions', 'metric_value_numeric', 'arch_category'])
    scatter_data = scatter_data[scatter_data['arch_category'].isin(ARCH_ORDER)]
    scatter_data = scatter_data[scatter_data['parameter_count_millions'] > 0]
    scatter_data = scatter_data[scatter_data['parameter_count_millions'] < 10000]

    # Normalize metric values: convert decimal (0-1) to percentage and filter non-percentage metrics
    scatter_data = scatter_data.copy()
    scatter_data['metric_normalized'] = scatter_data.apply(
        lambda row: normalize_metric_value(row['metric_value_numeric'], row.get('metric_name', '')),
        axis=1
    )
    scatter_data = scatter_data.dropna(subset=['metric_normalized'])
    scatter_data = scatter_data[scatter_data['metric_normalized'].between(5, 100)]

    # Create scatter plot
    for cat in ARCH_ORDER:
        cat_data = scatter_data[scatter_data['arch_category'] == cat]
        if len(cat_data) > 0:
            ax2.scatter(
                cat_data['parameter_count_millions'],
                cat_data['metric_normalized'],
                c=ARCH_COLORS.get(cat, '#878787'),
                label=cat,
                s=100,
                alpha=0.75,
                edgecolors='white',
                linewidth=1.2,
                zorder=3
            )

    ax2.set_xscale('log')
    ax2.set_xlim(0.05, 1000)  # 0.05M to 1000M
    ax2.set_ylim(-5, 105)

    # Reference zones
    ax2.axvspan(0.05, 100, alpha=0.05, color='#1a9850', zorder=0)
    ax2.axvspan(100, 1000, alpha=0.05, color='#e66101', zorder=0)
    ax2.text(5, -3, 'Lightweight', fontsize=8, color='#1a9850', ha='center', fontweight='medium', alpha=0.8)
    ax2.text(300, -3, 'Standard', fontsize=8, color='#e66101', ha='center', fontweight='medium', alpha=0.8)

    ax2.set_xlabel('Model Parameters (Millions, log scale)', fontsize=11, fontweight='medium')
    ax2.set_ylabel('Primary Performance Metric (%)', fontsize=11, fontweight='medium')
    ax2.text(
        0.5, -0.35, '(b) Performance vs. Computational Cost',
        fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes
    )

    # Legend with 2x2 layout
    legend_label_map = {
        'Pure ViT': 'Pure ViT',
        'Hierarchical ViT': 'Hierarchical ViT',
        'CNN-Transformer Hybrid': 'CNN-Trans. Hybrid',
        'Foundation Model': 'FM',
    }
    legend_handles = [
        mpatches.Patch(color=ARCH_COLORS[cat], label=legend_label_map.get(cat, cat), alpha=0.8)
        for cat in ARCH_ORDER
        if cat in scatter_data['arch_category'].unique()
    ]
    ax2.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        fontsize=9,
        framealpha=1.0,
        edgecolor='#cccccc',
        fancybox=True,
        facecolor='white',
        ncol=2,  # 2x2 grid
        columnspacing=1.5,
        handletextpad=0.5
    )

    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='both')

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.25)

    save_fig(fig, 'rq1_performance.jpg')
    plt.close(fig)


if __name__ == "__main__":
    generate()


