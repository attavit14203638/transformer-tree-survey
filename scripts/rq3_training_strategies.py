"""Figure 8: Training Strategy Analysis (Redesigned 3-Panel)

Panel (a): Horizontal grouped bar - Strategy distribution with FM/Specialist breakdown
Panel (b): Stacked bar - FM types and their tuning methods (DATA-DRIVEN)
Panel (c): Scatter plot - Label efficiency (DATA-DRIVEN, no fitted trend lines)

Data Source: training_clean.csv + performance_clean.csv
Output: /Users/fadil/Desktop/Survey/68f823600ac5436c4d362b39/figures/rq3_training.jpg
"""

from utils import load_data, save_fig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import re

# Metrics that should NOT appear on a 0-100% scale
NON_PERCENTAGE_METRICS = ['RMSE', 'MAE', 'MSE', 'PSNR', 'R²', 'R-squared', 'r', 'Abs Rel', 'RMSE Log']

def is_percentage_metric(metric_name):
    """Check if metric should be on a 0-100% scale."""
    if pd.isna(metric_name):
        return True  # Assume percentage if unknown
    primary = str(metric_name).split('|')[0]
    for non_pct in NON_PERCENTAGE_METRICS:
        # Use word-boundary regex to avoid false positives (e.g., 'r' matching 'Accuracy')
        pattern = r'\b' + re.escape(non_pct) + r'\b'
        if re.search(pattern, str(metric_name), re.IGNORECASE):
            if re.search(pattern, primary, re.IGNORECASE):
                return False
    return True

def normalize_metric_value(value, metric_name):
    """Convert decimal values (0-1) to percentage (0-100) for percentage metrics."""
    if pd.isna(value):
        return value
    if not is_percentage_metric(metric_name):
        return None  # Will be filtered out
    # If value is between 0 and 1 (exclusive of 1 to avoid edge cases), likely decimal format
    if 0 < value <= 1.0:
        return value * 100
    return value

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Professional color palette
COLORS = {
    'fm': '#c44e52',           # Muted red for foundation models
    'specialist': '#4c72b0',   # Muted blue for specialists
    'zero_shot': '#937860',    # Brown for zero-shot
    'peft': '#da8bc3',         # Pink for PEFT
    'full_ft': '#8172b3',      # Purple for full fine-tuning
    'accent': '#55a868',       # Green for highlights
    'grid': '#e0e0e0',
    'text_dark': '#2d2d2d',
    'text_light': '#666666',
}

# FM Adaptation data from Table 4 in the paper
FM_ADAPTATION_DATA = [
    {
        'model': 'SAM/SAM2',
        'methods': [
            {'name': 'Zero-shot', 'perf_low': 41, 'perf_high': 65, 'params_pct': 0},
            {'name': 'PEFT (LoRA)', 'perf_low': 85, 'perf_high': 95, 'params_pct': 5},
            {'name': 'Full FT', 'perf_low': 92, 'perf_high': 98, 'params_pct': 100},
        ]
    },
    {
        'model': 'Grounding-DINO',
        'methods': [
            {'name': 'Zero-shot', 'perf_low': 35, 'perf_high': 50, 'params_pct': 0},
            {'name': 'PEFT (ASCS)', 'perf_low': 58, 'perf_high': 62, 'params_pct': 0.28},
        ]
    },
    {
        'model': 'CLIP/DOFA-CLIP',
        'methods': [
            {'name': 'Full FT', 'perf_low': 75, 'perf_high': 85, 'params_pct': 100},
        ]
    },
    {
        'model': 'Prithvi-100M',
        'methods': [
            {'name': 'Head-only', 'perf_low': 78, 'perf_high': 82, 'params_pct': 2},
        ]
    },
]


def categorize_strategy(row):
    """Categorize learning strategy."""
    strategy = str(row.get('learning_strategy_clean', 'Fully Supervised'))
    
    if 'zero' in strategy.lower():
        return 'Zero-Shot'
    elif 'self' in strategy.lower():
        return 'Self-Supervised'
    elif 'semi' in strategy.lower():
        return 'Semi-Supervised'
    elif 'mixed' in strategy.lower():
        # Mixed strategies that combine multiple approaches -> count as Fully Supervised
        return 'Fully Supervised'
    else:
        return 'Fully Supervised'


def is_foundation_model(row):
    """Check if paper uses foundation model."""
    fm = str(row.get('foundation_model', ''))
    if pd.isna(fm) or fm.lower() in ['no', 'nan', '']:
        return 'Specialist'
    return 'Foundation Model'


def draw_panel_a(ax, train):
    """Panel (a): Horizontal grouped bar - Strategy distribution."""
    # Get counts
    strategy_order = ['Fully Supervised', 'Semi-Supervised', 'Self-Supervised', 'Zero-Shot']
    
    # Calculate FM and Specialist counts per strategy
    data = []
    for strat in strategy_order:
        strat_data = train[train['strategy_category'] == strat]
        fm_count = len(strat_data[strat_data['fm_category'] == 'Foundation Model'])
        spec_count = len(strat_data[strat_data['fm_category'] == 'Specialist'])
        if fm_count + spec_count > 0:
            # For self-supervised, collapse FM + specialist into a single total
            if strat == 'Self-Supervised':
                # Use unique paper count (3) rather than model-count (3+3)
                if 'paper_id' in strat_data.columns:
                    total_unique = strat_data['paper_id'].nunique()
                else:
                    total_unique = len(strat_data)
                data.append({
                    'strategy': strat.replace(' ', '\n'),
                    'FM': 0,
                    'Specialist': total_unique,
                    'total': total_unique,
                    'single_label': True,
                })
            else:
                data.append({
                    'strategy': strat.replace(' ', '\n'),
                    'FM': fm_count,
                    'Specialist': spec_count,
                    'total': fm_count + spec_count,
                    'single_label': False,
                })
    
    if not data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    df = pd.DataFrame(data)
    df = df.sort_values('total', ascending=True)  # Sort for horizontal bar
    
    y_pos = np.arange(len(df))
    bar_height = 0.6
    
    # Draw stacked horizontal bars
    bars_spec = ax.barh(y_pos, df['Specialist'], height=bar_height, 
                        color=COLORS['specialist'], label='Specialist', edgecolor='white', linewidth=0.5)
    bars_fm = ax.barh(y_pos, df['FM'], height=bar_height, left=df['Specialist'],
                      color=COLORS['fm'], label='Foundation Model', edgecolor='white', linewidth=0.5)
    
    # Add count labels
    for i, (idx, row) in enumerate(df.iterrows()):
        total = row['total']
        # Label at end of bar
        ax.text(total + 0.5, i, f"{total}", va='center', ha='left', 
                fontsize=9, fontweight='medium', color=COLORS['text_dark'])
        
        # Internal labels if space permits (skip for collapsed Self-Supervised row)
        single_label = bool(row.get('single_label', False))
        if not single_label:
            if row['Specialist'] >= 3:
                ax.text(row['Specialist']/2, i, f"{row['Specialist']}", va='center', ha='center',
                       fontsize=8, color='white', fontweight='medium')
            if row['FM'] >= 3:
                ax.text(row['Specialist'] + row['FM']/2, i, f"{row['FM']}", va='center', ha='center',
                       fontsize=8, color='white', fontweight='medium')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['strategy'], fontsize=11)
    ax.set_xlabel('Number of Papers', fontsize=12, fontweight='medium')
    ax.set_xlim(0, df['total'].max() * 1.15)
    # ax.set_title('(a) Learning Strategy Distribution', fontsize=13, fontweight='bold', pad=10)
    ax.text(0.5, -0.20, '(a) Learning Strategy Distribution', 
            fontsize=14, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    
    # Clean up
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='#cccccc')


def categorize_fm_type(fm):
    """Categorize foundation model into main types."""
    if pd.isna(fm) or str(fm).lower() == 'no':
        return None
    fm_lower = str(fm).lower()
    # Check self-built / non-standard FMs first to avoid false substring matches
    if 'timo' in fm_lower:
        return 'Other FM'
    if 'fomo' in fm_lower:
        return 'Other FM'
    if 'detr' in fm_lower:
        return 'Other FM'
    if 'swin' in fm_lower and 'imagenet' in fm_lower:
        return 'Other FM'
    if 'sam' in fm_lower:
        return 'SAM/SAM2'
    elif 'clip' in fm_lower or 'dofa' in fm_lower or 'siglip' in fm_lower:
        return 'CLIP-based'
    elif 'prithvi' in fm_lower:
        return 'Prithvi'
    elif 'dino' in fm_lower:
        return 'DINO-based'
    elif 'llava' in fm_lower or 'intern' in fm_lower or 'phi' in fm_lower or 'gpt' in fm_lower:
        return 'VLM'
    else:
        return 'Other FM'

def categorize_tuning_method(method):
    """Categorize tuning method into main types."""
    if pd.isna(method):
        return 'Not Specified'
    method_lower = str(method).lower()
    if 'zero' in method_lower:
        return 'Zero-shot'
    elif 'peft' in method_lower or 'lora' in method_lower or 'adapter' in method_lower or 'prompt' in method_lower:
        return 'PEFT'
    elif 'head' in method_lower or 'progressive' in method_lower:
        return 'Partial FT'
    elif 'full' in method_lower:
        return 'Full FT'
    elif 'not applicable' in method_lower:
        return 'Evaluation Only'
    else:
        return 'Other'

def draw_panel_b(ax, train):
    """Panel (b): Heatmap - FM types vs tuning methods (DATA-DRIVEN)."""
    import matplotlib.colors as mcolors
    
    # Filter to FM papers only
    fm_papers = train[train['foundation_model'] != 'No'].copy()
    fm_papers = fm_papers[fm_papers['foundation_model'].notna()]
    
    # Categorize FM types and tuning methods
    fm_papers['fm_type'] = fm_papers['foundation_model'].apply(categorize_fm_type)
    fm_papers['tuning_cat'] = fm_papers['finetuning_method'].apply(categorize_tuning_method)
    
    # Filter out None fm_types and evaluation-only papers (not adapting FMs)
    fm_papers = fm_papers[fm_papers['fm_type'].notna()]
    fm_papers = fm_papers[fm_papers['tuning_cat'] != 'Evaluation Only']
    
    # Define order (most common first for FM types, Other FM moved to last)
    fm_order = ['SAM/SAM2', 'Other FM', 'Prithvi', 'VLM', 'DINO-based', 'CLIP-based']
    tuning_order = ['Zero-shot', 'PEFT', 'Partial FT', 'Full FT']
    
    # Build data matrix (only include FM types with data)
    data_matrix = []
    fm_labels = []
    fm_totals = []
    for fm in fm_order:
        fm_data = fm_papers[fm_papers['fm_type'] == fm]
        if len(fm_data) > 0:
            row = []
            for tuning in tuning_order:
                count = len(fm_data[fm_data['tuning_cat'] == tuning])
                row.append(count)
            data_matrix.append(row)
            fm_labels.append(fm)
            fm_totals.append(len(fm_data))
    
    if len(data_matrix) == 0:
        ax.text(0.5, 0.5, 'Insufficient FM data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=11, color=COLORS['text_light'])
        return
    
    data_matrix = np.array(data_matrix)
    
    # Create custom colormap (white to deep purple)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom', ['#f7f7f7', '#d4b9da', '#c994c7', '#df65b0', '#980043'], N=256
    )
    
    # Draw heatmap
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=data_matrix.max())
    
    # Add cell annotations
    for i in range(len(fm_labels)):
        for j in range(len(tuning_order)):
            val = data_matrix[i, j]
            if val > 0:
                text_color = 'white' if val >= data_matrix.max() * 0.5 else '#333333'
                ax.text(j, i, str(int(val)), ha='center', va='center',
                       fontsize=11, fontweight='bold', color=text_color)
    
    # Add row totals on the right
    for i, (label, total) in enumerate(zip(fm_labels, fm_totals)):
        ax.text(len(tuning_order) - 0.5 + 0.55, i, f'{total}', ha='left', va='center',
               fontsize=9, fontweight='medium', color=COLORS['text_dark'])
    
    # Add column totals at top
    col_totals = data_matrix.sum(axis=0)
    for j, total in enumerate(col_totals):
        if total > 0:
            ax.text(j, -0.6, f'{int(total)}', ha='center', va='bottom',
                   fontsize=9, fontweight='medium', color=COLORS['text_dark'])
    
    # Axis labels
    ax.set_xticks(np.arange(len(tuning_order)))
    ax.set_xticklabels(tuning_order, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(fm_labels)))
    ax.set_yticklabels(fm_labels, fontsize=9)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(tuning_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(fm_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Title
    ax.text(0.5, -0.20, '(b) FM Adaptation Methods', 
            fontsize=14, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=15, pad=0.15)
    cbar.set_label('Papers', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_c(ax, train, perf):
    """Panel (c): Scatter plot - Label efficiency (DATA-DRIVEN, enhanced visualization)."""
    # Merge training and performance data (include metric_name for normalization)
    merged = pd.merge(train, perf[['paper_id', 'model_id', 'metric_value_numeric', 'metric_name']], 
                      on=['paper_id', 'model_id'], how='left')
    
    # Normalize metric values: convert decimal (0-1) to percentage and filter non-percentage metrics
    merged = merged.copy()
    merged['metric_normalized'] = merged.apply(
        lambda row: normalize_metric_value(row['metric_value_numeric'], row.get('metric_name', '')), axis=1
    )
    
    # Filter valid data points
    scatter_data = merged.dropna(subset=['labeled_samples_numeric', 'metric_normalized'])
    scatter_data = scatter_data[scatter_data['labeled_samples_numeric'] > 0]
    scatter_data = scatter_data[scatter_data['metric_normalized'] > 0]
    scatter_data = scatter_data[scatter_data['metric_normalized'] <= 100]
    # Exclude pre-training/instruction-tuning corpora (>100K samples are not task-specific)
    scatter_data = scatter_data[scatter_data['labeled_samples_numeric'] <= 100000]
    
    if len(scatter_data) == 0:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
               transform=ax.transAxes, fontsize=11, color=COLORS['text_light'])
        ax.text(0.5, -0.35, '(c) Label Efficiency', 
                fontsize=12, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Add subtle background zones (descriptive, not claiming thresholds)
    ax.axvspan(0.5, 100, alpha=0.04, color=COLORS['fm'], zorder=0)
    ax.axvspan(100, 1000, alpha=0.03, color='#ffd700', zorder=0)
    ax.axvspan(1000, 100000, alpha=0.04, color=COLORS['specialist'], zorder=0)
    
    # Zone labels at top
    ax.text(10, 103, 'Few-shot\n(<100)', fontsize=7, color=COLORS['fm'],
           ha='center', va='bottom', fontweight='medium', alpha=0.8)
    ax.text(300, 103, 'Low-data\n(100-1k)', fontsize=7, color='#b35806',
           ha='center', va='bottom', fontweight='medium', alpha=0.8)
    ax.text(7000, 103, 'Full-data\n(>1k)', fontsize=7, color=COLORS['specialist'],
           ha='center', va='bottom', fontweight='medium', alpha=0.8)
    
    # Plot by FM category with distinct markers and larger size
    markers = {'Specialist': 'o', 'Foundation Model': 'D'}  # Diamond for FM
    colors = {'Specialist': COLORS['specialist'], 'Foundation Model': COLORS['fm']}
    
    for cat in ['Specialist', 'Foundation Model']:
        cat_data = scatter_data[scatter_data['fm_category'] == cat]
        if len(cat_data) > 0:
            ax.scatter(cat_data['labeled_samples_numeric'], cat_data['metric_normalized'],
                      c=colors[cat], label=cat, s=90, alpha=0.8,
                      marker=markers[cat], edgecolors='white', linewidth=1.2, zorder=5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Labeled Samples (log scale)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='medium')
    ax.text(0.5, -0.20, '(c) Label Efficiency by Model Type', 
            fontsize=14, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    
    # Set axis limits
    ax.set_xlim(5, 200000)
    ax.set_ylim(0, 115)
    
    # Add sample count annotations
    n_spec = len(scatter_data[scatter_data['fm_category'] == 'Specialist'])
    n_fm = len(scatter_data[scatter_data['fm_category'] == 'Foundation Model'])
    # Moved to left to avoid overlap with legend
    ax.text(0.45, 0.02, f'n={n_spec+n_fm} experiments\n({n_spec} specialist, {n_fm} FM)',
           transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='#cccccc'))
    
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, which='both', zorder=0)
    ax.set_ylim(0, 108)
    ax.set_xlim(5, 200000)
    
    # Legend
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='#cccccc')


def generate():
    geo, models, perf, papers, train, datasets = load_data()
    if train is None:
        print("❌ Could not load training data")
        return
    
    # Merge with papers for year info
    if papers is not None:
        train = pd.merge(train, papers[['paper_id', 'year']], on='paper_id', how='left')
    
    # Categorize strategies
    train['strategy_category'] = train.apply(categorize_strategy, axis=1)
    train['fm_category'] = train.apply(is_foundation_model, axis=1)
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(15, 5.5))
    fig.patch.set_facecolor('white')
    
    # Grid: 3 panels with different widths
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.85, 1.15, 1.0], wspace=0.22)
    
    # Panel (a): Horizontal grouped bar
    ax1 = fig.add_subplot(gs[0])
    draw_panel_a(ax1, train)
    
    # Panel (b): FM tuning methods (DATA-DRIVEN)
    ax2 = fig.add_subplot(gs[1])
    draw_panel_b(ax2, train)
    
    # Panel (c): Scatter plot (DATA-DRIVEN, no trend lines)
    ax3 = fig.add_subplot(gs[2])
    draw_panel_c(ax3, train, perf)
    
    # Adjust layout
    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.25)
    
    # Save figure
    save_fig(fig, 'rq3_training.jpg')
    plt.close(fig)
    
    # Print summary
    print("\n" + "="*60)
    print("RQ3 FIGURE GENERATED (All panels DATA-DRIVEN)")
    print("="*60)
    print("\n✅ Panel (a): Learning strategy distribution (from CSV)")
    print("✅ Panel (b): FM tuning methods breakdown (from CSV)")
    print("✅ Panel (c): Label efficiency scatter (from CSV, no fitted curves)")
    
    # Statistics
    total_papers = len(train)
    fm_total = len(train[train['fm_category'] == 'Foundation Model'])
    print(f"\n📊 Summary: {total_papers} papers, {fm_total} FM ({100*fm_total/total_papers:.1f}%)")


if __name__ == "__main__":
    generate()
