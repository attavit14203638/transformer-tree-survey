"""Figure 7: Cross-Environment Generalization Analysis (Improved 2-Panel)

Panel (a): Horizontal bar chart showing key transfer experiments with performance drops
           - Grouped by transfer type (Cross-Biome, Cross-Geographic, Cross-Sensor)
Panel (b): Horizontal bar chart of generalization factors by frequency
           - Improved factor extraction with better categorization

Data Source: training_clean.csv (cross_env_tested, generalization_summary, factors_affecting_generalization)
Output: /Users/fadil/Desktop/Survey/68f823600ac5436c4d362b39/figures/rq2_generalization.jpg
"""

from utils import load_data, save_fig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

# Set publication-quality style (matching rq1_performance_efficiency.py)
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

# Color scheme - grouped by transfer type
TRANSFER_COLORS = {
    'Cross-Biome': '#d73027',       # Red - severe drops
    'Cross-Geographic': '#fc8d59',   # Orange - moderate drops  
    'Cross-Sensor/View': '#fee090',  # Yellow - varied
    'Within-Region': '#91bfdb',      # Light blue - mild drops
}

FACTOR_COLORS = {
    'Structural': '#1a9850',    # Green
    'Spectral': '#2166ac',      # Blue
    'Environmental': '#7b3294', # Purple
    'Data-Related': '#e66101',  # Orange
}

def extract_transfer_results(train_df):
    """Extract comprehensive transfer results with grouping by transfer type."""
    results = []
    
    # Comprehensive key transfer experiments from training_clean.csv analysis
    # Format: paper_id, label, source, target, in_domain, cross_domain, metric, transfer_type
    key_transfers = [
        # Cross-Biome Transfers (largest drops)
        {'paper_id': 3, 'label': 'HTC', 
         'transfer': 'Multi-site → Australia', 
         'in_domain': 40.98, 'cross_domain': 4.07, 'metric': 'AP50',
         'type': 'Cross-Biome'},
        # REMOVED: SegFormer Denmark→Rwanda was NOT a transfer experiment
        # Source paper (gominski_benchmarking_2023) trained separate models on each dataset
        {'paper_id': 49, 'label': 'SAM/G-DINO',
         'transfer': 'ID → OOD Zones',
         'in_domain': 81.0, 'cross_domain': 50.5, 'metric': 'AP',
         'type': 'Cross-Biome'},
        
        # Cross-Geographic (same biome type, different regions)
        # Note: ViTDet+SAM shows Quebec-trained → Global test (29% relative drop)
        # The reverse (FLICKRTREE-trained → Quebec) shows only 4pp drop, discussed in text
        {'paper_id': 29, 'label': 'ViTDet+SAM',
         'transfer': 'Quebec → Global',
         'in_domain': 81.0, 'cross_domain': 57.9, 'metric': 'AP50',
         'type': 'Cross-Geographic'},
        # Note: Shows RO → MA transfer only (45.0% F1); RO → PA is much worse (6.1% F1)
        {'paper_id': 7, 'label': 'DeepLabv3+',
         'transfer': 'Amazon RO → MA',
         'in_domain': 64.9, 'cross_domain': 45.0, 'metric': 'F1',
         'type': 'Cross-Geographic'},
        {'paper_id': 30, 'label': 'TransU-Net++',
         'transfer': 'Amazon → Atlantic',
         'in_domain': 97.2, 'cross_domain': 88.2, 'metric': 'OA',
         'type': 'Cross-Geographic'},
        {'paper_id': 62, 'label': 'MTCDNet',
         'transfer': 'Temperate → Mixed',
         'in_domain': 91.5, 'cross_domain': 87.3, 'metric': 'mAP',
         'type': 'Cross-Geographic'},
        
        # Cross-Sensor/Viewpoint
        {'paper_id': 25, 'label': 'ViT-B16',
         'transfer': 'Ground → UAV',
         'in_domain': 87.0, 'cross_domain': 68.0, 'metric': 'Acc',
         'type': 'Cross-Sensor/View'},
        {'paper_id': 46, 'label': 'ViT/DenseNet',
         'transfer': 'Summer → Fall',
         'in_domain': 78.0, 'cross_domain': 48.0, 'metric': 'Acc',
         'type': 'Cross-Sensor/View'},
        
        # Within-Region (different compositions/sites)
        {'paper_id': 18, 'label': 'DETR',
         'transfer': 'Mixed → Deciduous',
         'in_domain': 86.0, 'cross_domain': 71.0, 'metric': 'F1',
         'type': 'Within-Region'},
        {'paper_id': 36, 'label': 'WetMapFormer',
         'transfer': 'Multi-site Wetland',
         'in_domain': 98.2, 'cross_domain': 96.3, 'metric': 'AA',
         'type': 'Within-Region'},
    ]
    
    for t in key_transfers:
        drop = ((t['cross_domain'] - t['in_domain']) / t['in_domain']) * 100
        results.append({
            'paper_id': t['paper_id'],
            'label': t['label'],
            'transfer': t['transfer'],
            'in_domain': t['in_domain'],
            'cross_domain': t['cross_domain'],
            'drop': drop,
            'metric': t['metric'],
            'type': t['type']
        })
    
    return pd.DataFrame(results)

def extract_factors(train_df):
    """Extract and count generalization factors with improved categorization."""
    # More granular factor categories
    factor_mapping = {
        # Structural factors (canopy, crown, density)
        'Canopy density': ['canopy density', 'crown density', 'dense canopy', 'canopy structure', 
                          'crown overlap', 'overlapping crown', 'tree density', 'dense forest'],
        'Crown complexity': ['crown complexity', 'crown shape', 'tree crown', 'crown size', 
                            'crown boundary', 'crown structure', 'canopy complexity'],
        'Forest composition': ['forest type', 'forest composition', 'species composition', 
                              'vegetation type', 'mixed forest', 'deciduous', 'coniferous'],
        
        # Spectral/sensor factors
        'Spectral differences': ['spectral', 'band', 'wavelength', 'reflectance', 'signature'],
        'Resolution mismatch': ['resolution', 'gsd', 'scale', 'spatial resolution', 'pixel size'],
        'Sensor variation': ['sensor', 'modality', 'camera', 'lidar', 'sar'],
        
        # Environmental factors
        'Geographic shift': ['geographic', 'domain shift', 'location', 'region', 'spatial transfer'],
        'Lighting/Weather': ['lighting', 'illumination', 'weather', 'shadow', 'atmospheric'],
        'Seasonal variation': ['seasonal', 'temporal', 'phenolog', 'leaf-on', 'leaf-off'],
        
        # Data-related factors
        'Training data limits': ['training data', 'data quality', 'annotation', 'label', 
                                'sample', 'dataset diversity'],
        'Species diversity': ['species diversity', 'species variation', 'different species'],
    }
    
    factor_counts = Counter()
    factor_groups = {
        'Canopy density': 'Structural',
        'Crown complexity': 'Structural', 
        'Forest composition': 'Structural',
        'Spectral differences': 'Spectral',
        'Resolution mismatch': 'Spectral',
        'Sensor variation': 'Spectral',
        'Geographic shift': 'Environmental',
        'Lighting/Weather': 'Environmental',
        'Seasonal variation': 'Environmental',
        'Training data limits': 'Data-Related',
        'Species diversity': 'Data-Related',
    }
    
    # Get papers with factors
    factors_data = train_df[train_df['factors_affecting_generalization'].notna()]['factors_affecting_generalization']
    
    for factors_str in factors_data:
        factors_text = str(factors_str).lower()
        matched_factors = set()  # Avoid double-counting
        
        for factor_name, keywords in factor_mapping.items():
            for keyword in keywords:
                if keyword in factors_text and factor_name not in matched_factors:
                    factor_counts[factor_name] += 1
                    matched_factors.add(factor_name)
                    break
    
    # Build DataFrame with groups
    factor_df = pd.DataFrame([
        {'factor': k, 'count': v, 'group': factor_groups.get(k, 'Other')}
        for k, v in factor_counts.most_common(10) if v >= 3  # Only factors mentioned 3+ times
    ])
    
    return factor_df

def generate():
    geo, models, perf, papers, train, datasets = load_data()
    if train is None:
        print("❌ Could not load data")
        return
    
    # Count cross-env tested papers
    cross_env_count = train[train['cross_env_tested'] == 'Yes'].shape[0]
    total_count = train.shape[0]
    print(f"\n📊 Cross-environment testing: {cross_env_count}/{total_count} papers ({100*cross_env_count/total_count:.1f}%)")
    
    # Create figure with two panels - slightly taller for legend space below
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor('white')
    
    # === Panel (a): Transfer experiments grouped by type ===
    transfer_df = extract_transfer_results(train)
    
    # Sort by drop magnitude within groups, but group by type
    type_order = ['Cross-Biome', 'Cross-Geographic', 'Cross-Sensor/View', 'Within-Region']
    transfer_df['type_order'] = transfer_df['type'].map({t: i for i, t in enumerate(type_order)})
    transfer_df = transfer_df.sort_values(['type_order', 'drop'], ascending=[True, True])
    
    y_pos = np.arange(len(transfer_df))
    
    # Create horizontal bar chart with color by type
    bar_colors = [TRANSFER_COLORS[t] for t in transfer_df['type']]
    bars = ax1.barh(y_pos, transfer_df['drop'], color=bar_colors, alpha=0.85, height=0.7,
                   edgecolor='white', linewidth=0.5)
    
    # Add value annotations - show percentage on ALL bars
    for i, (idx, row) in enumerate(transfer_df.iterrows()):
        drop_abs = abs(row['drop'])
        
        if drop_abs < 6:
            # Small drops (<5%): place label outside bar with arrow pointing to it
            ax1.annotate(f"{row['drop']:.0f}%", 
                        xy=(row['drop'], i),  # Arrow points to end of bar
                        xytext=(-15, i),  # Label position (further left)
                        fontsize=8.5, color='#404040', fontweight='bold',
                        va='center', ha='right',
                        arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))
        else:
            # Larger drops: place label just outside the bar end
            ax1.text(row['drop'] + 2, i, f"{row['drop']:.0f}%", 
                    va='center', ha='left', fontsize=8.5, color='#404040', fontweight='bold')
        
        # Add in-domain → cross-domain values on right
        ax1.text(2, i, f"{row['in_domain']:.0f}→{row['cross_domain']:.0f} {row['metric']}", 
                va='center', ha='left', fontsize=7.5, color='#505050', style='italic')
    
    # Y-axis labels: Model + Transfer
    y_labels = [f"{row['label']}\n{row['transfer']}" for _, row in transfer_df.iterrows()]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=10.5)
    
    ax1.set_xlabel('Performance Drop (%)', fontsize=13, fontweight='medium')
    # ax1.set_title('(a) Cross-Environment Transfer Performance', fontsize=14, fontweight='bold', pad=15)
    ax1.text(0.5, -0.25, '(a) Cross-Environment Transfer Performance', 
             fontsize=14, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax1.axvline(x=0, color='#404040', linestyle='-', linewidth=1.2, alpha=0.6)
    ax1.set_xlim(-95, 12)
    ax1.xaxis.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    
    # Add mean drop annotation - position in the MIDDLE of the dashed line
    avg_drop = transfer_df['drop'].mean()
    ax1.axvline(x=avg_drop, color='#7b3294', linestyle='--', linewidth=2, alpha=0.8)
    # Place mean label in the middle of the bar chart (vertically centered)
    mid_y = (len(transfer_df) - 1) / 2
    ax1.text(avg_drop, mid_y, f'Mean: {avg_drop:.0f}%', 
            fontsize=9, color='#7b3294', fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#7b3294', alpha=0.9))
    
    # Add type separators and labels
    current_y = -0.5
    type_starts = {}
    for t in type_order:
        mask = transfer_df['type'] == t
        if mask.any():
            indices = transfer_df[mask].index
            first_idx = transfer_df.index.get_loc(indices[0])
            type_starts[t] = first_idx
    
    # Add legend for transfer types - positioned below chart to avoid blocking bars
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TRANSFER_COLORS[t], label=t, alpha=0.85) 
                      for t in type_order if t in transfer_df['type'].values]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=8, framealpha=0.95, edgecolor='#cccccc', fancybox=True)
    
    # === Panel (b): Generalization factors by frequency ===
    factors_df = extract_factors(train)
    
    if len(factors_df) > 0:
        # Sort by count
        factors_df = factors_df.sort_values('count', ascending=True)
        
        y_pos2 = np.arange(len(factors_df))
        
        # Color by group
        group_colors = {'Structural': '#1a9850', 'Spectral': '#2166ac', 
                       'Environmental': '#7b3294', 'Data-Related': '#e66101'}
        bar_colors2 = [group_colors.get(g, '#878787') for g in factors_df['group']]
        
        bars2 = ax2.barh(y_pos2, factors_df['count'], color=bar_colors2, 
                        alpha=0.85, height=0.7, edgecolor='white', linewidth=0.5)
        
        # Add value annotations
        for i, (idx, row) in enumerate(factors_df.iterrows()):
            ax2.text(row['count'] + 0.5, i, f"{row['count']}", 
                    va='center', ha='left', fontsize=9, color='#404040', fontweight='medium')
        
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(factors_df['factor'], fontsize=11)
        ax2.set_xlabel('Number of Papers Citing Factor', fontsize=13, fontweight='medium')
        # ax2.set_title('(b) Factors Affecting Generalization', fontsize=14, fontweight='bold', pad=15)
        ax2.text(0.5, -0.25, '(b) Factors Affecting Generalization', 
                 fontsize=14, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
        ax2.xaxis.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        
        # Set x-axis limit
        max_count = factors_df['count'].max()
        ax2.set_xlim(0, max_count + 4)
        
        # Add legend for factor groups - positioned below chart
        unique_groups = factors_df['group'].unique()
        legend_elements2 = [Patch(facecolor=group_colors[g], label=g, alpha=0.85) 
                           for g in ['Structural', 'Spectral', 'Environmental', 'Data-Related']
                           if g in unique_groups]
        ax2.legend(handles=legend_elements2, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=8, framealpha=0.95, edgecolor='#cccccc', fancybox=True)
    else:
        ax2.text(0.5, 0.5, 'Insufficient factor data', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12, color='#878787')
        # ax2.set_title('(b) Factors Affecting Generalization', fontsize=12, fontweight='bold', pad=15)
        ax2.text(0.5, -0.25, '(b) Factors Affecting Generalization', 
                 fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    
    # Adjust layout with extra bottom space for legends
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.25)
    
    # Save figure
    save_fig(fig, 'rq2_generalization.jpg')
    plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("RQ2 FIGURE SUMMARY STATISTICS (Improved)")
    print("="*70)
    print(f"\n📊 Panel (a) - Transfer Performance by Type:")
    print("-" * 60)
    for t in type_order:
        type_df = transfer_df[transfer_df['type'] == t]
        if len(type_df) > 0:
            avg = type_df['drop'].mean()
            print(f"\n  {t} (n={len(type_df)}, avg drop={avg:.1f}%):")
            for idx, row in type_df.iterrows():
                print(f"    Paper {row['paper_id']}: {row['in_domain']:.0f}% → {row['cross_domain']:.0f}% ({row['drop']:.1f}%)")
    
    print(f"\n  Overall mean drop: {avg_drop:.1f}%")
    
    if len(factors_df) > 0:
        print(f"\n📈 Panel (b) - Top Generalization Factors:")
        print("-" * 60)
        for idx, row in factors_df.sort_values('count', ascending=False).iterrows():
            print(f"  {row['factor']} ({row['group']}): {row['count']} papers")

if __name__ == "__main__":
    generate()
