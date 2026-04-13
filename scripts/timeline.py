"""Figure: Temporal Architecture Adoption Timeline

Professional stacked area chart showing adoption of different architecture types over time.
Enhanced with category breakdowns, growth indicators, and refined aesthetics.
Uses same classification logic as architecture_sankey.py for consistency.

Data Source: papers_clean.csv + models_clean.csv
Output: /Users/fadil/Desktop/Survey/68f823600ac5436c4d362b39/figures/timeline.jpg
"""

from utils import load_data, save_fig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# User-specified palette (8-color swatch)
COLORS = {
    'CNN-Transformer Hybrid': '#DA7C30',      # Orange  RGB(218,124,48)
    'Hierarchical ViT': '#3E9651',            # Green   RGB(62,150,81)
    'Pure ViT': '#396AB1',                    # Blue    RGB(57,106,177)
    'Foundation Model': '#CC2529',            # Red     RGB(204,37,41)
    'Vision-Language Model': '#6B4C9A',       # Purple  RGB(107,76,154)
    'Other': '#535154',                       # Dark gray RGB(83,81,84)
}

# Darker versions for text/accents
COLORS_DARK = {
    'CNN-Transformer Hybrid': '#A85E20',
    'Hierarchical ViT': '#2B6B39',
    'Pure ViT': '#274A7E',
    'Foundation Model': '#922428',
    'Vision-Language Model': '#4A356D',
    'Other': '#3A3A3C',
}

# Display order (bottom to top in stacked area) - largest categories at bottom
CATEGORY_ORDER = [
    'CNN-Transformer Hybrid',
    'Hierarchical ViT',
    'Pure ViT',
    'Foundation Model',
    'Vision-Language Model',
    'Other'
]

# Short labels for in-chart annotations (used inside stacked areas)
SHORT_LABELS = {
    'CNN-Transformer Hybrid': 'Hybrid',
    'Hierarchical ViT': 'Hierarchical',
    'Pure ViT': 'Pure ViT',
    'Foundation Model': 'FM',
    'Vision-Language Model': 'VLM',
    'Other': 'Other'
}

# Legend labels for the Yearly Breakdown panel (more descriptive)
PANEL_LABELS = {
    'CNN-Transformer Hybrid': 'CNN-Transformer Hybrid',
    'Hierarchical ViT': 'Hierarchical ViT',
    'Pure ViT': 'Pure ViT',
    'Foundation Model': 'FM',
    'Vision-Language Model': 'VLM',
    'Other': 'Other'
}

# Legend labels
LEGEND_LABELS = {
    'CNN-Transformer Hybrid': 'CNN-Transformer Hybrid',
    'Hierarchical ViT': 'Hierarchical ViT',
    'Pure ViT': 'Pure ViT',
    'Foundation Model': 'Foundation Model',
    'Vision-Language Model': 'Vision-Language Model',
    'Other': 'Other/Unspecified'
}


def smooth_data(x, y, num_points=200):
    """Create smooth interpolation for more aesthetic curves."""
    if len(x) < 4:
        x_smooth = np.linspace(x.min(), x.max(), num_points)
        y_smooth = np.interp(x_smooth, x, y)
        return x_smooth, y_smooth
    try:
        spline = make_interp_spline(x, y, k=2)
        x_smooth = np.linspace(x.min(), x.max(), num_points)
        y_smooth = spline(x_smooth)
        y_smooth = np.maximum(y_smooth, 0)
        return x_smooth, y_smooth
    except:
        return x, y


def darken_color(hex_color, factor=0.7):
    """Darken a hex color by a factor."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    darker = tuple(int(c * factor) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*darker)


def classify_architecture(row, hierarchical_backbones):
    """Classify paper into architecture category (matches Sankey logic)."""
    import re
    
    backbones_str = row.get('backbones', '')
    if pd.isna(backbones_str):
        backbones_str = ''
    backbone_str = str(backbones_str).lower()
    
    # Detect backbone types
    patterns = {
        'Swin': r'swin',
        'SegFormer': r'segformer|mit-b',
        'DaViT': r'davit',
        'Twins-SVT': r'twins',
        'PVT': r'\bpvt\b|pyramid vision',
        'TiMo': r'\btimo\b',
        'UniFormer': r'uniformer',
        'ViT': r'vit|vision transformer|\bvit\b',
        'PCT': r'\bpct\b|point cloud transformer',
        'SAM': r'\bsam\b|segment anything|hiera',
        'CLIP': r'\bclip\b',
        'Grounding DINO': r'grounding.?dino|\bglip\b',
        'Prithvi': r'prithvi',
        'DINOv2': r'dinov2|dino.?v2',
        'LLaVA': r'llava',
        'GPT': r'\bgpt\b',
        'InternViT': r'internvit|intern-?vit',
        'ResNet': r'resnet|resunet|res-?net',
        'TransUNet': r'transunet',
        'DETR': r'\bdetr\b|mask.?dino',
        'Mask2Former': r'mask2former',
    }
    
    detected = []
    for name, pat in patterns.items():
        if re.search(pat, backbone_str):
            detected.append(name)
    
    cat = row.get('category_clean', '')
    
    # Foundation Model
    if any(b in detected for b in ['SAM', 'Prithvi', 'Grounding DINO', 'CLIP', 'DINOv2']):
        return 'Foundation Model'
    if cat == 'Foundation Model Adaptation':
        return 'Foundation Model'
    
    # Vision-Language Model
    if any(b in detected for b in ['LLaVA', 'GPT', 'InternViT']):
        return 'Vision-Language Model'
    if cat == 'Vision-Language Model':
        return 'Vision-Language Model'
    
    # Hierarchical ViT (pure, not inside CNN hybrid)
    if cat == 'Pure Vision Transformer':
        if any(b in detected for b in hierarchical_backbones):
            return 'Hierarchical ViT'
        if any(b in detected for b in ['ViT', 'PCT']):
            return 'Pure ViT'
        # Check if backbone string mentions hierarchical architectures
        if any(h.lower() in backbone_str for h in hierarchical_backbones):
            return 'Hierarchical ViT'
        return 'Pure ViT'
    
    # CNN-Transformer Hybrid
    if cat == 'CNN-Transformer Hybrid':
        return 'CNN-Transformer Hybrid'
    
    # Other category - try to classify
    if cat == 'Other':
        if any(b in detected for b in hierarchical_backbones):
            return 'Hierarchical ViT'
        if any(b in detected for b in ['ViT', 'PCT']):
            return 'Pure ViT'
        if any(b in detected for b in ['ResNet', 'TransUNet', 'DETR', 'Mask2Former']):
            return 'CNN-Transformer Hybrid'
    
    return 'Other'


def generate():
    _, models, _, papers, _, _ = load_data()
    if models is None: 
        return

    hierarchical_backbones = ['Swin', 'SegFormer', 'Twins-SVT', 'PVT', 'DaViT', 'TiMo', 'UniFormer']
    
    # Merge models with papers to get year information
    merged = pd.merge(models, papers, on='paper_id', how='inner')
    merged = merged.dropna(subset=['year', 'category_clean'])
    merged['year'] = merged['year'].astype(int)
    
    # Apply architecture classification (same logic as Sankey)
    merged['arch_type'] = merged.apply(lambda r: classify_architecture(r, hierarchical_backbones), axis=1)
    
    # Count unique papers per category per year
    paper_counts = merged.groupby(['year', 'arch_type'])['paper_id'].nunique().reset_index()
    paper_counts.columns = ['year', 'category', 'count']
    
    # Pivot to get categories as columns
    pivot = paper_counts.pivot(index='year', columns='category', values='count').fillna(0)
    
    # Ensure all categories exist and are in correct order
    for cat in CATEGORY_ORDER:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[CATEGORY_ORDER]
    
    # Fill in all years from 2021-2025 (show full context even though first papers appeared in 2022)
    all_years = range(2021, 2026)
    pivot = pivot.reindex(all_years, fill_value=0)
    
    # Setup figure with professional styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Create figure with extra space on right for breakdown panel
    fig = plt.figure(figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Main timeline axes (left portion)
    # Move axes up to ensure bottom panel titles do not overlap with x-axis label
    ax = fig.add_axes([0.06, 0.18, 0.62, 0.76])
    ax.set_facecolor('#fafafa')
    
    # Prepare data
    years = pivot.index.values.astype(float)
    totals = pivot.sum(axis=1)
    
    # Smooth data for aesthetic curves
    num_points = 100
    years_smooth = np.linspace(years.min(), years.max(), num_points)
    
    data_smooth = []
    for cat in CATEGORY_ORDER:
        y = pivot[cat].values
        _, y_smooth = smooth_data(years, y, num_points)
        data_smooth.append(y_smooth)
    
    colors = [COLORS[cat] for cat in CATEGORY_ORDER]
    
    # Create stacked area chart
    ax.stackplot(years_smooth, data_smooth, labels=CATEGORY_ORDER, colors=colors, 
                 alpha=0.88, edgecolor='white', linewidth=0.3)
    
    # Add border lines on top of each stacked area
    cumulative = np.zeros(num_points)
    for i, cat in enumerate(CATEGORY_ORDER):
        cumulative = cumulative + data_smooth[i]
        ax.plot(years_smooth, cumulative, color=darken_color(colors[i], 0.8), 
                linewidth=1.8, alpha=0.9)
    
    # Add category count labels INSIDE the stacked areas
    for year in pivot.index:
        cumulative_y = 0
        for cat in CATEGORY_ORDER:
            count = int(pivot.loc[year, cat])
            if count > 0:
                # Position label in the middle of this category's area
                mid_y = cumulative_y + count / 2
                
                # Only show label if area is tall enough
                if count >= 2:
                    ax.annotate(f'{count}', xy=(year, mid_y), fontsize=9, 
                               fontweight='bold', ha='center', va='center',
                               color='white', alpha=0.95,
                               path_effects=[
                                   pe.withStroke(linewidth=2.5, foreground=darken_color(COLORS[cat], 0.6))
                               ])
                elif count == 1:
                    # Smaller font for single papers
                    ax.annotate(f'{count}', xy=(year, mid_y), fontsize=7.5, 
                               fontweight='bold', ha='center', va='center',
                               color='white', alpha=0.9,
                               path_effects=[
                                   pe.withStroke(linewidth=2, foreground=darken_color(COLORS[cat], 0.6))
                               ])
                cumulative_y += count
    
    # Add total count annotations above the stack
    for i, year in enumerate(pivot.index):
        total = int(totals.loc[year])
        y_pos = total + 1.5
        
        # Calculate year-over-year growth
        if i > 0:
            prev_year = list(pivot.index)[i-1]
            prev_total = int(totals.loc[prev_year])
            if prev_total > 0:
                growth = ((total - prev_total) / prev_total) * 100
                growth_text = f'+{growth:.0f}%' if growth > 0 else f'{growth:.0f}%'
                growth_color = '#2E7D32' if growth > 0 else '#C62828'
            else:
                growth_text = ''
                growth_color = '#666666'
        else:
            growth_text = ''
            growth_color = '#666666'
        
        # Total count badge
        ax.annotate(f'{total}', xy=(year, y_pos), fontsize=13, fontweight='bold',
                   ha='center', va='bottom', color='#222222',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#cccccc', alpha=0.95, linewidth=0.8))
        
        # Growth indicator (small text ABOVE the total badge)
        if growth_text:
            ax.annotate(growth_text, xy=(year, y_pos + 2.8), fontsize=9,
                       ha='center', va='bottom', color=growth_color, fontweight='semibold')
    
    # Add markers at actual data points
    for year in pivot.index:
        total = totals.loc[year]
        ax.plot(year, total, 'o', color='#333333', markersize=5, zorder=5,
               markeredgecolor='white', markeredgewidth=1)
    
    # Axis styling
    ax.set_xlabel('Publication Year', fontsize=12, fontweight='medium', labelpad=12, color='#333333')
    ax.set_ylabel('Number of Papers', fontsize=12, fontweight='medium', labelpad=12, color='#333333')
    
    ax.set_xlim(years.min() - 0.35, years.max() + 0.35)
    ax.set_ylim(0, pivot.sum(axis=1).max() * 1.42)  # More room for growth % above badges
    
    ax.set_xticks(pivot.index.values)
    ax.set_xticklabels([str(int(y)) for y in pivot.index.values], fontsize=12, fontweight='semibold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    
    # Grid and spines
    ax.yaxis.grid(True, linestyle='-', alpha=0.25, color='#aaaaaa', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#888888')
    
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#444444',
                   length=5, width=0.8, direction='out', pad=6)
    
    # =====================================================
    # RIGHT PANEL: Yearly breakdown table/summary
    # =====================================================
    ax_table = fig.add_axes([0.72, 0.21, 0.26, 0.72])
    ax_table.set_facecolor('white')
    ax_table.axis('off')
    
    # Add subtle border around panel
    for spine in ['top', 'bottom', 'left', 'right']:
        ax_table.spines[spine].set_visible(True)
        ax_table.spines[spine].set_color('#e0e0e0')
        ax_table.spines[spine].set_linewidth(1)
    
    # Title for the panel (removed to avoid duplicating the bottom "(b) Yearly Breakdown" caption)
    
    # Create visual breakdown for each year (exclude 2021 from breakdown panel)
    year_list = [y for y in pivot.index if y >= 2022]
    n_years = len(year_list)
    y_spacing = 0.72 / n_years  # Leave more room at bottom for legend
    
    for idx, year in enumerate(year_list):
        y_base = 0.88 - (idx + 0.5) * y_spacing
        
        # Year label with total in parentheses
        total = int(totals.loc[year])
        ax_table.text(0.05, y_base + 0.02, f'{int(year)}', fontsize=12, fontweight='bold',
                     ha='left', va='center', color='#333333', transform=ax_table.transAxes)
        ax_table.text(0.22, y_base + 0.02, f'({total})', fontsize=10, 
                     ha='left', va='center', color='#888888', transform=ax_table.transAxes)
        
        # Mini stacked bar for this year
        bar_left = 0.05
        bar_width = 0.90
        bar_height = 0.055
        
        if total > 0:
            cumulative_x = bar_left
            for cat in CATEGORY_ORDER:
                count = int(pivot.loc[year, cat])
                if count > 0:
                    width = (count / total) * bar_width
                    rect = FancyBboxPatch(
                        (cumulative_x, y_base - bar_height - 0.02), width, bar_height,
                        boxstyle='round,pad=0.003,rounding_size=0.012',
                        facecolor=COLORS[cat], edgecolor='white', linewidth=0.8,
                        alpha=0.92, transform=ax_table.transAxes
                    )
                    ax_table.add_patch(rect)
                    
                    # Count label inside bar - show for all counts
                    if width > 0.06:
                        # Larger font for bigger bars
                        ax_table.text(cumulative_x + width/2, y_base - bar_height/2 - 0.02, 
                                     str(count), fontsize=9, fontweight='bold', 
                                     ha='center', va='center', color='white', 
                                     transform=ax_table.transAxes)
                    elif count >= 1:
                        # Smaller font for narrow bars (single papers)
                        ax_table.text(cumulative_x + width/2, y_base - bar_height/2 - 0.02, 
                                     str(count), fontsize=7, fontweight='bold', 
                                     ha='center', va='center', color='white', 
                                     transform=ax_table.transAxes)
                    
                    cumulative_x += width
    
    # Legend at bottom of panel - 2-2-1 layout with VLM below Pure ViT
    box_size = 0.032
    legend_y_top = 0.12
    legend_y_mid = 0.06
    legend_y_bottom = 0.00
    
    # Only show categories with data
    active_cats = [cat for cat in CATEGORY_ORDER if pivot[cat].sum() > 0]
    
    # 2-2-1 layout positions (VLM aligned below Pure ViT)
    positions = [
        (0.05, legend_y_top),   # CNN-Transformer Hybrid
        (0.55, legend_y_top),   # Hierarchical ViT
        (0.05, legend_y_mid),   # Pure ViT
        (0.55, legend_y_mid),   # FM
        (0.05, legend_y_bottom), # VLM (below Pure ViT)
    ]
    
    for i, cat in enumerate(active_cats):
        if i < len(positions):
            x, y = positions[i]
            
            # Color box
            rect = FancyBboxPatch(
                (x, y - box_size/2), box_size, box_size,
                boxstyle='round,pad=0.003,rounding_size=0.008',
                facecolor=COLORS[cat], edgecolor='#cccccc', linewidth=0.5,
                alpha=0.95, transform=ax_table.transAxes
            )
            ax_table.add_patch(rect)
            
            # Label to the right of box - use PANEL_LABELS for descriptive names
            ax_table.text(x + box_size + 0.015, y, PANEL_LABELS[cat],
                         fontsize=7.5, ha='left', va='center', color='#444444',
                         fontweight='medium', transform=ax_table.transAxes)
    
    # Add key insight annotation and sample size
    total_papers = int(pivot.sum().sum())
    # Calculate growth from first year with papers to last year
    first_nonzero_idx = next((i for i, v in enumerate(totals) if v > 0), 0)
    first_nonzero = int(totals.iloc[first_nonzero_idx])
    growth_overall = ((int(totals.iloc[-1]) - first_nonzero) / first_nonzero) * 100 if first_nonzero > 0 else 0
    
    ax.annotate(f'n = {total_papers} papers', xy=(0.99, 0.97), xycoords='axes fraction',
               fontsize=10, ha='right', va='top', color='#666666', style='italic')
    
    # Add growth insight below the title
    ax.annotate(f'{growth_overall:.0f}% growth (2021→2025)', xy=(0.99, 0.91), 
               xycoords='axes fraction', fontsize=9, ha='right', va='top', 
               color='#2E7D32', fontweight='medium')

    # Add figure title at bottom
    # fig.text(0.5, 0.02, 'Temporal Architecture Adoption Timeline', fontsize=13, fontweight='bold',
    #          ha='center', va='bottom', color='#333333')
    
    # Panel (a) title - Main Chart
    # Center calculation: 0.06 (left) + 0.62 (width) / 2 = 0.37
    fig.text(0.37, 0.02, '(a) Temporal Architecture Adoption Timeline', fontsize=12, fontweight='bold',
             ha='center', va='bottom', color='#333333')
             
    # Panel (b) title - Yearly Breakdown
    # Center calculation: 0.72 (left) + 0.26 (width) / 2 = 0.85
    fig.text(0.85, 0.02, '(b) Yearly Breakdown', fontsize=12, fontweight='bold',
             ha='center', va='bottom', color='#333333')
    
    save_fig(fig, 'timeline.jpg')
    plt.close(fig)
    
    # Print summary statistics
    print("\n📊 Timeline Summary:")
    print(f"   Years covered: {int(years.min())} - {int(years.max())}")
    print(f"   Total papers: {total_papers}")
    print(f"   Overall growth: {growth_overall:.0f}%")
    print(f"\n   Papers by category:")
    for cat in CATEGORY_ORDER:
        total = int(pivot[cat].sum())
        if total > 0:
            pct = (total / total_papers) * 100
            print(f"   - {cat}: {total} ({pct:.1f}%)")
    print(f"\n   Papers per year:")
    for year in pivot.index:
        print(f"   - {year}: {int(totals.loc[year])}")


if __name__ == "__main__":
    generate()
