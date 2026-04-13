"""Figure: Full Transformer Architecture Taxonomy Sankey Diagram

Multi-level Sankey diagram with paper citation details:
Level 1: Root (Transformer Models)
Level 2: Main Categories (VFM, VLM)  
Level 3: Sub-categories (Hierarchical ViT, Pure ViT, Hybrid, Foundation)
Level 4: Specific Backbones (Swin, SegFormer, SAM, CLIP, etc.)
Level 5: Paper citations (shown in hover + annotations)

Data Source: models_clean.csv, papers_clean.csv
Output: /Users/fadil/Desktop/Survey/68f823600ac5436c4d362b39/figures/architecture_sankey.jpg
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

# Paths - relative to repository root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_citation(citation_key):
    """Format citation key to short author-year format."""
    if pd.isna(citation_key):
        return "Unknown"
    # Extract author and year from citation_key like "smith_title_2024"
    parts = str(citation_key).split('_')
    if len(parts) >= 2:
        author = parts[0].capitalize()
        year = parts[-1] if parts[-1].isdigit() else "20XX"
        return f"{author} ({year})"
    return str(citation_key)[:15]


def detect_backbones(backbones_str):
    """Extract key backbone architectures from the backbones field."""
    if pd.isna(backbones_str):
        return []
    
    backbones = []
    backbone_str = str(backbones_str).lower()
    
    patterns = {
        # Hierarchical ViT
        'Swin': r'swin',
        'SegFormer': r'segformer|mit-b',
        'DaViT': r'davit',
        'Twins-SVT': r'twins',
        'PVT': r'\bpvt\b|pyramid vision',
        'TiMo': r'\btimo\b',
        'UniFormer': r'uniformer',
        # Pure ViT
        'ViT': r'vit|vision transformer|\bvit\b',
        'PCT': r'\bpct\b|point cloud transformer',
        'BEiT': r'beit',
        # CNN-Transformer Hybrid
        'TransUNet': r'transunet',
        'DETR': r'\bdetr\b|mask.?dino',
        'Mask2Former': r'mask2former',
        'MViT': r'mvit',
        'DPT': r'\bdpt\b|dense prediction transformer',
        'ResNet': r'resnet|resunet|res-?net',
        'UNet': r'\bunet\b|u-net|spconv',
        'YOLO': r'yolo',
        'DenseNet': r'densenet',
        'EfficientNet': r'efficientnet',
        'ConvNeXt': r'convnext',
        'FPN': r'\bfpn\b',
        'Transformer': r'transformer encoder|transformer decoder',  # Generic transformer
        # Foundation Models
        'SAM': r'\bsam\b|segment anything|hiera',
        'CLIP': r'\bclip\b',
        'Grounding DINO': r'grounding.?dino|\bglip\b',
        'Prithvi': r'prithvi',
        'FoMo-Net': r'fomo',
        'DINOv2': r'dinov2|dino.?v2',
        # VLM
        'LLaVA': r'llava',
        'GPT': r'\bgpt\b',
        'InternViT': r'internvit|intern-?vit',
    }
    
    for name, pattern in patterns.items():
        if re.search(pattern, backbone_str):
            backbones.append(name)
    
    return backbones if backbones else ['Unspecified']


def generate():
    """Generate the full multi-level Sankey diagram with paper citations."""
    # Load data
    models = pd.read_csv(os.path.join(DATA_DIR, "models_clean.csv"))
    papers = pd.read_csv(os.path.join(DATA_DIR, "papers_clean.csv"))
    
    df = pd.merge(models, papers[['paper_id', 'citation_key']], on='paper_id', how='left')
    df['backbone_list'] = df['backbones'].apply(detect_backbones)
    df['citation_short'] = df['citation_key'].apply(format_citation)
    
    # Node management with manual XY coordinates
    nodes = []
    node_indices = {}
    node_customdata = []  # Store paper citations for hover
    node_x = []
    node_y = []
    
    def add_node(label, papers_list=None, x=None, y=None):
        if label not in node_indices:
            node_indices[label] = len(nodes)
            nodes.append(label)
            node_customdata.append(papers_list or [])
            if x is not None and y is not None:
                node_x.append(x)
                node_y.append(y)
            else:
                # Fallback if manual coordinates are not provided (shouldn't happen with updated logic)
                node_x.append(0.5)
                node_y.append(0.5)
        else:
            # If node exists, extend citations (used for shared nodes)
            if papers_list:
                existing_papers = node_customdata[node_indices[label]]
                # Merge and unique
                combined = list(set(existing_papers + papers_list))
                node_customdata[node_indices[label]] = combined
        return node_indices[label]
    
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # Color palette (matching reference image style)
    colors = {
        'root': 'rgba(220, 53, 69, 0.9)',  # Red for root
        'vfm': 'rgba(255, 193, 7, 0.85)',  # Yellow/Gold
        'Hierarchical ViT': 'rgba(40, 167, 69, 0.8)',  # Green
        'Pure ViT': 'rgba(23, 162, 184, 0.8)',  # Cyan
        'CNN-Transformer Hybrid': 'rgba(255, 127, 14, 0.8)',  # Orange
        'Foundation Model Adaptation': 'rgba(214, 39, 40, 0.8)',  # Red
        'Vision-Language Model': 'rgba(227, 119, 194, 0.8)',  # Pink
        'Other': 'rgba(127, 127, 127, 0.8)',  # Gray
        'Unspecified/Custom': 'rgba(220, 220, 220, 0.9)' # Light Gray
    }
    
    # ============ ARCHITECTURE CLASSIFICATION ============
    hierarchical_backbones = ['Swin', 'SegFormer', 'Twins-SVT', 'PVT', 'DaViT', 'TiMo', 'UniFormer']
    
    def get_vit_type(row):
        # PRIORITY: If backbone is unspecified, force to "Unspecified/Custom" category
        if 'Unspecified' in row['backbone_list']:
            return 'Unspecified/Custom'
            
        cat = row['category_clean']
        
        # Handle 'Other' category by attempting to classify based on backbones
        if cat == 'Other':
            bbs = row['backbone_list']
            # Foundation Models
            if any(b in ['SAM', 'Prithvi', 'FoMo-Net', 'Grounding DINO', 'CLIP', 'DINOv2'] for b in bbs):
                return 'Foundation Model'
            # Hierarchical
            if any(b in hierarchical_backbones for b in bbs):
                return 'Hierarchical ViT'
            # Hybrid (if it has CNNs or explicitly Hybrid-like)
            if any(b in ['ResNet', 'UNet', 'YOLO', 'DenseNet', 'EfficientNet', 'ConvNeXt', 'TransUNet'] for b in bbs):
                return 'CNN-Transformer Hybrid'
            # Standard
            if any(b in ['ViT', 'PCT', 'BEiT'] for b in bbs):
                return 'Pure ViT'
            # Default fallback for Other
            return 'Unspecified/Custom'

        if cat == 'Pure Vision Transformer':
            if any(b in hierarchical_backbones for b in row['backbone_list']):
                return 'Hierarchical ViT'
            return 'Pure ViT'
            
        return cat
    
    df['arch_type'] = df.apply(get_vit_type, axis=1)
    # Normalize naming to a single canonical label used throughout the figure.
    df['arch_type'] = df['arch_type'].replace({
        'Foundation Model Adaptation': 'Foundation Model'
    })
    
    # ============ COUNT UNIQUE PAPERS (not model entries) ============
    # Group by paper_id to get unique papers per category
    paper_arch = df.groupby('paper_id').agg({
        'arch_type': 'first',
        'citation_short': 'first',
        'backbone_list': lambda x: [item for sublist in x for item in sublist]  # flatten
    }).reset_index()

    # Re-route papers with no recognized backbone match for their category into Unspecified/Custom.
    # This creates a single gray stream from the root (instead of cross-category "other" links).
    backbone_mapping = {
        'Hierarchical ViT': ['Swin', 'SegFormer', 'Twins-SVT', 'PVT', 'DaViT', 'TiMo', 'UniFormer'],
        'Pure ViT': ['ViT', 'PCT', 'BEiT'],
        'Foundation Model': ['SAM', 'Prithvi', 'FoMo-Net', 'Grounding DINO', 'CLIP', 'DINOv2'],
        'Vision-Language Model': ['LLaVA', 'GPT', 'InternViT'],
        'CNN-Transformer Hybrid': ['ResNet', 'TransUNet', 'DETR', 'Mask2Former', 'MViT', 'DPT',
                                   'UNet', 'YOLO', 'DenseNet', 'EfficientNet', 'ConvNeXt', 'FPN', 'Transformer'],
    }

    # Manual backbone overrides - update for specific known papers
    manual_backbone_overrides = {
        54: 'Transformer', 36: 'Transformer', 2: 'ViT', 12: 'ViT',
    }

    def has_recognized_backbone(row) -> bool:
        cat = row['arch_type']
        if cat == 'Unspecified/Custom':
            return True
        allowed = set(backbone_mapping.get(cat, []))
        if not allowed:
            return False
        pid = row['paper_id']
        if pid in manual_backbone_overrides:
            return manual_backbone_overrides[pid] in allowed
        return any(bb in allowed for bb in set(row['backbone_list']))

    paper_arch['arch_type'] = paper_arch.apply(
        lambda r: r['arch_type'] if has_recognized_backbone(r) else 'Unspecified/Custom',
        axis=1
    )
    
    arch_counts = paper_arch['arch_type'].value_counts().to_dict()
    total_unique_papers = len(paper_arch)
    
    # ============ LAYOUT CONFIGURATION ============
    # X coordinates (horizontal)
    X_ROOT = 0.001
    X_CAT = 0.4
    X_BACKBONE = 0.999
    
    # ============ LEVEL 1: ROOT ============
    all_papers = paper_arch['citation_short'].unique().tolist()
    # Root in the middle vertically
    root_idx = add_node(f"Transformer\nArchitectures\nfor Tree Extraction\n({total_unique_papers} papers)", 
                        all_papers, x=X_ROOT, y=0.5)
    
    # ============ LEVEL 2: DIRECT TO CATEGORIES ============
    # Categories flow directly from root
    
    # User requested order: 
    # "Pure ViT" (mapped to Pure ViT), "Hierarchical ViT", "CNN-Transformer Hybrid", 
    # "Foundation Model", "Vision-Language Model", "Unspecfied/Custom"
    
    category_order = ['Pure ViT', 'Hierarchical ViT', 'CNN-Transformer Hybrid', 
                      'Foundation Model', 'Vision-Language Model', 'Unspecified/Custom']
    
    # (arch_type already normalized)
    paper_arch['arch_display'] = paper_arch['arch_type']
    
    category_colors = {
        'Hierarchical ViT': ('rgba(62, 150, 81, 0.8)', 'rgba(62, 150, 81, 0.5)'),  # Green
        'Pure ViT': ('rgba(57, 106, 177, 0.8)', 'rgba(57, 106, 177, 0.5)'),  # Blue
        'CNN-Transformer Hybrid': ('rgba(218, 124, 48, 0.8)', 'rgba(218, 124, 48, 0.5)'),  # Orange
        'Foundation Model': ('rgba(204, 37, 41, 0.8)', 'rgba(204, 37, 41, 0.5)'),  # Red
        'Vision-Language Model': ('rgba(107, 76, 154, 0.8)', 'rgba(107, 76, 154, 0.5)'),  # Purple
        'Other': ('rgba(83, 81, 84, 0.8)', 'rgba(83, 81, 84, 0.5)'),  # Dark gray
        'Unspecified/Custom': ('rgba(175, 175, 175, 0.9)', 'rgba(175, 175, 175, 0.5)') # Light gray
    }
    
    # Calculate Y positions based on counts, ensuring we *don't* exceed Plotly's [0, 1] range.
    #
    # Previous logic allocated 0.9 total block height *and then* added fixed gaps after every
    # category (including zero-count ones), which can push later nodes downward and create a
    # large "gap" in the middle. Here we:
    # - Only lay out categories that actually exist (count > 0)
    # - Allocate (blocks + gaps) to fit exactly within [TOP_MARGIN, 1 - BOTTOM_MARGIN]
    TOP_MARGIN = 0.05
    BOTTOM_MARGIN = 0.05
    AVAILABLE_HEIGHT = 1.0 - TOP_MARGIN - BOTTOM_MARGIN
    CATEGORY_GAP = 0.02

    # Get counts for requested order
    cat_counts = {}
    for cat in category_order:
        cat_counts[cat] = len(paper_arch[paper_arch['arch_type'] == cat])

    # Only place visible categories (avoid reserving gap space for zero-count categories)
    visible_categories = [cat for cat in category_order if cat_counts.get(cat, 0) > 0]
    total_count = sum(cat_counts[c] for c in visible_categories)

    # Guard: if something went wrong upstream, fall back to a safe layout
    if total_count <= 0:
        visible_categories = category_order
        total_count = max(1, sum(cat_counts.values()))

    total_gap = max(0, len(visible_categories) - 1) * CATEGORY_GAP
    # Ensure we never get negative content height (in case someone increases CATEGORY_GAP)
    content_height = max(0.0, AVAILABLE_HEIGHT - total_gap)

    current_y_start = TOP_MARGIN
    category_y_pos = {}
    category_block_height = {}

    for cat in visible_categories:
        ratio = cat_counts[cat] / total_count
        block_height = ratio * content_height
        center_y = current_y_start + (block_height / 2.0)
        category_y_pos[cat] = center_y
        category_block_height[cat] = block_height
        current_y_start += block_height + CATEGORY_GAP
    
    category_indices = {}
    
    for cat in category_order:
        # Map back to original arch_type for filtering
        cat_papers = paper_arch[paper_arch['arch_type'] == cat]
        count = len(cat_papers)
        if count > 0:
            citations = cat_papers['citation_short'].unique().tolist()
            cat_label = f"{cat}\n({count})"
            # Use calculated Y position (computed only for non-zero categories)
            cat_idx = add_node(cat_label, citations, x=X_CAT, y=category_y_pos.get(cat, 0.5))
            category_indices[cat] = cat_idx
            
            sources.append(root_idx)
            targets.append(cat_idx)
            values.append(count)
            # Use specific color or fallback
            color_tuple = category_colors.get(cat, category_colors['Other'])
            link_colors.append(color_tuple[1])
    
    # ============ LEVEL 3: SPECIFIC BACKBONES ============
    # Assign backbones to categories
    # Single grouped catch-all bucket on the right for the Unspecified/Custom stream.
    OTHER_GROUP_DISPLAY = "Other/\nUnmapped/\nUnspecified/\nCustom"
    
    # Prepare list of all backbones in order of categories
    # We need to compute total height available for backbones
    
    backbone_data = [] # (cat, backbone, count, flow)
    
    # (manual_backbone_overrides defined above for consistency)
    
    # Store backbone -> papers mapping
    backbone_papers = {}
    
    # Identify Unspecified/Custom papers specifically
    # Since we moved them to a category, we treat them as a "backbone" called Unspecified/Custom 
    # that flows from the category "Unspecified/Custom"
    
    for cat in category_order:
        if cat not in category_indices: continue
        
        # Determine backbones for this category
        if cat == 'Unspecified/Custom':
            backbones = [] # Special handling
        else:
            backbones = backbone_mapping.get(cat, [])
            
        cat_idx = category_indices[cat]
        cat_papers_df = paper_arch[paper_arch['arch_type'] == cat]
        
        # If this is the "Unspecified/Custom" category, we just create one node
        if cat == 'Unspecified/Custom':
             count = len(cat_papers_df)
             citations = cat_papers_df['citation_short'].unique().tolist()
             backbone_data.append({
                'cat': cat,
                'name': OTHER_GROUP_DISPLAY,
                'count': count,
                'flow': float(count),
                'citations': citations,
                'cat_idx': cat_idx,
                'color': category_colors.get(cat)[1]
             })
             continue

        # Normal category processing
        backbone_flow_counts = {}
        backbone_raw_counts = {}
        backbone_citation_lists = {}
        # No per-category unmapped sink: unmapped papers are routed to Unspecified/Custom upstream.
        
        for _, row in cat_papers_df.iterrows():
            paper_id = row['paper_id']
            matches = []
            if paper_id in manual_backbone_overrides:
                override_bb = manual_backbone_overrides[paper_id]
                if override_bb in backbones: matches.append(override_bb)
            else:
                for bb in row['backbone_list']:
                    if bb in backbones: matches.append(bb)
            
            if matches:
                weight = 1.0 / len(matches)
                for bb in matches:
                    backbone_flow_counts[bb] = backbone_flow_counts.get(bb, 0.0) + weight
                    backbone_raw_counts[bb] = backbone_raw_counts.get(bb, 0) + 1
                    if bb not in backbone_citation_lists: backbone_citation_lists[bb] = []
                    backbone_citation_lists[bb].append(row['citation_short'])
        
        # Add identified backbones to list
        # Order them by count descending
        sorted_bbs = sorted(backbones, key=lambda b: backbone_raw_counts.get(b, 0), reverse=True)
        
        for bb in sorted_bbs:
            count_raw = backbone_raw_counts.get(bb, 0)
            flow_val = backbone_flow_counts.get(bb, 0.0)
            if count_raw > 0:
                backbone_data.append({
                    'cat': cat,
                    'name': bb,
                    'count': count_raw,
                    'flow': flow_val,
                    'citations': list(set(backbone_citation_lists.get(bb, []))),
                    'cat_idx': cat_idx,
                    'color': category_colors.get(cat)[1]
                })


    # Assign Y positions for backbones
    # Key improvement: layout each category's backbones *within that category's allocated block height*
    # (including gaps). This prevents overlaps while allowing us to increase spacing for readability.

    for cat in category_order:
        # Get backbones for this category
        cat_items = [item for item in backbone_data if item['cat'] == cat]
        if not cat_items:
            continue
            
        block_flow = sum(float(item['flow']) for item in cat_items)
        bb_gap = 0.008  # more vertical separation to avoid label overlap

        # Category block height comes from the same layout used for categories.
        # Fallback to proportional height if missing (shouldn't happen).
        cat_block_h = category_block_height.get(cat)
        if cat_block_h is None:
            cat_block_h = (block_flow / total_count) * content_height if total_count else 0.1

        # Ensure we fit gaps + node heights inside the category block.
        total_gaps = max(0, len(cat_items) - 1) * bb_gap
        available_for_nodes = max(0.0, cat_block_h - total_gaps)
        scale_cat = (available_for_nodes / block_flow) if block_flow > 0 else 0.0

        # Center this stack within the category block.
        cat_center_y = category_y_pos.get(cat, 0.5)
        current_bb_y = cat_center_y - (cat_block_h / 2.0)
        
        # If this is the LAST category (Unspecified), force it to align to the bottom
        # to ensure it's visually separate, or let it float.
        # But 'Unspecified' often ends up too high if we just center it.
        if cat == 'Unspecified/Custom':
            # Force it lower if needed, or keep centered. 
            # Given user feedback about gaps, centered logic is safest for now.
            pass

        # Place items
        for item in cat_items:
            # Item height within this category block
            h = float(item['flow']) * scale_cat
            y_pos = current_bb_y + (h / 2)
            
            # Format label
            if 'Unspecified' in item['name']:
                bb_label = f"{item['name']}\n({item['count']})"
            else:
                bb_label = f"{item['name']}\n({item['count']})"
                
            # Add node
            bb_idx = add_node(bb_label, item['citations'], x=X_BACKBONE, y=y_pos)
            
            # Track citations per backbone (union if repeated)
            existing = backbone_papers.get(item['name'], [])
            backbone_papers[item['name']] = sorted(list(set(existing + item['citations'])))
            
            sources.append(item['cat_idx'])
            targets.append(bb_idx)
            values.append(item['flow'])
            link_colors.append(item['color'])
            
            # Advance Y
            current_bb_y += h + bb_gap

    # ============ NODE COLORS ============
    node_colors_list = []
    for node in nodes:
        if 'Transformer\nArchitectures' in node or 'Tree Extraction' in node:
            node_colors_list.append('rgba(83, 81, 84, 0.9)')
        elif 'Vision-Language' in node:
            node_colors_list.append('rgba(107, 76, 154, 0.85)')
        elif 'Hierarchical' in node:
            node_colors_list.append('rgba(62, 150, 81, 0.8)')
        elif 'Pure ViT' in node:
            node_colors_list.append('rgba(57, 106, 177, 0.8)')
        elif 'CNN-Transformer' in node:
            node_colors_list.append('rgba(218, 124, 48, 0.8)')
        elif 'Foundation Model' in node:
            node_colors_list.append('rgba(204, 37, 41, 0.8)')
        elif 'Unspecified' in node or 'Custom' in node:
            node_colors_list.append('rgba(175, 175, 175, 0.9)')
        else:
            # Check for backbones
            is_bb = False
            for cat, color_tuple in category_colors.items():
                bb_name = node.split('\n')[0]
                
                # Check mapping
                found = False
                for b_list in backbone_mapping.values():
                    if bb_name in b_list:
                         found = True
                         break
                
                if found or bb_name == 'Unspecified':
                    # Find which category it belongs to
                    if any(x in node for x in ['Swin', 'SegFormer', 'Twins', 'PVT', 'DaViT', 'TiMo', 'UniFormer']):
                        node_colors_list.append('rgba(62, 150, 81, 0.7)')
                        is_bb = True; break
                    if any(x in node for x in ['SAM', 'Prithvi', 'FoMo', 'Grounding', 'CLIP', 'DINOv2']):
                        node_colors_list.append('rgba(204, 37, 41, 0.7)')
                        is_bb = True; break
                    if any(x in node for x in ['TransUNet', 'DETR', 'Mask2Former', 'MViT', 'DPT', 'ResNet', 
                                              'UNet', 'YOLO', 'DenseNet', 'EfficientNet', 'ConvNeXt', 'FPN', 'Transformer']):
                        node_colors_list.append('rgba(218, 124, 48, 0.7)')
                        is_bb = True; break
                    if any(x in node for x in ['LLaVA', 'GPT', 'InternViT']):
                        node_colors_list.append('rgba(107, 76, 154, 0.7)')
                        is_bb = True; break
                    if 'ViT' in node or 'PCT' in node or 'DINOv2' in node or 'BEiT' in node:
                        node_colors_list.append('rgba(57, 106, 177, 0.7)')
                        is_bb = True; break
            
            if not is_bb:
                node_colors_list.append('rgba(150, 150, 150, 0.7)')
    
    # Build hover templates
    hover_templates = []
    for i, node in enumerate(nodes):
        papers_list = node_customdata[i]
        if papers_list:
            papers_str = '<br>'.join(papers_list[:10])
            if len(papers_list) > 10:
                papers_str += f'<br>...and {len(papers_list)-10} more'
            hover_templates.append(f"{node}<br><br><b>Papers:</b><br>{papers_str}")
        else:
            hover_templates.append(node)
    
    # ============ CREATE SANKEY FIGURE ============
    fig = go.Figure(data=[go.Sankey(
        # Use 'fixed' so Plotly does not "helpfully" re-snap nodes and introduce whitespace/gaps.
        arrangement='fixed',
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=nodes,
            x=node_x,
            y=node_y,
            color=node_colors_list,
            customdata=hover_templates,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}: %{value:.1f} papers<extra></extra>'
        )
    )])
    
    fig.update_layout(
        # Title removed per request (keeps the figure tighter and cleaner for paper layout)
        title=None,
        font=dict(size=11, family="Arial"),
        height=760,
        width=1600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=10, r=50, t=20, b=35)
    )

    # Save JPEG
    try:
        jpeg_path = os.path.join(OUTPUT_DIR, "architecture_sankey.jpg")
        fig.write_image(jpeg_path, scale=2, format='jpeg')
        print(f"✅ Saved JPG to {jpeg_path}")
    except Exception as e:
        print(f"⚠️ Could not save JPEG: {e}")
    
    return fig, backbone_papers


if __name__ == "__main__":
    fig, backbone_papers = generate()
