"""Taxonomy: studies -> architecture family -> model family of the study's principal transformer (Sankey diagram).

Model families come from the name of the principal model (named architectures such as DETR, Mask2Former, SAM), then from
its backbone (Swin, SegFormer MiT, PVT, ViT, point transformers). Two residual groups are kept apart and listed with their
references in figure_data/taxonomy_residual.csv (the manuscript describes their composition in the text):
- "Study-specific designs": the principal transformer is an architecture built in the study that uses none of the named
  model families or backbones (for example convolution blocks combined with transformer blocks of the authors' own design);
- "Other named models": a named model family that is the principal model of only one study in that architecture family
  (families with more than five studies only).
"""
import io
import re
import shutil

import pandas as pd
import plotly.graph_objects as go
from PIL import Image

import common
from common import COLOR, FAMILIES, FAMILY, INK, paths, set_style, studies, write

FRAMEWORKS = [  # named architectures, matched on the model name first
    (r"Point.?Voxel Transformer", "Point transformer"), (r"\bSAM ?2|SAM2", "SAM 2"), (r"ViTDet", "ViT"),
    (r"\bSAM\b|Segment Anything|SAM-HQ|HQ-SAM|MobileSAM|FastSAM", "SAM"), (r"Grounding.?DINO", "Grounding DINO"), (r"CLIP", "CLIP"),
    (r"LLaVA|GPT|Qwen|InternVL|LLM|language model", "Large language model"), (r"FoMo|Prithvi|SatMAE|Scale-MAE", "Remote sensing foundation model"),
    (r"RT-?DETR", "RT-DETR"), (r"Deformable DETR", "Deformable DETR"), (r"DETR|\bDINO\b|Co-?DINO", "DETR family"),
    (r"Mask2Former|MaskFormer|OneFormer", "Mask2Former family"), (r"YOLO", "YOLO-based"), (r"SegFormer", "SegFormer"),
    (r"TransUNet|Swin-?UNet|UNETR", "Transformer U-Net"), (r"Segmenter|SETR|\bDPT\b", "ViT segmenter"), (r"MaxViT", "MaxViT"),
    (r"DINOv2", "DINOv2"), (r"CSWin", "CSWin"), (r"PoinTr", "Point transformer"),
]
BACKBONES = [  # transformer backbone families, matched on the backbone, then on the name
    (r"CSWin", "CSWin"), (r"(?<!C)Swin|shifted.window", "Swin"), (r"Point.?Voxel Transformer", "Point transformer"),
    (r"Pyramid Vision Transformer|\bPVT\b", "PVT"), (r"\bMiT\b|SegFormer", "SegFormer"), (r"Twins", "Twins"), (r"MViT", "MViT"),
    (r"MaxViT", "MaxViT"), (r"DINOv2", "DINOv2"), (r"\bSAM\b", "SAM"),
    (r"Point Transformer|PTv\d|\bPCT\b|Point-?BERT|Point-?MAE|point cloud transformer|PointTransformer|Superpoint Transformer|PoinTr",
     "Point transformer"),
    (r"ViT|DeiT|BEiT|vision transformer", "ViT"),
]
SPECIFIC, OTHER_NAMED = "Study-specific designs", "Other named models"


def model_group(name, backbone):
    for rx, g in FRAMEWORKS:
        if re.search(rx, str(name), re.I):
            return g
    for text in (str(backbone), str(name)):
        for rx, g in BACKBONES:
            if re.search(rx, text, re.I):
                return g
    for rx, g in FRAMEWORKS:
        if re.search(rx, str(backbone), re.I):
            return g
    return SPECIFIC


def display(name):
    return re.sub(r"\s*\((?:proposed|ours|our|Ours|proposed PEFT)\)", "", str(name)).strip()


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def assign():
    """Studies with the model family of their principal transformer and the node it is drawn in."""
    st = studies()
    st["model_group"] = [model_group(n, b) for n, b in zip(st.principal_model, st.principal_backbone)]
    st["node"] = st.model_group
    for k in FAMILIES:
        sub = st[st.family == k]
        if len(sub) > 5:
            single = sub.model_group.value_counts()
            single = single[(single == 1) & (single.index != SPECIFIC)].index
            st.loc[(st.family == k) & st.model_group.isin(single), "node"] = OTHER_NAMED
    return st


def main():
    st = assign()
    write(st[["study", "label", "bib_key", "family", "basis", "principal_model", "principal_backbone", "model_group", "node"]], "taxonomy")
    residual = st[st.node.isin([SPECIFIC, OTHER_NAMED])].copy()
    residual["model"] = residual.principal_model.map(display)
    residual = residual.sort_values(["node", "family", "label"])
    write(residual[["node", "family", "label", "bib_key", "model", "model_group", "principal_backbone", "basis", "study"]], "taxonomy_residual")

    nodes, colours, src, dst, val, lcol = [f"Studies (n = {len(st)})"], [INK], [], [], [], []
    for k in FAMILIES:
        sub = st[st.family == k]
        if sub.empty:
            continue
        fi = len(nodes)
        nodes.append(f"{FAMILY[k]} ({len(sub)})")
        colours.append(COLOR[k])
        src.append(0), dst.append(fi), val.append(len(sub)), lcol.append(rgba(COLOR[k], 0.40))
        counts = sub.node.value_counts()
        order = [g for g in counts.index if g not in (OTHER_NAMED, SPECIFIC)] + [g for g in (OTHER_NAMED, SPECIFIC) if g in counts.index]
        for g in order:
            nodes.append(f"{g} ({counts[g]})")
            colours.append(COLOR[k])
            src.append(fi), dst.append(len(nodes) - 1), val.append(int(counts[g])), lcol.append(rgba(COLOR[k], 0.33))
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=nodes, color=colours, pad=14, thickness=16, line=dict(color="#555555", width=0.5)),
                              link=dict(source=src, target=dst, value=val, color=lcol)))
    fig.update_layout(font=dict(family="Arial", size=13, color=INK), width=1500, height=900, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="white")
    paths.ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
    common.finish_jpeg(Image.open(io.BytesIO(fig.to_image(format="png", scale=2))), paths.ANALYSIS_FIGURES / "taxonomy.jpg")  # preview
    if common.SUBMISSION:  # a JPEG at least 7600 px wide (the journal's minimum for line art at full page width is 7480 px)
        out = common.submission_dir() / "taxonomy.jpg"
        size = common.finish_jpeg(Image.open(io.BytesIO(fig.to_image(format="png", scale=5.2))), out, 780)
        shutil.copy2(out, paths.PAPER_FIGURES / "taxonomy.jpg")
        print(f"  submission figures/taxonomy.jpg: {size[0]} x {size[1]} px")
    print(f"  saved figures/taxonomy.jpg; residual groups: {residual.node.value_counts().to_dict()}")


if __name__ == "__main__":
    set_style()
    main()
