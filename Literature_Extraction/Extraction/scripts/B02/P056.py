import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T3, T4 = "P056", "Table 3", "Table 4"
URL = "https://github.com/Bolci/Palm_Extraction_Tools/tree/main"
ROWS = [
    R("task", "crown_segmentation", 1, "Title", "Spectral–Spatial transformer-based semantic segmentation for large-scale mapping of individual date palm trees using very high-resolution satellite data"),
    R("output_unit", "crown_mask", 10, "Sec. 2.8", "The semantic segmentation results for delineating date palm plantations are represented by binary masks",
      note="binary palm masks; a post-processing step separates and counts individual palms"),
    R("tree_setting", "mixed", 7, "Sec. 2.4", "The vector data encompassed a total of 76,950 date palm trees scattered across agricultural and urban landscapes"),
    R("tree_taxa", "date palm (Phoenix dactylifera)", 5, "Sec. 2.2", "including the date palm (Phoenix dactylifera L.)"),
    R("study_area", "east Ajman, Al Zubair, Kalba, Dibba and Masafi (Fujairah, Sharjah and Ajman), United Arab Emirates; 275 km2", 5, "Sec. 2.2",
      "the areas are located in east Ajman, Al Zubair, Kalba, Dibba, and Masafi. Collectively, these areas encompass an area of 275 km2"),
    D("biome", "arid", "UAE study sites (arid); not stated in the paper"),
    R("platform", "satellite", 5, "Sec. 2.3", "a total of 20 multidate VHR WV-3 satellite image tiles were utilized"),
    R("sensors", "rgb;multispectral", 5, "Sec. 2.3", "The WV-3 dataset features eight spectral bands with a GSD of 1.24 m alongside a panchromatic band with a 0.31 m GSD",
      note="pansharpened RGB, RGB with NIR, and eight-band inputs compared"),
    R("gsd", "0.31", 5, "Table 1", "Panchromatic 450–800 0.31", unit="m", note="multispectral bands pansharpened from 1.24 m"),
    R("annotation_type", "polygon_mask", 7, "Sec. 2.4", "verification of delineated palms, serving as ground-truth data"),
    R("dataset_name", "own WorldView-3 date palm dataset (UAE sites)", 7, "Sec. 2.4", "The WV-3 data and its annotations were cropped into consistent 512 × 512 image tiles", item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 2.4", "A total of 16,260 image tiles were dedicated to developing deep learning models. Meanwhile, approximately 1800", item="dataset1",
      note="about 1800 further tiles for evaluation and testing"),
    D("dataset_public", "no", "No dataset link; the Data availability statement refers only to code (p.17)", item="dataset1"),
    R("n_images", "16260", 7, "Sec. 2.4", "A total of 16,260 image tiles were dedicated to developing deep learning models", item="dataset1", unit="images", note="development tiles only"),
    R("n_trees", "76950", 7, "Sec. 2.4", "The vector data encompassed a total of 76,950 date palm trees", item="dataset1", unit="trees"),
    R("dataset_name", "Dibba WorldView-3 tiles (unseen region)", 10, "Sec. 2.7", "The WV-3 datasets collected in Dibba city (Fig. 1) were used to evaluate the transferability of the assessed ViTs", item="dataset2"),
    R("dataset_role", "transfer_target", 13, "Sec. 3.3", "underwent testing on an unseen testing dataset from the Dibba region", item="dataset2"),
    D("dataset_public", "no", "No dataset link in the paper", item="dataset2"),
    R("n_images", "680", 10, "Sec. 2.7", "These data entail a set of non-overlapping 680 image-mask pairs derived from an area of 77.1 km2", item="dataset2", unit="images"),
]
PRE = dict(pre_page=12, pre_loc="Sec. 2.9", pre_ev="The models were initialized with ADE20K pretrained weights (Zhou et al., 2017)", pre_note="ADE20K",
           strat_page=12, strat_loc="Sec. 2.9", strat_ev="The models were trained for 100,000 iterations")
T3M = [  # item, Table 3 label regex, name, category, backbone
    ("model1", r"^\s*SegFormer-MiT-", "SegFormer", "hierarchical_vit", "MiT-B3"),
    ("model2", r"^\s*UperNet-ViT", "UperNet with ViT backbone", "pure_vit", "ViT (size not stated)"),
    ("model3", r"^\s*UperNet-Swin-", "UperNet with Swin-tiny backbone", "hierarchical_vit", "Swin-T"),
    ("model4", r"^\s*Mask2Former-", "Mask2Former with Swin-tiny backbone", "hierarchical_vit", "Swin-T"),
    ("model5", r"^\s*UniFormer", "UniFormer", "cnn_transformer_hybrid", "UniFormer-XS"),
]
INPUTS = ["RGB", "RGB + NIR1 + NIR2", "eight multispectral bands", "eight multispectral bands + NDVI"]
res = Counter("res")
for k, (item, rx, name, cat, bb) in enumerate(T3M):
    ROWS += model(item, name, "transformer_baseline", cat, bb, "other", "full_fine_tuning", 8, T3, name,
                  cat_note=("UniFormer combines convolution and self-attention" if item == "model5" else "comparison of transformer models only; no CNN baseline"), **PRE)
    miou, ev_i = row(PID, 8, r"^\s+(?:B3|tiny|Swin-tiny)?\s*mIoU", k, cols=(0, 82))
    mf, ev_f = row(PID, 8, r"^\s+mF-score", k, cols=(0, 82))
    assert len(miou) == 4 and len(mf) == 4, (item, miou, mf)
    for j, inp in enumerate(INPUTS):
        st = f"test set; input: {inp}"
        ROWS += result(res(), "miou", item, "dataset1", miou[j], 8, T3, ev_i, setting=st, headline=(item == "model3" and j == 2))
        ROWS += result(res(), "f1", item, "dataset1", mf[j], 8, T3, ev_f, setting=st, note="mean F-score")
T4M = [  # item, regex (line), occurrence, name, category, backbone
    ("model1", r"^\s*SegFormer\s+MiT-B3", 0, None, None, None), ("model6", r"^\s*SegFormer\s+MiT-B4", 0, "SegFormer", "hierarchical_vit", "MiT-B4"),
    ("model7", r"^\s*SegFormer\s+MiT-B5", 0, "SegFormer", "hierarchical_vit", "MiT-B5"), ("model3", r"^\s*UperNet-Swin\s+Tiny", 0, None, None, None),
    ("model8", r"^\s*UperNet-Swin\s+Small", 0, "UperNet with Swin-small backbone", "hierarchical_vit", "Swin-S"),
    ("model9", r"^\s*UperNet-Swin\s+Base", 0, "UperNet with Swin-base backbone", "hierarchical_vit", "Swin-B"),
    ("model4", r"^\s*Mask2Former-\s+Tiny", 0, None, None, None),
    ("model10", r"^\s*Mask2Former-\s+Small", 0, "Mask2Former with Swin-small backbone", "hierarchical_vit", "Swin-S"),
    ("model11", r"^\s*Mask2Former-\s+Base", 0, "Mask2Former with Swin-base backbone", "hierarchical_vit", "Swin-B"),
    ("model5", r"^\s*UniFormer\s+XS", 0, None, None, None), ("model12", r"^\s*UniFormer\s+Small", 0, "UniFormer", "cnn_transformer_hybrid", "UniFormer-S"),
    ("model13", r"^\s*UniFormer\s+Base", 0, "UniFormer", "cnn_transformer_hybrid", "UniFormer-B"),
]
for item, rx, n, name, cat, bb in T4M:
    lines = open(str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Literature_Extraction" / "PDF_Text" / f"{PID}.layout.txt"), encoding="utf-8").read().split("\f")[11].split("\n")
    import re
    ln = next(" ".join(x[:84].split()) for x in lines if re.search(rx, x[:84]))
    toks = ln.split()
    par, fl, mi, mfs = toks[-6] + " " + toks[-5] if toks[-5] in ("M",) else toks[-5], None, toks[-2], toks[-1]
    m = re.search(r"(\S+)\s*(M|G)?\s+([\d.]+)\s*(G|T)?\s+([\d.]+)\s+([\d.]+)$", ln)
    p_val, p_unit, f_val, f_unit, mi, mfs = m.group(1), m.group(2) or "", m.group(3), m.group(4) or "", m.group(5), m.group(6)
    if name:
        ROWS += model(item, name, "transformer_baseline", cat, bb, "other", "full_fine_tuning", 12, T4, ln, **PRE)
        ROWS += result(res(), "miou", item, "dataset1", mi, 12, T4, ln, setting="test set; input: eight multispectral bands")
        ROWS += result(res(), "f1", item, "dataset1", mfs, 12, T4, ln, setting="test set; input: eight multispectral bands", note="mean F-score")
    if p_val.endswith("G"):
        ROWS += [D("total_params", "120", f"printed as '{p_val}' in Table 4 (p.12), read as 0.12 billion parameters", item=item, unit="M")]
    else:
        ROWS += [R("total_params", p_val, 12, T4, ln, item=item, unit="M")]
    ROWS += [R("flops", f_val, 12, T4, ln, item=item, unit={"G": "GFLOPs", "T": "TFLOPs"}.get(f_unit, "unit not printed"))]
tr = Counter("tr")
for item, ind, ood, lab in (("model4", "77.36", "84.36", "Mask2Former"), ("model5", "77.88", "84.25", "UniFormer"), ("model3", "78.10", "83.17", "UperNet-Swin transformer")):
    t = tr()
    ROWS += [R("transfer_source", "WorldView-3 training regions (east Ajman, Al Zubair, Kalba, Masafi); in-domain value on their test tiles", 13, "Sec. 3.3",
               "those trained with RGB and eight multispectral bands underwent testing on an unseen testing dataset from the Dibba region", item=t),
             R("transfer_target", "Dibba region (unseen), 680 tiles", 13, "Sec. 3.3", "an unseen testing dataset from the Dibba region", item=t),
             R("shift_type", "region", 13, "Sec. 3.3", "an unseen testing dataset from the Dibba region", item=t),
             R("transfer_model", item, 14, "Sec. 3.3", "the Mask2Former, UniFormer, and UperNet-Swin transformer models ... attained mIoU scores of 84.36 %, 84.25 %, and 83.17 %", item=t),
             R("transfer_metric", "miou", 14, "Sec. 3.3", "attained mIoU scores of 84.36 %, 84.25 %, and 83.17 %", item=t, note="eight-band input"),
             R("value_in_domain", ind, 8, T3, f"{lab}: eight MS bands mIoU {ind}", item=t, unit="%"),
             R("value_out_of_domain", ood, 14, "Sec. 3.3", "These models attained mIoU scores of 84.36 %, 84.25 %, and 83.17 %", item=t, unit="%")]
ROWS += avail(code_url=URL, page=10, loc="Sec. 2.8", ev=f"The source code for the aforementioned postprocessing technique is available ... {URL}")
ROWS += [D("notes", "All models are transformer-based; no CNN baseline. Released code covers the post-processing that separates and counts palms. Dibba results for the other models and for RGB input are only in Figure 9.", "Sec. 2.8 to 3.3 and Tables 3 to 4 (pp. 8 to 14)")]
