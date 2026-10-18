import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T4, T7, TF = "Table 4", "Table 7", "Table F.9"
ROWS = [
    R("task", "detection;counting;attribute_estimation", 8, "Sec. 5", "Tables 1, 2 and 3 display the results for detecting, localizing and measuring trees respectively"),
    R("output_unit", "tree_instance;tree_count;tree_attribute", 8, "Sec. 5", "Metrics (except for counting MAE) are computed with γ = {0.5, 1.0, 2.0}", note="tree positions, counts and crown area or diameter"),
    R("tree_setting", "mixed", 8, "Sec. 5", "The datasets include nordic forests, urban areas, savannas, farmlands, plantations, and rainforest"),
    NR("tree_taxa", note="all trees in mixed landscapes; species not given"),
    R("study_area", "Denmark; Rwanda", 8, "Sec. 5", "23.6k trees over different areas in Denmark ... 98.8k trees in Rwanda"),
    R("biome", "multiple", 8, "Sec. 5", "nordic forests, urban areas, savannas, farmlands, plantations, and rainforest", note="temperate (Denmark) and tropical (Rwanda)"),
    R("platform", "airborne", 2, "Sec. 2", "This study solely relies on aerial imagery"),
    R("sensors", "rgb;multispectral", 8, "Sec. 5", "labeled on RGB+NIR aerial imagery [21] at 20cm ... labeled on RGB aerial imagery [4] at 25cm", note="Denmark RGB+NIR; Rwanda RGB"),
    R("gsd", "20", 8, "Sec. 5", "RGB+NIR aerial imagery [21] at 20cm ground resolution", unit="cm", note="Denmark 20 cm; Rwanda 25 cm"),
    R("annotation_type", "polygon_mask", 8, "Sec. 5", "Tree centers are derived as the geometric centers of labelled polygons"),
    R("dataset_name", "Denmark aerial tree dataset", 8, "Sec. 5", "First dataset contains 23.6k trees over different areas in Denmark", item="dataset1"),
    R("dataset_role", "train_test", 8, "Sec. 5", "We train and validate on three different, randomly selected splits of approx. 60/20% of the labeled data, and test on the remaining, fixed test split", item="dataset1"),
    NR("dataset_public", item="dataset1", note="labels from a national mapping project [21]; availability not stated"),
    D("n_trees", "23600", "printed as 23.6k (p.8, Sec. 5)", item="dataset1", unit="trees"),
    R("dataset_name", "Rwanda aerial tree dataset", 8, "Sec. 5", "Second dataset contains 98.8k trees in Rwanda", item="dataset2"),
    R("dataset_role", "train_test", 8, "Sec. 5", "We train and validate on three different, randomly selected splits of approx. 60/20% of the labeled data", item="dataset2"),
    NR("dataset_public", item="dataset2", note="labels from a national mapping project [4]; availability not stated"),
    D("n_trees", "98800", "printed as 98.8k (p.8, Sec. 5)", item="dataset2", unit="trees"),
]
M = [  # item, framework, backbone label, role, category, params (M), forward ms, post ms
    ("model1", "Segmentation", "UNet-R50", "cnn_baseline", "cnn", "33", "1.7ms", "81.8ms"),
    ("model2", "Segmentation", "DeepLabV3", "cnn_baseline", "cnn", "40", "1.8ms", "92.4 ms"),
    ("model3", "Segmentation", "SegFormer", "transformer_baseline", "hierarchical_vit", "45", "3.0ms", "91.3ms"),
    ("model4", "Segmentation", "TransUNet", "transformer_baseline", "cnn_transformer_hybrid", "105", "2.5ms", "91.9ms"),
    ("model5", "Heatmap", "UNet-R50", "proposed", "cnn", "33", "1.7ms", "2.1ms"),
    ("model6", "Heatmap", "DeepLabV3", "cnn_baseline", "cnn", "40", "1.8ms", "14.9ms"),
    ("model7", "Heatmap", "SegFormer", "transformer_baseline", "hierarchical_vit", "45", "3.0ms", "11.0ms"),
    ("model8", "Heatmap", "TransUNet", "transformer_baseline", "cnn_transformer_hybrid", "105", "2.5ms", "4.7ms"),
]
for item, fw, bb, role, cat, par, fwd, post in M:
    ROWS += model(item, f"{fw} framework with {bb}", role, cat, bb, None, None, 9, T4, f"{fw} {bb}", pre_status="not_reported",
                  pre_note="pre-training not stated", strat_note="models are trained on the datasets; initialisation not stated",
                  cat_note="Heatmap is the framework this paper proposes" if fw == "Heatmap" else "")
    ROWS += [R("total_params", par, 16, TF, f"{bb} ... {par}M", item=item, unit="M", note="backbone architecture"),
             R("inference_speed", f"{fwd} forward pass and {post} post-processing per 256x256 patch on one RTX 3090", 11, T7, f"{fw} {bb} {fwd} {post}", item=item)]
VALS = {  # item: (counting MAE %, bF1 at gamma=1), Table 4, Denmark
    "model1": ("30.7", "42.1", "30.7±2.9", "42.1±2.6"), "model2": ("33.0", "40.4", "33.0±6.2", "40.4±0.8"),
    "model3": ("32.8", "40.3", "32.8±2.3", "40.3±3.0"), "model4": ("31.8", "41.8", "31.8±3.6", "41.8±2.3"),
    "model5": ("33.4", "41.8", "33.4±3.2", "41.8±3.5"), "model6": ("37.2", "42.8", "37.2±5.5", "42.8±2.1"),
    "model7": ("36.4", "42.0", "36.4±4.6", "42.0±2.4"), "model8": ("36.0", "40.0", "36.0±1.4", "40.0±2.4"),
}
res = Counter("res")
for item, (mae, f1, mae_ev, f1_ev) in VALS.items():
    fw, bb = next((m[1], m[2]) for m in M if m[0] == item)
    ROWS += result(res(), "mae", item, "dataset1", mae, 9, T4, f"{fw} {bb} Counting MAE (%) {mae_ev}", setting="test split; tree counting",
                   note="normalised counting MAE; mean ± SD over 5 runs")
    ROWS += result(res(), "f1", item, "dataset1", f1, 9, T4, f"{fw} {bb} bF1 γ=1 {f1_ev}", setting="test split; detection at distance threshold γ = 1",
                   note="mean ± SD over 5 runs; Table 4 also gives γ = 0.5 and γ = 2")
ROWS += avail()
ROWS += [D("notes", "Benchmark paper. The main comparison (Tables 1 to 3: Segmentation, CenterNet, Heatmap, P2P and Faster R-CNN on Denmark and Rwanda) uses only convolutional models; transformers (SegFormer, TransUNet) appear in the feature-extractor comparison (Table 4, Denmark only), which is the part recorded here. arXiv version (2311.07981).", "Sec. 5 and Tables 1 to 4 (pp. 8 to 9)")]
