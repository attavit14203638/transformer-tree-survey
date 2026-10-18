import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2 = "P135", "Table 2"
URL = "https://github.com/Global-Ecosystem-Health-Observatory/TreeMort"
ROWS = [
    R("task", "mortality_detection;crown_segmentation", 1, "Abstract", "Mapping standing dead trees is critical for assessing forest health", note="instance segmentation of standing dead trees"),
    R("output_unit", "crown_mask", 5, "Sec. 3.4", "These images encompass approximately 15,000 centroids identifying individual dead tree canopies"),
    R("tree_setting", "natural_forest", 5, "Sec. 3.4", "sourced from the National Land Survey of Finland (NLS), providing comprehensive coverage across Finland", note="Finnish forests"),
    R("tree_taxa", "not specified (dead trees in Finnish forests)", 5, "Sec. 3.4", "approximately 15,000 centroids identifying individual dead tree canopies"),
    R("study_area", "Finland (National Land Survey aerial campaigns, about 100 km2)", 6, "Table 1", "Total Area Covered approx. 100 km2"),
    R("biome", "boreal", 1, "Abstract", "Tested on high-resolution aerial imagery from boreal forests"),
    R("platform", "airborne", 5, "Sec. 3.4", "The dataset used in this study comprises high-resolution aerial orthophotos sourced from the National Land Survey of Finland (NLS)"),
    R("sensors", "rgb;multispectral", 6, "Table 1", "Spectral Bands Red, Green, Blue, Near-Infrared"),
    R("gsd", "0.25", 6, "Table 1", "Spatial Resolution 0.25 m/pixel", unit="m", note="Sec. 3.4 also mentions orthophotos at 0.5 m"),
    R("annotation_type", "polygon_mask", 6, "Table 1", "No. of Annotations ∼15,000 individual dead tree polygons"),
    R("dataset_name", "NLS Finland aerial images with dead-tree polygons (TreeMort dataset)", 5, "Sec. 3.4", "The dataset was assembled by downloading 125 aerial image tiles from the NLS web interface", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 3.4", "The dataset was partitioned into training, validation, and test sets in a 70:20:10 ratio using a spatially aware stratification approach", item="dataset1",
      note="spatially separated test regions; five random splits"),
    NR("dataset_public", item="dataset1", note="imagery available from the NLS web interface; annotation release not stated"),
    R("n_images", "125", 6, "Table 1", "Total Images 125", item="dataset1", unit="images", note="87,458 patches of 256 x 256"),
    R("n_trees", "15000", 6, "Table 1", "No. of Annotations ∼15,000 individual dead tree polygons", item="dataset1", unit="trees", note="approximately"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="not stated")
M = [  # item, Table 2 label, name, role, category, backbone, note
    ("model1", "TreeMort-3T-UNet", "TreeMort-3T-UNet", "proposed", "cnn", "ResNet-34 U-Net with decoder attention modules and three output heads",
     "attention modules not described as multi-head self-attention"),
    ("model2", "MaskFormer (Cheng et al., 2021)", "MaskFormer", "transformer_baseline", "cnn_transformer_hybrid", "MaskFormer (backbone not stated)", ""),
    ("model3", "BEiT (Bao et al., 2021)", "BEiT", "transformer_baseline", "pure_vit", "BEiT", ""),
    ("model4", "DINOv2 (Oquab et al., 2023)", "DINOv2", "transformer_baseline", "foundation_model", "DINOv2 ViT", ""),
    ("model5", "DETR (Carion et al., 2020)", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "DETR", ""),
    ("model6", "U-Net (Ronneberger et al., 2015)", "U-Net", "cnn_baseline", "cnn", "U-Net", "the paper's baseline"),
    ("model7", "U-Net (EfficientNet-B7) (Ronneberger et al., 2015)", "U-Net (EfficientNet-B7)", "cnn_baseline", "cnn", "EfficientNet-B7", "best CNN on pixel IoU and tree IoU"),
    ("model8", "UNet++ (EfficientNet-B7) (Zhou et al., 2018)", "UNet++ (EfficientNet-B7)", "cnn_baseline", "cnn", "EfficientNet-B7", "best CNN on F1"),
]
res = Counter("res")
for item, lab, name, role, cat, bb, note in M:
    nums, ev = row(PID, 8, rf"^\s*{re.escape(lab)}\s+(?=\d)", 0)
    assert len(nums) == 9, (item, nums, ev)  # pixel IoU, tree IoU, instance P, R, F1, pixel P, R, F1, centroid error
    if item == "model1":
        ROWS += model(item, name, role, cat, bb, "domain_specific", "full_fine_tuning", 8, T2, ev, cat_note=note, pre_page=3, pre_loc="Sec. 3.1",
                      pre_ev="we use a ResNet-34 backbone, pre-trained on ImageNet and fine-tuned on the FLAIR-INC dataset", pre_note="ImageNet, then FLAIR-INC aerial data",
                      strat_page=4, strat_loc="Sec. 3.1", strat_ev="A ResNet-34 encoder fine-tuned on the diverse FLAIR-INC dataset")
    else:
        ROWS += model(item, name, role, cat, bb, None, None, 8, T2, ev, cat_note=note, **NP)
    st = "spatially separated test set; mean of five splits"
    ROWS += result(res(), "f1", item, "dataset1", nums[4], 8, T2, ev, unit="fraction", setting=st + "; dead-tree instances")
    ROWS += result(res(), "iou", item, "dataset1", nums[1], 8, T2, ev, unit="fraction", setting=st + "; mean IoU over trees", headline=(item == "model1"))
    ROWS += result(res(), "iou", item, "dataset1", nums[0], 8, T2, ev, unit="fraction", setting=st + "; pixel IoU")
ROWS += avail(code_url=URL, page=10, loc="Data availability", ev="has been made publicly accessible via URL https://github.com/Global-Ecosystem-Health-Observatory/TreeMort")
ROWS += [D("notes", "Table 2 compares 13 models; following protocol 4.1, recorded: the four transformers, the proposed model, the U-Net baseline and the best CNN per metric. KokoNet, PSPNet, FPN, DeepLabv3+ and HCF-Net are in Table 2. The proposed model is CNN-based, so the paper qualifies through its transformer baselines.", "Sec. 4 and Table 2 (pp. 7 to 8)")]
