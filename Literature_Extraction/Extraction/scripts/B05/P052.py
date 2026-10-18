import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T4 = "P052", "Table 2", "Table 4"
# Binary tree crown masks scored by IoU and F1 of the tree class: crown segmentation at pixel level. The paper gives no train, validation and test sizes, so the split is not stated.
T = Tags(task="crown_segmentation", level="pixel", split="not_stated", agg="single")
DS_EV = "Although trained on a dataset of only 1200 image patches"
SAT_EV = "Our research is based on images of Hormoz Island, Kouhdasht, and Mahmoudabad, taken by WorldView-2, WorldView-3, and GeoEye-1 satellites"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "This study introduces a deep learning framework for urban tree crown segmentation that exclusively leverages high-resolution satellite imagery from GeoEye-1, WorldView-2, and WorldView-3"),
    R("output_unit", "crown_mask", 5, "Sec. 2.1", "The resulting dataset comprises 256 × 256 pixel RGBN image patches paired with their corresponding binary crown masks", note="binary masks; tree counts and crown areas are named as possible outputs (p.8)"),
    R("tree_setting", "urban", 1, "Abstract", "Urban tree detection is fundamental to effective forestry management"),
    NR("tree_taxa", note="urban trees; species not given"),
    R("study_area", "Mahmoudabad (north), Kouhdasht (west) and Hormoz Island (south), Iran", 7, "Sec. 2.2", SAT_EV),
    D("biome", "multiple", "humid north, temperate to semi-arid west and arid coastal south of Iran (Sec. 2.2); not coded in the paper"),
    R("platform", "satellite", 7, "Sec. 2.2", SAT_EV),
    R("sensors", "rgb;multispectral", 11, "Sec. 4", "The dataset consisted of 4-band RGBN satellite images", note="pan-sharpened red, green, blue and near-infrared"),
    NR("gsd", note="the three sensors give a 0.31 to 0.46 m resolution interval (p.15); no single GSD is printed"),
    R("annotation_type", "polygon_mask", 5, "Sec. 2.1", "all inaccurate or incomplete crown delineations were corrected manually using the polygon editing tools in ArcGIS Pro", note="NDVI watershed and SAM masks refined by hand"),
    R("dataset_name", "own multi-sensor urban tree crown dataset, three Iranian sites (1200 patches of 256 × 256 pixels)", 15, "Sec. 5", DS_EV, item="dataset1"),
    R("dataset_key", "own_p052_iran_urban_trees", 15, "Sec. 5", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.1", "was used to train and evaluate all segmentation models", item="dataset1"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_images", "1200", 15, "Sec. 5", DS_EV, item="dataset1", unit="tiles", note="256 × 256 pixel RGBN patches"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="not stated")
ROWS += model("model1", "Residual U-Net with Attention Gates (proposed)", "proposed", "cnn", "modified ResNet-50 backbone with four input channels, residual blocks and attention gates", None, None, 8, "Sec. 3",
              "We propose a tree crown segmentation framework that integrates a Residual U-Net architecture with Attention Gates (AGs)", cat_note="convolutional U-Net with attention gates, a generic attention module (protocol 4.5)",
              pre_status="not_reported", pre_note="not stated", strat_note="Adam, learning rate 0.001, batch size 16, 100 epochs, BCE plus Dice loss (Sec. 4.1)")
ROWS += model("model2", "DeepLab v3 (ResNet50)", "cnn_baseline", "cnn", "ResNet50", None, None, 16, "Sec. 5", "convolutional networks such as U-Net and DeepLab v3, with their strong local feature aggregation",
              cat_note="convolutional network; best baseline on all four metrics of Table 4", **NPRE)
ROWS += model("model3", "U-Net (VGG19)", "cnn_baseline", "cnn", "VGG19", None, None, 16, "Sec. 5", "convolutional networks such as U-Net and DeepLab v3, with their strong local feature aggregation",
              cat_note="convolutional network; the U-Net on which the proposed model builds", **NPRE)
ROWS += model("model4", "SegFormer (MiT-B2)", "transformer_baseline", "hierarchical_vit", "MiT-B2", None, None, 16, "Sec. 5", "transformer-based architectures such as SegFormer demonstrate strong performance on large-scale semantic segmentation benchmarks",
              cat_note="named as a transformer-based architecture (Sec. 5)", **NPRE)
ROWS += model("model5", "SegFormer (MiT-B0)", "transformer_baseline", "hierarchical_vit", "MiT-B0", None, None, 16, "Sec. 5", "transformer-based architectures such as SegFormer demonstrate strong performance on large-scale semantic segmentation benchmarks",
              cat_note="named as a transformer-based architecture (Sec. 5)", **NPRE)
ROWS += model("model6", "Residual U-Net without attention gates (ablation)", "cnn_baseline", "cnn", "modified ResNet-50 backbone with residual blocks, no attention gates", None, None, 10, "Sec. 3.4",
              "an ablation study compared two versions of the model: one with and one without AGs", cat_note="the proposed network without its attention gates", **NPRE)
ROWS += [D("total_params", "24.857985", "24,857,985 parameters (Table 2) in millions", item="model1", unit="M"),
         R("inference_speed", "about 0.15 s per 256 × 256 patch on an NVIDIA RTX 3070 (8 GB)", 15, "Table 3", "Inference (per 256 × 256 patch) ∼0.15 s", item="model1")]
res = Counter("res")
SRC = [("model1", r"Our Network\s+ResNet50"), ("model2", r"ResNet50"), ("model3", r"VGG19"), ("model4", r"MiT-B2"), ("model5", r"SegFormer\s+MiT-B0")]
RES = {}
for item, rx in SRC:
    nums, ev = row(PID, 16, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(nums) == 4, (item, nums)  # F1, IoU, precision, recall
    st = "Table 4: comparison of networks and backbones on the 1200-patch dataset"
    RES[(item, "f1")] = res()
    ROWS += result(RES[(item, "f1")], "f1", item, "dataset1", nums[0], 16, T4, ev, unit="fraction", setting=st, note="F1-score of the tree class", **T, primary=True)
    RES[(item, "iou")] = res()
    ROWS += result(RES[(item, "iou")], "iou", item, "dataset1", nums[1], 16, T4, ev, unit="fraction", setting=st, note="IoU of the tree class", **T, primary=True, headline=(item == "model1"))
EV_AB = "The inclusion of AGs improved the F1-score from 0.8772 to 0.9121 and IoU from 0.8017 to 0.8384"
ROWS += result(res(), "f1", "model6", "dataset1", "0.8772", 10, "Sec. 3.4", EV_AB, unit="fraction", setting="ablation without attention gates (text only)", note="F1-score of the tree class", **T)
ROWS += result(res(), "iou", "model6", "dataset1", "0.8017", 10, "Sec. 3.4", EV_AB, unit="fraction", setting="ablation without attention gates (text only)", note="IoU of the tree class", **T)
ROWS += [R("limitations", "Only three Iranian urban areas and a 0.31 to 0.46 m resolution interval were tested, so global generalisation is not shown", 18, "Sec. 6", "the current results should be interpreted as evidence of cross-sensor reliability and intra-regional adaptability rather than full global generalization"),
         D("code_available", "no", "No code repository or availability statement in the paper"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table 4 compares the proposed Residual U-Net with Attention Gates with 19 networks and backbones (F1, IoU, precision, recall); following protocol 4.1 the proposed model, the best baseline (DeepLab v3 ResNet50), the U-Net it builds on (VGG19) and the two SegFormer rows are recorded, with F1 and IoU; the ablation without attention gates is from the text (p.10). The text also prints F1 and IoU with four decimals for the proposed model and two baselines (0.9121 and 0.8384; DeepLab v3 0.8712 and 0.7612; U-Net 0.8568 and 0.7495); Table 4 is used and only the U-Net values differ beyond rounding (cf items).",
           "Sec. 5 and Tables 2 to 4 (pp. 13 to 16)")]
ROWS += conflict("cf1", f"{RES[('model3', 'f1')]}.value", "0.8568", 15, "Sec. 5", "U-Net with VGG19 (F1 = 0.8568, IoU = 0.7495)", "V1a",
                 "Table 4, the comparison table, gives 0.85; the text of Sec. 5 gives 0.8568 (which would round to 0.86)", unit="fraction")
ROWS += conflict("cf2", f"{RES[('model3', 'iou')]}.value", "0.7495", 15, "Sec. 5", "U-Net with VGG19 (F1 = 0.8568, IoU = 0.7495)", "V1a",
                 "Table 4, the comparison table, gives 0.74; the text of Sec. 5 gives 0.7495 (which would round to 0.75)", unit="fraction")
