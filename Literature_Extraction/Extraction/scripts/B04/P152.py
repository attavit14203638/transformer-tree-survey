import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T4 = "P152", "Table 3 (image)", "Table 4"
# Pixel-wise detection of unhealthy tree crowns (UTC): health classification at pixel level on the unseen test data. The per-class UTC scores are primary, the average over the
# background and UTC classes (Avg F, the paper's headline F1) is not, because the background is not a tree class (protocol 6.3).
T = Tags(task="health_classification", level="pixel", split="test", agg="single")
TASK_EV = "we aimed to determine the optimal approach for detecting unhealthy tree crowns (UTC)"
T2_EV = "Number of image patches used 400"

ROWS = [
    R("task", "health_classification", 1, "Abstract", TASK_EV, note="unhealthy tree crowns are segmented as pixels (binary UTC mask)"),
    R("output_unit", "crown_mask", 1, "Abstract", TASK_EV),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "Connecticut (CT) is a densely forested state of the USA with a mixed forest vegetation type"),
    R("tree_taxa", "mixed forest, predominantly oak-hickory (58 tree species in Connecticut); species not separated", 4, "Sec. 2.1", "with 58 species of trees in Connecticut [72]. It has a predominant oak-hickory species"),
    R("study_area", "eastern Windham County, Connecticut, USA", 4, "Sec. 2.1", "The study area is in the eastern part of Windham County, Connecticut, USA"),
    D("biome", "temperate", "Connecticut, USA; not stated in the paper"),
    R("platform", "airborne", 5, "Sec. 2.2", "NAIP aerial imagery was",
      note="NAIP aerial imagery from a Leica ADS-100 airborne digital sensor (Sec. 2.3.1)"),
    R("sensors", "multispectral", 10, "Table 2", "Spectral resolution 4 (Red, Green, Blue, Near infrared)"),
    R("gsd", "0.6", 10, "Table 2", "Image resolution 0.6 m", unit="m"),
    R("annotation_type", "polygon_mask", 5, "Sec. 2.2", "creating polygon labels for the UTCs in shapefile format, which were later turned into a binary mask image"),
    R("dataset_name", "UTC annotations on NAIP imagery, Windham County (400 patches of 254 × 254 pixels)", 10, "Table 2", T2_EV, item="dataset1"),
    R("dataset_key", "own_p152_ct_naip", 10, "Table 2", T2_EV, item="dataset1"),
    R("dataset_role", "train_test", 10, "Table 2", "Train:Test:Validation data ratio 80:10:10", item="dataset1"),
    R("dataset_public", "on_request", 30, "Data Availability Statement", "The raw data supporting the conclusions of this article will be made available by the authors on request", item="dataset1", note="the NAIP imagery itself is public"),
    R("n_images", "400", 10, "Table 2", T2_EV, item="dataset1", unit="tiles", note="254 × 254 pixel patches cut from 16 NAIP scenes; split 80:10:10"),
    R("n_trees", "5133", 10, "Table 2", "Total number of crowns annotated 5133", item="dataset1", unit="trees", note="unhealthy tree crowns; 1 to 170 crowns per patch"),
    R("dataset_name", "annotator study: 30 new patches of the same NAIP imagery annotated by nine annotators", 14, "Sec. 2.7", "we engaged nine individuals to annotate 30 new image patches, each containing at least one tree crown", item="dataset2"),
    R("dataset_key", "own_p152_ct_naip", 14, "Sec. 2.7", "we engaged nine individuals to annotate 30 new image patches, each containing at least one tree crown", item="dataset2", note="same NAIP imagery and study area"),
    R("dataset_role", "test_only", 25, "Sec. 3.3", "we detected UTC segmentation on the dataset provided to the annotators using our SegFormer-RGB model", item="dataset2"),
    R("dataset_public", "on_request", 30, "Data Availability Statement", "The raw data supporting the conclusions of this article will be made available by the authors on request", item="dataset2"),
    R("n_images", "30", 14, "Sec. 2.7", "we engaged nine individuals to annotate 30 new image patches, each containing at least one tree crown", item="dataset2", unit="tiles"),
]
PRE = dict(pre_page=27, pre_loc="Sec. 4", pre_ev="The feature extraction backbone was trained on standard RGB channels", pre_note="pre-trained on RGB images; the pre-training dataset is not named",
           strat_page=27, strat_loc="Sec. 4", strat_ev="we used a transfer learning strategy while keeping the backbone frozen, and only the segmentation head was trained",
           strat_note="batch size 12, 100 epochs with early stopping, Adam, learning rate 0.001, weighted focal loss (Sec. 2.5)")
M = [  # item, name, role, category, backbone, note
    ("model1", "SegFormer-B0", "transformer_baseline", "hierarchical_vit", "SegFormer-B0 (hierarchical transformer encoder with All-MLP decoder)", "the ViT-based model of the study (Sec. 2.4.1)"),
    ("model2", "U-Net (ResNet101 encoder)", "cnn_baseline", "cnn", "ResNet101", "CNN encoder-decoder (Sec. 2.4.2)"),
    ("model3", "DeepLab v3+ (ResNet-101)", "cnn_baseline", "cnn", "ResNet-101 with atrous spatial pyramid pooling", "CNN encoder-decoder (Sec. 2.4.3)"),
]
EVM = {"model1": "For this study, we used the SegFormer-B0 architecture", "model2": "We used ResNet101 as the encoder backbone", "model3": "This framework leverages ResNet-101 as its foundational"}
PAGE = {"model1": 11, "model2": 11, "model3": 12}
for item, name, role, cat, bb, note in M:
    ROWS += model(item, name, role, cat, bb, "other", "head_only", PAGE[item], "Sec. 2.4", EVM[item], cat_note=note, **PRE)
ROWS += [R("total_params", "3.75", 11, "Sec. 2.4.1", "has 3.75 million parameters", item="model1", unit="M"),
         R("total_params", "58", 12, "Sec. 2.4.3", "around 58 million parameters", item="model3", unit="M", note="printed for the DeepLab v3+ framework (the sentence attaches it to the decoder module)")]
# Table 3 (image): rows read from the rendered page. Columns: IoU (UTC class), OA, precision BG and UTC, recall BG and UTC, F1 BG and UTC, average precision, recall and F1.
T3V = {
    "model2": {"RGB": "0.30 0.96 0.98 0.46 0.98 0.46 0.98 0.46 0.72 0.72 0.72", "CIR": "0.30 0.96 0.98 0.43 0.98 0.47 0.98 0.45 0.71 0.73 0.72", "VI": "0.32 0.96 0.99 0.43 0.97 0.56 0.98 0.49 0.71 0.77 0.74",
               "PCA": "0.02 0.96 0.98 0.13 0.99 0.02 0.98 0.03 0.56 0.51 0.51", "RGBPC": "0.09 0.96 0.97 0.26 0.99 0.12 0.98 0.17 0.62 0.56 0.58"},
    "model3": {"RGB": "0.48 0.97 0.99 0.6 0.98 0.71 0.99 0.65 0.8 0.85 0.82", "CIR": "0.45 0.97 0.99 0.59 0.98 0.66 0.99 0.62 0.79 0.82 0.81", "VI": "0.4 0.96 0.99 0.49 0.97 0.68 0.98 0.57 0.74 0.83 0.78",
               "PCA": "0.24 0.96 0.98 0.39 0.98 0.38 0.98 0.38 0.69 0.68 0.68", "RGBPC": "0.30 0.95 0.99 0.37 0.96 0.61 0.97 0.46 0.68 0.79 0.72"},
    "model1": {"RGB": "0.54 0.98 0.99 0.62 0.98 0.8 0.99 0.7 0.81 0.89 0.85", "CIR": "0.48 0.97 0.99 0.53 0.97 0.83 0.98 0.65 0.76 0.9 0.82", "VI": "0.51 0.97 0.99 0.62 0.98 0.73 0.99 0.67 0.81 0.86 0.83",
               "PCA": "0.24 0.94 0.98 0.30 0.96 0.52 0.97 0.38 0.64 0.74 0.68", "RGBPC": "0.45 0.97 0.99 0.53 0.98 0.74 0.98 0.62 0.76 0.86 0.80"},
}
NAMES = {"model1": "SegFormer", "model2": "U-Net", "model3": "DeepLab v3+"}
res = Counter("res")
for item in ("model1", "model2", "model3"):
    for band, vals in T3V[item].items():
        v = vals.split()
        assert len(v) == 11, (item, band, v)
        ev = f"{NAMES[item]} {band} {vals}"
        st = f"Table 3: {band} input, unseen test data"
        prim = band == "RGB"
        ROWS += result(res(), "iou", item, "dataset1", v[0], 16, T3, ev, unit="fraction", setting=st, note="IoU column (the UTC class)", **T, primary=prim)
        ROWS += result(res(), "f1", item, "dataset1", v[7], 16, T3, ev, unit="fraction", setting=st, note="F1-score of the UTC class", **T, primary=prim)
        ROWS += result(res(), "f1", item, "dataset1", v[10], 16, T3, ev, unit="fraction", setting=st, note="Avg F: mean of the background and UTC F1-scores (not a tree-only score)", **T,
                       headline=(item == "model1" and band == "RGB"))
# Table 4: SegFormer-RGB on the annotator-study patches against three reference polygon sets
for lab, ref in (("Intersection", "intersection of all annotators (minimum consensus)"), ("Union", "union of all annotators (maximum extent)"), ("Median", "median polygons (most common consensus)")):
    nums, ev = row(PID, 25, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 6, (lab, nums)  # recall, precision, F1, IoU, relative change in IoU, relative change in F1
    st = f"Table 4: SegFormer-RGB on the 30 annotator-study patches, reference = {ref}"
    ROWS += result(res(), "f1", "model1", "dataset2", nums[2], 25, T4, ev, unit="fraction", setting=st, note="F1-score of the UTC class", **T)
    ROWS += result(res(), "iou", "model1", "dataset2", nums[3], 25, T4, ev, unit="fraction", setting=st, note="IoU of the UTC class", **T)
ROWS += [R("limitations", "Model scores depend strongly on subjective crown annotations: IoU and F1 change a lot with the reference polygons used", 1, "Abstract",
           "reflecting the significant sensitivity of model performance to inconsistencies in ground truth annotations")]
ROWS += avail(data="on_request", data_page=30, data_loc="Data Availability Statement", data_ev="The raw data supporting the conclusions of this article will be made available by the authors on request")
ROWS += [D("notes", "Table 3 (p.16, an image) gives IoU, overall accuracy and per-class precision, recall and F1 for three models and five band combinations (RGB, CIR, VI, PCA, RGBPC) on the unseen test data; recorded are the UTC-class IoU and F1 for all 15 configurations (primary: RGB) and the average F1 over background and UTC (Avg F, the paper's headline 0.85 and 0.82, not primary). Overall accuracy, precision and recall are in Table 3. Table 4 re-scores SegFormer-RGB against three reference polygon sets on 30 patches annotated by nine annotators; the abstract's 14.81 to 57.41 percent (IoU) and 8.57 to 47.14 percent (F1) are relative changes against the median reference, not scores.",
           "Sec. 3 and Tables 2 to 4 (pp. 10 to 25)")]
