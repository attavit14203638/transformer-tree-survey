import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1 = "P137", "Table 1"
# Street trees of Anyang-si (South Korea): U-Net, DeepLabV3 and Mask2Former are trained on 10,021 annotated trees of 25 cm aerial orthophotos (Table 1: precision, recall and F1; only F1 is recorded). The vitality
# index (0.7 NDVI + 0.3 NDMI from CAS500-1 and Sentinel-2) is computed per detected tree and is neither learned nor scored. The paper does not say which split Table 1 is computed on (the section is titled "Training
# Performance") nor whether precision and recall count pixels or trees, so split and level are not_stated. The paper names U-Net as the model it uses for the vitality step; Table 1 ranks Mask2Former first.
T = Tags(task="detection", level="not_stated", split="not_stated", agg="single")
TASK_EV = "Deep learning models such as U-Net, Mask2former, and DeepLabV3 were applied for tree detection"
TREES_EV = "A total of 10,021 individual trees were annotated across diverse urban environments"
MODELS_EV = "U-Net, DeepLabV3, and Mask2Former were trained on the constructed dataset"
AERIAL_EV = "Orthorectified aerial RGB imagery (25 cm resolution)"
ROWS = [
    R("task", "detection;health_classification", 1, "Abstract", "this study aims to analyze the vitality of individual street trees in urban areas by using high-resolution aerial and satellite imagery with deep learning models",
      note="the deep learning models only detect street trees (the scored task); the vitality index of each tree is computed from NDVI and NDMI (not learned, not scored) and classified into five levels"),
    R("output_unit", "crown_mask;tree_label", 3, "Sec. 3.4.3", "The resulting vitality index is classified into five levels to evaluate the vitality of each individual street tree",
      note="semantic masks of trees that are used as tree polygons (U-Net); five vitality levels per tree from the index"),
    R("tree_setting", "urban", 1, "Abstract", "Urban street trees provide various environmental benefits"),
    R("tree_taxa", "urban street trees (species not named)", 1, "Abstract", "Urban street trees provide various environmental benefits"),
    R("study_area", "Anyang-si, Gyeonggi-do, South Korea (training data from Gwanyang-dong)", 2, "Sec. 3.1", "The study was conducted in Anyang-si, Gyeonggi-do, South Korea"),
    D("biome", "temperate", "Anyang-si, Gyeonggi-do, South Korea; not stated in the paper"),
    R("platform", "multiple", 2, "Sec. 3.1", AERIAL_EV, note="aerial orthophotos of the National Geographic Information Institute of Korea (training data), CAS500-1 and Sentinel-2 satellite images (vegetation indices)"),
    R("sensors", "rgb;multispectral", 2, "Sec. 3.1", "CAS500-1 provides 2 m resolution imagery with Red, Green, Blue, and NIR bands", note="aerial RGB for detection; CAS500-1 (RGB and NIR) for NDVI; Sentinel-2 (13 bands, SWIR) for NDMI"),
    R("gsd", "25", 2, "Sec. 3.1", AERIAL_EV, unit="cm", note="aerial imagery used for detection; CAS500-1 2 m, Sentinel-2 10 to 60 m"),
    NR("annotation_type", note="form of the 10,021 tree annotations is not stated; the models are semantic segmentation models"),
    R("dataset_name", "own aerial street-tree dataset (25 cm orthophotos), Gwanyang-dong, Anyang-si", 2, "Sec. 3.2.1", TREES_EV, item="dataset1"),
    R("dataset_key", "own_p137_anyang_street_trees", 2, "Sec. 3.2.1", TREES_EV, item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 3.2.2", MODELS_EV, item="dataset1", note="the split of the 10,021 annotated trees is not described"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_trees", "10021", 2, "Sec. 3.2.1", TREES_EV, item="dataset1", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="NVIDIA RTX 5070, input tiles of 512 x 512 pixels, batch size 4, 20 epochs (Sec. 3.2.2); initialisation not stated")
res = Counter("res")
for k, (lab, rx, name, role, cat, cnote, cols) in enumerate((("U-Net", r"^\s*U-Net\s+(?=0\.)", "U-Net", "cnn_baseline", "cnn", "convolutional segmentation network", (80, 220)),
                                                           ("DeepLabV3", r"^\s*DeepLabV3\s+(?=0\.)", "DeepLabV3", "cnn_baseline", "cnn", "convolutional segmentation network", (80, 220)),
                                                           ("Mask2former", r"Mask2former\s+(?=0\.)", "Mask2Former", "transformer_baseline", "cnn_transformer_hybrid", "Mask2Former (transformer decoder) with a backbone that is not stated; the category used for Mask2Former without a named transformer backbone", (70, 220))), start=1):
    item = f"model{k}"
    nums, ev = row(PID, 3, rx, 0, cols=cols)
    assert len(nums) == 3, (lab, nums)  # precision, recall, F1
    ROWS += model(item, name, role, cat, "not stated in the paper", None, None, 3, T1, ev, cat_note=cnote, **NP)
    ROWS += result(res(), "f1", item, "dataset1", nums[2], 3, T1, ev, unit="fraction", setting="Table 1 (Training Performance of Each Model); split and scored unit not stated",
                   note="F1 score; precision and recall of the same row are not recorded", **T, primary=True)
ROWS += [R("limitations", "The 2020 aerial and 2023 satellite imagery are from different dates, and densely planted trees are hard to delineate at 25 cm", 4, "Sec. 5", "A limitation of this study is the temporal gap between the 2020 aerial and 2023 satellite imagery")]
ROWS += avail(code="no", weights="not_stated", data="no")
ROWS += [D("notes", "Only Table 1 is a quantitative evaluation of a learned model: precision, recall and F1 of U-Net (F1 0.756786), DeepLabV3 (0.716752) and Mask2Former (0.797997) for street tree detection. Only F1 is recorded; the paper does not state the split (the section is titled Training Performance) or whether the scores count pixels or trees (the text speaks of 80 percent of detected trees being street trees, but the models are semantic segmentation models), so level and split are not_stated. The abstract says U-Net had the best performance, whereas Table 1 ranks Mask2Former first on all metrics; the paper chooses U-Net for the vitality step because Mask2Former merges adjacent crowns (Fig. 4). The vitality index (0.7 NDVI + 0.3 NDMI) is computed per tree without a reference and is not scored. The backbone of Mask2Former is not stated, so it is recorded as cnn_transformer_hybrid (the category used for Mask2Former without a named transformer backbone).",
           "Sec. 3 and 4, Table 1 (pp. 2 to 4)")]
