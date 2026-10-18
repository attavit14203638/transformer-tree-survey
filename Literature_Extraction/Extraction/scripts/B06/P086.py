import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T3 = "P086", "Table II", "Table III"
# Binary tree against non-tree point segmentation of urban and suburban MS-LiDAR (Loosdorf-tree) with three transformer models. The IoU of the tree class is the primary result; the mIoU averages the
# tree and non-tree classes (never primary). Table III is the input ablation of the best model (SPT); its XYZ + green + NIR row is the SPT row of Table II. The authors' headline is the SPT mIoU of 85.28.
T = Tags(task="component_segmentation", level="point", split="test", agg="single")
OBJ_EV = "This study explores tree point extraction using MS-LiDAR and deep learning"
DS_EV = "and validation plots in a ratio of 68.53:16.28:15.19."
CODE_EV = "DL model (SPT) for both mono- and MS point clouds are pub-"

ROWS = [
    R("task", "component_segmentation", 1, "Abstract", OBJ_EV, note="binary tree against non-tree semantic segmentation of the point cloud (tree point extraction)"),
    R("output_unit", "tree_points", 1, "Abstract", OBJ_EV),
    R("tree_setting", "urban", 1, "Title", "Multispectral LiDAR data for extracting tree points in urban and suburban areas"),
    NR("tree_taxa", note="urban and suburban trees; species not given"),
    R("study_area", "Loosdorf and Melk, Lower Austria, Austria", 2, "Sec. 2.A", "RIEGL VQ-1560i-DW laser scanner over the cities of Loosdorf and Melk in Lower Austria, representing both urban and"),
    D("biome", "temperate", "Lower Austria; coded from the study area"),
    R("platform", "airborne", 2, "Sec. 2.A", "The dataset is collected using a multispectral airborne RIEGL VQ-1560i-DW laser scanner over the cities of Loosdorf and Melk in Lower Austria"),
    R("sensors", "lidar;multispectral", 2, "Sec. 2.A", "operates at wavelengths of 532 nm (green) and 1064 nm (NIR)", note="two wavelengths (green and NIR) of the RIEGL VQ-1560i-DW; pNDVI derived from their reflectance"),
    R("point_density", "21.11", 2, "Sec. 2.A", "resulting in an overall multispectral point cloud density of 21.11 points/m2", unit="pts/m2", note="9.1 (green) and 14.7 (NIR) points/m2 per scanner"),
    R("annotation_type", "point_label", 2, "Sec. 2.A", "annotated into tree and non-tree classes using CloudCompare software"),
    R("dataset_name", "Loosdorf-tree: MS-LiDAR urban and suburban tree benchmark, 1.68 km2 and 48,164,790 labelled points", 2, "Sec. 2.A", DS_EV, item="dataset1"),
    R("dataset_key", "loosdorf_tree_ms", 2, "Sec. 2.A", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 2.A", DS_EV, item="dataset1", note="training, testing and validation plots in the ratio 68.53:16.28:15.19"),
    R("dataset_public", "yes", 2, "Sec. 2.A", "Loosdorftree dataset is publicly available via the TU Wien dataset", item="dataset1", note="the link is to be given on acceptance (footnote 1 of the preprint)"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained for 300 epochs with AdamW and a class-weighted cross-entropy loss on the training plots (Sec. 2.C)")
ROWS += model("model1", "Superpoint Transformer (SPT)", "transformer_baseline", "hierarchical_vit", "hierarchical superpoint structure with self-attention between superpoints at several scales", None, None, 2, "Sec. 2.C",
              "it employs a self-attention mechanism to capture relationships between superpoints across multiple scales", cat_note="superpoint hierarchy with self-attention", **NPRE)
ROWS += model("model2", "Point Transformer V1 (PTv1)", "transformer_baseline", "pure_vit", "PointTransformerSeg50 with vector self-attention", None, None, 3, "Sec. 2.C", "The PointTransformer-", cat_note="self-attention point network", **NPRE)
ROWS += model("model3", "Point Transformer V3 (PTv3)", "transformer_baseline", "pure_vit", "serialised point attention with Z-order and Hilbert curve patterns", None, None, 2, "Sec. 2.C",
              "PTv3 is the latest version of the Point Transformer models", cat_note="self-attention point network with serialisation", **NPRE)
res = Counter("res")
for item, k in (("model1", 0), ("model2", 1), ("model3", 2)):
    n, ev = row(PID, 3, r"^\s*IoUTree \(%\)\s+(?=\d)", 0, cols=(80, 220))
    m, evm = row(PID, 3, r"^\s*mIoU \(%\)\s+(?=\d)", 0, cols=(80, 220))
    assert len(n) == 3 and len(m) == 3
    ROWS += result(res(), "iou", item, "dataset1", n[k], 3, T2, ev, unit="%", setting="Table II: tree points extraction from MS-LiDAR (XYZ + green + NIR), test plots", note="IoU of the tree class", **T, primary=True)
    ROWS += result(res(), "miou", item, "dataset1", m[k], 3, T2, evm, unit="%", setting="Table II: tree points extraction from MS-LiDAR (XYZ + green + NIR), test plots", note="mIoU of the tree and non-tree classes", **T, headline=(item == "model1"))
# Table III: spectral ablation of SPT (tree IoU and mIoU); the XYZ + green + NIR row repeats Table II and is not recorded again
for lab, rx in (("XYZ", r"XYZ"), ("XYZ + green", r"\+ green"), ("XYZ + NIR", r"\+ NIR"), ("XYZ + pNDVI", r"\+ pNDVI"), ("XYZ + green + NIR + pNDVI", r"\+ green \+ NIR \+ pNDVI")):
    n, ev = row(PID, 4, rf"^\s*{rx}\s+(?=\d)", 0, cols=(0, 66))
    assert len(n) == 5, (lab, n)  # IoU non-tree, IoU tree, mIoU, mAcc, OA
    st = f"Table III: SPT with input {lab}, test plots"
    ROWS += result(res(), "iou", "model1", "dataset1", n[1], 4, T3, ev, unit="%", setting=st, note="IoU of the tree class; spectral ablation of the best model", **T)
    ROWS += result(res(), "miou", "model1", "dataset1", n[2], 4, T3, ev, unit="%", setting=st, note="mIoU of the tree and non-tree classes; spectral ablation of the best model", **T)
ROWS += [D("total_params", "0.212", "printed as 212K (Table V); converted to millions", item="model1", unit="M"),
         R("total_params", "170.3", 4, "Table V", "Parameters 212K 170.3M 46.15M", item="model2", unit="M"),
         R("total_params", "46.15", 4, "Table V", "Parameters 212K 170.3M 46.15M", item="model3", unit="M"),
         R("limitations", "NIR alone lowers the tree IoU, relative reflectance ignores the incidence angle, and only one dataset of one sensor is used", 4, "Sec. 4", "However, since these values do not consider")]
ROWS += [R("code_available", "yes", 3, "Sec. 3", CODE_EV, note="GitHub repository of the paper (3DOM-FBK/TreePointExtractor); code and pre-trained model of SPT"),
         R("code_url", "https://github.com/3DOM-FBK/TreePointExtractor", 3, "Sec. 3", "5 https://github.com/3DOM-FBK/TreePointExtractor"),
         R("weights_available", "yes", 3, "Sec. 3", "and pre-trained model of the best-performing tree extraction", note="the pre-trained SPT for mono- and MS point clouds"),
         R("data_available", "yes", 2, "Sec. 2.A", "Loosdorftree dataset is publicly available via the TU Wien dataset")]
ROWS += [D("notes", "Table II compares SPT, PTv1 and PTv3 (IoU of the tree and non-tree classes, mIoU, mAcc, OA; tree IoU and mIoU recorded; the tree IoU is primary). Table III repeats SPT with six input feature sets (XYZ, + green, + NIR, + pNDVI, + green + NIR, + green + NIR + pNDVI); tree IoU and mIoU recorded for the five rows other than the Table II configuration. Table IV (error rate of points above 2 m normalized height: 39.63, 31.07, 41.57, 29.02, 35.12 and 35.93 percent for the same six inputs) is not recorded. Table V gives parameters (recorded) and training time (0.67, 26.37 and 2.32 h; not recorded). The mIoU averages tree and non-tree classes, so it is not primary; the authors' headline is the SPT mIoU of 85.28.",
           "Sec. 3 to 4 and Tables II to V (pp. 3 to 4)")]
