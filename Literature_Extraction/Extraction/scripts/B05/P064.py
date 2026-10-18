import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P064", "Table II"
# Part segmentation of single eucalyptus trees (stem regions at the diameter measurement heights) scored by mIoU over points with 5-fold cross-validation. The table does not say whether the single
# value per model is the fold mean, so the aggregation is not stated.
T = Tags(task="component_segmentation", level="point", split="cross_validation", agg="not_stated")
OBJ_EV = "A dataset comprising 55 clonal eucalyptus trees was constructed using an iPhone 12 Pro Max LiDAR sensor"
DS_EV = "The dataset comprises information collected from 55 clonal eucalyptus trees (Eucalyptus sp.) in June 2023"
CV_EV = "We use the 5-fold cross-validation strategy to evaluate our"

ROWS = [
    R("task", "component_segmentation", 1, "Abstract", OBJ_EV, note="regions of a single tree trunk that correspond to the diameter measurements (stem, d30cm, d70cm, d100cm, dbh)"),
    R("output_unit", "tree_component", 2, "Sec. 3.A", "The annotation of each region corresponding to diameter measurements was inspired by the approach presented by ShapeNetPart"),
    R("tree_setting", "plantation", 1, "Abstract", "particularly in eucalyptus plantations"),
    R("tree_taxa", "Eucalyptus sp. (clonal)", 2, "Sec. 3.A", DS_EV),
    NR("study_area", note="the place of the 55 trees is not stated; the authors are in Brazil"),
    NR("biome", note="not stated"),
    R("platform", "mobile_lidar", 2, "Sec. 3.A", "The acquisition process involved rotating the smartphone around each tree", note="hand-held iPhone LiDAR, 2 m range"),
    R("sensors", "lidar", 2, "Sec. 3.A", "Point clouds were acquired using a LiDAR sensor integrated into an iPhone 12 Pro Max"),
    R("annotation_type", "point_label", 2, "Sec. 3.A", "The annotation of each region corresponding to diameter measurements was inspired by the approach presented by ShapeNetPart", note="one label per point and region"),
    R("dataset_name", "own iPhone LiDAR point clouds of 55 clonal eucalyptus trees with stem region labels", 2, "Sec. 3.A", DS_EV, item="dataset1"),
    R("dataset_key", "own_p064_eucalyptus_iphone", 2, "Sec. 3.A", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 3.B", CV_EV, item="dataset1", note="five folds of 80 percent training and 20 percent validation data"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_trees", "55", 2, "Sec. 3.A", DS_EV, item="dataset1", unit="trees"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained with the settings of Table I (optimizer, learning rate, batch size, 2500 or 300 epochs)")
M = [  # item, key, name, role, category, backbone, page, loc, evidence, note
    ("model1", "PointNet", "PointNet", "other_baseline", "other", "PointNet", 2, "Sec. 2.B", "introduced PointNet, a deep learning architecture that processes raw point clouds directly", "MLP-based point network (Sec. 2.B)"),
    ("model2", "PointNet++", "PointNet++", "other_baseline", "other", "PointNet++", 2, "Sec. 2.B", "PointNet++ [15] was developed as a hierarchical neural network that captures local structures through recursive partitioning", "MLP-based point network (Sec. 2.B)"),
    ("model3", "DGCNN", "DGCNN", "cnn_baseline", "cnn", "EdgeConv on kNN graphs", 2, "Sec. 2.B", "Dynamic Graph CNN (DGCNN) [16] utilizes dynamic graph convolutions on kNN graphs", "graph convolution (Sec. 2.B)"),
    ("model4", "PointMLP", "PointMLP", "other_baseline", "other", "residual MLP", 2, "Sec. 2.B", "PointMLP [19] simplifies point cloud analysis by using a residual MLP network", "MLP-based point network (Sec. 2.B)"),
    ("model5", "PointNeXt-S", "PointNeXt-S", "other_baseline", "other", "PointNeXt-S", 2, "Sec. 2.B", "PointNeXt [18] modernizes PointNet++ with enhanced training strategies", "MLP-based point network, a PointNet++ successor (Sec. 2.B)"),
    ("model6", "PointT", "Point Transformer (PointT)", "transformer_baseline", "pure_vit", "self-attention on point neighbourhoods", 2, "Sec. 2.C", "a transformer model optimized for kNN computation", "attention-based point model (Sec. 2.C)"),
    ("model7", "PVT", "Point Voxel Transformer (PVT)", "transformer_baseline", "cnn_transformer_hybrid", "voxel-based and point-based branches with attention", 2, "Sec. 2.C",
     "the Point-Voxel Transformer (PVT) proposed by [24] integrates voxel-based and point-based advantages for effective 3D analysis", "voxel branch (3D convolution, Sec. 2.A) combined with attention (Sec. 2.C)"),
]
for item, key, name, role, cat, bb, pg, loc, ev, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, pg, loc, ev, cat_note=note, **NPRE)
res = Counter("res")
for item, key, name, role, cat, bb, pg, loc, ev, note in M:
    rx = {"PointNet": r"PointNet \[14\]", "PointNet++": r"PointNet\+\+ \[15\]", "DGCNN": r"DGCNN \[16\]", "PointMLP": r"PointMLP \[19\]", "PointNeXt-S": r"PointNeXt-S \[18\]", "PointT": r"PointT \[23\]", "PVT": r"PVT \[24\]"}[key]
    n, rev = row(PID, 3, rf"{rx}\s+(?=\d)", 0, cols=(80, 200))
    assert len(n) == 6, (key, n)  # mIoU, stem, d30cm, d70cm, d100cm, dbh
    ROWS += result(res(), "miou", item, "dataset1", n[0], 3, T2, rev, unit="fraction", setting="Table II: 5-fold cross-validation, mIoU over points", note="mIoU of the eucalyptus stem regions; the per-region values (stem, d30cm, d70cm, d100cm, dbh) are in the table", **T, primary=True)
ROWS += [R("limitations", "The dataset of 55 trees is small, so all mIoU values are lower than on ShapeNetPart", 3, "Sec. 4", "the reduced size of our dataset negatively impacted the mIoU metric"),
         D("code_available", "no", "No code repository or availability statement in the paper"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table II gives the mIoU and the per-region values (stem, d30cm, d70cm, d100cm, dbh) of seven point cloud networks with 5-fold cross-validation on 55 eucalyptus trees; the overall mIoU is recorded. Fig. 2 plots mIoU against inference time (not recorded). The abstract says transformer-based models were best; PVT has the highest mIoU (0.623) and PointNeXt-S the second (0.612).",
           "Sec. 4 and Table II (p. 3)")]
