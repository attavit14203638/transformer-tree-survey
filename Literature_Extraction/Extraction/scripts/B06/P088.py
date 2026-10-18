import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4 = "P088", "Table 4"
# Wood-leaf separation of single-tree point clouds (component segmentation, point level). Table 4 compares nine methods on five datasets in precision and mIoU; the mIoU (wood and leaf classes, both tree
# components) is recorded and the first listed dataset (rubber tree plantations) is primary. Precision is not recorded (an IoU is given). The authors' headline is the TreeNet3D mIoU of the enhanced PT (abstract).
T = Tags(task="component_segmentation", level="point", split="test", agg="single")
OBJ_EV = "we propose an innovative approach that integrates Local Surface Features (LSF) and a Position Encoding (PosE) module within the Point Transformer (PT) network"
DSN = ("The prepared dataset encompasses three distinct LiDAR point cloud collections, each featuring a different forest type")
SPLIT_EV = "For testing purposes, 30% of the data are utilized, while 70% are allocated for training"

ROWS = [
    R("task", "component_segmentation", 1, "Abstract", OBJ_EV, note="wood and leaf separation of individual tree point clouds"),
    R("output_unit", "tree_component", 1, "Abstract", "Afterward, the segmented branch point cloud was used to model tree skeletons automatically"),
    R("tree_setting", "mixed", 5, "Sec. 2.1", DSN, note="rubber plantation, mixed forest and urban forest (private data); tropical species and synthetic trees (public data)"),
    R("tree_taxa", "rubber tree (Hevea brasiliensis), Litsea glutinosa, Acacia auriculiformis, Streblus asper; tropical species (LeWoS); synthetic flamboyant tree (TreeNet3D)", 5, "Sec. 2.1", "a rubber tree plantation with Hevea brasiliensis"),
    R("study_area", "Danzhou, Hainan Island, China (private data)", 3, "Sec. 2.1", "The study sites were located in Danzhou, Hainan Island, China"),
    R("biome", "tropical", 3, "Sec. 2.1", "Danzhou is situated within the low-latitude tropics and experiences a tropical monsoon climate"),
    D("platform", "multiple", "terrestrial laser scanning for rubber, mixed forest and LeWoS, airborne laser scanning for the urban forest, synthetic point clouds for TreeNet3D (Table 1)"),
    R("sensors", "lidar", 3, "Sec. 2.1", "Terrestrial LiDAR scanning (TLS) was conducted on mild days with a Leica C10 Scanstation", note="Leica C10 TLS for the private TLS data"),
    R("annotation_type", "point_label", 4, "Sec. 2.2", "we initially applied an automated segmentation method based on machine vision", note="machine vision wood-leaf labels corrected manually where the IoU of branches or leaves fell below 80 percent"),
]
DS = [("dataset1", "own_p088_rubber_tls", "own TLS rubber trees, Danzhou (285 training and 122 test trees)", "Rubber Tree Plantations", "407", "285 training and 122 test trees", "on_request"),
      ("dataset2", "own_p088_mixed_tls", "own TLS mixed tropical forest, Danzhou (167 training and 70 test trees)", "Mixed Forest", "237", "167 training and 70 test trees", "on_request"),
      ("dataset3", "own_p088_urban_als", "own ALS urban rubber trees, Danzhou (66 training and 28 test trees)", "Urban Forest", "94", "66 training and 28 test trees", "on_request"),
      ("dataset4", "lewos", "LeWoS public TLS tropical trees (43 training and 18 test trees)", "LeWoS", "61", "43 training and 18 test trees", "yes"),
      ("dataset5", "treenet3d_flamboyant", "TreeNet3D public synthetic flamboyant trees (490 training and 210 test trees)", "TreeNet3D", "700", "490 training and 210 test trees", "yes")]
for ds, key, nm, col, ntrees, nnote, pub in DS:
    ROWS += [R("dataset_name", nm, 5, "Table 1", "Quantity (Training / Test)", item=ds), R("dataset_key", key, 5, "Table 1", "Quantity (Training / Test)", item=ds),
             R("dataset_role", "train_test", 5, "Sec. 2.2", SPLIT_EV, item=ds, note="70 percent training, 30 percent test"),
             D("n_trees", ntrees, f"sum of the printed training and test numbers ({nnote}; Table 1)", item=ds, unit="trees")]
    if pub == "yes":
        ROWS += [R("dataset_public", "yes", 5, "Sec. 2.2", "we integrated two publicly", item=ds)]
    else:
        ROWS += [R("dataset_public", "on_request", 25, "Data Availability Statement", "On reasonable request, raw point clouds of trees used in this study", item=ds)]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on 70 percent of each dataset (Sec. 2.2); training settings of the compared methods are not described")
M = [  # item, label, name, role, category, backbone, evidence, note
    ("model1", "ML", "Machine learning (machine vision method [30])", "other_baseline", "classical", "machine vision wood-leaf separation", "an automated segmentation method based on machine vision [30]", "classical, not a neural network"),
    ("model2", "PN2", "PointNet++ [21]", "other_baseline", "other", "PointNet++", "PointNet++ [21]", "MLP-based point network (protocol 4.5)"),
    ("model3", "PSeg", "PSegNet [20]", "other_baseline", "other", "double-neighbourhood feature extraction on point sets", "PSegNet [20] simplifies samples during semantic segmentation of plant structures by employing", "point network with a double-neighbourhood feature extraction block (Sec. 1)"),
    ("model4", "PT", "Point Transformer (PT) [26]", "transformer_baseline", "pure_vit", "Point Transformer", "Point Transformer network [26]", "self-attention point network, the base of the proposed method"),
    ("model5", "Rep", "RepSurf-U [44]", "other_baseline", None, "named in Table 4 only", "RepSurf-U [44]", "named in Table 4 only; the paper does not describe the architecture"),
    ("model6", "NeXt", "PointNeXt [22]", "other_baseline", "other", "PointNeXt", "PointNeXt [22]", "MLP-based point network (protocol 4.5)"),
    ("model7", "MLP", "PointMLP + TAP [45]", "other_baseline", "other", "PointMLP with TAP", "PointMLP + TAP", "PointMLP is an MLP-based point network (protocol 4.5); TAP is named in Table 4 only"),
    ("model8", "P2V", "Point2vec [46]", "other_baseline", None, "named in Table 4 only", "Point2vec [46]", "named in Table 4 only; the paper does not describe the architecture"),
    ("model9", "EPT", "Enhanced PT (PT with LSF and PosE, proposed)", "proposed", "pure_vit", "Point Transformer with local surface features (PFH) and a position encoding module (PosE)", OBJ_EV, "self-attention point network with added features and position encoding"),
]
for item, lab, name, role, cat, bb, ev, note in M:
    pg = 1 if item == "model9" else (2 if item == "model3" else (4 if item == "model1" else (3 if item == "model4" else 17)))
    ROWS += model(item, name, role, cat, bb, None, None, pg, "Sec. 1 to 4" if pg != 17 else T4, ev, cat_note=note, **NPRE)
res = Counter("res")
COLS = [("dataset1", "Rubber Tree Plantations"), ("dataset2", "Mixed Forest"), ("dataset3", "Urban Forest"), ("dataset4", "LeWoS"), ("dataset5", "TreeNet3D")]
for k, (item, lab, *_r) in enumerate(M):
    n, ev = row(PID, 17, r"mIoU \(%\)\s+(?=\d)", k)
    assert len(n) == 5, (item, n)
    for (ds, dn), v in zip(COLS, n):
        ROWS += result(res(), "miou", item, ds, v, 17, T4, ev, unit="%", setting=f"Table 4: wood-leaf separation, {dn}", note="mIoU of the wood and leaf classes (precision is also in the table)", **T, primary=(ds == "dataset1"),
                       headline=(item == "model9" and ds == "dataset5"))
ROWS += [R("limitations", "The datasets consist of individually segmented trees, so the effect of tree density was not tested, and PosE raises the time per tree to 4.62 s", 17, "Sec. 4.6", "the potential impact of tree density on segmentation accuracy was not directly considered"),
         D("code_available", "no", "No code repository in the paper; essential processing code is available on request (Data Availability Statement)"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "partial", 25, "Data Availability Statement", "On reasonable request, raw point clouds of trees used in this study", note="private point clouds on request; LeWoS and TreeNet3D are public")]
ROWS += [D("notes", "Table 4 compares nine methods (a machine vision method, PointNet++, PSegNet, PT, RepSurf-U, PointNeXt, PointMLP + TAP, Point2vec and the proposed enhanced PT) on five datasets in precision and mIoU; mIoU is recorded for all 45 cells and precision is not. Table 2 (ablation of the local surface features and PosE, with the number of PT blocks) and Table 3 (position encodings, on one dataset) are not recorded. The first listed dataset (rubber tree plantations) is primary; the abstract's maximum (96.23 precision and 91.51 mIoU) is the TreeNet3D column and carries the headline. RepSurf-U and Point2vec are named in Table 4 only and have no architecture category.",
           "Sec. 4.6 and Table 4 (p. 17)")]
