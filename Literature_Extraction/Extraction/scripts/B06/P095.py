import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P095", "Table 3"
# Point-level segmentation of single rubber trees (tree against other points) and of single-tree datasets; the mIoU of the RubberTree and ForestSemantic tasks averages the tree and the other class (never primary) and
# the paper's first listed dataset (RubberTree) is of that kind, so no mIoU is primary. The R2 of tree height, crown diameter and DBH of the field test (abstract) is the primary result. ShapeNetPart (objects other
# than trees) is not recorded.
TM = Tags(task="component_segmentation", level="point", split="test", agg="single")
def tag(attr):
    return Tags(task="attribute_estimation", level="tree", split="not_stated", agg="single", attribute=attr)
OBJ_EV = "we propose a high-precision segmentation network,TM-WSNet"
ARCH_EV = "SGTramba, a hybrid feature extraction module combining Grouped Transformer and Mamba architectures"
SPLIT_EV = "the data set was randomly divided into a training set and a test set at a ratio of 4:1"
RT_EV = "finally selected 600 rubber trees for experiments"
FT_EV = "The data were obtained by scanning the rubber plantation in Danzhou City, Hainan Province, using the LiDAR system of DJIM300RTK UAV"

ROWS = [
    R("task", "component_segmentation;attribute_estimation", 1, "Abstract", OBJ_EV, note="segmentation of single rubber trees (tree points against other points) and their height, crown width and DBH"),
    R("output_unit", "tree_points;tree_attribute", 1, "Abstract", OBJ_EV),
    R("tree_setting", "plantation", 1, "Abstract", "individual rubber tree segmentation in plantation environments remains challenging"),
    R("tree_taxa", "rubber tree (Hevea brasiliensis)", 1, "Title", "individual rubber trees"),
    R("study_area", "Danzhou, Hainan Province, China", 16, "Sec. 3.4.7", FT_EV),
    D("biome", "tropical", "Hainan, China; coded from the study area"),
    R("platform", "uav", 3, "Sec. 2.2", "DJI M300 RTK drone", note="DJI M300 RTK with a CBI Lite laser radar system"),
    R("sensors", "lidar", 3, "Sec. 2.2", "wavelength of 905 nm"),
    R("annotation_type", "point_label", 3, "Sec. 2.3", "we labeled the main points of the target rubber tree as", note="tree labelled 1 and other points 0 after a treeiso pre-segmentation"),
    R("dataset_name", "own UAV LiDAR RubberTree dataset (600 single rubber trees, tree and other labels)", 3, "Sec. 2.3", RT_EV, item="dataset1"),
    R("dataset_key", "own_p095_rubbertree", 3, "Sec. 2.3", RT_EV, item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.3", SPLIT_EV, item="dataset1", note="1334 trees were examined and 600 used; 4:1 split"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_trees", "600", 3, "Sec. 2.3", RT_EV, item="dataset1", unit="trees"),
    R("dataset_name", "ForestSemantic (Liang et al.), 84 single-tree samples augmented to 252", 9, "Sec. 3.2.2", "we selected 84 initial data samples", item="dataset2"),
    R("dataset_key", "forestsemantic_evo", 9, "Sec. 3.2.2", "we selected 84 initial data samples", item="dataset2"),
    R("dataset_role", "train_test", 9, "Sec. 3.2.2", "a dataset containing 252 independent samples", item="dataset2"),
    D("dataset_public", "yes", "ForestSemantic is a public dataset cited as [40]", item="dataset2"),
    R("dataset_name", "GBS single-tree dataset (56 trees, trunk and crown)", 9, "Sec. 3.2.2", "We selected 56 single trees of different types to segment between trunks and crowns", item="dataset3"),
    R("dataset_key", "gbs_leaf_wood", 9, "Sec. 3.2.2", "We selected 56 single trees of different types to segment between trunks and crowns", item="dataset3"),
    R("dataset_role", "train_test", 9, "Sec. 3.2.2", "We selected 56 single trees of different types to segment between trunks and crowns", item="dataset3"),
    D("dataset_public", "yes", "the GBS dataset is a public dataset cited as [39]", item="dataset3"),
    R("n_trees", "56", 9, "Sec. 3.2.2", "We selected 56 single trees of different types to segment between trunks and crowns", item="dataset3", unit="trees"),
    R("dataset_name", "own new UAV LiDAR rubber tree point clouds for the field test, Danzhou", 16, "Sec. 3.4.7", FT_EV, item="dataset4"),
    R("dataset_key", "own_p095_field_test", 16, "Sec. 3.4.7", FT_EV, item="dataset4"),
    R("dataset_role", "transfer_target", 16, "Sec. 3.4.7", FT_EV, item="dataset4", note="brand new point clouds used to test the trained model in practice"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset4"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained for 200 epochs with the scale optimisation algorithm SCPO on the training split (Sec. 3.2.2)")
M = [
    ("model1", "GDANet (2020)", "cnn_baseline", "cnn", "graph neural network feature extraction", 16, "Sec. 3.4.5", "GDANET performs feature extraction by employing graph neural network (GNN)", "graph neural network (graph convolution)"),
    ("model2", "SPoTr (2023)", "other_baseline", None, "named in Table 3 only", 14, "Sec. 3.4.1", "networks, including GDANET [48], SPOTR [49], PointNext [50],", "named only; the paper does not describe the architecture"),
    ("model3", "PointNeXt (2022)", "other_baseline", "other", "PointNeXt", 16, "Sec. 3.4.5", "PointNext implements an improvement of PointNet++", "MLP-based point network (protocol 4.5)"),
    ("model4", "Point Transformer V2 (PTv2, 2022)", "transformer_baseline", "pure_vit", "Point Transformer V2", 16, "Sec. 3.4.5", "PTv2 is able to effectively handle global contextual features", "self-attention point network"),
    ("model5", "PointMamba (2024)", "other_baseline", "other", "PointMamba state-space model", 16, "Sec. 3.4.5", "Mamba successfully captures the global dependencies of the point cloud through the powerful state-space model", "state-space (Mamba) model (protocol 4.5)"),
    ("model6", "PC-Mamba (2024)", "other_baseline", "other", "PC-Mamba state-space model", 16, "Sec. 3.4.5", "Mamba successfully captures the global dependencies of the point cloud through the powerful state-space model", "state-space (Mamba) model (protocol 4.5)"),
    ("model7", "TM-WSNet (proposed)", "proposed", "cnn_transformer_hybrid", "SGTramba (grouped transformer and Mamba), WGMS wavelet grid encoder and SCPO learning rate search", 1, "Abstract", ARCH_EV,
     "grouped transformers for local features and Mamba for global dependencies, no convolution: recorded as a transformer hybrid"),
]
for item, name, role, cat, bb, pg, loc, ev, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, pg, loc, ev, cat_note=note, **NPRE)
res = Counter("res")
RID = {}
LAB = (("model1", r"GDANET\(2020\)"), ("model2", r"SPOTR\(2023\)"), ("model3", r"PointNext\(2022\)"), ("model4", r"PTv2\(2022\)"), ("model5", r"PointMamba\(2024\)"), ("model6", r"PC-Mamba\(2024\)"), ("model7", r"TM-WSNet"))
for item, rx in LAB:
    for k, ds, dn, note in ((0, "dataset1", "RubberTree", "mIoU of the tree and other classes"), (1, "dataset2", "ForestSemantic", "mIoU of the tree and other classes after the treeiso pre-segmentation"), (3, "dataset3", "GBS", "mIoU of the trunk and crown classes")):
        n, ev = row(PID, 12, rf"{rx}\s+(?=\d)", k)
        RID[(item, ds)] = res()
        ROWS += result(RID[(item, ds)], "miou", item, ds, n[0], 12, T3, ev, unit="%", setting=f"Table 3: comparison with other networks, {dn}", note=note, **TM)
# Field test of Sec. 3.4.7: mIoU of TM-WSNet on the new data (text) and R2 of the structure parameters (abstract)
RID["field"] = res()
ROWS += result(RID["field"], "miou", "model7", "dataset4", "86.15", 16, "Sec. 3.4.7", "TM-WSNet achieves an mIoU of 86.15 %, an mF1 score of 92.55 %, and an mAcc of 92.56 %", unit="%", setting="Sec. 3.4.7: field test on new point clouds",
               note="mIoU of the tree and other classes on the new data", **TM)
for attr, v, lab in (("height", "1.00", "tree height"), ("crown_diameter", "0.99", "crown width"), ("dbh", "0.89", "diameter at breast height")):
    ROWS += result(res(), "r2", "model7", "dataset4", v, 1, "Abstract", "height, crown width, and diameter at breast height with coefficients of determination (R2) of 1.00, 0.99, and 0.89, respectively", unit="fraction",
                   setting="Sec. 3.4.7: structure parameters of the field test trees", note=f"R2 of the {lab} estimated from the segmented trees (practical field test)", **tag(attr), primary=True)
# Transfer pair: the RubberTree test set against the new field-test point clouds (mIoU of TM-WSNet)
tr = "tr1"
ROWS += [R("transfer_source", "RubberTree training data of Danzhou (UAV LiDAR)", 3, "Sec. 2.3", RT_EV, item=tr),
         R("transfer_target", "new rubber tree point clouds of the Danzhou plantation", 16, "Sec. 3.4.7", FT_EV, item=tr),
         R("shift_type", "site", 16, "Sec. 3.4.7", FT_EV, item=tr, note="brand new data from the same plantation city; the paper does not say what differs"),
         R("transfer_model", "model7", 16, "Sec. 3.4.7", "TM-WSNet achieves an mIoU of 86.15 %, an mF1 score of 92.55 %, and an mAcc of 92.56 %", item=tr),
         R("transfer_metric", "miou", 16, "Sec. 3.4.7", "TM-WSNet achieves an mIoU of 86.15 %, an mF1 score of 92.55 %, and an mAcc of 92.56 %", item=tr),
         R("value_in_domain", "88.78", 12, T3, row(PID, 12, r"TM-WSNet\s+(?=\d)", 0)[1], item=tr, unit="%", note="RubberTree test set, Table 3"),
         R("value_out_of_domain", "86.15", 16, "Sec. 3.4.7", "mIoU of 86.15 %", item=tr, unit="%"),
         R("transfer_level", "point", 16, "Sec. 3.4.7", "mIoU of 86.15 %", item=tr)]
ROWS += [R("limitations", "The model mistakes regular background noise for trunk points and fails where low rubber trees overlap with surrounding vegetation or the canopy boundary is blurred", 16, "Sec. 3.4.7", "the model misidentifies background noise with regular shapes as part of the rubber tree trunk"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data availability statement in the paper")]
ROWS += [D("notes", "Table 3 compares six networks with TM-WSNet on RubberTree, ForestSemantic, ShapeNetPart and GBS (mIoU, mAcc, mPrec, mF1, mRecall and an R2 of the structure parameters); the mIoU is recorded for RubberTree, ForestSemantic and GBS (ShapeNetPart is a benchmark of objects other than trees and is not recorded; the R2 column is not defined clearly and is not recorded). The mIoU of RubberTree and ForestSemantic averages the tree and the other class, so no mIoU is primary. Tables 1 and 2 (module and ablation results) are not recorded. The field test (Sec. 3.4.7) gives the mIoU of 86.15 percent and the R2 of height, crown width and DBH (abstract: 1.00, 0.99, 0.89); one transfer pair compares it with the RubberTree test mIoU of 88.78.",
           "Sec. 3.4 and Table 3 (pp. 12 to 16)")]
