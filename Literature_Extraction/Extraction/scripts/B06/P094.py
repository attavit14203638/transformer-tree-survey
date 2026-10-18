import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P094"
# Three linked tasks on ALS data calibrated with TLS: tree point classification (seven classifiers, mIoU over tree and non-tree classes, never primary), individual-tree segmentation (instance mean IoU
# and detection rate of the Average row of ten plots, ALS methods primary) and tree DBH regression (mean RMSE and r2 of the ten plots on independent test trees, ALS regressors primary).
TC = Tags(task="component_segmentation", level="point", split="test", agg="mean")
TS = Tags(task="tree_segmentation_3d", level="instance_3d", split="not_stated", agg="mean")
TR = Tags(task="attribute_estimation", level="tree", split="test", agg="mean", attribute="dbh")
OBJ_EV = "A transformer-based neural network was adapted to classify and segment 3D individual trees from ALS data"
DS1_EV = "To create a classification reference for ALS trees, four sample point clouds were clipped"
DS2_EV = "Ten plots from 2015 to 2021 were"
SPLIT1_EV = "The training point clouds covered 80 % of the common area"
CODE_EV = "Our codes are available at https://github.com/truebelief/artemis_treescaling"

ROWS = [
    R("task", "component_segmentation;tree_segmentation_3d;attribute_estimation", 1, "Abstract", OBJ_EV, note="ALS tree point classification, individual-tree segmentation and DBH estimation calibrated with TLS trees"),
    R("output_unit", "tree_points;tree_instance;tree_attribute", 1, "Abstract", OBJ_EV),
    R("tree_setting", "natural_forest", 1, "Abstract", "primarily in montane forests"),
    R("tree_taxa", "lodgepole pine, trembling aspen, red pine and sugar maple", 3, "Sec. 2.1", "selected as joint ALS and TLS sites, including seven lodgepole pine plots"),
    R("study_area", "Alberta (Castle and Jasper sites), the Canadian Prairies and Ontario, Canada", 3, "Sec. 2.1", "LPine#3–7 in Jasper National Park belonged to the Montane Cordillera ecozone"),
    D("biome", "temperate", "montane and mixed forests of Canada; coded from the study area"),
    D("platform", "multiple", "airborne laser scanning (Teledyne Optech Titan) and terrestrial laser scanning (Teledyne Optech Ilris HD and Polaris), Sec. 2.1"),
    R("sensors", "lidar", 4, "Sec. 2.1", "a multi-channel Teledyne", note="three-channel Teledyne Optech Titan ALS (532, 1064 and 1550 nm) and Optech TLS; the classifier uses only occupancy voxels"),
    R("annotation_type", "point_label;regression_target", 1, "Abstract", "Manual delineation of TLS trees provided a precise validation reference", note="tree and non-tree labels, TLS-delineated reference trees, TLS DBH by circle fitting as the regression target"),
    R("dataset_name", "own ALS tree and non-tree reference point clouds of four sample areas, Castle site", 5, "Sec. 3.1", DS1_EV, item="dataset1"),
    R("dataset_key", "own_p094_castle_als", 5, "Sec. 3.1", DS1_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 3.1", SPLIT1_EV, item="dataset1", note="80 percent of the area of each sample for training and 20 percent for testing"),
    D("dataset_public", "no", "No data availability statement in the paper (the codes are on GitHub)", item="dataset1"),
    R("dataset_name", "ten joint ALS and TLS plots (LPine#1 to 7, TAspen, RPine, SMaple)", 3, "Sec. 2.1", DS2_EV, item="dataset2"),
    R("dataset_key", "own_p094_joint_als", 3, "Sec. 2.1", DS2_EV, item="dataset2"),
    R("dataset_role", "train_test", 7, "Sec. 3.3", "544 and 223 trees were selected for training and testing", item="dataset2", note="DBH regression: 544 training and 223 test trees"),
    D("dataset_public", "no", "No data availability statement in the paper", item="dataset2"),
    D("n_trees", "767", "sum of the printed 544 training and 223 test trees of the DBH regression (Sec. 3.3)", item="dataset2", unit="trees"),
    R("dataset_name", "TLS scans of the ten joint plots with manually isolated trees", 3, "Sec. 2.1", DS2_EV, item="dataset3"),
    R("dataset_key", "own_p094_joint_tls", 3, "Sec. 2.1", DS2_EV, item="dataset3"),
    R("dataset_role", "train_test", 3, "Sec. 2.1", DS2_EV, item="dataset3", note="TLS reference trees and the TLS-based comparison rows"),
    D("dataset_public", "no", "No data availability statement in the paper", item="dataset3"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the training areas or trees of each task (Sec. 3)")
NA = dict(pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
M = [  # item, name, role, category, backbone, page, loc, evidence, note, kwargs
    ("model1", "SegFormer (3D voxel version) tree point classifier (proposed)", "proposed", "hierarchical_vit", "3D SegFormer on 96 x 96 x 192 occupancy voxels of 1 m", 6, "Sec. 3.1", "replaced convolution layers with a vision Transformer (ViT) and", "SegFormer replaces convolution layers with a vision transformer (Sec. 3.1)", NPRE),
    ("model2", "ResNet (3D FCN) tree point classifier", "cnn_baseline", "cnn", "3D ResNet as a fully convolutional network", 6, "Sec. 3.1", "The ResNet served as a baseline model for comparison", "convolutional baseline (Sec. 3.1)", NPRE),
    ("model3", "Random forest tree point classifier", "other_baseline", "classical", "200 bagged decision trees", 14, "Table B1", "200 bagged decision trees", "machine learning without a neural network", NA),
    ("model4", "LAStools tree point classification", "other_baseline", "classical", "lasground, lasheight and lasclassify with default settings", 14, "Table B1", "lasground, lasheight, lasclassify: settings all default", "rule-based filtering", NA),
    ("model5", "Terrascan tree point classification", "other_baseline", "classical", "classification of ground, building, surface and low-height points", 14, "Table B1", "Classify ground, building, surface, and low-height points", "rule-based filtering", NA),
    ("model6", "Intensity-threshold filter", "other_baseline", "classical", "optimal intensity threshold", 14, "Table B1", "Optimal threshold between the minimum and maximum range", "threshold filter", NA),
    ("model7", "Number-of-returns threshold filter", "other_baseline", "classical", "optimal threshold of the number of returns", 14, "Table B1", "Optimal threshold between the minimum and maximum range", "threshold filter", NA),
    ("model8", "Segma (ALS tree segmentation)", "other_baseline", "classical", "Gaussian kernel and height slices", 14, "Table B1", "Gaussian kernel sigma", "classical segmentation", NA),
    ("model9", "lidR-li2012 (ALS tree segmentation)", "other_baseline", "classical", "point-based region growing of Li et al. (2012)", 14, "Table B1", "Threshold 1: 1.5; Threshold 2: 2", "classical segmentation", NA),
    ("model10", "lidR-itcSegment (ALS tree segmentation)", "other_baseline", "classical", "canopy height model local maxima and region growing", 14, "Table B1", "Raster resolution: 0.5 m", "classical segmentation", NA),
    ("model11", "Mask R-CNN on the canopy height model (ALS tree segmentation)", "cnn_baseline", "cnn", "Mask R-CNN with ResNet50-FPN on 0.5 m CHM patches", 14, "Table B1", "Backbone: ResNet50FPN", "convolutional instance segmentation", NPRE),
    ("model12", "Fuzzy center segmentation (FCS: SegFormer near-center classification and cut-pursuit clustering, proposed)", "proposed", "hierarchical_vit", "3D SegFormer classifying near-center points on 128 voxel blocks, then graph clustering", 7, "Sec. 3.2", "workflow is referred to as the fuzzy center segmentation model (FCS)", "SegFormer (transformer) classifies the crown centres, a graph-based cut-pursuit step assigns the remaining points", NPRE),
    ("model13", "treeiso on TLS point clouds (TLS tree segmentation)", "other_baseline", "classical", "unsupervised treeiso (CloudCompare plugin) on TLS scans", 5, "Sec. 2.2", "Individual tree segmentation from TLS point clouds was performed using a CloudCompare plugin treeiso", "unsupervised segmentation of TLS data", NA),
    ("model14", "Random forest regression (RFR) on height profile metrics", "other_baseline", "classical", "random forest regressor on height-based metrics", 8, "Sec. 3.4", "Gaussian process regression (GPR), support vector regression (SVR), and random", "machine learning regression", NA),
    ("model15", "Support vector regression (SVR) on height profile metrics", "other_baseline", "classical", "support vector regressor on height-based metrics", 8, "Sec. 3.4", "Gaussian process regression (GPR), support vector regression (SVR), and random", "machine learning regression", NA),
    ("model16", "Gaussian process regression (GPR) on height profile metrics", "other_baseline", "classical", "Gaussian process regressor on height-based metrics", 8, "Sec. 3.4", "Gaussian process regression (GPR), support vector regression (SVR), and random", "machine learning regression", NA),
    ("model17", "CNN regression network (CNNR, VGG-like) on 3D voxels", "cnn_baseline", "cnn", "3D CNN-based regression network adapted from VGG on 0.2 m voxels", 7, "Sec. 3.3", "a conventional 3D CNN", "convolutional regression network", NPRE),
    ("model18", "CNN-GPR (3D voxel CNN with a Gaussian process layer)", "cnn_baseline", "cnn", "CNN layers on 5 x 5 x 20 voxels with a Gaussian process regression layer", 7, "Sec. 3.3", "CNN-GPR simply altered the input to be a block of 5", "convolutional layers with a Gaussian process output layer", NPRE),
    ("model19", "NN-GPR (attribute neural network with a Gaussian process layer, proposed)", "proposed", "other", "six MLP layers on a height-based profile with a Gaussian process regression layer", 7, "Sec. 3.3", "Gaussian Process Regression (NN-GPR)", "multilayer perceptron with a Gaussian process layer (not a transformer)", NPRE),
    ("model20", "NN-GPR on TLS-derived inputs", "proposed", "other", "the NN-GPR network trained on TLS trees", 11, "Table 5", "TLS(NN-GPR)", "the same network on TLS inputs (a comparison row)", NPRE),
]
for item, name, role, cat, bb, pg, loc, ev, note, kw in M:
    ROWS += model(item, name, role, cat, bb, None, None, pg, loc, ev, cat_note=note, **kw)
res = Counter("res")
RID = {}
# Table 3: ALS tree point classification, Average row of four sample areas (columns: slope, tree percent, TRI, then seven mIoU)
n3, e3 = row(PID, 8, r"^\s*Average\s+(?=\d)", 0)
assert len(n3) == 10
for item, v in zip(("model6", "model7", "model4", "model5", "model3", "model2", "model1"), n3[3:]):
    RID[item] = res()
    ROWS += result(RID[item], "miou", item, "dataset1", v, 8, "Table 3", e3, unit="fraction", setting="Table 3: ALS tree point classification, average of the four test sample areas", note="mIoU of the tree and non-tree classes (the tree class alone is not given)", **TC)
# Table 4: Average row of ten plots (point densities, then rate and mIoU of Segma, lidR-li2012, lidR-itcSegment, Mask R-CNN, FCS and TLS treeiso)
n4, e4 = row(PID, 9, r"^\s*Average\s+(?=\d)", 0)
assert len(n4) == 16
for k, (item, ds, prim) in enumerate((("model8", "dataset2", True), ("model9", "dataset2", True), ("model10", "dataset2", True), ("model11", "dataset2", True), ("model12", "dataset2", True), ("model13", "dataset3", False))):
    rate, miou = n4[4 + 2 * k], n4[5 + 2 * k]
    st = "Table 4: tree segmentation, average of the ten joint plots, " + ("TLS" if item == "model13" else "ALS")
    ROWS += result(res(), "miou", item, ds, miou, 9, "Table 4", e4, unit="fraction", setting=st, note="mean IoU of the matched trees (mIoU of tree segmentation)", **TS, primary=prim)
    ROWS += result(res(), "recall", item, ds, rate, 9, "Table 4", e4, unit="%", setting=st, note="detection rate: matched reference trees (IoU above 0.5) over all reference trees", **TS, primary=prim)
# Table 5: DBH regression, mean over the ten plots (RFR, SVR, GPR, CNNR, CNN-GPR, NN-GPR on ALS, NN-GPR on TLS)
rm, erm = row(PID, 11, r"^\s*Mean RMSE \(cm\)\s+(?=\d)", 0)
r2, er2 = row(PID, 11, r"^\s*Mean r2\s+(?=\d)", 0)
assert len(rm) == 7 and len(r2) == 7
for item, ds, a, b in (("model14", "dataset2", rm[0], r2[0]), ("model15", "dataset2", rm[1], r2[1]), ("model16", "dataset2", rm[2], r2[2]), ("model17", "dataset2", rm[3], r2[3]), ("model18", "dataset2", rm[4], r2[4]),
                       ("model19", "dataset2", rm[5], r2[5]), ("model20", "dataset3", rm[6], r2[6])):
    prim = item != "model20"
    st = "Table 5: DBH estimation of the independent test trees, mean over the ten plots, " + ("TLS input" if item == "model20" else "ALS input")
    ROWS += result(res(), "rmse", item, ds, a, 11, "Table 5", erm, unit="cm", setting=st, note="mean RMSE of the DBH over the ten plots", **TR, primary=prim)
    ROWS += result(res(), "r2", item, ds, b, 11, "Table 5", er2, unit="fraction", setting=st, note="mean r2 of the DBH over the ten plots", **TR, primary=prim)
ROWS += conflict("cf1", f"{RID['model1']}.value", "0.92", 1, "Abstract", "mean intersectionover-union (mIoU) of 0.92 for ALS tree classification", "V1a",
                 "Table 3 and Sec. 4.1 give a mean mIoU of 0.89 for the SegFormer ALS tree classification; the abstract gives 0.92, which is the SegFormer mIoU for TLS tree points (Sec. 5.1)", unit="fraction")
ROWS += [R("limitations", "Understory and occluded trees are hard to detect from ALS (79 percent detection by FCS), and the supervised models need TLS references of the same sites", 14, "Sec. 5.2", "our average ALS detection rate of 79 % using FCS", note="the paper reports 79 percent ALS detection against 87 percent for TLS in a comparable study"),
         R("code_available", "yes", 1, "Abstract", CODE_EV), R("code_url", "https://github.com/truebelief/artemis_treescaling", 1, "Abstract", CODE_EV),
         D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data availability statement in the paper")]
ROWS += [D("notes", "Table 3 compares seven ALS tree point classifiers (mIoU of tree and non-tree classes; the Average row of four sample areas is recorded and no value is primary because the mIoU includes the non-tree class). Table 4 compares four ALS segmentation methods, the proposed FCS and TLS treeiso (detection rate and mean IoU per plot; the Average row of ten plots is recorded, ALS rows primary, the TLS row is a reference). Table 5 gives mean RMSE, RMSE% and r2 of six ALS DBH regressors and the proposed network on TLS inputs (RMSE and r2 recorded; RMSE% is not). Per-plot and per-area values, the other classifiers' parameters (Table B1) and the biomass maps are not recorded. The abstract's mIoU of 0.92 for ALS classification differs from Table 3 (cf item).",
           "Sec. 4 and Tables 3 to 5 (pp. 8 to 11)")]
