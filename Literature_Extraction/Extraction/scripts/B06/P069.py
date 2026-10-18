import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T3 = "P069", "Table 2", "Table 3"
# Apple tree point clouds (46 trees, Kinect V2) segmented into trunk and branches, branches into instances, and measured for architectural traits. Semantic segmentation: point-level mIoU of the
# trunk and branch classes (component segmentation). Table 3 (post-processed, test data of fold 1: the increases of Table 3 fit fold 1 of Table 2) is primary; Table 2 is the 4-fold mean before post-processing.
TS3 = Tags(task="component_segmentation", level="point", split="test", agg="single")
TS2 = Tags(task="component_segmentation", level="point", split="cross_validation", agg="mean")
TI = Tags(task="component_segmentation", level="instance_3d", split="not_stated", agg="single")
TH = Tags(task="attribute_estimation", level="tree", split="not_stated", agg="single", attribute="height")
TO = Tags(task="attribute_estimation", level="tree", split="not_stated", agg="single", attribute="other")
TB = Tags(task="attribute_estimation", level="not_stated", split="not_stated", agg="single", attribute="other")
OBJ_EV = "The overall goal of this study was to develop a deep learning-based organ-level instance segmentation method to quantify the 3D architectural traits of apple trees"
DS_EV = "a total of 46 apple tree point clouds were collected, with 34 used for training and 12 for testing"
AREA_EV = "collected from Qian County (latitude: 34.33 N, longitude: 108.71 W) in Shaanxi Province, China"
INST_EV = "instance segmentation results based on SoftGroup++ and PointNeXt achieved mAP_50 of 0.815 and 0.842"

ROWS = [
    R("task", "component_segmentation;attribute_estimation", 1, "Abstract", OBJ_EV, note="trunk and branch segmentation, branch instances, and seven architectural traits"),
    R("output_unit", "tree_component;tree_attribute", 1, "Abstract", OBJ_EV),
    R("tree_setting", "orchard", 3, "Sec. 2.1", "mounted on a tripod was used to scan apple tree samples randomly selected in each row", note="apple trees in rows"),
    R("tree_taxa", "apple (Malus domestica), two-year-old cross-bred trees", 3, "Sec. 2.1", "the age of the trees was 2 years old"),
    R("study_area", "Qian County, Shaanxi Province, China", 3, "Sec. 2.1", AREA_EV),
    D("biome", "temperate", "Shaanxi, China; coded from the study area"),
    R("platform", "ground_camera", 3, "Sec. 2.1", "A Kinect V2 sensor (Microsoft Corp., Redmond, WA, USA) mounted on a tripod was used to scan apple tree samples randomly selected in each row", note="Kinect V2 time-of-flight depth camera on a tripod, about 1.5 m from the trunk"),
    R("sensors", "rgb;lidar", 3, "Sec. 2.1", "The RGB images have a resolution of 1920", note="the Kinect V2 gives colour images and a depth point cloud (x, y, z and RGB); it is a depth camera, not a laser scanner"),
    R("annotation_type", "point_label", 3, "Sec. 2.2", "The point cloud of the apple tree has been meticulously categorized into two distinct classes: branches and trunks"),
    R("dataset_name", "own Kinect V2 point clouds of 46 two-year-old apple trees, Qian County", 3, "Sec. 2.1", DS_EV, item="dataset1"),
    R("dataset_key", "own_p069_apple_kinect", 3, "Sec. 2.1", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.1", DS_EV, item="dataset1", note="34 training and 12 test trees; Table 2 uses four folds"),
    R("dataset_public", "on_request", 15, "Data availability", "Data will be made available on request", item="dataset1"),
    R("n_trees", "46", 3, "Sec. 2.1", DS_EV, item="dataset1", unit="trees"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the apple tree dataset (500 epochs for PointNeXt, Sec. 2.3.1); the Point Transformer V2 uses AdamW with a learning rate of 0.006")
ROWS += model("model1", "PointNeXt with cylinder post-processing (proposed pipeline)", "proposed", "other", "PointNeXt (InvResMLP blocks on PointNet++ set abstraction), cylinder constraint, DBSCAN and skeleton-graph branch separation", None, None, 3, "Sec. 2.3.1",
              "PointNeXt is an expansion of PointNet++, enhanced by innovative", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model2", "PointNet", "other_baseline", "other", "PointNet", None, None, 5, "Sec. 2.3.3", "PointNet (Qi et al., 2017a) is a deep learning model designed for", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model3", "PointNet++", "other_baseline", "other", "PointNet++", None, None, 5, "Sec. 2.3.3", "PointNet++ (Qi et al., 2017b) is an improved version of PointNet", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model4", "Point Transformer V2 (PTv2)", "transformer_baseline", "pure_vit", "grouped vector attention, position encoding multiplier and partition-based pooling", None, None, 5, "Sec. 2.3.2",
              "Point Transformer V2 (PTv2) (Wu et al., 2022) is a recently developed effective and efficient point cloud semantic segmentation model", cat_note="self-attention point network (Sec. 2.3.2)", **NPRE)
ROWS += model("model5", "SoftGroup++", "cnn_baseline", "cnn", "U-Net backbone with soft grouping and a refinement U-Net", None, None, 7, "Sec. 2.4.2", "SoftGroup++ (Vu et al., 2024) is an advanced 3D instance segmentation network",
              cat_note="U-Net backbone with bottom-up grouping and top-down refinement (Sec. 2.4.2)", **NPRE)
res = Counter("res")
M3 = [("model2", r"PointNet", "PointNet"), ("model3", r"PointNet\+\+", "PointNet++"), ("model4", r"PTv2", "PTv2"), ("model1", r"PointNeXt", "PointNeXt")]
for item, rx, lab in M3:
    n, ev = row(PID, 10, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(n) == 2, n  # mIoU, increase over the model before post-processing
    ROWS += result(res(), "miou", item, "dataset1", n[0], 10, T3, ev, unit="fraction", setting="Table 3: after post-processing, test data of fold 1", note="mIoU of the trunk and branch classes after the cylinder constraint", **TS3, primary=True, headline=(item == "model1"))
na, eva = row(PID, 9, r"^\s*Average\s+(?=\d)", 0)
assert len(na) == 4, na
for (item, rx, lab), v in zip(((("model2", "", "PointNet"), ("model3", "", "PointNet++"), ("model4", "", "PTv2"), ("model1", "", "PointNeXt"))), na):
    ROWS += result(res(), "miou", item, "dataset1", v, 9, T2, "Average " + " ".join(na), unit="fraction", setting="Table 2: before post-processing, average of four folds", note="mIoU of the trunk and branch classes, average of the four folds (Average row)", **TS2)
# Branch instance segmentation (mAP at IoU 0.5), from the abstract and Sec. 3.2
ROWS += result(res(), "ap50", "model5", "dataset1", "0.815", 1, "Abstract", INST_EV, unit="fraction", setting="branch and trunk instance segmentation, SoftGroup++", note="mAP_50 of the organ instances (trunk and branches); JSIS3D is reported only as 37 percent lower than PointNeXt with post-processing", **TI, primary=True)
ROWS += result(res(), "ap50", "model1", "dataset1", "0.842", 1, "Abstract", INST_EV, unit="fraction", setting="branch and trunk instance segmentation, PointNeXt with post-processing and branch separation", note="mAP_50 of the organ instances (trunk and branches)", **TI, primary=True)
# Architectural traits (R2 and RMSE against manual measurements). PointNeXt: R2 from the abstract (three decimals), RMSE from Figs. 14 and 15; PTv2 and SoftGroup++: Fig. 14 (image)
TR = [  # item, trait, R2, RMSE, unit, tags, page, loc, evidence
    ("model1", "tree height", "0.987", "0.02", "m", TH, 13, "Fig. 15 (image)", "tree height R2 = 0.99 RMSE = 0.02 MAE = 0.01 MAPE = 0.86%"),
    ("model1", "trunk length", "0.788", "2.58", "cm", TO, 13, "Fig. 14 (image)", "PointNeXt trunk length R2 = 0.79 RMSE = 2.58 MAE = 2.22 MAPE = 2.17%"),
    ("model1", "trunk diameter", "0.877", "0.29", "cm", TO, 13, "Fig. 15 (image)", "trunk diameter R2 = 0.88 RMSE = 0.29 MAE = 0.15 MAPE = 5.93%"),
    ("model1", "branch length", "0.796", "11.25", "cm", TB, 13, "Fig. 14 (image)", "PointNeXt branch length R2 = 0.80 RMSE = 11.25 MAE = 6.88 MAPE = 10.24%"),
    ("model1", "branch number", "0.934", "1.00", "count", TO, 13, "Fig. 14 (image)", "PointNeXt branch number R2 = 0.93 RMSE = 1.00 MAE = 0.67 MAPE = 13.55%"),
    ("model4", "branch length", "0.84", "8.06", "cm", TB, 13, "Fig. 14 (image)", "PTv2 branch length R2 = 0.84 RMSE = 8.06 MAE = 5.27 MAPE = 7.31%"),
    ("model4", "branch number", "0.87", "1.26", "count", TO, 13, "Fig. 14 (image)", "PTv2 branch number R2 = 0.87 RMSE = 1.26 MAE = 0.75 MAPE = 14.25%"),
    ("model4", "trunk length", "0.84", "1.88", "cm", TO, 13, "Fig. 14 (image)", "PTv2 trunk length R2 = 0.84 RMSE = 1.88 MAE = 1.47 MAPE = 1.44%"),
    ("model5", "branch length", "0.85", "8.81", "cm", TB, 13, "Fig. 14 (image)", "SoftGroup++ branch length R2 = 0.85 RMSE = 8.81 MAE = 5.53 MAPE = 7.66%"),
    ("model5", "branch number", "0.84", "1.35", "count", TO, 13, "Fig. 14 (image)", "SoftGroup++ branch number R2 = 0.84 RMSE = 1.35 MAE = 0.83 MAPE = 13.45%"),
    ("model5", "trunk length", "0.84", "3.37", "cm", TO, 13, "Fig. 14 (image)", "SoftGroup++ trunk length R2 = 0.84 RMSE = 3.37 MAE = 3.07 MAPE = 3.00%"),
]
for item, trait, r2, rmse, unit, tags, pg, loc, ev in TR:
    prim = trait == "tree height"
    st = f"{trait}: predicted against manual measurement" + (" (R2 of PointNeXt from the abstract)" if item == "model1" else "")
    ROWS += result(res(), "r2", item, "dataset1", r2, pg if item != "model1" else 1, loc if item != "model1" else "Abstract", ev if item != "model1" else "achieved R2 values of 0.987, 0.788, 0.877, 0.796, and 0.934", unit="fraction", setting=st, note=f"R2 of {trait}", **tags, primary=prim)
    ROWS += result(res(), "rmse", item, "dataset1", rmse, pg, loc, ev, unit=unit, setting=f"{trait}: predicted against manual measurement", note=f"RMSE of {trait}", **tags, primary=prim)
ROWS += [R("limitations", "Only 46 two-year-old trees of one site and sensor; trunk diameter is sensitive to depth sensor noise outdoors, and the end-to-end instance models still err where branches touch", 11, "Sec. 3.3", "When utilizing Kinect V2 outdoors, susceptibility to lighting variations may lead to point cloud data loss"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "on_request", 15, "Data availability", "Data will be made available on request")]
ROWS += [D("notes", "Tables 2 and 3 give the mIoU of PointNet, PointNet++, PTv2 and PointNeXt for trunk and branch segmentation (Table 2: four folds before post-processing; Table 3: after the cylinder post-processing on the test data of fold 1); the Average row of Table 2 and the four Table 3 values are recorded. mAP_50 of the branch instances is only printed in the abstract and Sec. 3.2 (SoftGroup++ 0.815, PointNeXt 0.842); JSIS3D is described as 37 percent lower. The traits of Figs. 14 and 15 (R2, RMSE) are recorded for PointNeXt (all five traits), PTv2 and SoftGroup++ (branch length, branch number, trunk length); only tree height is a standard attribute and primary. MAE and MAPE of the figures are not recorded.",
           "Sec. 3 and Tables 2 and 3, Figs. 14 and 15 (pp. 9 to 13)")]
