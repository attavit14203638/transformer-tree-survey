import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P099", "Table 3"
# TreeAIBox workflow on UAV LiDAR (and multispectral) data of five reclaimed wellsites: TreeLoc (3D-SegFormer, treetop detection), TreeOff (3D-SegFormer, tree instances) and SpeciesCls (3D-ResNet, coniferous or
# deciduous). The paper uses LiDAR-only models for delineation and LiDAR + RGB models for classification (Sec. 3.3), so those columns of Table 3 are primary and the other inputs are not. The test data
# of TreeLoc are the spring 2024 scans, whereas its training data are the summer and autumn 2023 scans (Table 2). The vegetation filter (TreeFilter) is not a tree task and is not recorded.
TD = Tags(task="detection", level="object", split="test", agg="mean")
TS = Tags(task="tree_segmentation_3d", level="instance_3d", split="test", agg="mean")
TC = Tags(task="species_classification", level="tree", split="test", agg="mean")
TH = Tags(task="attribute_estimation", level="tree", split="validation", agg="single", attribute="height")  # the paper calls the field reference trees a validation dataset (p.6)
OBJ_EV = "for individual tree detection, crown delineation, and classification across five reclaimed wellsites in Alberta, Canada"
DATA_EV = "These annotations were split into training and testing sets (Table 2)"
CODE_EV = "through a GitHub repository: https://github.com/NRCan/TreeAIBox"
PAGE_EV = "we used LiDAR-only models for individual-tree delineation and LiDAR + RGB models for classification"

ROWS = [
    R("task", "detection;tree_segmentation_3d;species_classification;attribute_estimation", 1, "Abstract", OBJ_EV, note="treetop detection, individual-tree segmentation, coniferous or deciduous classification and tree height"),
    R("output_unit", "tree_instance;tree_label;tree_attribute", 7, "Sec. 3.1", "TreeOff then extracts individual tree characteristics (ITC): location, height"),
    R("tree_setting", "mixed", 4, "Sec. 2.1", "sites were planted with either Picea glauca (white spruce) or Pinus contorta (lodgepole pine)", note="reclaimed wellsites: planted conifers and naturally regenerating balsam poplar and willow"),
    R("tree_taxa", "white spruce (Picea glauca), lodgepole pine (Pinus contorta), balsam poplar (Populus balsamifera), willow (Salix sp.)", 4, "Sec. 2.1", "sites were planted with either Picea glauca (white spruce) or Pinus contorta (lodgepole pine)"),
    R("study_area", "near Grande Prairie, northwestern Alberta, Canada (five reclaimed wellsites)", 4, "Sec. 2.1", "The study area was located near Grande Prairie in northwestern Alberta, Canada"),
    R("biome", "boreal", 2, "Sec. 1", "Many of these wellsites are in the Canadian boreal forest region"),
    R("platform", "uav", 5, "Sec. 2.2.1", "mounted to a remote-controlled quadcopter DJI Matrice 300 RTK"),
    R("sensors", "lidar;multispectral;rgb", 5, "Sec. 2.2.1", "LiDAR data were collected using a DJI Zenmuse L1 sensor", note="L1 LiDAR with its RGB camera, and a MicaSense RedEdge-P five-band multispectral camera on the same UAV"),
    R("gsd", "1.36", 5, "Sec. 2.2.1", "with 1.36 cm ground sample distance (GSD)", unit="cm", note="LiDAR; the multispectral imagery has a GSD of 3.3 cm per pixel and band (Sec. 2.2.2)"),
    R("point_density", "1159", 5, "Sec. 2.2.1", "an average point density of 1159 points", unit="pts/m2"),
    R("annotation_type", "point;point_label", 9, "Sec. 3.2", "Reference data were manually labeled for", note="vegetation, treetop locations, individual-tree IDs and species (coniferous or deciduous); field-measured heights and locations of 342 reference trees for validation"),
    R("dataset_name", "own UAV LiDAR and multispectral data of five reclaimed wellsites, Alberta (three seasons)", 9, "Sec. 3.2", DATA_EV, item="dataset1"),
    R("dataset_key", "own_p099_wellsites_alberta", 9, "Sec. 3.2", DATA_EV, item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 3.2", DATA_EV, item="dataset1", note="Table 2: TreeLoc trains on summer and autumn 2023 scans and tests on spring 2024; TreeOff and SpeciesCls use one wellsite and matched field trees"),
    R("dataset_public", "yes", 20, "Data Availability Statement", "The data that support the findings of this study are freely available", item="dataset1", note="Open Government Portal (Canada)"),
    R("dataset_url", "https://open.canada.ca/data/en/dataset/15c99460-599b423a-90e0-1af8a4f81913", 20, "Data Availability Statement", "The data that support the findings of this study are freely available", item="dataset1",
      note="as printed on p.20; the reference list prints 15c99460-599b-423a-90e0-1af8a4f81913"),
    R("n_trees", "342", 6, "Sec. 2.2.4", "In total, 342 reference trees were sampled", item="dataset1", unit="trees", note="field-measured reference trees (84 white spruce, 78 lodgepole pine, 55 balsam poplar, 125 willow)"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the training samples of Table 2 with AdamW (checkpoint with the highest testing accuracy)", strat_page=9, strat_loc="Sec. 3.2",
            strat_ev="Training minimized the prediction error against the reference labels using the AdamW optimizer")
ROWS += model("model1", "TreeLoc (3D-SegFormer for treetop locations)", "proposed", "hierarchical_vit", "3D SegFormer with skip-fusion on 128 x 128 x 128 voxel blocks of 20 cm", None, None, 6, "Sec. 3.1",
              "TreeLoc: 3DSegFormer producing 2D tree top locations", cat_note="SegFormer: transformer-based long-range encoding with multi-scale feature extraction (Sec. 3.1)", **NPRE)
ROWS += model("model2", "TreeOff (3D-SegFormer for individual tree segmentation)", "proposed", "hierarchical_vit", "3D SegFormer with skip-fusion on 128 x 128 x 128 voxel blocks of 10 x 10 x 20 cm", None, None, 6, "Sec. 3.1",
              "for individual tree segmentation, adapted from [44]", cat_note="SegFormer: transformer-based long-range encoding with multi-scale feature extraction (Sec. 3.1)", **NPRE)
ROWS += model("model3", "SpeciesCls (3D-ResNet, coniferous or deciduous)", "proposed", "cnn", "3D ResNet on 96 x 96 x 96 voxel blocks of 30 cm", None, None, 6, "Sec. 3.1",
              "SpeciesCls: 3D-ResNet for", cat_note="3D residual convolutional network", **NPRE)
res = Counter("res")
RID = {}
# Table 3 (p.10): LiDAR only, LiDAR + RGB, LiDAR + MS
f1, ev_f1 = row(PID, 10, r"^\s*F1 = (?=\d)", 0)
to, ev_to = row(PID, 10, r"TreeOff: Tree segmentation\s+mIoU = (?=\d)", 0)
sp, ev_sp = row(PID, 10, r"Species classification\s+mIoU = (?=\d)", 0)
assert len(f1) == len(to) == len(sp) == 3
INP = ("LiDAR point cloud only", "LiDAR + RGB", "LiDAR + five MS bands")
for k, inp in enumerate(INP):
    st = f"Table 3: {inp}, testing accuracy averaged over the wellsites"
    RID[("f1", k)] = res()
    ROWS += result(RID[("f1", k)], "f1", "model1", "dataset1", f1[k], 10, T3, ev_f1, unit="fraction", setting=st, note="F1 of the treetop detection", **TD, primary=(k == 0))
    ROWS += result(res(), "miou", "model2", "dataset1", to[k], 10, T3, ev_to, unit="fraction", setting=st, note="mIoU of the individual tree instances", **TS, primary=(k == 0))
    ROWS += result(res(), "miou", "model3", "dataset1", sp[k], 10, T3, ev_sp, unit="fraction", setting=st, note="mIoU of the coniferous and deciduous classes", **TC, primary=(k == 1))
# Seasonal accuracy of SpeciesCls (LiDAR + RGB), p.14
for season, v in (("summer", "0.75"), ("autumn", "0.90"), ("spring", "0.93")):
    ROWS += result(res(), "miou", "model3", "dataset1", v, 14, "Sec. 4.4", "drops to 0.75 in summer, compared to 0.93 in spring and 0.90 in autumn", unit="fraction", setting=f"Sec. 4.4: SpeciesCls (LiDAR + RGB), {season} data",
                   note="mIoU of the coniferous and deciduous classes by season of the scan", **TC(split="not_stated", agg="single"))
# Field validation of the LiDAR heights (p.11): 321 of the 342 reference trees matched
EVH = "LiDAR-derived heights yielded an RMSE of 0.40 m (12.9% RMSE%)"
ROWS += result(res(), "rmse", "model2", "dataset1", "0.40", 11, "Sec. 4.2", EVH, unit="m", setting="Sec. 4.2: LiDAR-only tree heights against field-measured heights, 321 matched reference trees", note="RMSE of the tree height (RMSE% 12.9); split: the field reference trees form what the paper calls a validation dataset (p.6, creating a validation dataset of individual-tree heights and locations)", **TH, primary=True)
ROWS += result(res(), "r2", "model2", "dataset1", "0.95", 11, "Sec. 4.2", "with a strong correlation (R2 = 0.95)", unit="fraction", setting="Sec. 4.2: LiDAR-only tree heights against field-measured heights, 321 matched reference trees", note="R2 of the tree height; split: the field reference trees form what the paper calls a validation dataset (p.6)", **TH, primary=True)
# Transfer pair: TreeLoc trained on summer and autumn 2023 scans, tested on spring 2024 scans (no same-season test value is printed)
t = "tr1"
ROWS += [R("transfer_source", "summer and autumn 2023 scans (training data of TreeLoc)", 9, "Table 2", "19,030 trees, All (August/October 2023)", item=t),
         R("transfer_target", "spring 2024 scans (testing data of TreeLoc)", 9, "Table 2", "10,175 trees, All (May 2024)", item=t),
         R("shift_type", "season", 18, "Sec. 5.3", "Models trained on summer and autumn data maintained high accuracy when predicting spring condition datasets", item=t),
         R("transfer_model", "model1", 9, "Table 2", "10,175 trees, All (May 2024)", item=t),
         R("transfer_metric", "f1", 10, T3, ev_f1, item=t),
         NR("value_in_domain", item=t, note="no test value on the training seasons is printed"),
         R("value_out_of_domain", f1[0], 10, T3, ev_f1, item=t, unit="fraction", note="LiDAR point cloud only column of Table 3"),
         R("transfer_level", "object", 10, T3, ev_f1, item=t)]
ROWS += conflict("cf1", f"{RID[('f1', 0)]}.value", "0.82", 1, "Abstract", "an F1-score of 0.82 for treetop detection", "V1a",
                 "Table 3 and Sec. 4.1 give an F1 of 0.83 for the LiDAR-only treetop detection; the abstract gives 0.82 (the bar of Fig. 4 is about 0.815)", unit="fraction")
ROWS += [R("limitations", "The tree segmentation model was trained and validated on one wellsite under leaf-on conditions, and short (below 2 m) and multi-stemmed trees are detected and classified worse", 19, "Sec. 5.4",
           "TreeOff model was trained and validated on a single wellsite (#262) under leaf-on conditions"),
         R("code_available", "yes", 20, "Data Availability Statement", CODE_EV, note="TreeAIBox plugin for CloudCompare"), R("code_url", "https://github.com/NRCan/TreeAIBox", 20, "Data Availability Statement", CODE_EV),
         R("weights_available", "yes", 20, "Data Availability Statement", "The trained DL models except for SpeciesCls are available from the TreeAIBox plugin in CloudCompare", note="all trained models except SpeciesCls"),
         R("data_available", "yes", 20, "Data Availability Statement", "The data that support the findings of this study are freely available")]
ROWS += [D("notes", "Table 3 gives the accuracy of the modules with three inputs (LiDAR only, LiDAR + RGB, LiDAR + five MS bands): vegetation filtering (TreeFilter, mIoU 0.93; not a tree task, not recorded), treetop detection (TreeLoc, precision, recall and F1; F1 recorded), tree segmentation (TreeOff, mIoU) and species classification (SpeciesCls, mIoU). The paper uses LiDAR-only models for delineation and LiDAR + RGB models for classification (Sec. 3.3), so those columns are primary. Table 2: TreeLoc trains on 19,030 trees of the summer and autumn 2023 scans and tests on 10,175 trees of spring 2024; TreeOff is trained on 1000 and tested on 297 trees of wellsite 262; SpeciesCls on 172 and 74 matched field trees. The abstract's 0.94 for vegetation filtering and 0.82 for treetop detection are the rounded bars of Fig. 4 (Table 3 prints 0.93 and 0.83). The detection rate against the 342 field trees (93.8 percent matched) and the seasonal figures (Figs. 9 to 11) other than the species accuracy are not recorded.",
           "Sec. 3.2 to 4.4 and Tables 2 and 3 (pp. 9 to 14)")]
