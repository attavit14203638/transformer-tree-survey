import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P068"
# Two linked steps on three test plots: (1) tree point extraction, a six-class point cloud segmentation scored here by the IoU of the tree class (component segmentation, point level, as for the
# semantic stage of other 3D papers); (2) individual tree segmentation of the extracted tree points, scored by the F1 of the overall row of Tables IV, VI and VII (3D tree instances).
TP = Tags(task="component_segmentation", level="point", split="test", agg="single")
TT = Tags(task="tree_segmentation_3d", level="instance_3d", split="test", agg="single")
OBJ_EV = "an improved multistage method was proposed for tree points extraction and individual tree segmentation in urban scenes using multispectral LiDAR"
PT_EV = "Second, the Point Transformer deep learning network was employed for extracting urban tree points"
DS_EV = "Our study area consists of 17 plots of size"
SPLIT_EV = "Three of these plots were selected as test sets for tree points extraction and ITC segmentation experiments, while the remaining plots were exclusively utilized to train the deep learning network"

ROWS = [
    R("task", "tree_segmentation_3d;component_segmentation", 1, "Abstract", OBJ_EV),
    R("output_unit", "tree_points;tree_instance", 1, "Abstract", OBJ_EV, note="tree points of the scene, then one segmented crown per tree"),
    R("tree_setting", "urban", 1, "Abstract", OBJ_EV),
    NR("tree_taxa", note="urban trees; the paper mentions diverse species without naming them"),
    R("study_area", "University of Houston and surroundings, Texas, United States", 2, "Sec. 2.A", "The flight area is located in and around the University of Houston, Texas, United States"),
    D("biome", "subtropical", "Houston, Texas; coded from the study area"),
    R("platform", "airborne", 2, "Sec. 2.A", "obtained using the airborne Optech Titan system"),
    R("sensors", "lidar;multispectral", 2, "Sec. 2.B", "an ALS operating at three different wavelengths", note="three-wavelength (532, 1064 and 1550 nm) intensity channels fused into a multispectral point cloud"),
    R("annotation_type", "point_label", 4, "Sec. 3.A", "we manually assign semantic labels to the point cloud data", note="six classes; the reference trees of the test plots were labelled manually with the orthophoto"),
    R("dataset_name", "Optech Titan multispectral LiDAR of the University of Houston area: 17 plots, 3 test plots", 2, "Sec. 2.A", DS_EV, item="dataset1"),
    R("dataset_key", "titan_houston_ms_lidar", 2, "Sec. 2.A", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 2.A", SPLIT_EV, item="dataset1"),
    R("dataset_public", "partial", 3, "Sec. 2.B", "The final production dataset was in LASer file (LAS) format", item="dataset1", note="the raw Titan data are public (GRSS data fusion contest site); whether the authors' labels are released is not stated"),
    R("n_images", "17", 2, "Sec. 2.A", DS_EV, item="dataset1", unit="tiles", note="plots of 300 m by 300 m; three are the test plots"),
    R("n_trees", "593", 9, "Table IV (image)", "Overall 593 605 556 37 49 0.938 0.919 0.928", item="dataset1", unit="trees", note="reference trees of the three test plots"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the training plots with the validation plots (Fig. 1); hyperparameters are not summarised")
M = [  # item, name, role, category, backbone, page, loc, evidence, note
    ("model1", "Point Transformer (tree point extraction)", "proposed", "pure_vit", "Point Transformer with vector self-attention and position encoding, three-channel multispectral input", 1, "Abstract", PT_EV, "self-attention point network (Sec. 3.B)"),
    ("model2", "Point Transformer with improved tree mapping (proposed pipeline)", "proposed", "pure_vit", "Point Transformer tree points, then CHM-based improved tree mapping", 1, "Abstract", PT_EV, "the transformer extracts the tree points; a CHM-based algorithm segments the crowns (Sec. 3.C)"),
    ("model3", "Random Forest (tree point extraction)", "other_baseline", "classical", "random forest on point features", 8, "Table III (image)", "Random Forest [20] Machine learning based 79.77", "machine learning baseline without a neural network"),
    ("model4", "PointNet (tree point extraction)", "other_baseline", "other", "PointNet", 8, "Table III (image)", "PointNet [38] Deep learning network 57.04", "MLP-based point network (protocol 4.5)"),
    ("model5", "PointNet++ (tree point extraction)", "other_baseline", "other", "PointNet++", 8, "Table III (image)", "PointNet++ [50] Deep learning network 84.63", "MLP-based point network (protocol 4.5)"),
    ("model6", "RandLA-Net (tree point extraction)", "other_baseline", "other", "RandLA-Net", 8, "Table III (image)", "RandLA-Net [39] Deep learning network 93.26", "random sampling and MLP-based local aggregation (protocol 4.5)"),
    ("model7", "DGCNN (tree point extraction)", "cnn_baseline", "cnn", "DGCNN", 8, "Table III (image)", "DGCNN [40] Deep learning network 94.28", "graph convolution (EdgeConv, protocol 4.5)"),
    ("model8", "DGCNN with improved tree mapping", "cnn_baseline", "cnn", "DGCNN tree points, then CHM-based improved tree mapping", 13, "Sec. 5.B", "DGCNN with our enhanced tree mapping algorithm", "graph convolution (EdgeConv) tree points with the same crown algorithm"),
    ("model9", "PointNet with gradient-based clustering (Chen et al.)", "other_baseline", "other", "PointNet tree points, gradient-based clustering", 13, "Sec. 5.B", "PointNet with gradient-based clustering proposed by Chen et al. [41]", "MLP-based point network (protocol 4.5)"),
    ("model10", "Faster R-CNN on raster feature maps (Windrim and Bryson)", "cnn_baseline", "cnn", "Faster R-CNN on feature maps of the raw point data", 13, "Sec. 5.B", "Faster R-CNN proposed by Windrim and Bryson [52]", "convolutional detector"),
    ("model11", "RandLA-Net with YOLO-v3 (Chang et al.)", "cnn_baseline", "cnn", "RandLA-Net tree points, YOLO-v3 instance detection", 13, "Sec. 5.B", "RandLA-Net with YOLO-v3 proposed by Chang et al. [53]", "convolutional YOLO-v3 detection on RandLA-Net tree points"),
    ("model12", "Layer-stacking", "other_baseline", "classical", "layer-stacking clustering of the raw points", 13, "Sec. 5.B", "watershed [51] and layer stacking [34], the input data consist of the raw data", "no neural network"),
    ("model13", "Watershed", "other_baseline", "classical", "raster watershed on the raw points", 13, "Sec. 5.B", "watershed [51] and layer stacking [34], the input data consist of the raw data", "no neural network"),
    ("model14", "Random Forest with the original tree mapping algorithm (ablation)", "other_baseline", "classical", "random forest tree points, original tree mapping", 13, "Table VI (image)", "RF with original tree mapping algorithm Overall 593/716", "no neural network"),
    ("model15", "Point Transformer with the original tree mapping algorithm (ablation)", "transformer_baseline", "pure_vit", "Point Transformer tree points, original tree mapping", 13, "Table VI (image)", "Point Transformer with original tree mapping algorithm Overall 593/652", "self-attention point network with the original crown algorithm"),
]
for item, name, role, cat, bb, pg, loc, ev, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, pg, loc, ev, cat_note=note, **NPRE)
res = Counter("res")
# Table III (p.8, image): overall IoU of the tree class (%) of six tree point extraction methods
T3 = [("model3", "79.77", "Random Forest [20]"), ("model4", "57.04", "PointNet [38]"), ("model5", "84.63", "PointNet++ [50]"), ("model6", "93.26", "RandLA-Net [39]"), ("model7", "94.28", "DGCNN [40]"), ("model1", "95.97", "Point Transformer [46]")]
for item, v, lab in T3:
    ROWS += result(res(), "iou", item, "dataset1", v, 8, "Table III (image)", f"{lab} {v}", unit="%", setting="Table III: tree point extraction, overall tree IoU of the three test plots", note="IoU of the tree class (six-class urban scene segmentation)", **TP, primary=True)
# Table V (p.10, image): Point Transformer, tree IoU by test area and input band (the 3-band row is the fused input)
T5 = {"2-1": ("89.8", "87.9", "88.6", "92.9"), "3-3": ("95.4", "93.7", "95.3", "97.3"), "5-3": ("95.9", "92.2", "96.1", "97.8")}
for area, vals in T5.items():
    for band, v in zip(("1550 nm", "1064 nm", "532 nm", "3-bands"), vals):
        ROWS += result(res(), "iou", "model1", "dataset1", v, 10, "Table V (image)", f"{area} {band} tree {v}", unit="%", setting=f"Table V: area {area}, input {band}", note="IoU of the tree class in one test area (percent)", **TP)
# Tables VII and VI (p.13, images): F1 of individual tree segmentation, overall row of the three test plots
T7 = [("model12", "Layer-stacking Overall 593/704 498 95 206 0.840 0.707 0.768", "0.768", True), ("model13", "Watershed Overall 593/755 478 115 277 0.806 0.633 0.709", "0.709", True),
      ("model9", "Chen et al. [41] PointNet with gradient based clustering Overall 593/699 513 80 186 0.865 0.734 0.794", "0.794", True),
      ("model8", "DGCNN with our improved tree mapping method Overall 593/668 562 31 106 0.948 0.841 0.891", "0.891", True),
      ("model10", "Windrim and Bryson [52] Faster R-CNN Overall 593/576 519 74 57 0.875 0.901 0.888", "0.888", True),
      ("model11", "Chang et al. [53] RandLA-Net with YOLO-v3 Overall 593/594 543 50 51 0.916 0.914 0.915", "0.915", True),
      ("model2", "Our method Overall 593/605 556 37 49 0.938 0.919 0.928", "0.928", True)]
for item, ev, v, prim in T7:
    ROWS += result(res(), "f1", item, "dataset1", v, 13, "Table VII (image)", ev, unit="fraction", setting="Table VII: individual tree segmentation, overall row of the three test plots", note="F1 of the segmented trees (TP, FN, FP counts in the table)", **TT, primary=prim, headline=(item == "model2"))
for item, ev, v in (("model14", "RF with original tree mapping algorithm Overall 593/716 528 65 188 0.890 0.737 0.806", "0.806"), ("model15", "Point Transformer with original tree mapping algorithm Overall 593/652 560 33 92 0.944 0.859 0.899", "0.899")):
    ROWS += result(res(), "f1", item, "dataset1", v, 13, "Table VI (image)", ev, unit="fraction", setting="Table VI: ablation with the original tree mapping algorithm, overall row", note="F1 of the segmented trees", **TT)
ROWS += [R("limitations", "Individual trees are still over- or under-segmented where crowns touch or heights differ strongly, and the accuracy depends on the tree density of the area", 9, "Sec. 4.B", "the accuracy of the algorithm is more influenced by the density of tree distribution"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "partial", "the raw Titan data are public (GRSS data fusion contest site); the authors' labels are not mentioned")]
ROWS += [D("notes", "Table III compares six tree point extraction methods by the overall IoU of the tree class; Table V repeats the Point Transformer with single-band and fused three-band input per test area (OA, mIoU and the other class IoUs are not recorded). Table VII compares seven individual tree segmentation pipelines (TP, FN, FP, recall, precision, F1; the overall F1 is recorded); Table VI is the ablation with the original tree mapping algorithm. Table IV, the per-area results of the proposed method, is not recorded separately (its overall row is the last row of Table VII). Primary results are the overall rows of Tables III and VII.",
           "Sec. 4 to 5 and Tables III to VII (pp. 8 to 13)")]
