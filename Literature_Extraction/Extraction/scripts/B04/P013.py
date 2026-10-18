import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T6 = "P013", "Table 3", "Table 6"
URL = "https://github.com/Taoorwell/TOF_Detection/"
H3 = "Detection models DT GT TP FP DR (%) Precision (%) Recall (%) F1 (%) AP (%)"
H6 = "Locations DT GT TP FP DR (%) Precision (%) Recall (%) F1 (%)"
# Individual tree crown detection with boxes; Table 3 is the main comparison on the 15 test patches (IoU threshold 0.5), Table 6 the large-area approach on 25 test plots per city.
T = Tags(task="detection", level="object", split="test", agg="single")
REPO = "Data and code used in this study are publicly available through the GitHub repository: https://github.com/Taoorwell/TOF_Detection/"

ROWS = [
    R("task", "detection", 1, "Abstract", "we proposed an end-to-end approach for large-area TOF detection based on an established detection transformer architecture"),
    R("output_unit", "tree_instance", 1, "Abstract", "We labelled 23,643 tree crowns with bounding boxes in 330 WorldView-3 image patches"),
    R("tree_setting", "mixed", 3, "Sec. 2.1", "the transect spans from urban to rural areas, capturing a diverse range of TOF forms"),
    R("tree_taxa", "trees outside forests (all trees; species not specified)", 3, "Sec. 3.2", "we defined trees as woody plants that are visible from above (excluding understory)"),
    R("study_area", "Bengaluru, Karnataka, India (training and test); Delhi, India and Shanghai, China (additional test locations)", 3, "Sec. 2.1 and 2.2",
      "the capital of Karnataka in Southern India ... We collected two additional WorldView-3 images from urban areas in Delhi"),
    D("biome", "multiple", "Bengaluru (tropical), Delhi and Shanghai (subtropical); not stated in the paper"),
    R("platform", "satellite", 3, "Sec. 2.2", "The satellite imagery used for Bengaluru in this study was acquired from WorldView-3 on November 16, 2016"),
    R("sensors", "multispectral", 3, "Sec. 2.2", "The imagery includes a panchromatic band with a spatial resolution of 0.3 meters and eight multispectral (MS) bands with a spatial resolution of 1.24 m",
      note="seven bands used (Blue, Green, Yellow, Red, Red Edge, Near IR1, Near IR2), pan-sharpened"),
    R("gsd", "0.3", 3, "Sec. 2.2", "Pan-sharpening was employed to improve the spatial resolution of the multispectral imagery to 0.3 meters", unit="m"),
    R("annotation_type", "bounding_box", 3, "Sec. 3.2", "we used bounding boxes to cover entire tree crowns for isolated trees"),
    R("dataset_name", "Bengaluru WorldView-3 tree crown detection dataset (330 patches of 100 m × 100 m)", 3, "Sec. 3.2", "A total of 330 image patches of 100 m × 100 m on the ground each were extracted from the image of Bengaluru", item="dataset1"),
    R("dataset_key", "own_p013_bengaluru", 3, "Sec. 3.2", "A total of 330 image patches of 100 m × 100 m on the ground each were extracted from the image of Bengaluru", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 3.2", "randomly divided into three sets: the training set (300 image patches, 21160 bounding boxes)", item="dataset1"),
    R("dataset_public", "yes", 12, "Data availability", REPO, item="dataset1"),
    R("dataset_url", URL, 12, "Data availability", REPO, item="dataset1"),
    R("n_images", "330", 3, "Sec. 3.2", "A total of 330 image patches of 100 m × 100 m on the ground each were extracted from the image of Bengaluru", item="dataset1", unit="images",
      note="300 training, 15 validation and 15 test patches (Table 2)"),
    R("n_trees", "23643", 1, "Abstract", "We labelled 23,643 tree crowns with bounding boxes in 330 WorldView-3 image patches", item="dataset1", unit="trees",
      note="21,160 training, 1,567 validation and 916 test boxes (Table 2)"),
    R("dataset_name", "Bengaluru urban test plots (25 plots of 100 m × 100 m, reference for the transfer test)", 9, "Sec. 4.3", "The Bengaluru urban area was chosen as the reference", item="dataset2"),
    R("dataset_key", "own_p013_bengaluru", 9, "Sec. 4.3", "The Bengaluru urban area was chosen as the reference", item="dataset2", note="same Bengaluru WorldView-3 scene"),
    R("dataset_role", "test_only", 9, "Sec. 4.3", "25 plots, each with a size of 100 m × 100 m, were randomly selected and labelled as ground truth in each location", item="dataset2"),
    R("dataset_public", "yes", 12, "Data availability", REPO, item="dataset2"),
    R("n_images", "25", 9, "Sec. 4.3", "25 plots, each with a size of 100 m × 100 m, were randomly selected and labelled as ground truth in each location", item="dataset2", unit="plots"),
    R("n_trees", "852", 10, T6, "Bengaluru Urban 743 852 677 66", item="dataset2", unit="trees", note="GT column of Table 6"),
    R("dataset_name", "Delhi WorldView-3 test plots (25 plots of 100 m × 100 m, light haze)", 4, "Table 1", "Delhi 28 September 2017 16116 × 18096 26 km2 Light Urban", item="dataset3"),
    R("dataset_key", "own_p013_delhi", 4, "Table 1", "Delhi 28 September 2017 16116 × 18096 26 km2 Light Urban", item="dataset3"),
    R("dataset_role", "transfer_target", 9, "Sec. 4.3", "further tested with images of urban areas from the cities of Delhi and Shanghai", item="dataset3"),
    R("dataset_public", "yes", 12, "Data availability", REPO, item="dataset3"),
    R("n_images", "25", 9, "Sec. 4.3", "25 plots, each with a size of 100 m × 100 m, were randomly selected and labelled as ground truth in each location", item="dataset3", unit="plots"),
    R("n_trees", "627", 10, T6, "Delhi 639 627 548 91", item="dataset3", unit="trees", note="GT column of Table 6"),
    R("dataset_name", "Shanghai WorldView-3 test plots (25 plots of 100 m × 100 m, heavy haze)", 4, "Table 1", "Shanghai 18 April 2016 13640 × 21636 26 km2 Heavy Urban", item="dataset4"),
    R("dataset_key", "own_p013_shanghai", 4, "Table 1", "Shanghai 18 April 2016 13640 × 21636 26 km2 Heavy Urban", item="dataset4"),
    R("dataset_role", "transfer_target", 9, "Sec. 4.3", "further tested with images of urban areas from the cities of Delhi and Shanghai", item="dataset4"),
    R("dataset_public", "yes", 12, "Data availability", REPO, item="dataset4"),
    R("n_images", "25", 9, "Sec. 4.3", "25 plots, each with a size of 100 m × 100 m, were randomly selected and labelled as ground truth in each location", item="dataset4", unit="plots"),
    R("n_trees", "1148", 10, T6, "Shanghai 1059 1148 807 252", item="dataset4", unit="trees", note="GT column of Table 6"),
]
PRE = dict(pre_page=5, pre_loc="Sec. 3.3", pre_ev="For all models, the weights were initialized with those trained on the COCO object detection dataset", pre_note="COCO object detection weights",
           strat_page=5, strat_loc="Sec. 3.3", strat_ev="the weights were initialized with those trained on the COCO object detection dataset and then trained on our tree crown detection dataset",
           strat_note="initialised from COCO weights and trained on the tree dataset; frozen layers are not mentioned")
M = [  # item, row label in Table 3, name, role, category, backbone, note
    ("model1", "Faster RCNN", "Faster RCNN", "cnn_baseline", "cnn", "backbone not stated", "described as a conventional CNN-based detector (p.5)"),
    ("model2", "YOLO", "YOLO (YOLO v3)", "cnn_baseline", "cnn", "backbone not stated", "described as a conventional CNN-based detector (p.5)"),
    ("model3", "RetinaNet", "RetinaNet", "cnn_baseline", "cnn", "backbone not stated", "described as a conventional CNN-based detector (p.5)"),
    ("model4", "DETR", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", "transformer-based detection model (p.5); DETR has a CNN backbone, which the paper does not name"),
    ("model5", "Deformable-DETR", "Deformable-DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", "transformer-based detection model (p.5); backbone not named"),
    ("model6", "DINO-Res50", "DINO-Res50", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50 with transformer encoder and decoder", "DINO (DETR-type detector) with a ResNet50 backbone"),
    ("model7", "DINO-SWIN", "DINO-SWIN", "proposed", "hierarchical_vit", "SWIN transformer with transformer encoder and decoder", "DINO (DETR-type detector) with a Swin transformer backbone; the Swin size is not stated"),
]
res = Counter("res")
for item, lab, name, role, cat, bb, note in M:
    nums, ev = row(PID, 7, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 9, (lab, nums)  # DT, GT, TP, FP, DR, precision, recall, F1, AP
    ROWS += model(item, name, role, cat, bb, "other", "full_fine_tuning", 7, T3, H3 + " ... " + ev, cat_note=note, **PRE)
    ROWS += result(res(), "f1", item, "dataset1", nums[7], 7, T3, H3 + " ... " + ev, setting="Table 3: test set (15 patches, 916 trees); IoU threshold 0.5", **T, primary=True, headline=(item == "model7"))
    ROWS += result(res(), "ap50", item, "dataset1", nums[8], 7, T3, H3 + " ... " + ev, setting="Table 3: test set (15 patches, 916 trees); IoU threshold 0.5", note="AP at IoU 0.5 (p.6)", **T, primary=True)
# The proposed large-area approach: DINO-SWIN with two-level overlapping tiling and R-tree-based box merging (RBM), Table 6
ROWS += model("model8", "DINO-SWIN with two-level tiling and RBM (proposed large-area approach)", "proposed", "hierarchical_vit", "SWIN transformer with transformer encoder and decoder", "other", "full_fine_tuning", 9, "Sec. 4.3",
              "The proposed approach, DINO-SWIN in combination with the ... overlapping tiling scheme including RBM", cat_note="DINO-SWIN as in Table 3 with tiling and box merging added at inference", **PRE)
TR = {}
for lab, ds in (("Bengaluru Urban", "dataset2"), ("Delhi", "dataset3"), ("Shanghai", "dataset4")):
    nums, ev = row(PID, 10, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 8, (lab, nums)  # DT, GT, TP, FP, DR, precision, recall, F1
    TR[lab] = (nums[7], H6 + " ... " + ev)
    ROWS += result(res(), "f1", "model8", ds, nums[7], 10, T6, H6 + " ... " + ev, setting=f"Table 6: 25 test plots, {lab}; large-area approach", **T)
# Transfer (V8): the Bengaluru urban plots are the reference (in-domain), Delhi and Shanghai are unseen cities
tr = Counter("tr")
for lab, tgt, note in (("Delhi", "Delhi urban test plots (WorldView-3, light haze)", "different city and haze level, same sensor"),
                       ("Shanghai", "Shanghai urban test plots (WorldView-3, heavy haze)", "different city and haze level, same sensor")):
    t = tr()
    ROWS += [R("transfer_source", "Bengaluru WorldView-3 scene (training region); reference: Bengaluru urban test plots", 9, "Sec. 4.3", "The Bengaluru urban area was chosen as the reference", item=t),
             R("transfer_target", tgt, 9, "Sec. 4.3", "further tested with images of urban areas from the cities of Delhi and Shanghai", item=t),
             R("shift_type", "region", 3, "Sec. 2.2", "which aligns with our objective of assessing the model’s robustness to varying image quality", item=t, note=note),
             R("transfer_model", "model8", 10, T6, TR[lab][1], item=t),
             R("transfer_metric", "f1", 10, T6, TR[lab][1], item=t),
             R("value_in_domain", TR["Bengaluru Urban"][0], 10, T6, TR["Bengaluru Urban"][1], item=t, unit="%"),
             R("value_out_of_domain", TR[lab][0], 10, T6, TR[lab][1], item=t, unit="%"),
             R("transfer_level", "object", 10, T6, TR[lab][1], item=t)]
ROWS += [R("limitations", "Computational efficiency (inference time, resource use) was not analysed and accuracy falls under heavy haze (Shanghai)", 11, "Sec. 5.2",
           "a detailed analysis of computational efficiency, including inference time, resource usage, and their trade-offs, was not fully explored")]
ROWS += avail(code_url=URL, page=12, loc="Data availability", ev=REPO, data="yes", data_page=12, data_loc="Data availability", data_ev=REPO)
ROWS += [D("notes", "Table 3 (p.7) compares seven detectors on the 15 Bengaluru test patches; Table 4 ablates the input bands of DINO-SWIN (the default NIR1-R-G equals the Table 3 row) and is not recorded; Table 5 times NMS against R-tree-based box merging. Table 6 applies the large-area approach (DINO-SWIN with two-level overlapping tiling and RBM) to 25 test plots in each of Bengaluru urban, Delhi and Shanghai; Bengaluru urban is the reference. Precision, recall and detection rate are in Tables 3 and 6.",
           "Sec. 4 and Tables 2 to 6 (pp. 4 to 10)")]
