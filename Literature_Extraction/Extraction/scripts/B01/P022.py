import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, T2, T3, T4 = "Table 1", "Table 2", "Table 3", "Table 4"
ROWS = [
    R("task", "detection", 1, "Abstract", "Accurate detection of individual tree crowns is a critical prerequisite"),
    R("output_unit", "tree_instance", 10, "Sec. 2.2", "the CIoU loss function is used, which improves the robustness in detecting tree crowns", note="crown bounding boxes"),
    R("tree_setting", "mixed", 3, "Sec. 2.1", "one is a broadleaf forest dataset (Dataset_A) ... and the other is a mixed conifer forest dataset (Dataset_B)",
      note="Catalpa bungei stand (Dataset_A) and mixed conifer forest (Dataset_B)"),
    R("tree_taxa", "Catalpa bungei; Pinus massoniana; Cunninghamia lanceolata", 3, "Sec. 2.1", "primarily composed of Catalpa bungei ... The main tree species include Pinus massoniana and Cunninghamia lanceolata"),
    R("study_area", "Wenxian County, Henan; Liping County, Guizhou; China", 3, "Sec. 2.1", "Wenxian County, Henan Province ... Liping County, Guizhou Province"),
    R("biome", "multiple", 3, "Sec. 2.1", "temperate continental monsoon climate ... subtropical monsoon climate"),
    R("platform", "uav", 4, "Sec. 2.1", "a DJI Matrice 300 RTK UAV ... integrated the Zenmuse L1 module, which includes a LiDAR sensor, RGB camera"),
    R("sensors", "rgb;lidar", 4, "Sec. 2.1", "orthophotos with a resolution of 3 cm ... The CHM was generated from the normalized point cloud data", note="LiDAR used as a canopy height model raster"),
    R("gsd", "3", 4, "Sec. 2.1", "orthophotos with a resolution of 3 cm", unit="cm", note="CHM also 3 cm"),
    R("annotation_type", "bounding_box", 10, "Sec. 2.2", "the predicted bounding box and the ground truth bounding box"),
]
for ds, name, n, ev in (("dataset1", "Dataset_A (broadleaf, Wenxian County, Henan)", "879", "The dataset contains a total of 879 trees"),
                        ("dataset2", "Dataset_B (mixed conifer, Liping County, Guizhou)", "1618", "The dataset contains a total of 1618 trees")):
    ROWS += [R("dataset_name", name, 3, "Sec. 2.1", "two forest remote sensing datasets collected using UAV", item=ds),
             R("dataset_role", "train_test", 11, "Sec. 3.1.2", "all methods were trained and evaluated under identical conditions using the same datasets", item=ds,
               note="train/test split not described"),
             R("dataset_public", "on_request", 19, "Data Availability Statement", "The data used in this study are available from the corresponding author upon reasonable request", item=ds),
             R("n_trees", n, 3 if ds == "dataset1" else 4, "Sec. 2.1", ev, item=ds, unit="trees")]
M = [  # item, name, role, category, backbone, label
    ("model1", "MTCDNet", "proposed", "cnn_transformer_hybrid", "two CSP convolutional branches (RGB and CHM) with a multimodal transformer fusion module", "MTCDNet (Ours)"),
    ("model2", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "not stated", "DETR"),
    ("model3", "MS-DETR", "transformer_baseline", "cnn_transformer_hybrid", "not stated", "MS-DETR"),
    ("model4", "YOLOX", "cnn_baseline", "cnn", "YOLOX", "YOLOX"),
    ("model5", "YOLOv5", "cnn_baseline", "cnn", "YOLOv5", "YOLOV5"),
    ("model6", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8", "YOLOV8"),
    ("model7", "YOLOv10", "cnn_baseline", "cnn", "YOLOv10", "YOLOV10"),
    ("model8", "Early Fusion (RGB and depth concatenated at the input)", "other_baseline", "cnn", "not stated", "Early Fusion"),
]
for item, name, role, cat, bb, lab in M:
    ROWS += model(item, name, role, cat, bb, None, None, 12, T1, lab, pre_status="not_reported", pre_note="not stated",
                  strat_note="trained with SGD, initial learning rate 0.01 (p.10); initialisation not stated" if item == "model1" else "not stated",
                  cat_note="RGB only" if item in ("model2", "model3", "model4", "model5", "model6", "model7") else "")
ROWS += [R("total_params", "44.52", 16, T4, "✓ ✓ 45.38 93.12 35.74 44.52 45.31", item="model1", unit="M"),
         R("flops", "45.31", 16, T4, "✓ ✓ 45.38 93.12 35.74 44.52 45.31", item="model1", unit="GFLOPs")]
ROWS += model("model9", "MTCDNet without MM-Transformer and learnable positional encoding (addition fusion)", "other_baseline", "cnn", "two CSP convolutional branches",
              None, None, 16, T4, "× × 41.27 91.25 32.56 43.10 41.84", pre_status="not_reported", pre_note="not stated", strat_note="not stated",
              cat_note="ablation: features from RGB and depth fused by addition")
ROWS += [R("total_params", "43.10", 16, T4, "× × 41.27 91.25 32.56 43.10 41.84", item="model9", unit="M"),
         R("flops", "41.84", 16, T4, "× × 41.27 91.25 32.56 43.10 41.84", item="model9", unit="GFLOPs")]
VALS = {  # item: (A mAP, A AP50, B mAP, B AP50)
    "model2": ("38.03", "88.35", "37.61", "89.50"), "model3": ("39.46", "89.92", "42.56", "90.87"), "model4": ("40.25", "90.37", "42.49", "92.20"),
    "model5": ("38.98", "89.21", "45.16", "92.98"), "model6": ("42.83", "91.29", "45.63", "93.05"), "model7": ("43.03", "91.29", "46.88", "93.69"),
    "model8": ("44.25", "92.50", "45.31", "93.22"), "model1": ("45.38", "93.12", "48.41", "94.58"),
}
res = Counter("res")
for item, (am, a50, bm, b50) in VALS.items():
    lab = next(m[5] for m in M if m[0] == item)
    labA = "MMFTNet (Ours)" if item == "model1" else lab
    ROWS += result(res(), "ap50_95", item, "dataset1", am, 12, T1, f"{labA} mAP {am}", setting="Dataset_A test", headline=(item == "model1"),
                   note="mAP averaged over IoU thresholds" + ("; Table 1 labels the proposed model 'MMFTNet (Ours)'" if item == "model1" else ""))
    ROWS += result(res(), "ap50", item, "dataset1", a50, 12, T1, f"{labA} AP50 {a50}", setting="Dataset_A test")
    ROWS += result(res(), "ap50_95", item, "dataset2", bm, 13, T2, f"{lab} mAP {bm}", setting="Dataset_B test",
                   note="the text (p.12) gives 45.31/93.22/31.79 for MTCDNet on Dataset_B, which are the Early Fusion values in Table 2" if item == "model1" else "mAP averaged over IoU thresholds")
    ROWS += result(res(), "ap50", item, "dataset2", b50, 13, T2, f"{lab} AP50 {b50}", setting="Dataset_B test")
ROWS += result(res(), "ap50_95", "model9", "dataset1", "41.27", 16, T4, "× × 41.27 91.25", setting="Dataset_A test (ablation)")
ROWS += result(res(), "ap50", "model9", "dataset1", "91.25", 16, T4, "× × 41.27 91.25", setting="Dataset_A test (ablation)")
tr = Counter("tr")
for src, tgt, ind, outd, lab in (("Dataset_A", "Dataset_B", "45.38", "43.50", "Dataset_A-to-B 43.50"), ("Dataset_B", "Dataset_A", "48.41", "44.17", "Dataset_B-to-A 44.17")):
    i = tr()
    ROWS += [R("transfer_source", f"{src} (training data)", 16, T3, "Performance of MTCDNet on cross-dataset validation (Dataset_A-to-B and Dataset_B-to-A)", item=i),
             R("transfer_target", f"{tgt} (test data)", 16, T3, lab, item=i),
             R("shift_type", "site;species", 3, "Sec. 2.1", "broadleaf forest dataset (Dataset_A) ... mixed conifer forest dataset (Dataset_B)", item=i),
             R("transfer_model", "model1", 16, T3, lab, item=i),
             R("transfer_metric", "ap50_95", 16, T3, "mAP (%)", item=i),
             R("value_in_domain", ind, 12 if src == "Dataset_A" else 13, T1 if src == "Dataset_A" else T2, f"MTCDNet mAP {ind}", item=i, unit="%"),
             R("value_out_of_domain", outd, 16, T3, lab, item=i, unit="%")]
ROWS += avail(data="on_request", data_page=19, data_loc="Data Availability Statement",
              data_ev="The data used in this study are available from the corresponding author upon reasonable request")
ROWS += [D("notes", "Baselines use RGB only; MTCDNet and Early Fusion use RGB and CHM. Ablation (Table 4): adding the MM-Transformer raised Dataset_A mAP from 41.27 to 43.93, and learnable positional encoding to 45.38. Single-modality MTCDNet (Table 5): RGB only 42.96 mAP, CHM only 37.20.", "Sec. 3 and Tables 1 to 5 (pp. 11 to 17)")]
