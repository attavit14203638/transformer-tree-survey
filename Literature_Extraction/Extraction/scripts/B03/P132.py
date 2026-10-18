import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T5, T6 = "P132", "Table 5", "Table 6"
ROWS = [
    R("task", "detection;health_classification", 1, "Abstract", "a deep learning-based pine nematode tree identification method is proposed by fusing visible and multispectral imagery",
      note="boxes for pine wilt infected trees"),
    R("output_unit", "tree_instance", 1, "Abstract", "to realize the identification of infected pine trees with high detection speed and accuracy"),
    R("tree_setting", "natural_forest", 3, "Sec. 1", "PWD usually occurs in a complex environment with various tree species, vegetation, bare ground and shadows", note="pine forest"),
    R("tree_taxa", "pine (pine wilt disease hosts)", 1, "Abstract", "the identification of infected pine trees"),
    R("study_area", "eastern Qingdao City, Shandong Province, China", 3, "Sec. 2.1", "The study area is located in the eastern region of Qingdao City, Shandong Province"),
    D("biome", "temperate", "Qingdao, Shandong (warm temperate); not stated in the paper"),
    R("platform", "uav", 4, "Sec. 2.1", "The images were collected by a UAV (DJI M300 RTK) equipped with a multispectral camera (Changguang Yuchen MS600 Pro) and a visible light camera (Zenith H20)"),
    R("sensors", "rgb;multispectral", 4, "Sec. 2.1", "The multispectral camera captures images in six bands: red, green, blue, near-infrared, red edge and red edge 750 nm"),
    R("annotation_type", "bounding_box", 1, "Abstract", "The YOLOv5 (You Only Look Once v5)-based detection algorithm is adopted", note="box labels for infected trees"),
    R("dataset_name", "own UAV visible and multispectral pine wilt dataset, Qingdao (dataset 3: visible plus multispectral)", 12, "Sec. 3.2",
      "dataset 3 is an expansion of dataset 2 by adding 5706 augmented multispectral images, with a total of 15,256 images", item="dataset1"),
    R("dataset_role", "train_test", 5, "Table 2", "Visible 1958 1615 9550 343 2058", item="dataset1", note="1615 training and 343 validation visible images (2058 after splitting); results on the validation set"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    R("n_images", "1958", 5, "Table 2", "Visible 1958 1615 9550 343 2058", item="dataset1", unit="images", note="original visible images; 965 multispectral images used for training"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained 500 epochs with SGD (p.12); initialisation not stated")
res = Counter("res")
# Table 6: dataset 3
T6M = [("model1", r"Improved", "Improved YOLOv5l (GhostNet, CBAM and CA, transformer encoder, BiFPN)", "proposed", "cnn_transformer_hybrid",
        "YOLOv5l with GhostNet, CBAM+CA, transformer encoder and BiFPN"),
       ("model2", r"Base YOLOv5l", "YOLOv5l", "cnn_baseline", "cnn", "YOLOv5l"),
       ("model3", r"YOLOv7 \[36\]", "YOLOv7", "cnn_baseline", "cnn", "YOLOv7"),
       ("model4", r"YOLOX \[37\]", "YOLOX", "cnn_baseline", "cnn", "YOLOX"),
       ("model5", r"SSD300 \[14\]", "SSD300", "cnn_baseline", "cnn", "SSD300"),
       ("model6", r"YOLOv4 \[12\]", "YOLOv4", "cnn_baseline", "cnn", "YOLOv4"),
       ("model7", r"Faster R-CNN \[11\]", "Faster R-CNN", "cnn_baseline", "cnn", "Faster R-CNN")]
for item, rx, name, role, cat, bb in T6M:
    if item == "model1":
        nums, ev = row(PID, 15, r"^\s+(?=0\.981)", 0)
        ev = "Improved YOLOv5l " + ev
    else:
        nums, ev = row(PID, 15, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(nums) == 5, (item, nums, ev)
    ROWS += model(item, name, role, cat, bb, None, None, 15, T6, ev, **NP)
    ROWS += result(res(), "ap50", item, "dataset1", nums[2], 15, T6, ev, unit="fraction", setting="dataset 3 (visible and multispectral); validation images",
                   headline=(item == "model1"))
# Table 5 ablation (text layer duplicated by a peer-review overlay, so values read from the rendered page): the transformer encoder is added between M3 and M4
T5I = "Table 5 (image)"
for item, mark, name, cat, note, vals in (
        ("model8", "M3", "YOLOv5l with GhostNet and CBAM+CA (M3, before the transformer encoder)", "cnn", "ablation step before the transformer encoder",
         ("0.967", "0.910", "0.944", "44.40", "0.064")),
        ("model9", "M4", "YOLOv5l with GhostNet, CBAM+CA and transformer encoder (M4)", "cnn_transformer_hybrid", "ablation step adding the transformer encoder",
         ("0.987", "0.936", "0.973", "45.60", "0.066"))):
    ev = f"{mark} " + " ".join(vals)
    ROWS += model(item, name, "cnn_baseline" if cat == "cnn" else "transformer_baseline", cat, "YOLOv5l variant", None, None, 13, T5I, ev, cat_note=note, **NP)
    ROWS += result(res(), "ap50", item, "dataset1", vals[2], 13, T5I, ev, unit="fraction", setting="dataset 3; ablation")
ROWS += avail()
ROWS += [D("notes", "The ablation (Table 5) shows mAP@0.5 rising from 0.944 (M3) to 0.973 when the transformer encoder is added (M4) and to 0.987 with BiFPN (M5, the final model). Table 4 compares the base and improved YOLOv5l on datasets 1 to 3. Model sizes (MB) and detection times are in Table 6.", "Sec. 3.2 and Tables 4 to 6 (pp. 12 to 15)")]
