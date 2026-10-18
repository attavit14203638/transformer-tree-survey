import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T4, T5 = "Table 4", "Table 5"
ROWS = [
    R("task", "detection", 1, "Abstract", "a novel feature-oriented individual tree identification network suitable for UAV high-resolution images"),
    R("output_unit", "tree_instance", 9, "Sec. 3.5", "Precision, Recall, F1-score and mAP50 evaluation metrics", note="bounding boxes of individual trees"),
    R("tree_setting", "plantation", 3, "Sec. 2", "The study area is located in the experimental forest for selecting hybrid Liriodendron in Lishan forest farm", note="experimental forest for hybrid Liriodendron selection"),
    R("tree_taxa", "hybrid Liriodendron (Liriodendron sinoamericanum)", 3, "Sec. 2", "selecting hybrid Liriodendron (Liriodendron sinoamericanum) in Lishan forest farm"),
    R("study_area", "Lishan forest farm, Ji'an City, Jiangxi Province, China", 3, "Sec. 2", "Lishan forest farm, Ji'an City, Jiangxi Province, China"),
    R("biome", "subtropical", 3, "Sec. 2", "The region has a subtropical monsoon climate"),
    R("platform", "uav", 3, "Sec. 2", "This study used the DJI Phantom 4 RTK UAV to collect high-resolution three-band images"),
    R("sensors", "rgb", 3, "Sec. 2", "high-resolution three-band images of the study area"),
    R("gsd", "0.05", 4, "Sec. 2", "a complete orthoimage of the study area with a ground resolution of 0.05 m", unit="m"),
    R("annotation_type", "bounding_box", 4, "Sec. 2", "this study chooses visual interpretation for manual data annotation", note="box annotations implied by the YOLO detector and box loss"),
    R("dataset_name", "own UAV Liriodendron dataset (640 x 640 tiles)", 4, "Sec. 2", "crop 413 images with a resolution of 640 × 640 pixels at a fixed step size of 320 pixels, and constitute the original dataset", item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 2", "divided into training set, validation set and test set according to the ratio of 7:1:2", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    R("n_images", "1239", 4, "Sec. 2", "a total of 1239 images including 7824 individual tree samples were obtained", item="dataset1", unit="images",
      note="after offline augmentation of 413 original tiles"),
    D("n_trees", "7112", "5548 + 686 + 878 trees in the training, validation and test sets (p.4, Sec. 2); the printed total 7824 is in cf1 (rule V1c)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "FO-Net", "proposed", "cnn_transformer_hybrid", "YOLOv7-Tiny with variable position drift convolution, gather-and-distribute neck and unified self-attention head",
              None, None, 4, "Sec. 3.1", "the YOLOv7-Tiny deep learning model is used as the baseline model framework to construct ... FO-Net",
              pre_status="not_reported", pre_note="pre-training not stated", strat_note="training details not stated",
              cat_note="transformer block in the high-level gather-and-distribute fusion: Ffuse = Transformer(Falign) (p.8, Eq. 8)")
ROWS += [R("total_params", "8.9", 10, T5, "FO-Net (ours) 90.7 85.8 82.8 0.85 8.9 15.4 453", item="model1", unit="M"),
         R("flops", "15.4", 10, T5, "FO-Net (ours) ... 8.9 15.4 453", item="model1", unit="GFLOPs")]
ROWS += model("model2", "YOLOv7-Tiny (baseline)", "cnn_baseline", "cnn", "YOLOv7-Tiny", None, None, 10, T5, "YOLOv7-Tiny 85.7 84.3 78.8 0.8 6 13.2 227",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated", cat_note="the model FO-Net is built on")
ROWS += [R("total_params", "6", 10, T5, "YOLOv7-Tiny ... 6 13.2 227", item="model2", unit="M"), R("flops", "13.2", 10, T5, "YOLOv7-Tiny ... 6 13.2 227", item="model2", unit="GFLOPs")]
ROWS += model("model3", "YOLOX-S", "cnn_baseline", "cnn", "YOLOX-S", None, None, 10, T5, "YOLOX-S 89.94 89.46 71.53 0.79 9 26.8", pre_status="not_reported",
              pre_note="not stated", strat_note="not stated", cat_note="best CNN on mAP50 in Table 5")
ROWS += [R("total_params", "9", 10, T5, "YOLOX-S ... 9 26.8", item="model3", unit="M"), R("flops", "26.8", 10, T5, "YOLOX-S ... 9 26.8", item="model3", unit="GFLOPs")]
ROWS += model("model4", "YOLOv7", "cnn_baseline", "cnn", "YOLOv7", None, None, 10, T5, "YOLOv7 87.6 85.9 81.7 0.84 36.4 103.2 319", pre_status="not_reported",
              pre_note="not stated", strat_note="not stated", cat_note="best CNN on F1 in Table 5")
ROWS += [R("total_params", "36.4", 10, T5, "YOLOv7 ... 36.4 103.2 319", item="model4", unit="M"), R("flops", "103.2", 10, T5, "YOLOv7 ... 36.4 103.2 319", item="model4", unit="GFLOPs")]
ROWS += model("model5", "YOLOv7-Tiny with gather-and-distribute neck (ablation A)", "other_baseline", "cnn_transformer_hybrid", "YOLOv7-Tiny", None, None, 9, T4,
              "A √ 88.4 86.8 80.0 0.83 8.9", pre_status="not_reported", pre_note="not stated", strat_note="not stated",
              cat_note="ablation: baseline plus the transformer-based gather-and-distribute fusion only")
ROWS += [R("total_params", "8.9", 9, T4, "A √ 88.4 86.8 80.0 0.83 8.9", item="model5", unit="M")]
res = Counter("res")
for m, lab, ap, f1, loc, pg in (("model1", "FO-Net (ours)", "90.7", "0.85", T5, 10), ("model2", "YOLOv7-Tiny", "85.7", "0.8", T5, 10),
                                ("model3", "YOLOX-S", "89.94", "0.79", T5, 10), ("model4", "YOLOv7", "87.6", "0.84", T5, 10),
                                ("model5", "A (GD only)", "88.4", "0.83", T4, 9)):
    ROWS += result(res(), "ap50", m, "dataset1", ap, pg, loc, f"{lab} mAP50 {ap}", setting="test set", headline=(m == "model1"))
    ROWS += result(res(), "f1", m, "dataset1", f1, pg, loc, f"{lab} F1 {f1}", unit="fraction", setting="test set")
ROWS += avail()
ROWS += [D("notes", "Table 5 lists 26 convolutional detectors (Faster R-CNN, SSD, EfficientDet-D1 and YOLOv3 to YOLOX variants) plus FO-Net; following protocol 4.1, only the baseline FO-Net is built on (YOLOv7-Tiny) and the best CNN per metric (YOLOX-S for mAP50, YOLOv7 for F1) are recorded. Ablation (Table 4): adding the gather-and-distribute neck to YOLOv7-Tiny raised mAP50 from 85.7 to 88.4. The 'unified self-attention identification head' follows Dynamic Head (attention-based but not a transformer block).", "Tables 4 to 5 (pp. 9 to 10) and Sec. 3.4")]
