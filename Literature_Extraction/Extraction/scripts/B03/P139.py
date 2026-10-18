import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1 = "Table I (image)"
S = "Sec. IV-A"
ROWS = [
    R("task", "detection;health_classification", 7, S, "The resulting PWD dataset comprises 2478 images, categorized into two classes: diseased tree and dead tree"),
    R("output_unit", "tree_instance;tree_label", 7, T1, "AP(%) Diseased Tree Dead Tree"),
    R("tree_setting", "natural_forest", 7, S, "including mixed coniferous-broadleaf forests, evergreen broadleaf forests, and planted coniferous forests"),
    R("tree_taxa", "pine (pine wilt disease hosts)", 1, "Abstract", "a novel PWD detection method called feature interaction and dependency capturing-YOLO (FIDC-YOLO) is proposed"),
    R("study_area", "Qimen County, Huangshan City, Anhui Province, China", 7, S, "The study area is located in Qimen County, Huangshan City, Anhui Province"),
    R("biome", "subtropical", 7, S, "Characterized by a subtropical monsoon climate"),
    R("platform", "uav", 7, S, "The original images are captured by an M300RTK UAV equipped with a Zenmuse P1 lens"),
    R("sensors", "rgb", 7, S, "equipped with a Zenmuse P1 lens in the study area, with a resolution of 8192 × 5460"),
    R("annotation_type", "bounding_box", 7, S, "30 high-resolution, low-distortion images are selected from the original data for labeling and image cropping", note="box labels by class"),
    R("dataset_name", "own UAV PWD dataset, Qimen County, Huangshan (640 x 640 crops)", 7, S, "The resulting PWD dataset comprises 2478 images", item="dataset1"),
    R("dataset_role", "train_test", 7, S, "split into 1982 for training, 248 for validation, and 248 for testing", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    R("n_images", "2478", 7, S, "The resulting PWD dataset comprises 2478 images", item="dataset1", unit="images"),
]
DET = dict(pre_status="not_reported", pre_note="pre-trained ResNet50 backbone (pre-training data not stated)", strat_page=8, strat_loc="Sec. IV-D",
           strat_ev="Faster R-CNN and transformer-based methods (DINO, and RT-DETR) are trained using the MMDetection [38] framework with a pre-trained ResNet50 backbone")
YOLO = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained in the Ultralytics framework with the same hyperparameters as FIDC-YOLO (p.8)")
M = [  # item, name, role, category, backbone, (diseased AP, dead AP, mAP, params, FLOPs), evidence, kind, note
    ("model1", "FIDC-YOLO", "proposed", "other", "YOLOv10l with SELAN, FA-SPPF and a Hydra (Mamba) dependency module", ("81.0", "78.9", "80.0", "26.1", "88.8"),
     "FIDC-YOLO(Ours) 81.0 78.9 80.0 26.1 88.8", "own", "state-space (Mamba) module, no transformer"),
    ("model2", "RT-DETR", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50 with transformer encoder-decoder", ("72.5", "71.7", "72.1", "42.8", "107"),
     "RT-DETR [14] 72.5 71.7 72.1 42.8 107", "det", ""),
    ("model3", "DINO", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50 with deformable transformer", ("68.0", "69.2", "68.6", "47.5", "179.0"),
     "DINO [26] 68.0 69.2 68.6 47.5 179.0", "det", "DETR-type detector, not the DINO self-supervised ViT"),
    ("model4", "Faster R-CNN", "cnn_baseline", "cnn", "ResNet50", ("65.6", "67.1", "66.4", "41.4", "134.0"), "Faster R-CNN [22] 65.6 67.1 66.4 41.4 134.0", "det", ""),
    ("model5", "YOLOv8m", "cnn_baseline", "cnn", "YOLOv8m", ("79.5", "77.3", "78.4", "25.8", "78.7"), "YOLOv8m 79.5 77.3 78.4 25.8 78.7", "yolo", ""),
    ("model6", "YOLOv10l", "cnn_baseline", "cnn", "YOLOv10l", ("77.4", "76.8", "77.1", "25.7", "126.3"), "YOLOv10l [27] 77.4 76.8 77.1 25.7 126.3", "yolo", "baseline of FIDC-YOLO"),
    ("model7", "Mamba-YOLO", "other_baseline", "other", "Mamba-YOLO", ("80.0", "77.4", "78.7", "57.6", "155.9"), "Mamba-YOLO [19] 80.0 77.4 78.7 57.6 155.9", "yolo", "state-space model"),
    ("model8", "YOLO-PWD (reproduced on YOLOv5m)", "cnn_baseline", "cnn", "YOLOv5m", ("75.7", "73.0", "74.3", "42.4", "61.6"), "YOLO-PWD [10] 75.7 73.0 74.3 42.4 61.6", "yolo", ""),
    ("model9", "L-FIDC-YOLO", "proposed", "other", "lightweight FIDC-YOLO on YOLOv10n", ("75.9", "76.2", "76.0", "2.9", "9.1"), "L-FIDC-YOLO(Ours) 75.9 76.2 76.0 2.9 9.1", "own", "lightweight variant"),
]
res = Counter("res")
for item, name, role, cat, bb, v, ev, kind, note in M:
    if kind == "own":
        ROWS += model(item, name, role, cat, bb, "none", "from_scratch", 7, T1, ev, cat_note=note, pre_page=7, pre_loc="Sec. IV-B",
                      pre_ev="We train the model from scratch using the Stochastic Gradient Descent (SGD) optimizer", strat_page=7, strat_loc="Sec. IV-B",
                      strat_ev="We train the model from scratch using the Stochastic Gradient Descent (SGD) optimizer")
    elif kind == "det":
        ROWS += model(item, name, role, cat, bb, None, "full_fine_tuning", 7, T1, ev, cat_note=note, **DET)
    else:
        ROWS += model(item, name, role, cat, bb, None, None, 7, T1, ev, cat_note=note, **YOLO)
    ROWS += [R("total_params", v[3], 7, T1, ev, item=item, unit="M"), R("flops", v[4], 7, T1, ev, item=item, unit="GFLOPs")]
    ROWS += result(res(), "ap50", item, "dataset1", v[2], 7, T1, ev, setting="PWD test set; mean of diseased and dead tree AP", headline=(item == "model1"),
                   note="mAP at IoU 0.50 (Sec. IV-C)")
ROWS += avail()
ROWS += [D("notes", "FIDC-YOLO uses a Mamba (Hydra) module rather than attention; the transformer models in the paper are the DETR-type baselines RT-DETR and DINO. Results on the DIOR benchmark (Table II) are not tree data.", "Sec. IV-D and Table I (pp. 7 to 8)")]
