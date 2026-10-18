import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T3, T4 = "P127", "Table 3", "Table 4"
ROWS = [
    R("task", "detection;health_classification", 3, "Sec. 2 (Study area; Data acquisition and marking)", "infected pines were divided into two categories: 'diseased tree' and 'dead tree'",
      note="boxes for pine wilt diseased and dead trees"),
    R("output_unit", "tree_instance;tree_label", 3, "Sec. 2 (Study area; Data acquisition and marking)", "Annotation included the coordinates of the rectangular bounding box around the infected tree and the category of the infection stage"),
    R("tree_setting", "natural_forest", 3, "Sec. 2 (Study area; Data acquisition and marking)", "The Huangshan area is a famous scenic spot with a forest cover more than 70%"),
    R("tree_taxa", "Pinus hwangshanensis; Pinus massoniana; Pinus thunbergii", 3, "Sec. 2 (Study area; Data acquisition and marking)", "There are a several pine species in the area, including Pinus hwangshanensis, Pinus massoniana, and Pinus thunbergii"),
    R("study_area", "Huangshan, southern Anhui Province, China", 3, "Sec. 2 (Study area; Data acquisition and marking)", "Huangshan was selected as the research area and is in Huangshan City, southern Anhui Province"),
    R("biome", "subtropical", 3, "Sec. 2 (Study area; Data acquisition and marking)", "It has a subtropical monsoon climate"),
    R("platform", "uav", 3, "Sec. 2 (Study area; Data acquisition and marking)", "The flight platform was M300RTK with a Zen p1 lens"),
    R("sensors", "rgb", 3, "Sec. 2 (Study area; Data acquisition and marking)", "UAV took visible light remote sensing images October 18th and 19th"),
    R("gsd", "6.15", 3, "Sec. 2 (Study area; Data acquisition and marking)", "The pixel count was 45 million, the ground resolution 6.15 cm", unit="cm"),
    R("annotation_type", "bounding_box", 3, "Sec. 2 (Study area; Data acquisition and marking)", "the open-source annotation tool LabelImg used to annotate infected trees in the thirty images"),
    R("dataset_name", "own UAV pine wilt dataset, Huangshan (640 x 640 tiles)", 3, "Sec. 2 (Study area; Data acquisition and marking)", "A total of 2478 images with a resolution of 640 × 640 pixels were cut out and made into data sets", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2 (Study area; Data acquisition and marking)", "According to the ratio 8:1:1, 1966 training sets, 256 validation sets, and 256 test sets were obtained", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    R("n_images", "2478", 3, "Sec. 2 (Study area; Data acquisition and marking)", "A total of 2478 images with a resolution of 640 × 640 pixels were cut out", item="dataset1", unit="images"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained 200 epochs with SGD (p.8); initialisation not stated")
res = Counter("res")
T4M = [("model1", r"A\+B\+C\+D", "Improved YOLOv8 (MHSA, SPD-Conv, GD neck, Focal-EIoU)", "proposed", "cnn_transformer_hybrid", "YOLOv8 with multi-head self-attention in the backbone", "proposed model (A+B+C+D)"),
       ("model2", r"A\+B", "YOLOv8 with MHSA (A+B)", "transformer_baseline", "cnn_transformer_hybrid", "YOLOv8 with multi-head self-attention in the backbone", "ablation: attention only"),
       ("model3", r"A", "YOLOv8 (benchmark model A)", "cnn_baseline", "cnn", "YOLOv8", "same-framework baseline")]
for item, lab, name, role, cat, bb, note in T4M:
    nums, ev = row(PID, 9, rf"^{lab}\s+(?=\d)", 0, cols=(0, 72))
    assert len(nums) == 5, (item, nums, ev)
    ROWS += model(item, name, role, cat, bb, None, None, 9, T4, ev, cat_note=note, **NP)
    ROWS += result(res(), "ap50", item, "dataset1", nums[0], 9, T4, ev, unit="fraction", setting="test set; diseased and dead trees", headline=(item == "model1"))
T3M = [("model4", "YOLOv5"), ("model5", "SSD-VGG16"), ("model6", "YOLOX"), ("model7", "YOLOv4"), ("model8", "YOLOv7"), ("model9", "EfficientDet"), ("model10", "Faster-RCNN")]
for item, lab in T3M:
    nums, ev = row(PID, 8, rf"(?<![\w-]){re.escape(lab)}\s+(?=\d)", 0)
    ev = ev[ev.index(lab):]  # drop left-column text on the same printed line
    assert len(nums) == 5, (item, nums, ev)
    ROWS += model(item, lab, "cnn_baseline", "cnn", lab, None, None, 8, T3, ev, **NP)
    ROWS += result(res(), "ap50", item, "dataset1", nums[0], 8, T3, ev, unit="fraction", setting="test set; diseased and dead trees")
ROWS += avail()
ROWS += [D("notes", "The proposed model appears only in the ablation (Table 4, A+B+C+D, mAP50 0.785); Table 3 lists the other detectors. Precision, recall and per-class mAP are in Tables 3 to 4.", "Sec. 3 and Tables 3 to 4 (pp. 8 to 9)")]
