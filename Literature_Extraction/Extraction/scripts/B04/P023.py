import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T4 = "P023", "Table 2", "Table 4"
DS_EV = "We divide the 218 images into 1471 patches with certain size and mark 87135 trees from them"
MAP = "M1, M2, M3, M4, and M5 correspond to swin transformer, YOLO, RCNN, faster RCNN, and DETR, respectively"
# Tree detection with boxes on the test patches; AP 0.50-0.95 and AP 0.50 are COCO-style (IoU steps of 0.05 from 0.50; IoU 0.50). AP (Small) is not recorded.
T = Tags(task="detection", level="object", split="test", agg="single")

ROWS = [
    R("task", "detection", 1, "Abstract", "we picked the swin transformer, RCNN, faster RCNN, YOLO, and DETR for tree detection from satellite images"),
    R("output_unit", "tree_instance", 2, "Sec. 1", "considering the marked boxes and their confidence score of different methods on the same image"),
    R("tree_setting", "mixed", 4, "Sec. 4.1", "It contains mostly beech, oak, pine and olive trees, as well as shrubs"),
    R("tree_taxa", "beech, oak, pine and olive trees, and shrubs", 4, "Sec. 4.1", "It contains mostly beech, oak, pine and olive trees, as well as shrubs"),
    R("study_area", "Turkey (mainly Bursa and Izmir)", 4, "Sec. 4.1", "satellite images taken from different regions of Turkey ... We mainly collected images from Bursa and Izmir"),
    D("biome", "mediterranean", "Bursa and Izmir, Turkey; not stated in the paper"),
    R("platform", "satellite", 4, "Sec. 4.1", "The dataset includes satellite images taken from different regions of Turkey"),
    D("sensors", "rgb", "imagery type not stated; the sample images in Fig. 1 (p.5) are colour images"),
    D("annotation_type", "bounding_box", "the marked trees are fused and scored as boxes (weighted boxes fusion, p.2 to p.4); the paper does not describe the annotation form"),
    R("dataset_name", "own satellite tree dataset, Turkey (Bursa, Izmir), 1471 patches", 4, "Sec. 4.1", DS_EV, item="dataset1"),
    R("dataset_key", "own_p023_turkey_trees", 4, "Sec. 4.1", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 5, "Table 1", "Images 1023 226 222 Trees 60356 13227 13552", item="dataset1"),
    R("dataset_public", "no", 8, "Data availability", "No datasets were generated or analysed during the current study", item="dataset1", note="data and code are announced as available upon publication"),
    R("n_images", "1471", 4, "Sec. 4.1", DS_EV, item="dataset1", unit="tiles", note="218 satellite images cut into patches: 1023 training, 226 validation and 222 test (Table 1)"),
    R("n_trees", "87135", 4, "Sec. 4.1", DS_EV, item="dataset1", unit="trees", note="60,356 training, 13,227 validation and 13,552 test trees (Table 1)"),
]
NPRE = dict(pre_status="not_reported", pre_note="MMDetection pretrained weights are mentioned (p.4); the pretraining data are not stated", strat_note="trained for 100 epochs in MMDetection; initialisation not stated")
SINGLE = [  # item, row label, name, role, category, backbone, note
    ("model1", "M1", "Swin transformer (M1)", "transformer_baseline", "hierarchical_vit", "swin transformer (MMDetection detector; head not stated)", "Swin transformer detector"),
    ("model2", "M2", "YOLO (M2, YOLOv3)", "cnn_baseline", "cnn", "YOLOv3", "YOLOv3 (p.3)"),
    ("model3", "M3", "RCNN (M3)", "cnn_baseline", "cnn", "backbone not stated", "region-based CNN"),
    ("model4", "M4", "Faster RCNN (M4)", "cnn_baseline", "cnn", "backbone not stated", "region-based CNN with a region proposal network"),
    ("model5", "M5", "DETR (M5, Deformable DETR)", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", "Deformable DETR is the DETR version used (p.3); its CNN backbone is not named"),
]
res = Counter("res")
for item, lab, name, role, cat, bb, note in SINGLE:
    nums, ev = row(PID, 5, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 3, (lab, nums)  # AP 0.50-0.95, AP 0.50, AP small
    ROWS += model(item, name, role, cat, bb, None, None, 5, "Sec. 4.3.2", MAP, cat_note=note, **NPRE)
    st = "Table 2: single model, test set (222 patches, 13,552 trees)"
    ROWS += result(res(), "ap50_95", item, "dataset1", nums[0], 5, T2, ev, unit="fraction", setting=st, note="AP (0.50-0.95)", **T, primary=True)
    ROWS += result(res(), "ap50", item, "dataset1", nums[1], 5, T2, ev, unit="fraction", setting=st, note="AP (0.50)", **T, primary=True)
FUSION = [  # item, row label, name
    ("model6", "M1, M3", "WBF fusion of M1 and M3 (swin transformer and RCNN)"),
    ("model7", "M1, M2, M3", "WBF fusion of M1, M2 and M3 (swin transformer, YOLO and RCNN)"),
    ("model8", "M1, M2, M3, M5", "WBF fusion of M1, M2, M3 and M5 (swin transformer, YOLO, RCNN and DETR)"),
    ("model9", "M1, M2, M3, M4, M5", "WBF fusion of all five models (M1 to M5)"),
]
for item, lab, name in FUSION:
    nums, ev = row(PID, 6, rf"^\s*{lab}\s+(?=\d)", 0, cols=(63, 140))
    assert len(nums) == 3, (lab, nums)
    ROWS += model(item, name, "proposed", "cnn_transformer_hybrid", "weighted boxes fusion (post-processing) of the outputs of the listed detectors", None, None, 6, T4, "Fusion ... " + ev,
                  cat_note="ensemble of detectors fused at post-processing (weighted boxes fusion): a Swin transformer is always included with CNN detectors and DETR; not a single network", **NPRE)
    st = "Table 4: best fusion of its size, test set (222 patches, 13,552 trees)"
    ROWS += result(res(), "ap50_95", item, "dataset1", nums[0], 6, T4, ev, unit="fraction", setting=st, note="AP (0.50-0.95)", **T, primary=True, headline=(item == "model8"))
    ROWS += result(res(), "ap50", item, "dataset1", nums[1], 6, T4, ev, unit="fraction", setting=st, note="AP (0.50)", **T, primary=True)
ROWS += [R("code_available", "no", 8, "Code Availability", "Data and code will be available upon publication", note="announced, not yet released"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "no", 8, "Data availability", "No datasets were generated or analysed during the current study", note="data and code are announced as available upon publication")]
ROWS += [D("notes", "Table 2 (p.5) gives the five single detectors and Table 4 (p.6) the best fusion of each size (two to five models, always including the Swin transformer); Table 3 lists all 15 fusion combinations and is not recorded (AP (Small) is in all three tables). On the 13,552 test trees the Swin transformer detected 11,434 (84.4 percent) and the four-model fusion 12,353 (91.2 percent); the paper reports a 7.8 percent gain. The text names the four-model fusion as Swin transformer, faster RCNN, YOLO and DETR while Table 4 lists M1, M2, M3 and M5 (RCNN); Table 4 is followed.",
           "Sec. 4.3 to 4.4 and Tables 1 to 4 (pp. 5 to 7)")]
