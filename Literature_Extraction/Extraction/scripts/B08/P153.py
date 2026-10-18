import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T5 = "P153", "Table 3", "Table 5"
# YOLOv8-MFD: YOLOv8 with a MobileViT (CNN and transformer) backbone, Focal Modulation and a Dynamic Head for pine wilt disease (PWD) infected trees in UAV RGB images. Table 5 is the main comparison (seven models:
# precision, recall, F1, mAP@0.5, model size, GFLOPS, FPS): F1 and mAP@0.5 are recorded (precision and recall are not, F1 and AP are given) with GFLOPS and FPS and are primary for every row; the headline is the
# proposed model. The test set (8:2 split of the images) is the evaluation set: the paper says the test set ensures reliable and consistent model evaluation. Table 3 (YOLOv8 with other backbones; its YOLOv8_Base row repeats
# YOLOv8 of Table 5) is recorded without primary; the module ablation of Table 4 is not recorded (its last row repeats the proposed model of Table 5).
TH = Tags(task="health_classification", level="object", split="test", agg="single")
TASK_EV = "we propose an improved object detection model, YOLOv8-MFD, designed for accurate and efficient detection of PWD-infected trees from UAV imagery"
BOX_EV = "precise annotation of PWD-infected trees using the LabelImg tool"
SPLIT_EV = "The training set comprises 16,552 images (corresponding to 4138 original images), while the test set consists of 5191 images (from 1298 original images), ensuring reliable and consistent model evaluation"
ROWS = [
    R("task", "health_classification", 1, "Abstract", TASK_EV, note="detection of pine wilt disease infected trees of four stages (early, middle, late, death) as one class"),
    R("output_unit", "tree_instance", 6, "Sec. 2.2", BOX_EV),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "The study area is located in Shaanxi Province, China", note="forested area; forest type is not described"),
    R("tree_taxa", "pine trees infected by pine wilt disease (the area description names Cunninghamia lanceolata and Abies spp. as dominant species)", 4, "Sec. 2.1", "The dominant tree species in the area are Cunninghamia lanceolata (China fir) and Abies spp. (firs)"),
    R("study_area", "Shaanxi Province, China (about 1000 km2 flown)", 4, "Sec. 2.1", "The study area is located in Shaanxi Province, China", note="Sec. 2.2 names Shanxi Province for the same forested area"),
    R("biome", "subtropical", 4, "Sec. 2.1", "This region features a subtropical monsoon climate"),
    R("platform", "uav", 5, "Sec. 2.2", "a DJI UAV equipped with an onboard camera was used as the flight platform", note="altitudes 80 to 300 m, July to September 2024"),
    R("sensors", "rgb", 5, "Sec. 2.2", "high-resolution RGB images equipped with a Global Positioning System (GPS)"),
    R("annotation_type", "bounding_box", 6, "Sec. 2.2", BOX_EV),
    R("dataset_name", "own UAV RGB dataset of PWD trees in four stages, Shaanxi (six orthoimages, 640 x 640 sub-images)", 5, "Sec. 2.2", "This study utilized 5436 original UAV-acquired images of Pine Wilt Disease (PWD)-", item="dataset1"),
    R("dataset_key", "own_p153_shaanxi_pwd", 5, "Sec. 2.2", "This study utilized 5436 original UAV-acquired images of Pine Wilt Disease (PWD)-", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 2.2", "The dataset was randomly partitioned into training and test sets at a ratio of 8:2", item="dataset1", note="training 16,552 augmented images (4138 originals), test 5191 images (1298 originals); no validation set is described"),
    R("dataset_public", "no", 20, "Data Availability Statement", "The data are not publicly available due to privacy", item="dataset1"),
    R("n_images", "5436", 5, "Sec. 2.2", "This study utilized 5436 original UAV-acquired images of Pine Wilt Disease (PWD)-", item="dataset1", unit="images", note="640 x 640 pixel sub-images of six orthoimages; augmentation multiplies the images by four"),
    R("n_trees", "21774", 6, "Sec. 2.2", "The resulting labeled dataset includes 21,774 annotated instances of PWD-infected trees", item="dataset1", unit="trees", note="annotated instances after augmentation"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="Adam, initial learning rate 0.001, batch size 16, 150 epochs, 640 x 640 inputs (Sec. 2.4.1); initialisation not stated")
M5 = [  # row regex, name, role, category, backbone, category note
    (r"YOLOv3", "YOLOv3", "cnn_baseline", "cnn", "YOLOv3", "convolutional YOLO detector"),
    (r"YOLOv5", "YOLOv5", "cnn_baseline", "cnn", "YOLOv5", "convolutional YOLO detector"),
    (r"YOLOv8", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8", "convolutional YOLO detector; the baseline that YOLOv8-MFD is built on"),
    (r"RT-DETR", "RT-DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", "RT-DETR (protocol 4.5: hybrid)"),
    (r"YOLOv9", "YOLOv9", "cnn_baseline", "cnn", "YOLOv9", "convolutional YOLO detector"),
    (r"YOLOv10", "YOLOv10", "cnn_baseline", "cnn", "YOLOv10", "convolutional YOLO detector"),
    (r"Proposed", "YOLOv8-MFD (proposed)", "proposed", "cnn_transformer_hybrid", "MobileViT v2 backbone (convolutions, MV2 blocks and MobileViT blocks with transformer layers) in YOLOv8, with Focal Modulation and a Dynamic Head",
     "YOLOv8 with a MobileViT backbone that fuses CNN and transformer global modeling (protocol 4.5: hybrid)"),
]
res = Counter("res")
for k, (rx, name, role, cat, bb, cnote) in enumerate(M5, start=1):
    item = f"model{k}"
    nums, line = row(PID, 17, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(nums) == 7, (name, nums)  # precision, recall, F1, mAP@0.5, size (MB), GFLOPS, FPS
    ROWS += model(item, name, role, cat, bb, None, None, 17, T5, line, cat_note=cnote, **NP)
    ROWS += [R("flops", nums[5], 17, T5, line, item=item, unit="GFLOPs"), R("inference_speed", f"{nums[6]} FPS (RTX 4060 Ti, 640 x 640 inputs)", 17, T5, line, item=item)]
    head = rx == "Proposed"
    ROWS += result(res(), "ap50", item, "dataset1", nums[3], 17, T5, line, unit="fraction", setting="Table 5: test set", note="mAP@0.5; precision and recall of the row and the model size in MB are not recorded", **TH, primary=True, headline=head)
    ROWS += result(res(), "f1", item, "dataset1", nums[2], 17, T5, line, unit="fraction", setting="Table 5: test set", note="F1 of the row", **TH, primary=True)
# Table 3: YOLOv8 with other backbones (the Base row repeats YOLOv8 of Table 5), not primary
B3 = [(r"YOLOv8_EfficientNetv1", "YOLOv8 with EfficientNetv1 backbone", "cnn", "EfficientNetv1", "YOLOv8 with a convolutional EfficientNet backbone"),
      (r"YOLOv8_EfficientViT", "YOLOv8 with EfficientViT backbone", "cnn_transformer_hybrid", "EfficientViT", "YOLOv8 with an EfficientViT backbone (EfficientViT inside a CNN detector is hybrid, protocol 4.5)"),
      (r"YOLOv8_MobileNetv1", "YOLOv8 with MobileNetv1 backbone", "cnn", "MobileNetv1", "YOLOv8 with a convolutional MobileNet backbone"),
      (r"YOLOv8_MobileNetv2", "YOLOv8 with MobileNetv2 backbone", "cnn", "MobileNetv2", "YOLOv8 with a convolutional MobileNet backbone"),
      (r"YOLOv8_MobileViTv2", "YOLOv8 with MobileViTv2 backbone only", "cnn_transformer_hybrid", "MobileViT v2", "YOLOv8 with the MobileViT v2 backbone, without Focal Modulation and Dynamic Head (protocol 4.5: hybrid)")]
for j, (rx, name, cat, bb, cnote) in enumerate(B3):
    item = f"model{len(M5) + 1 + j}"
    nums, line = row(PID, 13, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(nums) == 5, (name, nums)  # precision, recall, F1, mAP@0.5, size (MB)
    ROWS += model(item, name, "other_baseline", cat, bb, None, None, 13, T3, line, cat_note=cnote, **NP)
    ROWS += result(res(), "ap50", item, "dataset1", nums[3], 13, T3, line, unit="fraction", setting="Table 3: backbone comparison in YOLOv8, test set", note="mAP@0.5; precision, recall and model size of the row are not recorded", **TH)
    ROWS += result(res(), "f1", item, "dataset1", nums[2], 13, T3, line, unit="fraction", setting="Table 3: backbone comparison in YOLOv8, test set", note="F1 of the row", **TH)
ROWS += [R("limitations", "Trees in the death stage (no needles, pale dry branches, like other dead trees) are missed, and that stage is underrepresented in the training data", 18, "Sec. 4.2.1",
           "the model still exhibits certain limitations in identifying trees in the final (fourth) stage of Pine Wilt Disease, namely, the death stage")]
ROWS += avail(code="no", weights="not_stated", data="no", data_page=20, data_loc="Data Availability Statement", data_ev="The data are not publicly available due to privacy")
ROWS += [D("notes", "Table 5 compares YOLOv3, YOLOv5, YOLOv8, RT-DETR, YOLOv9, YOLOv10 and the proposed YOLOv8-MFD (precision, recall, F1, mAP@0.5, size, GFLOPS, FPS) on the test set: F1 0.884 and mAP@0.5 0.882 for the proposed model (primary, headline; precision and recall are not recorded), RT-DETR is the weakest (F1 0.637). Table 3 compares six backbones in YOLOv8 (EfficientNetv1, EfficientViT, MobileNetv1, MobileNetv2, MobileViTv2; recorded without primary, the Base row repeats YOLOv8 of Table 5); Table 4 (ablation of MobileViT, Focal Modulation and Dynamic Head; its last row repeats the proposed model) is not recorded. The evaluation set is the test set of an 8:2 split of augmented images (the split is by original image). Sec. 2.1 names Shaanxi Province and Sec. 2.2 Shanxi Province for the same area (Sec. 2.1 is used). The data are not public.",
           "Sec. 2 and 3, Tables 1 and 3 to 5 (pp. 4 to 17)")]
