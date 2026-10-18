import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P134", "Table 3"
# Dual-Scale Interaction (DSI: intra-scale spatial self-attention with cross-channel fusion, and cross-scale interaction) added to FPN-based detectors (YOLOv5, YOLOv7, YOLOv8, YOLOv9 and RT-DETR) for the detection of
# pine wilt disease trees at three infection stages (early, middle, late) in UAV RGB images. Table 3 is the main comparison (14 models; mAP@50 over the three stage classes, AP@50 per stage and ms per frame): all 14 rows are
# recorded (mAP@50 only) and primary, DSI-YOLOv9 carries the headline. The test set (8:1:1 split) is evaluated. Ablation Table 4 and the altitude groups of Table 5 (DSI-YOLOv8) are not recorded.
TH = Tags(task="health_classification", level="object", split="test", agg="single")
OBJ_EV = "Accurate multi-stage detection of pine wilt disease in trees is critical"
SPLIT_EV = "We split the dataset into training set, validation set, and test set by the ratio of"
ROWS = [
    R("task", "health_classification", 2, "Abstract", OBJ_EV, note="multi-class detection of pine wilt disease trees by infection stage (early, middle, late)"),
    R("output_unit", "tree_instance", 5, "Sec. 2.3", "the dataset was annotated using DarkLabel", note="boxes of infected trees with the infection stage as class"),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "The dominant vegetation types are Masson pine and Metasequoia coniferous forests", note="forest coverage above 80 percent; forest type otherwise not described"),
    R("tree_taxa", "Masson pine (dominant) and Metasequoia", 4, "Sec. 2.1", "with Masson pine being the dominant tree species"),
    R("study_area", "Fuzhou City, Fujian Province, southeastern China (about 85 hectares)", 4, "Sec. 2.1", "The study site is situated in Fuzhou City, Fujian Province, in southeastern China"),
    D("biome", "subtropical", "Fuzhou (26.3 N), Fujian; not stated in the paper"),
    R("platform", "uav", 5, "Sec. 2.2", "The DAJI Mini3 pro was used to conduct cruise flights in the experimental sites from October 8 to 15, 2023", note="DJI Mini 3 Pro; flight altitudes 62 to 500 m (Sec. 4.3)"),
    R("sensors", "rgb", 5, "Sec. 2.2", "Colour mode RGB", note="48 million pixel camera, 4K video frames"),
    R("annotation_type", "bounding_box", 5, "Sec. 2.3", "the dataset was annotated using DarkLabel", note="boxes drawn after field investigations and with forestry experts"),
    R("dataset_name", "own UAV RGB video frames of PWD trees at three infection stages, Fuzhou (DJI Mini 3 Pro)", 5, "Sec. 2.2", "8,443 images with dimensions of 3840", item="dataset1"),
    R("dataset_key", "own_p134_fuzhou_pwd", 5, "Sec. 2.2", "8,443 images with dimensions of 3840", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.3", SPLIT_EV, item="dataset1", note="split 8:1:1 into training, validation and test sets"),
    R("dataset_public", "on_request", 21, "Data availability statement", "Data can be obtained by contacting the corresponding authors", item="dataset1"),
    R("n_images", "8443", 5, "Sec. 2.2", "8,443 images with dimensions of 3840", item="dataset1", unit="images", note="3840 x 2160 frames extracted every 30 frames from about 2.5 hours of 4K video"),
    D("n_trees", "77592", "sum of the printed samples per stage: 16,870 early, 34,056 middle and 26,666 late-stage infected trees (Sec. 2.3 and Table 2)", item="dataset1", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_page=13, strat_loc="Sec. 3.5", strat_ev="The model was trained for 100 epochs with a batch size of 8 samples",
          strat_note="SGD with momentum 0.937 and weight decay 5e-4, learning rate 1e-3 decaying, 100 epochs, batch size 8, 640 x 640 inputs; initialisation not stated")
DSI = "adds the dual-scale interaction: spatial self-attention with cross-channel fusion at the highest semantic FPN layer and cross-scale interaction at the FPN fusion points"
DSICAT = "convolutional detector with a spatial self-attention layer (query, key, value) added by the dual-scale interaction (protocol 4.5: self-attention layers added)"
MODELS = [  # table label, model name, role, category, backbone, category note
    ("RetinaNet", "RetinaNet", "cnn_baseline", "cnn", "RetinaNet", "convolutional one-stage detector"),
    ("SSD", "SSD", "cnn_baseline", "cnn", "SSD", "convolutional one-stage detector"),
    ("YOLOv5", "YOLOv5", "cnn_baseline", "cnn", "YOLOv5 (FPN-based)", "convolutional detector"),
    ("YOLOv7", "YOLOv7", "cnn_baseline", "cnn", "YOLOv7 (FPN-based)", "convolutional detector"),
    ("Faster-RCNN", "Faster R-CNN", "cnn_baseline", "cnn", "Faster R-CNN", "convolutional two-stage detector"),
    ("DETR", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "DETR (transformer encoder and decoder)", "DETR with a CNN backbone (protocol 4.5)"),
    ("RT-DETR", "RT-DETR", "transformer_baseline", "cnn_transformer_hybrid", "RT-DETR (FPN-based)", "RT-DETR with a CNN backbone (protocol 4.5)"),
    ("YOLOv8", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8 (FPN-based)", "convolutional detector"),
    ("YOLOv9", "YOLOv9", "cnn_baseline", "cnn", "YOLOv9 (FPN-based)", "convolutional detector"),
    ("DSI-YOLOv5", "DSI-YOLOv5", "proposed", "cnn_transformer_hybrid", "YOLOv5 " + DSI, DSICAT),
    ("DSI-YOLOv7", "DSI-YOLOv7", "proposed", "cnn_transformer_hybrid", "YOLOv7 " + DSI, DSICAT),
    ("DSI-RT-DETR", "DSI-RT-DETR", "proposed", "cnn_transformer_hybrid", "RT-DETR " + DSI, "RT-DETR (hybrid) with the dual-scale interaction (spatial self-attention) added"),
    ("DSI-YOLOv8", "DSI-YOLOv8", "proposed", "cnn_transformer_hybrid", "YOLOv8 " + DSI, DSICAT),
    ("DSI-YOLOv9", "DSI-YOLOv9", "proposed", "cnn_transformer_hybrid", "YOLOv9 " + DSI, DSICAT),
]
res = Counter("res")
RID = {}
for k, (lab, name, role, cat, bb, cnote) in enumerate(MODELS, start=1):
    item = f"model{k}"
    nums, ev = row(PID, 14, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 5, (lab, nums)  # mAP@50, AP@50 early, middle, late, ms per frame
    ROWS += model(item, name, role, cat, bb, None, None, 14, T3, ev, cat_note=cnote, **NP)
    ROWS.append(R("inference_speed", f"{nums[4]} ms per frame (hardware of the measurement not stated; training PC with an RTX 2080Ti, inputs resized to 640 x 640)", 14, T3, ev, item=item))
    RID[item] = res()
    ROWS += result(RID[item], "ap50", item, "dataset1", nums[0], 14, T3, ev, unit="%", setting="Table 3: test set, three infection stage classes", note="mAP@50 over the early, middle and late stage classes",
                   **TH, primary=True, headline=(lab == "DSI-YOLOv9"))
ROWS += [R("limitations", "Most background targets are still identified as infected trees, some infected trees are not detected, and the gain of the method diminishes as the accuracy of the original model increases", 20, "Sec. 5",
           "Although the improvement effect diminishes as the original model accuracy increases")]
ROWS += avail(code="no", weights="not_stated", data="on_request", data_page=21, data_loc="Data availability statement", data_ev="Data can be obtained by contacting the corresponding authors")
ROWS += [D("notes", "Table 3 compares RetinaNet, SSD, Faster R-CNN, DETR, YOLOv5, YOLOv7, YOLOv8, YOLOv9 and RT-DETR with their dual-scale interaction (DSI) versions on the test set (mAP@50, AP@50 of the three stages, ms per frame). All 14 rows are recorded (mAP@50 only; the stage-wise AP@50 are not recorded) and primary; the abstract's headline is DSI-YOLOv9 (96.4). DSI-YOLOv5 has the largest gain (+13.8 points, 62.9 to 76.7), the gain of RT-DETR is the smallest (+0.7). The DSI models are recorded as cnn_transformer_hybrid because the intra-scale interaction is a spatial self-attention layer with query, key and value matrices (a judgement call: a single self-attention layer, not a multi-head transformer block). Table 4 (ablation of ISI, CSI1 and CSI2 in YOLOv8) and Table 5 (DSI-YOLOv8 in four flight altitude groups of the test set, mAP@50 93.1 to 96.1) are not recorded. The dataset has 16,870 early, 34,056 middle and 26,666 late-stage samples.",
           "Sec. 3 and 4, Tables 2 to 5 (pp. 5 to 20)")]
