import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P136", "Table 3"
# Early-stage pine wilt disease (PWD) detection in UAV RGB images by fusing frequency-domain (DCT) and spatial-domain features (CBAM-enhanced frequency features, deformable-convolution fusion) added to YOLO detectors.
# No transformer is proposed: the only transformer is the DETR (Res101) baseline of Table 3 (K8). Table 3 is the main comparison: 16 rows (Faster R-CNN, DETR, YOLOv3, YOLOv4 and YOLOv5n/s/m/l/x, each YOLO with and without the
# frequency branch). All 16 rows are recorded (AP0.5 and F1, level object, the validation set of the early-stage dataset) and primary; the headline is the best row (K18, YOLOv5m with frequency features, Sec. 4.1).
# Ablation Tables 1 and 2 and the held-out plot Y-0 (qualitative, Fig. 10) are not recorded.
TH = Tags(task="health_classification", level="object", split="validation", agg="single")
OBJ_EV = "we propose a detection method based on deep learning for early-stage PWD by comprehensively utilizing the features in the frequency domain and the spatial domain"
BOX_EV = "which annotates the early-stage PWD targets in the original images, recording the bounding box positions"
PLAT_EV = "using a DJI Elf 4 Pro V2.0 drone equipped with an optical camera"
IMG_EV = "A total of 547 aerial images containing red pine were obtained"
ROWS = [
    R("task", "health_classification", 1, "Abstract", OBJ_EV, note="detection of pine trees with early-stage pine wilt disease (needles fade from green to yellow-green)"),
    R("output_unit", "tree_instance", 4, "Sec. 2.2", BOX_EV),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "most of which are mixed broadleaf", note="the forest is mostly mixed broadleaf and conifer forest; no further description"),
    R("tree_taxa", "pine trees with early-stage pine wilt disease (red pine as printed; species not named)", 4, "Sec. 2.1", IMG_EV),
    R("study_area", "Dahuofang Experimental Forest, Fushun, Liaoning Province, China", 4, "Sec. 2.1", "The study area for our study is located in the Dahuofang Experimental Forest in Fushun, Liaoning Province, China"),
    D("biome", "temperate", "Fushun, Liaoning; not stated in the paper"),
    R("platform", "uav", 4, "Sec. 2.1", PLAT_EV, note="flight altitude 100 to 240 m, August 2020 and April to August 2021"),
    R("sensors", "rgb", 4, "Sec. 2.1", PLAT_EV, note="JPEG images of 5472 x 3648 pixels; the frequency-domain input is computed from the RGB images by DCT"),
    R("annotation_type", "bounding_box", 4, "Sec. 2.2", BOX_EV, note="LabelImg 3.18"),
    R("dataset_name", "own UAV RGB images of early-stage PWD, Dahuofang Experimental Forest (plots Y-0 to Y-3 and PY-4)", 4, "Sec. 2.1", IMG_EV, item="dataset1"),
    R("dataset_key", "own_p136_dahuofang_early", 4, "Sec. 2.1", IMG_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 2.2", "consisting of 718 images in the training set and 436 images in the validation set", item="dataset1",
      note="plot Y-0 (20 images) is held out and only used to assess generalisation (Fig. 10, qualitative); there is no separate test set"),
    R("dataset_public", "on_request", 16, "Data Availability Statement", "available on request from the corresponding author, H.W., upon reasonable request", item="dataset1"),
    R("n_images", "547", 4, "Sec. 2.1", IMG_EV, item="dataset1", unit="images", note="527 images of Y-1, Y-2, Y-3 and PY-4 and 20 of Y-0; after cropping to 2500 x 1500 there are 718 training and 436 validation images"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="SGD with momentum 0.937, learning rate 1e-2, batch size 16 for training, 200 epochs (Sec. 3.1); initialisation not stated")
OURS = "with the frequency branch of the paper: DCT frequency features enhanced by CBAM, fused with the spatial features by deformable convolution (FFM)"
OURS_CAT = "convolutional detector; CBAM is a generic attention module and does not make a model transformer-based (protocol 4.5)"
MODELS = [  # table label, name, role, category, backbone, category note
    ("K7", "Faster R-CNN (Res101)", "cnn_baseline", "cnn", "ResNet-101", "convolutional two-stage detector"),
    ("K8", "DETR (Res101)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-101 with transformer encoder and decoder", "DETR with a ResNet-101 backbone; the paper states DETR is built on the Transformer architecture (p.13)"),
    ("K9", "YOLOv3", "cnn_baseline", "cnn", "YOLOv3", "convolutional detector"),
    ("K10", "YOLOv3 with frequency features (ours, K10)", "proposed", "cnn", "YOLOv3 " + OURS, OURS_CAT),
    ("K11", "YOLOv4", "cnn_baseline", "cnn", "YOLOv4", "convolutional detector"),
    ("K12", "YOLOv4 with frequency features (ours, K12)", "proposed", "cnn", "YOLOv4 " + OURS, OURS_CAT),
    ("K13", "YOLOv5n", "cnn_baseline", "cnn", "YOLOv5n", "convolutional detector"),
    ("K14", "YOLOv5n with frequency features (ours, K14)", "proposed", "cnn", "YOLOv5n " + OURS, OURS_CAT),
    ("K15", "YOLOv5s", "cnn_baseline", "cnn", "YOLOv5s", "convolutional detector"),
    ("K16", "YOLOv5s with frequency features (ours, K16)", "proposed", "cnn", "YOLOv5s " + OURS, OURS_CAT),
    ("K17", "YOLOv5m", "cnn_baseline", "cnn", "YOLOv5m", "convolutional detector"),
    ("K18", "YOLOv5m with frequency features (ours, K18)", "proposed", "cnn", "YOLOv5m " + OURS, OURS_CAT),
    ("K19", "YOLOv5l", "cnn_baseline", "cnn", "YOLOv5l", "convolutional detector"),
    ("K20", "YOLOv5l with frequency features (ours, K20)", "proposed", "cnn", "YOLOv5l " + OURS, OURS_CAT),
    ("K21", "YOLOv5x", "cnn_baseline", "cnn", "YOLOv5x", "convolutional detector"),
    ("K22", "YOLOv5x with frequency features (ours, K22)", "proposed", "cnn", "YOLOv5x " + OURS, OURS_CAT),
]
res = Counter("res")
for k, (lab, name, role, cat, bb, cnote) in enumerate(MODELS, start=1):
    item = f"model{k}"
    nums, line = row(PID, 14, rf"^\s*{lab}\s*(?:\(ours\))?\s+(?=\S)", 0)
    assert len(nums) == 2, (lab, nums)  # AP0.5, F1
    ev = "Faster- ... " + line + " ... RCNN(Res101)" if lab == "K7" else line
    ROWS += model(item, name, role, cat, bb, None, None, 14, T3, ev, cat_note=cnote, **NP)
    for metric, v in (("ap50", nums[0]), ("f1", nums[1])):
        ROWS += result(res(), metric, item, "dataset1", v, 14, T3, ev, unit="%", setting="Table 3: validation set of the early-stage PWD dataset", note=("AP at IoU 0.5 (printed AP0.5)" if metric == "ap50" else ""),
                       **TH, primary=True, headline=(lab == "K18" and metric == "ap50"))
ROWS += [R("limitations", "The cause of the colour change of a detected pine cannot be determined by the model (drought also changes the colour), and trees with incomplete shapes at the image edges are detected worse", 15, "Sec. 4.3",
           "the specific cause of the color change in pine trees cannot be determined solely through model detection")]
ROWS += avail(code="no", weights="not_stated", data="on_request", data_page=16, data_loc="Data Availability Statement", data_ev="available on request from the corresponding author, H.W., upon reasonable request")
ROWS += [D("notes", "Table 3 compares Faster R-CNN (Res101), DETR (Res101), YOLOv3, YOLOv4 and YOLOv5n/s/m/l/x with and without the frequency branch on the validation set (AP0.5 and F1). All 16 rows are recorded and primary; the headline is the best row (K18, YOLOv5m with frequency features, AP 84.9, F1 79.0 as stated in Sec. 4.1). The proposed method is convolutional (CBAM is a generic attention module); the only transformer is the DETR baseline (K8, AP 78.7, F1 75.1), which the paper says has precision 2.3 percent above YOLOv5s. The ablation of Tables 1 and 2 (K0 to K6 and METD, where METD has the values of K16) and the held-out plot Y-0 (Fig. 10, qualitative only) are not recorded. The evaluation set is called the validation set; there is no separate test set.",
           "Sec. 3 and 4, Tables 1 to 3 (pp. 9 to 15)")]
