import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T8, T9 = "P123", "Table 8 (image)", "Table 9 (image)"
# MFTD (YOLOv5 with visible and multispectral red-band branches fused by a multi-band feature fusion transformer, MFFT) detects pine trees infected with pine wilt disease at three infection stages on UAV images of
# Korean pines (KP, first listed and the dataset of the abstract's first result: primary) and Chinese pines (CP). Tables 8 and 9 (YOLO series and MFTD; their rows are interleaved in the text layer, so the values are read
# from the rendered page) are the main comparison; mAP@50 is the mean of the AP@50 of the early, middle and late stage classes (health_classification). The 30 percent part of the 7:3 split is called the validation set.
# More than ten rows per table: all four MFTD variants and the YOLOv5 models of the same sizes (RGB and red-band input) are recorded, YOLOv3 and YOLOv4 are in the notes.
TH = Tags(task="health_classification", level="object", split="validation", agg="single")
OBJ_EV = "a multi-band image-fusion infected pine tree detector (MFTD) based on deep learning was proposed"
SPLIT_EV = "used to divide the two datasets into training sets and validation sets"
ROWS = [
    R("task", "health_classification", 1, "Abstract", OBJ_EV, note="detection of pine trees infected by pine wilt disease and of the infection stage (early, middle, late)"),
    R("output_unit", "tree_instance", 7, "Sec. 2.3", "The annotation contents include the coordinates of the rectangular bounding boxes", note="boxes of infected pine trees with the infection stage as class"),
    R("tree_setting", "plantation", 3, "Sec. 2.1", "the majority are planted forests such as Korean pines (Pinus koraiensis), Chinese"),
    R("tree_taxa", "Korean pine (Pinus koraiensis) and Chinese pine (Pinus tabulaeformis)", 3, "Sec. 2.1", "pines (P. tabulaeformis) and larches (Larix spp.)"),
    R("study_area", "Dahuofang Experimental Forest Farm, Fushun, Liaoning, Northeast China", 3, "Sec. 2.1", "The experiment was conducted in the Dahuofang Experimental Forest Farm"),
    R("biome", "temperate", 3, "Sec. 2.1", "the region is in the intermediate temperate zone"),
    D("platform", "uav", "DJI Phantom 4 multispectral UAV (Sec. 2.2, Table 1)"),
    R("sensors", "rgb;multispectral", 5, "Table 1", "Red edge: 730 nm ± 16 nm", note="DJI Phantom 4 multispectral UAV: a visible camera and five multispectral bands; the models use the visible and the red-band images"),
    R("annotation_type", "bounding_box", 7, "Sec. 2.3", "The open-source annotation tool, LabelImage, was", note="boxes of infected pine trees at three infection stages in the visible and red-band images"),
    R("dataset_name", "own UAV visible and red-band images of Korean pines with pine wilt disease (KP dataset)", 5, "Table 2", "Image sets amount 180 150 239 135", item="dataset1"),
    R("dataset_key", "own_p123_kp_pwd", 5, "Table 2", "Image sets amount 180 150 239 135", item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 2.3", SPLIT_EV, item="dataset1", note="7:3 split into training and validation sets without overlap between images"),
    D("dataset_public", "no", "Data Availability Statement: Not applicable (p.21)", item="dataset1"),
    D("n_images", "389", "sum of the image sets of the Korean pine study areas 2 and 3 (150 and 239, Table 2); the text gives 389", item="dataset1", unit="images"),
    R("dataset_name", "own UAV visible and red-band images of Chinese pines with pine wilt disease (CP dataset)", 5, "Table 2", "Image sets amount 180 150 239 135", item="dataset2"),
    R("dataset_key", "own_p123_cp_pwd", 5, "Table 2", "Image sets amount 180 150 239 135", item="dataset2"),
    R("dataset_role", "train_test", 7, "Sec. 2.3", SPLIT_EV, item="dataset2", note="7:3 split into training and validation sets without overlap between images"),
    D("dataset_public", "no", "Data Availability Statement: Not applicable (p.21)", item="dataset2"),
    D("n_images", "315", "sum of the image sets of the Chinese pine study areas 1 and 4 (180 and 135, Table 2); the text gives 315", item="dataset2", unit="images"),
]
COCO = dict(pre_page=14, pre_loc="Sec. 3.1", pre_ev="for infected pine trees trained in this paper were pre-trained on the MS COCO dataset", pre_note="MS COCO pre-training",
            strat_page=14, strat_loc="Sec. 3.1", strat_ev="A total of 500 epochs", strat_note="SGD, 500 epochs, batch size 16, initialised from MS COCO weights")
YEV = "Table 8 YOLO series and MFTD"
MODELS = []  # key -> item
def add_model(key, name, role, cat, bb, ev, cnote):
    item = f"model{len(MODELS) + 1}"
    MODELS.append((key, item))
    if key.startswith("M"):
        pg, loc, ev = 1, "Abstract", "Firstly, the Halfway Fusion mode was adopted to fuse the network based on four YOLOv5 variants"
    else:
        pg, loc, ev = 9, "Sec. 2.5", "The lightweight object detection network YOLOv5 contains four variants: YOLOv5s,"
    ROWS.extend(model(item, name, role, cat, bb, "other", "full_fine_tuning", pg, loc, ev, cat_note=cnote, **COCO))
    return item
for v, sz in (("s", "small"), ("m", "medium"), ("l", "large"), ("x", "extra large")):
    add_model(f"M{v}", f"MFTD (YOLOv5{v}, visible and red-band fusion)" + (" (proposed, best)" if v == "l" else ""), "proposed", "cnn_transformer_hybrid", f"dual-branch YOLOv5{v} (visible and multispectral subnets) with MFFT modules before fusion",
              "K21(MFTD) v5s RGB + R" if v == "s" else f"MFTD v5{v}", "YOLOv5 convolutional backbone with a multi-band feature fusion transformer (multi-head attention and feed-forward network)")
for v in ("s", "m", "l", "x"):
    add_model(f"Y{v}RGB", f"YOLOv5{v} (visible RGB images)", "cnn_baseline", "cnn", f"YOLOv5{v} on visible images", f"YOLOv5{v}", "YOLOv5 convolutional detector")
for v in ("s", "m", "l", "x"):
    add_model(f"Y{v}R", f"YOLOv5{v} (red-band images)", "cnn_baseline", "cnn", f"YOLOv5{v} on red-band images", f"YOLOv5{v}", "YOLOv5 convolutional detector")
IT = dict(MODELS)
res = Counter("res")
RID = {}
# Table 8 (KP, p.17) and Table 9 (CP): label, mAP@50, AP early, middle, late, size MB, FPS
KP = {"Ms": ("K21(MFTD) v5s RGB + R", "80.8", "77.1", "84.6", "80.7", "89.6", "49.2"), "Mm": ("K22(MFTD) v5m RGB + R", "85.3", "84.3", "90.1", "81.5", "216.8", "40.2"),
      "Ml": ("K11(MFTD) v5l RGB + R", "88.5", "87.2", "93.5", "84.8", "413.4", "35.6"), "Mx": ("K23(MFTD) v5x RGB + R", "85.9", "84.7", "93.1", "80.0", "690.1", "30.4"),
      "YsRGB": ("K15 v5s RGB", "77.9", "71.3", "87.4", "75.1", "14.4", "146.3"), "YmRGB": ("K17 v5m RGB", "79.2", "73.4", "86.3", "77.9", "42.5", "97.1"),
      "YlRGB": ("K0 v5l RGB", "79.9", "72.9", "87.3", "78.6", "93.7", "86.1"), "YxRGB": ("K19 v5x RGB", "79.3", "75.2", "86.2", "76.5", "175.1", "58.9"),
      "YsR": ("K16 v5s R", "77.8", "69.5", "82.8", "81.1", "14.4", "146.3"), "YmR": ("K18 v5m R", "77.6", "72.7", "76.1", "84.0", "42.5", "97.1"),
      "YlR": ("K1 v5l R", "77.8", "70.6", "81.6", "81.3", "93.7", "86.1"), "YxR": ("K20 v5x R", "78.4", "71.0", "82.4", "81.8", "175.1", "58.9")}
CP = {"Ms": ("C21(MFTD) v5s RGB + R", "80.2", "79.7", "81.6", "79.3", "89.6", "49.2"), "Mm": ("C22(MFTD) v5m RGB + R", "82.0", "80.2", "84.4", "81.3", "216.8", "40.2"),
      "Ml": ("C11(MFTD) v5l RGB + R", "86.8", "81.2", "92.9", "86.2", "413.4", "35.6"), "Mx": ("C23(MFTD) v5x RGB + R", "84.7", "80.8", "88.5", "84.6", "690.1", "30.4"),
      "YsRGB": ("C15 v5s RGB", "68.6", "65.0", "75.5", "65.2", "14.4", "146.3"), "YmRGB": ("C17 v5m RGB", "72.6", "68.8", "82.2", "66.7", "42.5", "97.1"),
      "YlRGB": ("C0 v5l RGB", "76.0", "69.6", "85.5", "72.8", "93.7", "86.1"), "YxRGB": ("C19 v5x RGB", "75.6", "70.6", "86.0", "70.3", "175.1", "58.9"),
      "YsR": ("C16 v5s R", "64.5", "62.3", "64.9", "66.3", "14.4", "146.3"), "YmR": ("C18 v5m R", "69.2", "68.0", "68.6", "70.9", "42.5", "97.1"),
      "YlR": ("C1 v5l R", "73.6", "68.1", "74.4", "78.2", "93.7", "86.1"), "YxR": ("C20 v5x R", "73.3", "63.9", "75.6", "80.5", "175.1", "58.9")}
for ds, tab, pg, loc, prim in (("dataset1", KP, 17, T8, True), ("dataset2", CP, 17, T9, False)):
    for key, item in MODELS:
        lab, mp, e, m, l, mb, fps = tab[key]
        ev = f"{lab} {mp} {e} {m} {l} {mb} {fps}"
        RID[(ds, key)] = res()
        ROWS += result(RID[(ds, key)], "ap50", item, ds, mp, pg, loc, ev, unit="%", setting=f"{loc.split(' (')[0]}: {'Korean pine' if ds == 'dataset1' else 'Chinese pine'} dataset, validation images, mean of the three infection stages",
                       note="mAP@50, the mean of the AP@50 of the early, middle and late infection stage classes", **TH, primary=prim, headline=(prim and key == "Ml"))
        if ds == "dataset1":
            ROWS.append(R("inference_speed", f"{fps} FPS (hardware not stated for the speed; training server with an RTX 3090)", pg, loc, ev, item=item))
ROWS += [D("notes", "Tables 8 and 9 (KP and CP datasets) compare YOLOv3, YOLOv4, YOLOv5 s, m, l and x on visible (RGB) or red-band (R) images and the MFTD variants (RGB + R): mAP@50, AP@50 of the early, middle and late stage, model size in MB and FPS. Recorded: the four MFTD variants and the eight YOLOv5 rows of both tables (mAP@50 only; the stage-wise AP@50 are not recorded); YOLOv3 (KP 73.9, CP 66.9 mAP@50) and YOLOv4 (KP 78.7, CP 73.3) are not recorded. The KP dataset is primary because the abstract's first result (88.5) is on it. The 30 percent part of the 7:3 split is called the validation set and is used for the test. The text layer of the tables is interleaved on p.17, so the values were read from the rendered page. Table 4 and 5 (network fusion modes of YOLOv5l), Tables 6 and 7 (ablation of MFFT) and the model sizes in MB are not recorded.",
           "Sec. 3 and Tables 8 and 9 (p. 17)")]
ROWS += [D("code_available", "no", "No code repository of the authors in the paper (only the YOLOv5 repository is cited)"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "no", 21, "Data Availability Statement", "Data Availability Statement: Not applicable")]
