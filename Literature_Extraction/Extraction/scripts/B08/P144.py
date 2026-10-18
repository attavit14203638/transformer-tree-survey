import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T3 = "P144", "Table II", "Table III"
# YOLOv11 with a modified loss (class weights, weighted focal loss, SmoothL1 box loss, approximated DFL; architecture unchanged) against YOLOv8, Deformable DETR (the only transformer) and the standard YOLOv11 for the five
# palm condition classes of MOPAD Site 2 (healthy, yellowish, smallish, dead, mismanaged). Table II gives the overall precision, recall, F1 and FPS of the four models and Table III the per-class values: the overall
# F1 is recorded and primary (precision and recall are not, F1 is given); the per-class F1 are recorded without primary (Dead is mortality_detection, the other classes health_classification). The paper has a training and
# a validation set only (1,803 and 500 images) and does not name the split of the tables, so split is not_stated. The numbers are taken from the sentences of the results section, which print them verbatim.
TH = Tags(task="health_classification", level="object", split="not_stated", agg="single")
DS_EV = "The dataset is divided into a training set of 1,803 images and a validation set of 500"
MOPAD_EV = "a UAV imagery collection annotated with five oil palm condition classes: healthy, dead, mismanaged, yellowish, and smallish"
OV_EV = "F1-score of 99.64%, surpassing the standard YOLOv11 (97.77%), YOLOv8 (98.45%), and Deformable DETR (90.73%)"
ROWS = [
    R("task", "health_classification", 1, "Abstract", "This study proposes a modified YOLOv11 model for oil palm condition detection using the MOPAD dataset, which includes five condition classes", note="detection of five palm condition classes: healthy, yellowish, smallish, dead and mismanaged"),
    R("output_unit", "tree_instance", 2, "Fig. 2", "Sample images from the MOPAD site 2 dataset with original bounding box annotations"),
    R("tree_setting", "plantation", 1, "Introduction", "Oil palm (Elaeis guineensis) is one of the most important commodities in the world", note="oil palm plantation"),
    R("tree_taxa", "oil palm (Elaeis guineensis)", 1, "Introduction", "Oil palm (Elaeis guineensis) is one of the most important commodities in the world"),
    R("study_area", "MOPAD Site 2, Papua, Indonesia", 1, "Introduction", "Only Site 2, located in Papua, Indonesia, was used, as it is the only publicly available subset"),
    D("biome", "tropical", "Papua, Indonesia; not stated in the paper"),
    R("platform", "uav", 1, "Introduction", MOPAD_EV),
    NR("sensors", note="UAV imagery of MOPAD; the colour channels are not named in the paper"),
    R("annotation_type", "bounding_box", 2, "Fig. 2", "Sample images from the MOPAD site 2 dataset with original bounding box annotations"),
    R("dataset_name", "MOPAD Site 2 (Papua), five oil palm condition classes, 1024 x 1024 images", 2, "Sec. III.A", DS_EV, item="dataset1"),
    R("dataset_key", "mopad", 2, "Sec. III.A", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. III.A", DS_EV, item="dataset1", note="training and validation sets only; no test set"),
    R("dataset_public", "yes", 6, "Dataset Availability Statement", "The dataset used in this study is publicly available on GitHub: https://github.com/rs-dl/MOPAD", item="dataset1"),
    R("dataset_url", "https://github.com/rs-dl/MOPAD", 6, "Dataset Availability Statement", "The dataset used in this study is publicly available on GitHub: https://github.com/rs-dl/MOPAD", item="dataset1"),
    D("n_images", "2303", "sum of the 1,803 training and 500 validation images (Sec. III.A); 1024 x 1024 pixels, resized to 640 x 640", item="dataset1", unit="images"),
    D("n_trees", "151583", "sum of the class counts of Table I: training 231 + 89,958 + 431 + 26,644 + 1,473 and validation 61 + 24,884 + 95 + 7,395 + 411 instances of the five classes", item="dataset1", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="YOLOv11: Ultralytics default configuration, 640 x 640 inputs, 100 epochs, batch size 32; the training settings of the other models and the initialisation are not stated")
NAMES_EV = "benchmark it against YOLOv8, the MOPAD baseline, and Deformable DETR"
ROWS += model("model1", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8 (size not stated)", None, None, 1, "Introduction", NAMES_EV, cat_note="convolutional YOLO detector", **NP)
ROWS += model("model2", "Deformable DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", None, None, 1, "Introduction", "Deformable DETR, in particular, offers a balance of accuracy and speed",
              cat_note="DETR-style detector with a transformer encoder and decoder; the backbone is not stated (protocol 4.5: DETR variants are cnn_transformer_hybrid)", **NP)
ROWS += model("model3", "YOLOv11 Standard", "cnn_baseline", "cnn", "YOLOv11 (C3k2 blocks, SPPF and C2PSA modules)", None, None, 3, "Sec. III.B", "YOLOv11 is used as the baseline model without architectural or loss function modifications",
              cat_note="the paper describes YOLOv11 as a single-stage detector whose C2PSA modules enhance spatial attention; no self-attention or transformer block is described", **NP)
ROWS += model("model4", "YOLOv11 Modified", "proposed", "cnn", "YOLOv11 with a modified loss: inverse-frequency class weights, weighted focal loss (gamma 2.0), SmoothL1 box loss and an MSE approximation of DFL", None, None, 3, "Sec. III.C",
              "the YOLOv11 loss function was modified without altering the model", cat_note="the architecture of YOLOv11 is unchanged; only the loss function is modified", **NP)
res = Counter("res")
SPEED = (("model1", "137", "YOLOv8 and YOLOv11 Standard achieved the highest throughput at 137 FPS and 132 FPS"), ("model2", "6.6", "Deformable DETR, at only 6.6 FPS"),
         ("model3", "132", "YOLOv8 and YOLOv11 Standard achieved the highest throughput at 137 FPS and 132 FPS"), ("model4", "46", "the Modified YOLOv11 showed a reduction in throughput (46 FPS)"))
for item, fps, ev in SPEED:
    ROWS.append(R("inference_speed", f"{fps} FPS (hardware not stated; 640 x 640 inputs)", 4, T2, ev, item=item))
for item, f1, head in (("model1", "98.45", False), ("model2", "90.73", False), ("model3", "97.77", False), ("model4", "99.64", True)):
    ROWS += result(res(), "f1", item, "dataset1", f1, 4, T2, OV_EV, unit="%", setting="Table II: overall (five classes); split not stated", note="overall F1 of Table II; it equals the mean of the five per-class F1 of Table III within rounding (checked), and the overall precision and recall are not recorded",
                   **TH, primary=True, headline=head)
CLS = [  # class, task, page, evidence, F1 by model (YOLOv8, Deformable DETR, YOLOv11 Standard, YOLOv11 Modified)
    ("Dead", "mortality_detection", 4, "Dead class, it reached an F1-score of 99.82%, higher than Standard YOLOv11 (99.47%), YOLOv8 (99.63%), and far above Deformable DETR (90.16%)", ("99.63", "90.16", "99.47", "99.82")),
    ("Healthy", "health_classification", 4, "Healthy class, performance also improved, achieving 99.80% F1 compared to Standard YOLOv11 (99.44%), YOLOv8 (99.39%), and Deformable DETR (98.50%)", ("99.39", "98.50", "99.44", "99.80")),
    ("Mismanaged", "health_classification", 4, "Mismanaged class, where the Modified YOLOv11 achieved 99.36% F1, outperforming YOLOv8 (95.68%), Standard YOLOv11 (92.65%), and Deformable DETR (74.17%)", ("95.68", "74.17", "92.65", "99.36")),
    ("Smallish", "health_classification", 5, "Smallish class, where the Modified YOLOv11 reached 99.75% F1, surpassing YOLOv8 (99.02%), Standard YOLOv11 (98.85%), and Deformable DETR (97.51%)", ("99.02", "97.51", "98.85", "99.75")),
    ("Yellowish", "health_classification", 5, "Yellowish class, the Modified YOLOv11 also achieved the highest performance with 99.45% F1, outperforming Standard YOLOv11 (98.44%), YOLOv8 (98.57%), and Deformable DETR (93.30%)", ("98.57", "93.30", "98.44", "99.45")),
]
for cname, task, pg, ev, vals in CLS:
    for item, v in zip(("model1", "model2", "model3", "model4"), vals):
        ROWS += result(res(), "f1", item, "dataset1", v, pg, T3, ev, unit="%", setting=f"Table III: {cname} class; split not stated", note=f"F1 of the {cname} class (precision and recall of the class are not recorded)", **TH(task=task))
ROWS += [R("limitations", "Only MOPAD Site 2 is used, so there is no cross-site validation, and the loss components are not ablated", 5, "Sec. V", "as this study is limited to MOPAD Site 2")]
ROWS += avail(code="no", weights="not_stated", data="yes", data_page=6, data_loc="Dataset Availability Statement", data_ev="The dataset used in this study is publicly available on GitHub: https://github.com/rs-dl/MOPAD")
ROWS += [D("notes", "Overall precision, recall, F1 and FPS of YOLOv8, Deformable DETR, standard YOLOv11 and the loss-modified YOLOv11 on MOPAD Site 2 (Table II) and per-class values (Table III). Recorded: the overall F1 (98.45, 90.73, 97.77, 99.64; primary, headline the modified YOLOv11) and the per-class F1 (not primary), with the FPS of Table II. The overall F1 equals the mean of the five per-class F1 of Table III within rounding (checked on all four models), so the overall values are class averages over the five condition classes, all of which are palms. The paper names no split (the data has training and validation sets of 1,803 and 500 images), so split is not_stated. Deformable DETR is the only transformer and has by far the lowest recall (86.39) and speed (6.6 FPS).",
           "Sec. III and IV and Tables I to III (pp. 2 to 5)")]
