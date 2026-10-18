import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P147", "Table III"
# CoFA-DETR (RT-DETR based, frequency-domain modules and collaborative hybrid assignment training with a ResNet-50 backbone) for the five growth status classes of oil palms (healthy, dead, mismanaged, smallish,
# yellowish) in UAV images of MOPAD (Sites 1 and 2 combined). Table III is the main comparison: 11 models with precision, recall, F1 (printed as integer percentages) and GFLOPs. The MOPAD row is taken from the MOPAD
# paper [7] and the YOLOv5, YOLOv7, YOLOv8 and HR-YOLOv8 rows from [20]; they carry copied_from. F1 is recorded (precision and recall are not) and primary for every row; the headline is CoFA-DETR-R50. The paper does
# not describe a split, so split is not_stated.
TH = Tags(task="health_classification", level="object", split="not_stated", agg="single")
TASK_EV = "our model categorizes each detected tree into one of five visually distinguishable growth condition classes: healthy palms, dead palms, mismanaged palms, smallish palms, and yellowish palms"
MOPAD_EV = "The dataset used in this study is sourced from [7] available at https://github.com/rs-"
SITE_EV = "covering a total area of 28.85 km² across two different locations in Indonesia, namely Kalimantan and Papua"
COUNT_EV = "includes approximately 363,877 oil palm trees"
COPY_EV = "For MOPAD, we report results based on findings from [7], while the results for YOLOv5, YOLOv7, YOLOv8, and HR-YOLOv8 are taken from [20]"
ROWS = [
    R("task", "health_classification", 1, "Introduction", TASK_EV, note="detection of oil palms and classification of five growth status classes; the dead class is one of the five"),
    R("output_unit", "tree_instance", 1, "Introduction", TASK_EV),
    R("tree_setting", "plantation", 1, "Abstract", "we propose the use of CoFA-DETR, a real-time end-to-end object detection framework based on Detection Transformer (DETR) and optimized for Unmanned Aerial Vehicle (UAV) imagery in oil palm plantations", note="oil palm plantations"),
    R("tree_taxa", "oil palm", 1, "Abstract", "we propose the use of CoFA-DETR, a real-time end-to-end object detection framework based on Detection Transformer (DETR) and optimized for Unmanned Aerial Vehicle (UAV) imagery in oil palm plantations"),
    R("study_area", "Kalimantan and Papua, Indonesia (MOPAD Sites 1 and 2, 28.85 km2)", 4, "Sec. IV.A", SITE_EV),
    D("biome", "tropical", "Kalimantan and Papua, Indonesia; not stated in the paper"),
    R("platform", "uav", 4, "Sec. IV.A", "The data were collected using Skywalker X8 aerial", note="Skywalker X8 aerial platform, Sony A6000 camera; the paper recommends a fixed flight altitude of 500 m (p.5)"),
    D("sensors", "rgb", "Sony A6000 camera of MOPAD (Sec. IV.A, p.4); the colour channels are not named in this paper"),
    R("annotation_type", "bounding_box", 1, "Introduction", TASK_EV, note="boxes with one of five growth status classes (MOPAD annotations); the paper does not describe the annotation form"),
    R("dataset_name", "MOPAD (oilpalmuav) multi-class oil palm UAV dataset, Sites 1 and 2 combined (Kalimantan and Papua)", 4, "Sec. IV.A", MOPAD_EV, item="dataset1", note="the abstract calls the dataset oilpalmuav"),
    R("dataset_key", "mopad", 4, "Sec. IV.A", MOPAD_EV, item="dataset1"),
    D("dataset_role", "train_test", "no split of the dataset is described in the paper", item="dataset1"),
    D("dataset_public", "yes", "MOPAD is public (GitHub link in Sec. IV.A, p.4)", item="dataset1"),
    R("n_trees", "363877", 4, "Sec. IV.A", COUNT_EV, item="dataset1", unit="trees", note="printed as approximately 363,877 oil palm trees in five classes"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="CoFA-DETR: 36 epochs, AdamW, batch size 2, learning rate 0.0002, Quality Focal, L1 and GIoU losses (Table II); the others: same hardware (RTX 6000 Ada), initialisation not stated")
M = [  # row regex, name, role, category, backbone, copied (reference or None), category note
    (r"MOPAD", "MOPAD baseline model", "cnn_baseline", "cnn", "Faster R-CNN with a Refined Pyramid Feature module and a hybrid class-balanced loss (as described in the related work of this paper)", "ref. [7] (Zheng et al., ISPRS J. Photogramm. Remote Sens. 2021)",
     "Faster R-CNN based model of the MOPAD paper, described in Sec. II.A"),
    (r"YOLOv5", "YOLOv5", "cnn_baseline", "cnn", "YOLOv5", "ref. [20] (HR-YOLOv8 paper, Electronics 2024)", "convolutional YOLO detector"),
    (r"YOLOv7", "YOLOv7", "cnn_baseline", "cnn", "YOLOv7", "ref. [20] (HR-YOLOv8 paper, Electronics 2024)", "convolutional YOLO detector"),
    (r"YOLOv8", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8", "ref. [20] (HR-YOLOv8 paper, Electronics 2024)", "convolutional YOLO detector"),
    (r"HR-YOLOv8", "HR-YOLOv8", "cnn_baseline", "cnn", "YOLOv8 based crop growth status detector", "ref. [20] (HR-YOLOv8 paper, Electronics 2024)", "YOLOv8 variant (the architecture is not described in this paper)"),
    (r"DETR", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", None, "DETR with a CNN backbone (protocol 4.5: hybrid)"),
    (r"Deformable-DETR", "Deformable DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", None, "DETR-style detector with deformable attention (protocol 4.5: hybrid)"),
    (r"RT-DETR-R50", "RT-DETR-R50", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-50", None, "RT-DETR with a ResNet-50 backbone (protocol 4.5: hybrid)"),
    (r"UAV-DETR", "UAV-DETR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated", None, "DETR-style detector for UAV imagery (protocol 4.5: hybrid)"),
    (r"CoDino-R50", "CoDINO-R50 (Co-DETR with DINO, ResNet-50)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-50", None, "DETR-style detector with a ResNet-50 backbone (protocol 4.5: hybrid)"),
    (r"CoFA-DETR-R50 \[ours\]", "CoFA-DETR-R50 (ours)", "proposed", "cnn_transformer_hybrid", "ResNet-50 with a channel mapper, a deformable encoder, convolutional cross-scale feature fusion with frequency modules (FFT) and a decoder with auxiliary ATSS and ROI heads",
     None, "RT-DETR based detector with a ResNet-50 backbone, deformable attention and FFT frequency modules (protocol 4.5: hybrid)"),
]
res = Counter("res")
for k, (rx, name, role, cat, bb, copied, cnote) in enumerate(M, start=1):
    item = f"model{k}"
    nums, line = row(PID, 4, rf"^\s*{rx}\s+(?=[\d-])", 0, cols=(80, 200))
    assert len(nums) in (1, 4), (name, nums)  # (precision, recall,) F1, GFLOPs; the MOPAD row has F1 only
    f1 = nums[0] if rx == "MOPAD" else nums[2]
    ROWS += model(item, name, role, cat, bb, None, None, 4, T3, line, cat_note=cnote + ("; row copied from another paper" if copied else ""), **NP)
    if rx != "MOPAD":
        ROWS.append(R("flops", nums[3], 4, T3, line, item=item, unit="GFLOPs", note=("copied from " + copied) if copied else ""))
    ROWS += result(res(), "f1", item, "dataset1", f1, 4, T3, line, unit="%", setting="Table III: five growth status classes; split not stated", note="F1 printed as an integer percentage; precision and recall of the row are not recorded",
                   **TH, primary=True, headline=(rx == r"CoFA-DETR-R50 \[ours\]"), copied_from=copied)
ROWS += [R("limitations", "The model may underperform in haze or strong sunlight glare, is sensitive to image noise and motion blur, and is less reliable where canopies overlap heavily", 5, "Sec. IV.D",
           "The model may underperform under extreme weather conditions such as haze, or strong sunlight glare")]
ROWS += avail(code="no", weights="not_stated", data="yes", data_page=4, data_loc="Sec. IV.A", data_ev=MOPAD_EV, note_d="MOPAD is public; this paper does not describe a release of its own splits")
ROWS += [D("notes", "Table III compares MOPAD, YOLOv5, YOLOv7, YOLOv8, HR-YOLOv8, DETR, Deformable DETR, RT-DETR-R50, UAV-DETR, CoDINO-R50 and CoFA-DETR-R50 on the five growth status classes of MOPAD (Sites 1 and 2 combined, about 363,877 trees); F1 is recorded for all 11 (primary; precision and recall are not recorded) with GFLOPs. The MOPAD row (F1 63 percent) is taken from ref. [7] and the YOLOv5, YOLOv7, YOLOv8 and HR-YOLOv8 rows from ref. [20], so they carry copied_from. CoFA-DETR-R50 and CoDINO-R50 both reach 99 percent F1 (CoFA-DETR needs 77 GFLOPs, CoDINO 673), the abstract's 22 point gain is over HR-YOLOv8 (77 percent). The values are printed as integer percentages. The paper describes no split. The abstract names the dataset oilpalmuav, Sec. IV.A names MOPAD.",
           "Sec. IV and Table III (pp. 4 to 5)")]
