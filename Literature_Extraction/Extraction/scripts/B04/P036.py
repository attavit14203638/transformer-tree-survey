import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P036"
URL = "https://github.com/Lihy256/CEDAnet"
# Individual tree (lychee) instance segmentation: box AP is detection at object level, mask AP is crown segmentation at instance-mask level (protocol 6.2). AP is the COCO AP over IoU 0.50 to 0.95.
TB = Tags(task="detection", level="object", split="test", agg="single")
TM = Tags(task="crown_segmentation", level="instance_mask", split="test", agg="single")
PUB = "Both two datasets and the code of CEDAnet will be publicly available at https://github.com/Lihy256/CEDAnet"

ROWS = [
    R("task", "crown_segmentation;detection", 1, "Abstract", "Individual tree segmentation (ITS) is a pivotal technique in orchard research, estimating tree counts and delineating crown contours"),
    R("output_unit", "crown_mask;tree_instance", 1, "Abstract", "Individual tree segmentation (ITS) is a pivotal technique in orchard research, estimating tree counts and delineating crown contours",
      note="instance masks and boxes (Tables I and II report both)"),
    R("tree_setting", "orchard", 3, "Sec. 3.A", "Both datasets, captured through UAV imagery, showcase dense lychee orchards in China’s Guangdong province"),
    R("tree_taxa", "lychee", 3, "Sec. 3.A", "Both datasets, captured through UAV imagery, showcase dense lychee orchards in China’s Guangdong province"),
    R("study_area", "Conghua District (Guangzhou) and Dianbai District (Maoming), Guangdong Province, China", 3, "Sec. 3.A",
      "Li-Bo Park area in Conghua District, Guangzhou City, Guangdong Province, China ... Dianbai District of Maoming City, Guangdong Province"),
    D("biome", "subtropical", "Guangdong Province, China; not stated in the paper"),
    R("platform", "uav", 3, "Sec. 3.A", "The images of the study area were taken by DJI M30 UAV"),
    D("sensors", "rgb", "imagery type not stated; the UAV images in Fig. 1 and Fig. 5 (p.3, p.7) are colour images"),
    D("annotation_type", "polygon_mask", "individual trees annotated in the Labelme software (p.3); the ground truth is instance masks (Fig. 1)"),
    R("dataset_name", "iSCHID: instance segmentation in Conghua images dataset (dense lychee orchard, 512 × 512 slices)", 3, "Sec. 3.A.1", "The training and testing datasets contain the 265 and 64 pairs of data", item="dataset1"),
    R("dataset_key", "own_p036_ischid", 3, "Sec. 3.A.1", "The training and testing datasets contain the 265 and 64 pairs of data", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 3.A.1", "The training and testing datasets contain the 265 and 64 pairs of data", item="dataset1"),
    R("dataset_public", "yes", 1, "Abstract", PUB, item="dataset1", note="announced as 'will be publicly available' (p.1) and 'publicly available' (p.3)"),
    R("dataset_url", URL, 1, "Abstract", PUB, item="dataset1"),
    D("n_images", "329", "sum of the printed split counts: 265 training pairs plus 64 test pairs (p.3)", item="dataset1", unit="images"),
    R("dataset_name", "iSMMID: instance segmentation in Maoming images dataset (lychee orchard, 512 × 512 slices)", 3, "Sec. 3.A.2", "The training and testing datasets were randomly split and contained 270 and 55 pairs of images, respectively", item="dataset2"),
    R("dataset_key", "own_p036_ismmid", 3, "Sec. 3.A.2", "The training and testing datasets were randomly split and contained 270 and 55 pairs of images, respectively", item="dataset2"),
    R("dataset_role", "train_test", 3, "Sec. 3.A.2", "The training and testing datasets were randomly split and contained 270 and 55 pairs of images, respectively", item="dataset2"),
    R("dataset_public", "yes", 1, "Abstract", PUB, item="dataset2", note="announced as 'will be publicly available' (p.1) and 'publicly available' (p.3)"),
    R("dataset_url", URL, 1, "Abstract", PUB, item="dataset2"),
    D("n_images", "325", "sum of the printed split counts: 270 training pairs plus 55 test pairs (p.3)", item="dataset2", unit="images"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="initialisation not stated; all models trained for 250 epochs in MMDetection (Adam, learning rate 0.01)")
M = [  # item, label in Tables I and II, name, role, category, backbone, note
    ("model1", "Ours", "CEDAnet", "proposed", "cnn_transformer_hybrid", "Mask R-CNN with ResNet50, transformer-based contextual aggregation module (TCAM) and density-guided NMS (DG-NMS)",
     "Mask R-CNN with a transformer-based context module added (Sec. 3.B)"),
    ("model2", "Mask R-CNN", "Mask R-CNN", "cnn_baseline", "cnn", "backbone not stated", "also the Base model of the ablation (Tables III and IV)"),
    ("model3", "Cascade Mask R-CNN", "Cascade Mask R-CNN", "cnn_baseline", "cnn", "backbone not stated", "Mask R-CNN based"),
    ("model4", "MS R-CNN", "Mask Scoring R-CNN (MS R-CNN)", "cnn_baseline", "cnn", "backbone not stated", "Mask R-CNN based"),
    ("model5", "SOTR", "SOTR", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated; row-column separated attention with convolutional layers", "described as transformer-based (Sec. 4.B)"),
    ("model6", "Instaboost", "Instaboost", "other_baseline", None, "not stated", "a data augmentation method (Sec. 4.B); the instance segmentation model it is applied to is not stated"),
]
# Rows of Tables I and II, read from the rendered pages (the tables are images): AP, AP50, AP75, AP small, AP medium, AP large for boxes, then the same for masks
T1 = {  # iSCHID, Table I (p.6)
    "Mask R-CNN": (("0.466", "0.753", "0.514", "0.171", "0.418", "0.593"), ("0.462", "0.742", "0.532", "0.112", "0.413", "0.611")),
    "Cascade Mask R-CNN": (("0.469", "0.755", "0.519", "0.175", "0.421", "0.601"), ("0.466", "0.748", "0.541", "0.12", "0.433", "0.625")),
    "MS R-CNN": (("0.481", "0.748", "0.524", "0.18", "0.432", "0.597"), ("0.466", "0.741", "0.525", "0.12", "0.406", "0.595")),
    "SOTR": (("0.469", "0.741", "0.495", "0.19", "0.406", "0.592"), ("0.467", "0.713", "0.484", "0.144", "0.451", "0.606")),
    "Instaboost": (("0.453", "0.75", "0.483", "0.179", "0.407", "0.57"), ("0.433", "0.727", "0.469", "0.099", "0.389", "0.566")),
    "Ours": (("0.498", "0.769", "0.561", "0.217", "0.449", "0.63"), ("0.493", "0.793", "0.571", "0.19", "0.458", "0.636")),
}
T2 = {  # iSMMID, Table II (p.8); AP small is -1 (no small trees)
    "Mask R-CNN": (("0.691", "0.973", "0.894", "-1", "0.674", "0.714"), ("0.667", "0.959", "0.963", "-1", "0.635", "0.701")),
    "Cascade Mask R-CNN": (("0.695", "0.979", "0.899", "-1", "0.681", "0.723"), ("0.671", "0.963", "0.961", "-1", "0.643", "0.712")),
    "MS R-CNN": (("0.699", "0.977", "0.889", "-1", "0.689", "0.738"), ("0.694", "0.977", "0.913", "-1", "0.68", "0.739")),
    "SOTR": (("0.701", "0.979", "0.892", "-1", "0.685", "0.751"), ("0.693", "0.988", "0.938", "-1", "0.669", "0.715")),
    "Instaboost": (("0.701", "0.977", "0.886", "-1", "0.69", "0.745"), ("0.688", "0.986", "0.904", "-1", "0.678", "0.722")),
    "Ours": (("0.706", "0.983", "0.899", "-1", "0.697", "0.752"), ("0.703", "0.988", "0.971", "-1", "0.661", "0.745")),
}
res = Counter("res")
for item, lab, name, role, cat, bb, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, 6, "Table I (image)", f"{lab} " + " ".join(T1[lab][0]), cat_note=note, **NP)
    for ds, tab, page, loc, prim in (("dataset1", T1, 6, "Table I (image)", True), ("dataset2", T2, 8, "Table II (image)", False)):
        box, mask = tab[lab]
        for kind, vals, tags in (("bbox", box, TB), ("mask", mask, TM)):
            ev = f"{lab} {kind} " + " ".join(vals)
            nm = "iSCHID" if ds == "dataset1" else "iSMMID"
            for metric, v, mn in (("ap50_95", vals[0], "AP"), ("ap50", vals[1], "AP50")):
                ROWS += result(res(), metric, item, ds, v, page, loc, ev, unit="fraction", setting=f"{loc.split(' ')[0]} {loc.split(' ')[1]}: {nm} test set; {kind} {mn}",
                               note=f"{mn} of the {kind} output", **tags, primary=prim, headline=(item == "model1" and ds == "dataset1" and kind == "bbox" and metric == "ap50_95"))
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=PUB, data="yes", data_page=1, data_loc="Abstract", data_ev=PUB, note_d="announced as 'will be publicly available'")
ROWS += [D("notes", "Tables I and II compare CEDAnet with Mask R-CNN, Cascade Mask R-CNN, Mask Scoring R-CNN, SOTR and Instaboost on the iSCHID (dense orchard, first dataset and primary) and iSMMID test sets; all six rows are recorded for AP and AP50 of boxes and masks (AP75 and the size-wise APs are in the tables; AP small is -1 on iSMMID). Tables III and IV ablate TCAM and DG-NMS on top of Mask R-CNN (Base) and are not recorded; Table V varies the NMS threshold. The segmentation block of Table I is headed AP-bbox in the paper but the values are mask APs (they match the text).",
           "Sec. 4 to 5 and Tables I to V (pp. 6 to 9)")]
