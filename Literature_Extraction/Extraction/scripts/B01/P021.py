import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T2, T4 = "Table 2", "Table 4"
ROWS = [
    R("task", "detection", 1, "Abstract", "MRV-YOLO: A Multi-Channel Remote Sensing Object Detection Method for Identifying Reclaimed Vegetation"),
    R("output_unit", "tree_instance", 10, "Sec. 3.1.2", "includes 1356 images with 8344 labeled reclaimed plants", note="one box per reclaimed plant"),
    R("tree_setting", "orchard", 10, "Fig. 5", "(C) is some samples of reclaimed navel orange vegetation", note="navel orange trees planted for reclamation of a rare-earth mining area"),
    R("tree_taxa", "navel orange (Citrus sinensis)", 10, "Fig. 5", "some samples of reclaimed navel orange vegetation", note="species name not given; navel orange per Figure 5"),
    R("study_area", "Huangkeng Village, Anyuan County, Ganzhou City, Jiangxi, China (rare-earth mining area)", 9, "Sec. 3.1.1", "a typical mining area in Huangkeng Village, Anyuan County, Ganzhou City, China"),
    D("biome", "subtropical", "Ganzhou, southern Jiangxi (subtropical); not stated in the paper"),
    R("platform", "uav", 10, "Sec. 3.1.2", "a high-precision multi-spectral imaging system mounted on the MS 600 PRO UAV (Yusense, Qingdao, China)"),
    R("sensors", "rgb;multispectral", 11, T1 if False else "Table 1", "Red ... Green ... Blue ... Red Edge1 ... NIR", note="models run on RGB or on all six bands"),
    R("gsd", "0.04", 10, "Sec. 3.1.2", "a multi-spectral image with a spatial resolution of 0.04 m", unit="m"),
    R("annotation_type", "bounding_box", 10, "Sec. 3.1.2", "The annotation of reclaimed vegetation was conducted through manual visual interpretation and on-site investigation",
      note="TXT annotations for a YOLO-type detector"),
    R("dataset_name", "own multispectral UAV reclaimed-vegetation dataset (Huangkeng mining area)", 10, "Sec. 3.1.2", "we constructed a reclaimed vegetation dataset from the acquired multispectral remote sensing images", item="dataset1"),
    R("dataset_role", "train_test", 10, "Sec. 3.1.2", "The dataset was subsequently divided into a training set and a verification set in a 9:1 ratio", item="dataset1",
      note="results on the 10% verification split; no separate test set"),
    R("dataset_public", "on_request", 20, "Data Availability Statement", "further inquiries can be directed to the corresponding author", item="dataset1"),
    R("n_images", "1356", 10, "Sec. 3.1.2", "which includes 1356 images with 8344 labeled reclaimed plants", item="dataset1", unit="images"),
    R("n_trees", "8344", 10, "Sec. 3.1.2", "1356 images with 8344 labeled reclaimed plants", item="dataset1", unit="trees"),
]
M = [  # item, name, role, category, backbone, Table 4 params, FLOPs, label in Table 2
    ("model1", "MRV-YOLO", "proposed", "cnn", "YOLOv8 with CAGP-SPPF and C2f_DynamicConv", "3.83", "7.9", "our"),
    ("model2", "RT-DETR", "transformer_baseline", "cnn_transformer_hybrid", "not stated", "31.99", "103.6", "RT-DETR"),
    ("model3", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8", "3.01", "8.2", "YOLOv8"),
    ("model4", "YOLOv8-AS", "cnn_baseline", "cnn", "YOLOv8-AS", "2.66", "7.4", "YOLOv8-AS"),
]
NOTES = {"model1": "channel attention, not a transformer", "model3": "the model MRV-YOLO is built on (ablation baseline 'SPPF + C2f')",
         "model4": "best CNN in Table 2 on mAP@0.5, mAP@0.5:0.95 and F1 for both inputs"}
for item, name, role, cat, bb, par, fl, lab in M:
    ROWS += model(item, name, role, cat, bb, None, "from_scratch" if False else None, 13, T2, f"{lab}(rgb) ... {lab}(bands)", pre_status="not_reported",
                  pre_note="not stated", strat_note="trained for 100 epochs with AdamW (p.12); initialisation not stated", cat_note=NOTES.get(item, ""))
    ROWS += [R("total_params", par, 18, T4, f"{lab} ... {par} {fl}", item=item, unit="M", note="column headed 'Parameters (MB)'"),
             R("flops", fl, 18, T4, f"{lab} ... {par} {fl}", item=item, unit="GFLOPs")]
VALS = {  # item: {input: (mAP@0.5, mAP@0.5:0.95, F1)}
    "model1": {"rgb": ("87.6", "52.6", "84"), "bands": ("91.6", "54.9", "88")},
    "model2": {"rgb": ("77.1", "31", "77"), "bands": ("84.8", "35.8", "83")},
    "model3": {"rgb": ("82.7", "43.7", "79"), "bands": ("87", "44.1", "85")},
    "model4": {"rgb": ("84.8", "50.3", "81"), "bands": ("89.5", "52.2", "86")},
}
res = Counter("res")
for item, ins in VALS.items():
    lab = next(m[7] for m in M if m[0] == item)
    for inp, (a50, a5095, f1) in ins.items():
        st = f"verification split; {'RGB' if inp == 'rgb' else 'six bands (RGB, red edge, NIR)'} input"
        ROWS += result(res(), "ap50", item, "dataset1", a50, 13, T2, f"{lab}({inp}) mAP@0.5 {a50}", setting=st, headline=(item == "model1" and inp == "bands"))
        ROWS += result(res(), "ap50_95", item, "dataset1", a5095, 13, T2, f"{lab}({inp}) mAP@0.5:0.95 {a5095}", setting=st)
        ROWS += result(res(), "f1", item, "dataset1", f1, 13, T2, f"{lab}({inp}) F1-Score {f1}", setting=st)
ROWS += avail(data="on_request", data_page=20, data_loc="Data Availability Statement", data_ev="further inquiries can be directed to the corresponding author")
ROWS += [D("notes", "Table 2 has 22 rows (11 detectors, each on RGB and on six bands); following protocol 4.1, recorded: MRV-YOLO, the transformer RT-DETR, the YOLOv8 baseline and the best CNN (YOLOv8-AS). The other seven YOLO variants are in Table 2 (p.13). All models gained from six-band input. RT-DETR trains far slower (12.125 h versus 0.247 h, Table 4).", "Tables 2 to 4 (pp. 13 to 18)")]
