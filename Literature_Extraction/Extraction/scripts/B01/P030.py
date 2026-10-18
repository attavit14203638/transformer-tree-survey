import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, T2, T3, T4 = "Table I (image)", "Table II (image)", "Table III (image)", "Table IV (image)"  # tables are images in the TGRS version
URL = "https://github.com/HAAClassic/TreeFormer"
ROWS = [
    R("task", "counting", 1, "Abstract", "TreeFormer: A Semi-Supervised Transformer-Based Framework for Tree Counting From a Single High-Resolution Image"),
    R("output_unit", "tree_count", 8, "Sec. V-B", "estimated tree number for the ith sample", note="density maps summed to counts"),
    R("tree_setting", "mixed", 8, T1, "urban residential area, dense park ... cropland, urban, residential area ... wooded mountainous"),
    NR("tree_taxa", note="all trees; species not given"),
    R("study_area", "London, United Kingdom; Jiangsu, China; Yosemite National Park, California, USA", 7, "Sec. V-A",
      "captured from Jiangsu, China ... centered at Yosemite National Park, CA, USA"),
    D("biome", "multiple", "London and Yosemite (temperate), Jiangsu (subtropical); not stated in the paper"),
    R("platform", "multiple", 7, "Sec. V-A", "24 Gaofen-II satellite images with 0.8-m GSD ... collected from Google Maps", note="Google Maps imagery (KCL-London, Yosemite) and Gaofen-II satellite (Jiangsu)"),
    R("sensors", "rgb", 4, "Sec. III-B", "stitched together from Google Maps at 0.2-m ground sampling distance", note="RGB image products"),
    R("gsd", "0.20", 8, T1, "KCL-London ... 0.20", unit="m", note="KCL-London 0.20 m; Jiangsu 0.80 m; Yosemite 0.12 m"),
    R("annotation_type", "point", 8, "Sec. V-C", "The GT contains the coordinates of tree locations which are specified by annotations dots"),
]
DS = [("dataset1", "KCL-London", "613", "95,067", "95067", "urban residential area, dense park", "1024 × 1024 0.20 613", "labelled set plus 308 unlabelled images",
       8, "Sec. V-B", "there are also 308 additional unlabeled images"),
      ("dataset2", "Jiangsu (Gaofen-II)", "2400", "664,487", "664487", "cropland, urban, residential area", "256 × 256 0.80 2400", "training 1920, test 480",
       7, "Sec. V-A", "a training set that contains a total of 1920 images and a test set that contains 480 images"),
      ("dataset3", "Yosemite", "2700", "98,949", "98949", "wooded mountainous", "512 × 512 0.12 2700", "training 1350, test 1350",
       7, "Sec. V-A", "This data is divided into training (1350 images) and test data (1350 images)")]
for item, name, n_img, n_tr_ev, n_tr, land, ev, note, rpage, rloc, rev in DS:
    ROWS += [R("dataset_name", name, 8, T1, f"{name.split(' (')[0]} {land}", item=item),
             R("dataset_role", "train_test", rpage, rloc, rev, item=item, note=note),
             R("dataset_public", "yes", 1, "Abstract", f"The codes and datasets are available at {URL}", item=item),
             R("n_images", n_img, 8, T1, ev, item=item, unit="images"),
             R("n_trees", n_tr, 8, T1, f"{name.split(' (')[0]} ... {n_tr_ev}", item=item, unit="trees")]
M = [  # item, name, role, category, backbone, note
    ("model1", "MCNN [47]", "cnn_baseline", "cnn", "MCNN", "reproduced for tree counting"),
    ("model2", "CSRNet [73]", "cnn_baseline", "cnn", "CSRNet (VGG-16 front end)", "reproduced for tree counting"),
    ("model3", "Swin-UNet [77]", "transformer_baseline", "hierarchical_vit", "Swin-UNet", "the paper calls it transformer-based"),
    ("model4", "FusionNet [76]", "cnn_baseline", "cnn", "FusionNet", "reproduced for tree counting"),
    ("model5", "SASNet [75]", "cnn_baseline", "cnn", "SASNet", "reproduced for tree counting"),
    ("model6", "EDNet [17]", "cnn_baseline", "cnn", "EDNet", ""),
    ("model7", "DENT [31]", "transformer_baseline", "cnn_transformer_hybrid", "convolutional feature extractor with transformer encoder",
     "DENT employs a convolutional architecture ... Then a transformer encoder is used"),
    ("model8", "TreeCountNet [19]", "cnn_baseline", "cnn", "TreeCountNet", ""),
    ("model9", "S-TreeFormer (supervised)", "proposed", "hierarchical_vit", "pyramid vision transformer (PVT) encoder", "supervised version with the same backbone as TreeFormer"),
    ("model10", "TreeFormer (semi-supervised)", "proposed", "hierarchical_vit", "pyramid vision transformer (PVT) encoder", "uses labelled and unlabelled images"),
]
for item, name, role, cat, bb, note in M:
    if item in ("model9", "model10"):
        ROWS += model(item, name, role, cat, bb, None, None, 2, "Sec. I", "We base our network encoder on a pyramid vision transformer (PVT) [30]", pre_status="not_reported",
                      pre_note="not stated", strat_note="trained 500 epochs with Adam (p.9); initialisation not stated", cat_note=note)
    else:
        ROWS += model(item, name.split(" [")[0], role, cat, bb, None, None, 11, T3, name, pre_status="not_reported", pre_note="not stated",
                      strat_note="not stated", cat_note=note)
for item in ("model9", "model10"):
    lab = "S-TreeFormer" if item == "model9" else "TreeFormer"
    ROWS += [R("total_params", "116.144", 11, T4, f"{lab} 36.296 116.144 16", item=item, unit="M"),
             R("flops", "36.296", 11, T4, f"{lab} 36.296 116.144 16", item=item, unit="GFLOPs"),
             R("inference_speed", "16 ms per image (batch size 1)", 11, T4, f"{lab} 36.296 116.144 16", item=item)]
T3V = {  # item: {dataset: (MAE, RMSE, R2)} from Table III
    "model1": {"dataset1": ("25.87", "34.12", "0.45"), "dataset2": ("81.09", "125.45", "0.84"), "dataset3": ("10.44", "12.45", "0.61")},
    "model2": {"dataset1": ("23.27", "29.62", "0.59"), "dataset2": ("47.22", "83.14", "0.93"), "dataset3": ("9.34", "11.48", "0.65")},
    "model3": {"dataset1": ("36.45", "47.56", "0.24"), "dataset2": ("70.17", "110.46", "0.88"), "dataset3": ("13.34", "16.09", "0.52")},
    "model4": {"dataset1": ("28.45", "35.67", "0.47"), "dataset2": ("54.75", "89.45", "0.92"), "dataset3": ("6.88", "9.10", "0.78")},
    "model5": {"dataset1": ("24.33", "30.12", "0.56"), "dataset2": ("47.32", "76.90", "0.94"), "dataset3": ("6.33", "8.46", "0.81")},
    "model6": {"dataset1": ("26.18", "32.02", "0.52"), "dataset2": ("99.10", "153.47", "0.77"), "dataset3": ("9.92", "12.39", "0.60")},
    "model7": {"dataset3": ("7.50", "12.30", None)},
    "model8": {"dataset2": ("45.08", "77.96", "0.96")},
    "model9": {"dataset1": ("18.52", "24.32", "0.72"), "dataset2": ("41.06", "72.06", "0.95"), "dataset3": ("4.29", "5.85", "0.91")},
    "model10": {"dataset1": ("16.70", "22.98", "0.75")},
}
DSN = {"dataset1": "KCL-London", "dataset2": "Jiangsu", "dataset3": "Yosemite"}
res = Counter("res")
for item, per in T3V.items():
    lab = next(m[1] for m in M if m[0] == item).split(" (")[0]
    for ds, vals in per.items():
        st = "test set; all training images labelled" + ("; plus 308 unlabelled images" if item == "model10" else "")
        for metric, v, unit in zip(("mae", "rmse", "r2"), vals, ("trees", "trees", "fraction")):
            if v is None:
                continue
            ROWS += result(res(), metric, item, ds, v, 11, T3, f"{lab} {DSN[ds]} {metric.upper()} {v}", unit=unit, setting=st,
                           headline=(item == "model10" and metric == "mae"))
de = Counter("de")
T2V = {"10": {"dataset1": ("27.83", "0.41"), "dataset2": ("73.32", "0.86"), "dataset3": ("5.72", "0.84")},
       "30": {"dataset1": ("20.61", "0.66"), "dataset2": ("56.38", "0.91"), "dataset3": ("4.69", "0.89")}}
for pct, per in T2V.items():
    for ds, (mae, r2) in per.items():
        for metric, v, unit in (("mae", mae, "trees"), ("r2", r2, "fraction")):
            i = de()
            ROWS += [R("label_model", "model10", 10, T2, f"{pct}%Labeled {100 - int(pct)}%Unlabeled TreeFormer", item=i),
                     R("label_amount", pct, 10, T2, f"{pct}%Labeled {100 - int(pct)}%Unlabeled", item=i, note=f"rest of the training set used unlabelled; dataset {DSN[ds]}"),
                     R("label_unit", "percent_of_training_set", 9, "Sec. V-D", "we use 10% or 30% of training data as Dltr while the rest as Dutr", item=i),
                     R("label_metric", metric, 10, T2, f"{DSN[ds]} {metric.upper()}", item=i, note=DSN[ds]),
                     R("label_value", v, 10, T2, f"TreeFormer {DSN[ds]} {metric.upper()} {v} ({pct}% labelled)", item=i, unit=unit)]
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=f"The codes and datasets are available at {URL}", data="yes", data_page=1, data_loc="Abstract",
              data_ev=f"The codes and datasets are available at {URL}")
ROWS += [D("notes", "Table II compares semi-supervised methods (MT, ICT, L2R, CCT, TreeFormer) that all use the same transformer backbone at 10% and 30% labels; only TreeFormer's values are recorded (as data-efficiency items). Fig. 6a extends the label fraction to 20% to 100% (figure only). The four crowd-counting CNNs were reproduced by the authors; DENT and TreeCountNet values are partial in Table III.", "Sec. V-D, Fig. 6 and Tables II to IV (pp. 9 to 12)")]
