import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T2 = "Table II"
ROWS = [
    R("task", "counting", 1, "Abstract", "state-of-the-art results when evaluating TreeVision on two benchmark tree-counting datasets (Yosemite and KCL-London)"),
    R("output_unit", "tree_count", 4, "Sec. IV-B", "target and predicted counts of trees which are produced by taking the total value of all pixels in the corresponding density maps"),
    R("tree_setting", "mixed", 4, "Sec. IV-A", "The study area is centered on London ... The research site is located at Yosemite National Park", note="urban (London) and natural forest (Yosemite)"),
    NR("tree_taxa", note="all trees; species not given"),
    R("study_area", "London, United Kingdom; Yosemite National Park, California, USA", 4, "Sec. IV-A", "The study area is centered on London, United Kingdom ... Yosemite National Park, California, United States of America"),
    D("biome", "temperate", "London and Yosemite (temperate; Yosemite montane); not stated in the paper"),
    R("platform", "airborne", 1, "Abstract", "density map-based methods, in aerial images", note="Yosemite imagery collected from Google Maps"),
    R("sensors", "rgb", 4, "Sec. IV-A", "The data are collected from Google Maps", note="RGB implied by Google Maps imagery"),
    D("gsd", "11.8", "Yosemite only: 2262.5 m / 19,200 px = 0.118 m per pixel (p.4, Sec. IV-A); KCL-London resolution not given", unit="cm"),
    R("annotation_type", "point", 1, "Introduction", "the use of emerging and trending point-based annotations"),
    R("dataset_name", "KCL-London", 4, "Table I", "KCL-London 452 : 161 1024 × 1024 613 ... 95,067", item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. IV-A", "The labeled dataset comprises 452 images for training and 161 images for testing", item="dataset1"),
    NR("dataset_public", item="dataset1", note="benchmark from [10]; availability not stated in this paper"),
    R("n_images", "613", 4, "Table I", "KCL-London 452 : 161 1024 × 1024 613", item="dataset1", unit="images", note="plus 308 unannotated images, not used"),
    R("n_trees", "95067", 4, "Table I", "KCL-London ... 95,067", item="dataset1", unit="trees"),
    R("dataset_name", "Yosemite", 4, "Table I", "Yosemite 1350 : 1350 512 × 512 2700 ... 98,949", item="dataset2"),
    R("dataset_role", "train_test", 4, "Sec. IV-A", "regions B and D are used for training, while regions A and C are used for testing", item="dataset2"),
    NR("dataset_public", item="dataset2", note="benchmark from [6]; availability not stated in this paper"),
    R("n_images", "2700", 4, "Table I", "Yosemite 1350 : 1350 512 × 512 2700", item="dataset2", unit="images"),
    R("n_trees", "98949", 4, "Table I", "Yosemite ... 98,949", item="dataset2", unit="trees"),
]
ROWS += model("model1", "TreeVision", "proposed", "pure_vit", "ViT-Base (ViTDet, 16x16 patches) with dilated convolutions", "other", "full_fine_tuning",
              3, "Sec. III", "We choose ViT-Base from [22] with a 16×16 input patch size as the backbone", pre_page=4, pre_loc="Sec. IV-C",
              pre_ev="initialize our dilated convolutions with the weights of CSRNet trained in dataset ShanghaiB [36] and our Vision Transformer backbone with the weights of ViTDet [27]",
              pre_note="ViTDet weights for the backbone; CSRNet (ShanghaiB) weights for the dilated convolutions", strat_page=4, strat_loc="Sec. IV-C",
              strat_ev="trained ... TreeVision for 20 epochs on each dataset", cat_note="plain ViT backbone with a convolutional density head")
BASE = [("model2", "MCNN [36]", "MCNN", "cnn", "cnn_baseline"), ("model3", "CSRNet [13]", "CSRNet (VGG16 front end)", "cnn", "cnn_baseline"),
        ("model4", "SASNet [37]", "SASNet", "cnn", "cnn_baseline"), ("model5", "EDNet [21]", "EDNet", "cnn", "cnn_baseline"),
        ("model6", "S-TreeFormer [10]", "TreeFormer (pyramid vision transformer encoder)", "hierarchical_vit", "transformer_baseline")]
for item, lab, bb, cat, role in BASE:
    ROWS += model(item, lab.split(" [")[0], role, cat, bb, None, None, 5, T2, lab, pre_status="not_reported",
                  pre_note="values taken from [10]; training details not given here", strat_note="values taken from [10]")
VALS = {  # item: KCL (MAE, RMSE, R2), Yosemite (MAE, RMSE, R2)
    "model2": (("25.87", "34.12", "0.45"), ("10.44", "12.45", "0.61")), "model3": (("20.21", "25.69", "0.69"), ("7.25", "9.35", "0.76")),
    "model4": (("24.33", "30.12", "0.56"), ("6.33", "8.46", "0.81")), "model5": (("26.18", "32.02", "0.52"), ("9.92", "12.39", "0.60")),
    "model6": (("18.52", "24.32", "0.72"), ("4.29", "5.85", "0.91")), "model1": (("16.80", "21.98", "0.78"), ("4.56", "6.15", "0.89")),
}
LAB = {"model1": "TreeVision (ours)", **{b[0]: b[1] for b in BASE}}
res = Counter("res")
for item, (kcl, yos) in VALS.items():
    for ds, vals, name in (("dataset1", kcl, "KCL-London"), ("dataset2", yos, "Yosemite")):
        note = "" if item == "model1" else "value reported from [10], not re-run"
        for metric, v, unit in zip(("mae", "rmse", "r2"), vals, ("trees", "trees", "fraction")):
            ROWS += result(res(), metric, item, ds, v, 5, T2, f"{LAB[item]} {name} {metric.upper()} {v}", unit=unit, setting="test set; per-image tree count",
                           note=note, headline=(item == "model1" and ds == "dataset1" and metric == "mae"))
ROWS += avail()
ROWS += [D("notes", "Baseline values in Table II are taken from the TreeFormer paper [10]. Table III shows that the proposed ground-truth generation (Gaussian filtering before cropping) lowers Yosemite MAE from 4.56 to 4.31 for TreeVision and from 7.25 to 4.92 for CSRNet.", "Sec. IV-D, IV-E and Tables II to III (p.5)")]
