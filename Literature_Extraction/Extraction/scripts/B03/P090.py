import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2, T3, T5 = "P090", "Table 2", "Table 3", "Table 5"
ROWS = [
    R("task", "crown_segmentation;tree_segmentation_3d", 1, "Abstract",
      "we first introduce a novel Multi-Source Feature Fusion Network (MSFFNet) to achieve accurate instance-level segmentation of individual trees in the 2D image domain",
      note="2D crown instances transferred to the LiDAR point cloud for 3D individual-tree segmentation"),
    R("output_unit", "crown_mask;tree_points", 19, "Sec. 4.3", "the individual tree segmentation of 3D point clouds based on cross-modal semantic information"),
    R("tree_setting", "plantation", 4, "Sec. 2.1", "which consists of typical commercial forests", note="commercial Chinese fir and Masson pine forest with bamboo"),
    R("tree_taxa", "Cunninghamia lanceolata; Pinus massoniana", 6, "Sec. 2.3", "267 individually annotated tree samples from two representative species: Chinese fir (72%) and Pinus massoniana (28%)"),
    R("study_area", "southern Ganzhou, Jiangxi Province, China", 4, "Sec. 2.1", "we selected a representative study area in the southern part of Ganzhou, Jiangxi Province, China"),
    R("biome", "subtropical", 4, "Sec. 2.1", "features a subtropical monsoon humid climate"),
    R("platform", "uav", 4, "Sec. 2.2", "using the DJI Matrice 350 RTK UAV platform"),
    R("sensors", "rgb;lidar", 4, "Sec. 2.2", "Zenmuse L2 integrated multi-sensor system. This advanced system comprises a solid-state LiDAR module ... a 20-megapixel RGB imaging unit",
      note="MSFFNet input: RGB orthophoto and LiDAR-derived CHM"),
    R("gsd", "3", 5, "Table 1", "Spatial resolution 3 cm/pixel", unit="cm"),
    R("point_density", "150", 4, "Sec. 2.2", "with an average point cloud density of 150 points/m2", unit="pts/m2"),
    R("annotation_type", "polygon_mask;point_label", 6, "Sec. 2.3", "manual labeling was performed on UAV-acquired RGB imagery, resulting in approximately 1364 tree instance masks in 2D",
      note="crown masks in 2D; tree-by-tree point labels for the 3D test trees"),
    R("dataset_name", "own UAV RGB and LiDAR dataset, Ganzhou (crown instances; Forest 1 and Forest 2 test plots)", 6, "Sec. 2.3",
      "Construction of the Individual Tree Crown Segmentation Dataset", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 2.3", "these two subregions were not involved in the training and validation stages of the MSFFNet but were reserved for model testing", item="dataset1"),
    R("dataset_public", "on_request", 23, "Data Availability Statement", "The data presented in this study are available on request from the corresponding author due to privacy", item="dataset1"),
    D("n_images", "7112", "5548 + 686 + 878 images in the training, validation and test sets (p.6, Sec. 2.3); 256 x 256 patches after augmentation; the printed total 7239 is in cf1 (rule V1c)", item="dataset1", unit="images"),
    R("n_trees", "1364", 6, "Sec. 2.3", "resulting in approximately 1364 tree instance masks in 2D", item="dataset1", unit="trees", note="approximately; 267 of them in the two test plots"),
]
PRE = dict(pre_page=15, pre_loc="Sec. 3.4", pre_ev="the model was pre-trained on the ImageNet dataset [39]. The pre-trained weights were then used to initialize the MSFFNet",
           strat_page=15, strat_loc="Sec. 3.4", strat_ev="The batch size and number of epochs were set to 8 and 60")
res = Counter("res")
FOREST = ["Forest 1 test plot", "Forest 2 test plot"]
# MSFFNet backbones (Table 2); the MiT-B4 model is the one compared with other methods in Table 3
for k, b in enumerate(("B4", "B1", "B2", "B3")):
    item = f"model{k + 1}"
    name = f"MSFFNet with MiT-{b} (SegFormer) encoder"
    ROWS += model(item, name, "proposed" if b == "B4" else "transformer_baseline", "hierarchical_vit", f"MiT-{b}", "imagenet", "full_fine_tuning", 17, T2, f"MiT-{b}",
                  cat_note=("SegFormer branches for RGB and CHM with fusion modules" if b == "B4" else "backbone variant of the proposed model"), **PRE)
    for n in range(2):
        nums, ev = row(PID, 17, rf"MiT-{b}\s+(?=\d)", n)
        st = f"{FOREST[n]}; crown instances"
        ROWS += result(res(), "f1", item, "dataset1", nums[3], 17, T2, ev, unit="fraction", setting=st)
        ROWS += result(res(), "iou", item, "dataset1", nums[4], 17, T2, ev, unit="fraction", setting=st, headline=(b == "B4" and n == 0),
                       note=("abstract: IoU of 87.60%" if (b == "B4" and n == 0) else ""))
NOPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="not stated")
for item, lab, name, role, cat, note in (("model5", "Mask R-CNN", "Mask R-CNN", "cnn_baseline", "cnn", ""),
                                          ("model6", "SOLOV2", "SOLOv2", "cnn_baseline", "cnn", ""),
                                          ("model7", "Mask2Former", "Mask2Former", "transformer_baseline", "cnn_transformer_hybrid", "backbone not stated")):
    ROWS += model(item, name, role, cat, "not stated", None, None, 18, T3, lab, cat_note=note, **NOPRE)
    for n in range(2):
        nums, ev = row(PID, 18, rf"{lab}\s+(?=\d)", n)
        st = f"{FOREST[n]}; crown instances"
        ROWS += result(res(), "f1", item, "dataset1", nums[2], 18, T3, ev, unit="fraction", setting=st)
        ROWS += result(res(), "iou", item, "dataset1", nums[3], 18, T3, ev, unit="fraction", setting=st)
# 3D individual-tree segmentation of the 267 test trees (Table 5)
ROWS += model("model8", "Cross-modal semantic transfer (MSFFNet crowns mapped to the LiDAR point cloud)", "proposed", "hierarchical_vit", "MSFFNet (MiT-B4) with probabilistic label transfer",
              "imagenet", "full_fine_tuning", 19, "Sec. 4.3", "the proposed cross-modal semantic transfer framework", cat_note="3D labels come from the 2D transformer model; no 3D training", **PRE)
ROWS += model("model9", "K-means clustering", "other_baseline", "classical", "K-means on points (no learning)", "none", None, 20, "Sec. 4.3", "K-means [41]", strat_note="not a trained model")
ROWS += model("model10", "PDE-Net", "other_baseline", "other", "point-based deep network (non-transformer)", None, None, 20, "Sec. 4.3", "PDE-Net [42]", **NOPRE)
for item, n in (("model9", 0), ("model10", 1), ("model8", 2)):
    nums, ev = row(PID, 21, r"SAC\s+(?=\d)", n)
    for k in range(2):
        ROWS += result(res(), "other", item, "dataset1", nums[k], 21, T5, ev, unit="fraction", setting=f"{FOREST[k]}; 3D individual-tree point segmentation",
                       note="segmentation accuracy (SAC); omission and commission errors also in Table 5")
ROWS += avail(data="on_request", data_page=23, data_loc="Data Availability Statement",
              data_ev="The data presented in this study are available on request from the corresponding author due to privacy")
ROWS += [D("notes", "The MSFFNet row in Table 3 equals the MiT-B4 row of Table 2. Precision, recall and overall accuracy are also reported (Tables 2 to 3).", "Sec. 4.1 to 4.3 and Tables 2, 3 and 5 (pp. 16 to 21)")]
