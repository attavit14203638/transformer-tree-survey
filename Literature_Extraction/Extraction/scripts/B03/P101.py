import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T4, T5, T6 = "Table 4 (image)", "Table 5 (image)", "Table 6 (image)"
URL = "https://github.com/WHU-USI3DV/WHU-STree"
ROWS = [
    R("task", "species_classification;tree_segmentation_3d", 1, "Abstract", "We benchmark representative baselines for two key", note="tree species classification and individual tree segmentation benchmarks"),
    R("output_unit", "tree_label;tree_points", 21, "Sec. 4.3", "mean class precision (mPrec), mean class recall (mRec), mean class coverage (mCov)"),
    R("tree_setting", "urban", 1, "Abstract", "we introduce WHU-STree, a cross-city, richly annotated, and multi-modal urban street tree dataset"),
    R("tree_taxa", "50 street-tree species", 1, "Abstract", "encompassing 21,007 annotated tree instances across 50 species"),
    R("study_area", "Nanjing and Shenyang, China", 10, "Sec. 3.1", "We collected data in Nanjing and Shenyang to construct the dataset"),
    D("biome", "multiple", "Nanjing (subtropical) and Shenyang (temperate); the paper notes the cities lie in distinct climatic zones (p.13)"),
    R("platform", "mobile_lidar", 11, "Sec. 3.1", "Both the Hiscan-Z and Alpha3D systems share similar configurations, integrating a laser scanner, a GNSS receiver, an inertial measurement unit (IMU), and a 360° panoramic camera",
      note="vehicle mobile mapping systems"),
    R("sensors", "lidar;rgb", 11, "Sec. 3.1", "integrating a laser scanner ... and a 360° panoramic camera"),
    R("point_density", "616", 11, "Sec. 3.2", "averaging 616 and 1377 points/m? for Nanjing and Shenyang, respectively", unit="pts/m2", note="Nanjing 616; Shenyang 1377 (text layer is OCR)"),
    R("annotation_type", "point_label", 12, "Sec. 3.2", "tators delineated tree crowns based on morphological characteristics, ensuring each instance maintains reasonable 3D structure representation",
      note="tree instances and species labels on the point clouds"),
    R("dataset_name", "WHU-STree-NJ (Nanjing, class-balanced split)", 11, "Sec. 3.1", "The Nanjing dataset, designated as WHU-STree-NJ, was collected using the Hiscan-Z MMS in 2020 and 2021", item="dataset1"),
    R("dataset_role", "train_test", 15, "Sec. 3.3", "Class-balanced split: considering the ... This split maintains roughly consistent class ratio between the training and test sets", item="dataset1"),
    R("dataset_public", "yes", 2, "Abstract", f"The WHU-STree dataset is accessible at: {URL}", item="dataset1"),
    R("dataset_url", URL, 2, "Abstract", "is accessible at: https://github.com/WHU-USI3DV /WHU-STree", item="dataset1"),
    R("n_trees", "21007", 1, "Abstract", "encompassing 21,007 annotated tree instances", item="dataset1", unit="trees", note="both cities together"),
    R("dataset_name", "WHU-STree-SY (Shenyang, cross-city test)", 11, "Sec. 3.1", "The Shenyang dataset, referred to as WHU-STree-SY, was acquired in 2021 using the Alpha3D system", item="dataset2"),
    R("dataset_role", "transfer_target", 29, "Sec. 5.3", "we conducted cross-city experiments by training models on WHU-STree-NJ and evaluating them on WHU-STree-SY", item="dataset2"),
    R("dataset_public", "yes", 2, "Abstract", f"The WHU-STree dataset is accessible at: {URL}", item="dataset2"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on WHU-STree-NJ; details not stated")
CLS = [("model1", "MinkNet", "cnn_baseline", "cnn", "sparse voxel convolution (MinkowskiNet)", "49.94", "81.67", "MinkNet(Choy et al., 2019) 49.94 81.67"),
       ("model2", "PointMLP", "other_baseline", "other", "point-based MLP network", "58.00", "84.51", "PointMLP(Ma et al., 2022) 58.00 84.51"),
       ("model3", "Point Transformer V2 (PTv2)", "transformer_baseline", "pure_vit", "Point Transformer V2", "64.09", "87.99", "PTv2(Wu et al., 2022) 64.09 87.99"),
       ("model4", "TSCMDL (PointMLP and ResNet fusion)", "other_baseline", "other", "PointMLP point branch with ResNet image branch", "60.49", "86.20", "TSCMDL(Liu et al., 2023a) 60.49 86.20")]
res = Counter("res")
for item, name, role, cat, bb, miou, oa, ev in CLS:
    ROWS += model(item, name, role, cat, bb, None, None, 22, T4, ev, cat_note=("leverages self-attention (p.22)" if item == "model3" else ""), **NP)
    st = "WHU-STree-NJ class-balanced test set; species classification of tree point clouds"
    ROWS += result(res(), "miou", item, "dataset1", miou, 22, T4, ev, setting=st, headline=(item == "model3"), note="mean IoU over species")
    ROWS += result(res(), "oa", item, "dataset1", oa, 22, T4, ev, setting=st)
SEG = [  # item, name, role, category, backbone, NJ F1, SY F1, Table 5 evidence, species-specific F1, Table 6 evidence
    ("model5", "SegmentAnyTree", "cnn_baseline", "cnn", "sparse convolution with embedding clustering", "86.8", "86.7",
     "SegmentAnyTree(Wielgosz et al., 2024) 87.6 12.4 14.1 86.8 86.0 14.0 12.6 86.7", "51.2", "SegmentAnyTree(Wielgosz et al., 2024) 56.9 46.5 43.2 46.8 51.2"),
    ("model6", "SoftGroup", "cnn_baseline", "cnn", "sparse convolution with grouping", "62.7", "79.9",
     "SoftGroup(Vu et al., 2022) 59.0 41.0 33.0 62.7 79.9 20.1 20.1 79.9", "33.4", "SoftGroup(Vu et al., 2022) 36.0 31.1 31.4 31.6 33.4"),
    ("model7", "SPFormer", "transformer_baseline", "cnn_transformer_hybrid", "sparse convolution backbone with superpoint transformer decoder", "82.2", "84.9",
     "SPFormer(Sun et al., 2023) 93.2 6.8 26.6 82.2 87.4 12.6 17.5 84.9", "57.8", "SPFomer(Sun et al., 2023) 52.8 63.9 59.4 61.8 57.8"),
    ("model8", "LCPS (LiDAR-camera panoptic segmentation)", "other_baseline", "other", "LiDAR and panoramic image fusion network", "78.8", "87.4",
     "LCPS (Zhang et al., 2023) 75.6 24.4 17.8 78.8 83.0 17.0 7.8 87.4", "39.9", "LCPS(Zhang et al., 2023) 41.9 38.0 35.3 35.0 39.9"),
]
tr = Counter("tr")
for item, name, role, cat, bb, nj, sy, ev5, f6, ev6 in SEG:
    ROWS += model(item, name, role, cat, bb, None, None, 24, T5, ev5, cat_note=("transformer-based paradigm in the paper's discussion (p.26)" if item == "model7" else ""), **NP)
    ROWS += result(res(), "f1", item, "dataset1", nj, 24, T5, ev5, setting="WHU-STree-NJ test set; species-agnostic tree segmentation, IoU 0.5")
    ROWS += result(res(), "f1", item, "dataset2", sy, 24, T5, ev5, setting="WHU-STree-SY (trained on NJ); species-agnostic tree segmentation, IoU 0.5")
    ROWS += result(res(), "f1", item, "dataset1", f6, 25, T6, ev6, setting="WHU-STree-NJ test set; species-specific tree segmentation", note="segmentation and species together")
    t = tr()
    ROWS += [R("transfer_source", "WHU-STree-NJ (Nanjing, Hiscan-Z MMS); in-domain value on its test set", 29, "Sec. 5.3", "training models on WHU-STree-NJ and evaluating them on WHU-STree-SY", item=t),
             R("transfer_target", "WHU-STree-SY (Shenyang, Alpha3D MMS)", 29, "Sec. 5.3", "WHU-STree-NJ and WHU-STree-SY differ substantially in point density, species composition, and tree morphology", item=t),
             R("shift_type", "region;sensor;species", 29, "Sec. 5.3", "differ substantially in point density, species composition, and tree morphology", item=t),
             R("transfer_model", item, 24, T5, ev5, item=t),
             R("transfer_metric", "f1", 24, T5, "F1↑(%)", item=t, note="species-agnostic tree segmentation"),
             R("value_in_domain", nj, 24, T5, ev5, item=t, unit="%"),
             R("value_out_of_domain", sy, 24, T5, ev5, item=t, unit="%")]
ROWS += avail(code_url=URL, page=2, loc="Abstract", ev="The WHU-STree dataset is accessible at: https://github.com/WHU-USI3DV /WHU-STree", data="yes",
              data_page=2, data_loc="Abstract", data_ev="The WHU-STree dataset is accessible at: https://github.com/WHU-USI3DV /WHU-STree")
ROWS += [D("notes", "The PDF is a scanned copy with an OCR text layer, so all table values were read from the rendered pages (Tables 4 to 6). Detection, omission and commission rates are also in Table 5; Tables 7 and 8 are ablations of LCPS and SPFormer.", "Sec. 4.4 and 5 (pp. 21 to 29)")]
