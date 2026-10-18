import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T3, T4 = "Table III (image)", "Table IV (image)"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "This study presents tree morphology multirepresentation fusion network, integrating 3-D point cloud data with 2-D projections for enhanced tree species classification"),
    R("output_unit", "tree_label", 6, "Sec. IV-A", "Given a point cloud dataset D = {T1 , T2 , . . . , Tm } consisting of m individual trees"),
    R("tree_setting", "urban", 3, "Sec. III-A", "Our study focuses on two urban sites in Baoding City, Hebei Province, China"),
    R("tree_taxa", "eight urban species (seven broadleaf, one conifer), e.g. AECH, AIAL, GIBI, KOPA, LISI, MASP, PITA, YUSO", 5, "Sec. III-C",
      "species: Seven broadleaf (e.g., MASP, AIAL) and one coniferous (PITA)"),
    R("study_area", "Baoding City, Hebei Province, China (Millennium Xiulin, Xiong'an New Area; Hebei Agricultural University west campus)", 3, "Sec. III-A",
      "two urban sites in Baoding City, Hebei Province, China ... Site 1, located in the Millennium Xiulin of Xiong’an New Area"),
    R("biome", "temperate", 3, "Sec. III-A", "both experiencing a warm temperate continental monsoon climate"),
    R("platform", "mobile_lidar", 4, "Sec. III-B", "we used a backpack-mounted mobile scanning unit"),
    R("sensors", "lidar", 4, "Sec. III-B", "The handheld frame houses an AVIA LiDAR sensor and an Intel RealSense D435i RGBD camera", note="camera used only for visual reference"),
    R("annotation_type", "image_label", 4, "Sec. III-C", "Table I summarizes our UMTPC dataset, which includes 830 trees from eight", note="one species label per tree point cloud"),
    R("dataset_name", "UMTPC (urban MLS tree point cloud dataset, own)", 4, "Sec. III-C", "We introduce the urban MLS tree point cloud (UMTPC) dataset, a new benchmark dataset created for this study", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. III-C", "The dataset is split into training and test sets with a 7:3 ratio", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    R("n_trees", "830", 4, "Sec. III-C", "Table I summarizes our UMTPC dataset, which includes 830 trees from eight", item="dataset1", unit="trees"),
    R("dataset_name", "STPCTLS (single tree point clouds from terrestrial laser scanning) [25]", 5, "Sec. III-C",
      "we use the single tree point clouds from terrestrial laser scanning (STPCTLS) dataset [25]", item="dataset2"),
    R("dataset_role", "train_test", 13, "Sec. V-C", "Tables III and IV present the average results over ten trials on our UMTPC dataset and the STPCTLS dataset [25], respectively", item="dataset2"),
    R("dataset_public", "yes", 5, "Sec. III-C", "the single tree point clouds from terrestrial laser scanning (STPCTLS) dataset [25]", item="dataset2", note="published dataset"),
    R("n_trees", "684", 5, "Sec. III-C", "This dataset contains 684 trees from seven categories, collected using TLS systems in artificial and natural forests in Germany and the United States",
      item="dataset2", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on each dataset; mean of ten trials")
M = [  # item, name, role, category, backbone, note, (UMTPC F1, OA), (STPCTLS F1, OA), Table III row text, Table IV row text
    ("model1", "TM2F (tree morphology multirepresentation fusion network)", "proposed", "other", "3-D point branch with adaptive hierarchical sampling and a 2-D projection branch",
     "no attention or transformer component described", ("98.57", "98.77"), ("94.05", "94.55"),
     "TM2F 98.76 1.03 98.43 1.15 98.57 1.20 98.77 0.94 0.99 0.01", "TM2F 94.44 1.17 94.45 1.21 94.05 1.04 94.55 1.19 0.94 0.01"),
    ("model2", "OctFormer", "transformer_baseline", "hierarchical_vit", "octree-based transformer", "", ("95.46", "95.43"), ("86.58", "86.49"),
     "OctFormer [103] 95.47 1.79 95.42 2.06 95.46 1.18 95.43 1.93 0.95 0.02", "OctFormer [103] 86.50 2.77 85.80 2.36 86.58 2.14 86.49 2.22 0.86 0.02"),
    ("model3", "PoinTramba", "transformer_baseline", "cnn_transformer_hybrid", "Transformer and Mamba encoders", "point network with Transformer and Mamba encoders (protocol 4.5)", ("96.09", "96.43"), ("87.07", "87.23"),
     "PoinTramba [15] 96.71 1.40 95.89 1.76 96.09 1.81 96.43 1.42 0.96 0.02", "PoinTramba [15] 87.44 1.19 86.92 1.02 87.07 1.32 87.23 1.36 0.87 0.02"),
    ("model4", "PointNeXt", "other_baseline", "other", "PointNeXt", "MLP-based point network; best previous model", ("96.64", "96.65"), ("90.39", "91.04"),
     "PointNeXt [30] 96.69 1.29 96.64 1.40 96.64 1.81 96.65 2.15 0.96 0.02", "PointNeXt [30] 91.16 1.36 90.44 1.66 90.39 1.89 91.04 2.39 0.89 0.02"),
    ("model5", "PointMLP", "other_baseline", "other", "PointMLP", "MLP-based point network", ("96.21", "96.46"), ("87.71", "87.98"),
     "PointMLP [32] 96.30 1.28 96.20 1.57 96.21 1.47 96.46 1.23 0.96 0.02", "PointMLP [32] 88.29 1.70 87.76 1.54 87.71 1.43 87.98 1.69 0.85 0.02"),
    ("model6", "LayerNet", "other_baseline", "other", "LayerNet", "point network", ("95.08", "95.63"), ("85.96", "85.99"),
     "LayerNet [72] 95.05 1.86 95.16 2.00 95.08 1.90 95.63 2.12 0.95 0.02", "LayerNet [72] 86.37 1.66 85.81 2.23 85.96 1.81 85.99 1.24 0.82 0.01"),
    ("model7", "PointNet++", "other_baseline", "other", "PointNet++", "MLP-based point network", ("95.07", "95.24"), ("84.62", "84.06"),
     "PointNet++ [11] 95.27 1.25 95.04 1.35 95.07 1.43 95.24 1.31 0.95 0.02", "PointNet++ [11] 87.14 1.29 84.06 1.63 84.62 1.07 84.06 1.21 0.80 0.02"),
    ("model8", "PointNet", "other_baseline", "other", "PointNet", "MLP-based point network", ("91.45", "92.46"), ("78.80", "78.92"),
     "PointNet [10] 92.89 1.90 91.32 2.18 91.45 1.94 92.46 2.09 0.91 0.02", "PointNet [10] 80.20 1.16 78.76 1.35 78.80 1.69 78.92 1.49 0.73 0.02"),
    ("model9", "Random forest", "other_baseline", "classical", "RF on phenotypic features", "classical machine learning", ("85.95", "86.51"), ("60.16", "61.84"),
     "RF [102] 86.48 1.96 86.35 2.15 85.95 2.04 86.51 2.01 0.85 0.02", "RF [102] 59.75 1.35 62.00 1.63 60.16 1.08 61.84 2.09 0.51 0.02"),
    ("model10", "Support vector machine", "other_baseline", "classical", "SVM on phenotypic features", "classical machine learning", ("83.30", "85.32"), ("62.58", "68.24"),
     "SVM [101] 83.56 1.04 83.32 1.98 83.30 1.79 85.32 1.80 0.83 0.01", "SVM [101] 58.15 1.46 70.44 2.35 62.58 1.86 68.24 1.24 0.54 0.01"),
]
res = Counter("res")
for item, name, role, cat, bb, note, um, st_, ev3, ev4 in M:
    ROWS += model(item, name, role, cat, bb, None, None, 13, T3, ev3, cat_note=note, **NP)
    ROWS += result(res(), "f1", item, "dataset1", um[0], 13, T3, ev3, setting="UMTPC test set; mean of ten trials", headline=(item == "model1"))
    ROWS += result(res(), "oa", item, "dataset1", um[1], 13, T3, ev3, setting="UMTPC test set; mean of ten trials")
    ROWS += result(res(), "f1", item, "dataset2", st_[0], 14, T4, ev4, setting="STPCTLS test set; mean of ten trials")
    ROWS += result(res(), "oa", item, "dataset2", st_[1], 14, T4, ev4, setting="STPCTLS test set; mean of ten trials")
ROWS += [R("total_params", "2.25", 1, "Abstract", "with moderate computational resources (2.25M parameters, 1.11G FLOPs)", item="model1", unit="M"),
         R("flops", "1.11", 1, "Abstract", "with moderate computational resources (2.25M parameters, 1.11G FLOPs)", item="model1", unit="GFLOPs")]
ROWS += avail()
ROWS += [D("notes", "The proposed TM2F has no transformer component; the transformer-based models are the baselines OctFormer and PoinTramba. Tables III and IV are images (values read from the rendered page; evidence lists mean and std pairs for precision, recall, F1, OA, then kappa).", "Sec. V-C and Tables III to IV (pp. 13 to 14)")]
