import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T4 = "Table 4 (image)"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "Six different 3D deep learning neural networks and random forest algorithms were trained and tested to classify the point clouds of three tree species"),
    R("output_unit", "tree_label", 7, "Sec. 2.3.3", "the point cloud system of single trees segmented in this study was used as the ModelNet40-like data set"),
    R("tree_setting", "plantation", 3, "Sec. 2.1", "There are 46 sample plots and 18 kinds of artificial forests in the forest farm"),
    R("tree_taxa", "birch; Quercus mongolica; Pinus sylvestris", 7, "Table 2", "Birch ... Quercus mongolica ... Sylvestris"),
    R("study_area", "urban forestry demonstration base of Northeast Forestry University, Harbin, Heilongjiang, China; 43.95 ha", 3, "Sec. 2.1",
      "the urban forestry demonstration base of Northeast Forestry University ... Harbin, Heilongjiang Province ... covering an area of 43.95 ha"),
    D("biome", "temperate", "Harbin, northeast China (cold temperate); not stated in the paper"),
    R("platform", "uav", 4, "Sec. 2.2", "carried by DJI MTK R300", note="DJI M300 RTK with Zenmuse L1 (Table 1)"),
    R("sensors", "lidar", 4, "Table 1", "Zenmuse L1 Settings"),
    R("point_density", "1772", 4, "Table 1", "1772 points/m2", unit="pts/m2"),
    R("annotation_type", "image_label", 7, "Sec. 2.3.3", "The files were converted to txt format, saved in different folders according to different labels",
      note="one species label per segmented tree point cloud"),
    R("dataset_name", "own UAV LiDAR single-tree point clouds (ModelNet3), Northeast Forestry University", 7, "Sec. 2.3.3", "Finally, ModelNet3 was obtained, which belongs to us", item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 2.3.3", "divided into a training set and test set according to an 8:2 ratio", item="dataset1", note="1600 training and 400 test samples after augmentation (Table 2)"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    R("n_images", "2000", 7, "Sec. 2.3.3", "a total of 2000 files were obtained for three species of trees", item="dataset1", unit="samples", note="after augmentation"),
    R("n_trees", "1109", 6, "Sec. 2.3.2", "In this way, a total of 1109 trees were segmented as experimental data", item="dataset1", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the study data; initialisation not stated")
M = [("model1", "Point Cloud Transformer (PCT)", "proposed", "pure_vit", "PCT", "88.3", "0.82", "PCT 88.3 0.82", "the study's focal model (published PCT)"),
     ("model2", "Point Transformer (Zhao et al.), PT1", "transformer_baseline", "pure_vit", "Point Transformer", "87.3", "0.80", "PT1 87.3 0.80", ""),
     ("model3", "Point Transformer (Engel et al.), PT2", "transformer_baseline", "pure_vit", "Point Transformer", "87.8", "0.81", "PT2 87.8 0.81", ""),
     ("model4", "PointNet++ (MSG)", "other_baseline", "other", "PointNet++ multi-scale grouping", "85.5", "0.77", "PN + MSG 85.5 0.77", "MLP-based point network"),
     ("model5", "PointNet++ (SSG)", "other_baseline", "other", "PointNet++ single-scale grouping", "85.0", "0.76", "PN + SSG 85.0 0.76", "MLP-based point network"),
     ("model6", "PointNet", "other_baseline", "other", "PointNet", "80.5", "0.70", "PointNet 80.5 0.70", "MLP-based point network"),
     ("model7", "Random forest", "other_baseline", "classical", "random forest on point-cloud features", "63.3", "0.43", "RF 63.3 0.43", "classical machine learning")]
res = Counter("res")
for item, name, role, cat, bb, oa, kp, ev, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, 12, T4, ev, cat_note=note, **NP)
    st = "test set (400 samples, 3 species); 4096 points per tree"
    ROWS += result(res(), "oa", item, "dataset1", oa, 12, T4, ev, setting=st, headline=(item == "model1"))
    ROWS += result(res(), "kappa", item, "dataset1", kp, 12, T4, ev, unit="fraction", setting=st)
ROWS += avail()
ROWS += [D("notes", "The PDF carries an overlaid peer-review text layer, so Tables 2 and 4 were read from the rendered pages. Accuracy depends on the number of sampled points (Fig. 13); 4096 points was best.", "Sec. 2.3 and 3 (pp. 6 to 13)")]
