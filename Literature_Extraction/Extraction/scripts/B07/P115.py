import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P115", "Table 2"
# PCTreeS (Point Cloud Transformer on 3D airborne LiDAR tree point clouds) against two ResNet18 CNNs on six orthogonal 2D projections (baseline) and on the projections stacked as channels (baseline++), six classes
# (five species and other). Table 2 gives the AUC and the overall accuracy of the last epoch; the split of the results is not stated. AUC is metric other (never primary).
TS = Tags(task="species_classification", level="tree", split="not_stated", agg="single")
OBJ_EV = "we introduce the approach of directly feeding 3D point cloud images into a vision transformer model (PCTreeS)"
DS_EV = "In total, there are 136,752 trees (main stems) with 67 species labels"

ROWS = [
    R("task", "species_classification", 1, "Abstract", OBJ_EV, note="tree species classification of single-tree airborne LiDAR point clouds"),
    R("output_unit", "tree_label", 1, "Abstract", OBJ_EV),
    R("tree_setting", "natural_forest", 1, "Introduction", "LiDAR data collected at the Mpala Research Center, located on the Laikipia Plateau, Kenya", note="savanna and woodland"),
    R("tree_taxa", "Acacia drepanolobium, Croton dichogamus, Euclea divinorum, Acacia brevispica, Acacia mellifera and an other class (41 species in the matched data)", 3, "Data Matching and Processing", "The resulting dataset contains 41 species of trees in total"),
    R("study_area", "Mpala Research Center, Laikipia Plateau, Kenya", 1, "Introduction", "LiDAR data collected at the Mpala Research Center, located on the Laikipia Plateau, Kenya"),
    R("biome", "tropical", 1, "Abstract", "we apply the deep learning framework for tree classification in tropical savannas"),
    R("platform", "uav", 2, "Data", "from a lowaltitude (50m) UAV"),
    R("sensors", "lidar", 1, "Abstract", "we use Airborne LiDAR images"),
    R("annotation_type", "image_label", 2, "Data", "The ground-truth labels {yi } were provided by ForestGEO’s Mpala plot census", note="species label of each tree from the ForestGEO census, matched to the segmented LiDAR trees by location"),
    R("dataset_name", "own UAV airborne LiDAR tree segments of the Mpala plot matched to ForestGEO census labels (about 4000 trees, six classes)", 2, "Data", DS_EV, item="dataset1"),
    R("dataset_key", "own_p115_mpala_lidar", 2, "Data", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Experiments and Results", "we trained models on images with over 1,000 points", item="dataset1", note="models trained on images with over 1000 points; the split into training, validation and test data is not described"),
    R("dataset_public", "on_request", 4, "Conclusion", "For data access, please contact the authors and Davies lab", item="dataset1", note="LiDAR images from the Davies Lab; the census data from ForestGEO by request"),
    R("n_trees", "4000", 3, "Data Matching and Processing", "allows us to match about ∼4,000 LiDAR images with ground-truth labels", item="dataset1", unit="trees", note="printed as about 4,000 matched LiDAR images; 43,709 trees were segmented from the scan"),
]
ROWS += model("model1", "PCTreeS (Point Cloud Transformer for tree species classification, proposed)", "proposed", "pure_vit", "Point Cloud Transformer (PCT) with neighbor embedding and offset-attention on 3D point clouds", None, None, 4, "Sec. 3D Point Cloud Transformer",
              "This paper implements PCT on our dataset by feeding pure point cloud images", cat_note="transformer with point and neighbor embedding and offset-attention (Guo et al. 2021)", pre_status="not_reported", pre_note="not stated",
              strat_note="trained with batch size 32, 100 epochs and learning rate 1e-5 like the baselines; initialisation not stated")
ROWS += model("model2", "Baseline (ResNet18 with 1/4 of the filters on six 2D projections)", "cnn_baseline", "cnn", "ResNet18 with 1/4 of the filters and single channel input; six orthogonal projections as separate data points", "none", "from_scratch", 3, "Methods",
              "The backbone CNN maintains a ResNet18 architecture with 1/4 the filters and single channel input", cat_note="ResNet18 convolutional network", pre_page=3, pre_loc="Methods", pre_ev="Since this is a custom architecture, there were no pre-trained weights",
              pre_note="custom architecture without pre-trained weights", strat_page=3, strat_loc="Methods", strat_ev="Since this is a custom architecture, there were no pre-trained weights", strat_note="no pre-trained weights")
ROWS += model("model3", "Baseline++ (the six projections as channels of one ResNet18 input, height normalised)", "cnn_baseline", "cnn", "ResNet18 with 1/4 of the filters on the six projections stacked as channels", "none", "from_scratch", 4, "Methods",
              "we feed all six projections as channels of the same data point into the same ResNet18 architecture", cat_note="ResNet18 convolutional network", pre_page=3, pre_loc="Methods", pre_ev="Since this is a custom architecture, there were no pre-trained weights",
              pre_note="custom architecture without pre-trained weights", strat_page=3, strat_loc="Methods", strat_ev="Since this is a custom architecture, there were no pre-trained weights", strat_note="no pre-trained weights")
res = Counter("res")
RID = {}
for item, lab, auc, acc in (("model2", "Baseline", "0.75", "0.68"), ("model3", "Baseline++", "0.75", "0.70"), ("model1", "PCTreeS", "0.81", "0.72")):
    ev = f"{lab} {auc} {acc}"
    st = "Table 2: last epoch, six classes (five species and other)"
    RID[(item, "oa")] = res()
    ROWS += result(RID[(item, "oa")], "oa", item, "dataset1", acc, 4, T2, ev, unit="fraction", setting=st, note="overall accuracy of the last epoch (printed as a fraction)", **TS, primary=True, headline=(item == "model1"))
    ROWS += result(res(), "other", item, "dataset1", auc, 4, T2, ev, unit="fraction", setting=st, note="AUC of the last epoch (area under the ROC curve; averaging over the classes is not stated)", **TS)
ROWS += [R("limitations", "The overall accuracy is lower than that of terrestrial LiDAR studies because of the lower point density and errors of the tree segmentation (cropping, sparse points, merged trees, non-tree objects)", 4, "Experiments and Results",
           "While our best performance in overall accuracy is lower than Allen et al. 2022 (81%)"),
         R("code_url", "https://github.com/mattynaz/pctrees", 4, "Conclusion", "Available at https://github.com/mattynaz/pctrees"), R("code_available", "yes", 4, "Conclusion", "we have made the repository public"),
         D("weights_available", "not_stated", "No statement about trained weights"), R("data_available", "on_request", 4, "Conclusion", "For data access, please contact the authors and Davies lab")]
ROWS += [D("notes", "Table 2 compares the 2D CNN baseline, baseline++ and PCTreeS by AUC, overall accuracy and training time (about 90, 90 and 45 minutes); the values are those of the last epoch and the paper does not say whether they are computed on a test or validation set (split not_stated). The classes are the five most common species (90.9 percent of the matched images) and an other class; trees with fewer than 1000 points were filtered out. The paper compares its overall accuracy (0.72) with that of Allen et al. 2022 (81 percent, terrestrial LiDAR).",
           "Sec. Experiments and Results and Table 2 (p. 4)")]
