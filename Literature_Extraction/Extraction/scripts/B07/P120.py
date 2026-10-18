import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P120", "Table 3 (image)"
# Eight point cloud deep learning models (PointNet, PointNet++ MSG and SSG, PointMLP, PointMLP-elite, PointConv, DGCNN and the attention-based PCT) classify the eight species of the TS8 individual tree point clouds
# (80/20 stratified split). Table 3 (the text layer of its rows is garbled, so its values are read from the rendered page) gives balanced accuracy, precision, recall, F and kappa on the training and test sets;
# the paper's precision, recall and F are weighted averages and the weighted recall is the overall accuracy (Sec. 3.4). Test OA (Re), F and kappa are recorded; Table 4 (ModelNet40 results from the literature) gives
# parameters and FLOPs.
TS = Tags(task="species_classification", level="tree", split="test", agg="single")
OBJ_EV = "the PointConv model achieved the highest classification accuracy for tree species classification"
DS_EV = "An individual tree point cloud data set for tree species classification, called TS8, was successfully built"
SPLIT_EV = "we used a stratified random sampling method within species, randomly selecting 80% of each tree species as the training data set and the remaining 20% as the test set"

ROWS = [
    R("task", "species_classification", 1, "Abstract", OBJ_EV, note="classification of individual tree point clouds of eight species"),
    R("output_unit", "tree_label", 1, "Abstract", "classify and identify the individual tree point clouds of eight tree species"),
    D("tree_setting", "mixed", "plots of planted and natural forests, a remote sensing experimental station and a forest farm in three study areas of China; not stated as mixed"),
    R("tree_taxa", "birch, larch, locust, willow, poplar, elm, eucalyptus and Chinese fir", 4, "Table 1", "Betula platyphylla Suk."),
    R("study_area", "Greater Khingan Station (GKS), Huailai Remote Sensing Comprehensive Experimental Station (HL) and Gaofeng forest farm (GF), China", 4, "Table 1", "Table 1. Study area and tree species information"),
    D("biome", "multiple", "boreal (Greater Khingan), temperate (Huailai) and subtropical (Gaofeng) study areas in China; coded from the study areas"),
    R("platform", "mobile_lidar", 1, "Abstract", "mobile laser scanning (MLS) data collected in the field", note="LiBackpack DGC50 backpack laser scanning (SLAM) system"),
    R("sensors", "lidar", 1, "Abstract", "mobile laser scanning (MLS) data collected in the field"),
    R("annotation_type", "image_label", 4, "Sec. 2.2", DS_EV, note="one species label per segmented single-tree point cloud, in the file organisation of ModelNet40"),
    R("dataset_name", "own TS8 backpack laser scanning individual tree point clouds of eight species (1312 trees)", 4, "Sec. 2.2", DS_EV, item="dataset1"),
    R("dataset_key", "own_p120_ts8", 4, "Sec. 2.2", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 2.2", SPLIT_EV, item="dataset1", note="80 percent of each species for training and 20 percent for testing"),
    D("dataset_public", "no", "Data Availability Statement: Not applicable (p.20); no data link in the paper", item="dataset1"),
    D("n_trees", "1312", "sum of the numbers of trees of the eight species printed in Table 1 (215, 295, 142, 174, 165, 85, 131 and 105)", item="dataset1", unit="trees"),
]
M = (("model1", "PointNet", "other_baseline", "other", "PointNet (shared MLPs with max pooling)", 8, "PointNet is a pioneering work that deals directly with disordered point sets using multiple shared MLPs", "MLP-based point network (pointwise MLP category of the paper)", 3.47, 0.45, "PointNet 2017 0.860 0.892 3.47 M 0.45 G [25]"),
     ("model2", "PointNet++ (MSG)", "other_baseline", "other", "PointNet++ with multi-scale grouping", 8, "PointNet++ introduces a hierarchical feature learning paradigm", "MLP-based point network (pointwise MLP category of the paper)", 1.74, 4.09, "PointNet++(MSG) 2017 — 0.919 1.74 M 4.09 G [26]"),
     ("model3", "PointNet++ (SSG)", "other_baseline", "other", "PointNet++ with single-scale grouping", 8, "PointNet++ introduces a hierarchical feature learning paradigm", "MLP-based point network (pointwise MLP category of the paper)", 1.48, 1.68, "PointNet++(SSG) 2017 — 0.907 1.48 M 1.68 G [26]"),
     ("model4", "PointMLP", "other_baseline", "other", "deep residual MLP network with a geometric affine module", 9, "PointMLP is a deep residual MLP network", "MLP-based point network (pointwise MLP category of the paper)", 12.6, None, "PointMLP 2022 0.914 0.945 12.6 M — [24]"),
     ("model5", "PointMLP-elite", "other_baseline", "other", "lightweight PointMLP with a quarter of the channels", 9, "which reduces the number of channels in the PointMLP intermediate FC layer by a factor of 4", "MLP-based point network (lightweight PointMLP)", 0.68, None, "PointMLP-elite 2022 0.907 0.940 0.68 M — [24]"),
     ("model6", "PointConv", "cnn_baseline", "cnn", "PointConv, a Monte Carlo approximation of 3D continuous convolution", 9, "PointConv [33] is one of the best convolution-based methods", "convolution-based point network", None, None, "PointConv 2018 — 0.925 — — [33]"),
     ("model7", "DGCNN", "cnn_baseline", "cnn", "dynamic graph CNN with EdgeConv", 10, "called EdgeConv", "graph convolutional network (protocol 4.5: cnn)", 1.81, 2.43, "DGCNN 2019 0.902 0.929 1.81 M 2.43 G [34]"),
     ("model8", "PCT", "transformer_baseline", "pure_vit", "Point Cloud Transformer: input embedding and four stacked offset-attention modules with local neighbourhood aggregation", 11, "The PCT model encoder is made up of an input embedding module and four stacked attention modules", "attention-based point network (transformer)", 2.88, 2.32, "PCT 2021 — 0.932 2.88 M 2.32 G [35]"))
for item, name, role, cat, bb, pg, ev, cnote, prm, fl, sizeev in M:
    ROWS += model(item, name, role, cat, bb, None, None, pg, "Sec. 3.2", ev, cat_note=cnote, pre_status="not_reported", pre_note="not stated",
                  strat_note="hyperparameters of Table 2 (original settings kept as far as possible); initialisation not stated")
    if prm:
        ROWS.append(R("total_params", str(prm), 16, "Table 4", sizeev, item=item, unit="M", note="from the literature review of Table 4 (models trained on ModelNet40), not measured on the tree data"))
    if fl:
        ROWS.append(R("flops", str(fl), 16, "Table 4", sizeev, item=item, unit="GFLOPs", note="from the literature review of Table 4 (ModelNet40 models)"))
res = Counter("res")
RID = {}
T3V = {"model1": ("PointNet", "0.6251 0.7288 0.6277 0.7492 0.6351 0.7241 0.6288 0.7183 0.5726 0.679", ("0.7241", "0.7183", "0.679")),
       "model2": ("PointNet++(MSG)", "0.9642 0.9768 0.9648 0.974 0.9646 0.9732 0.9645 0.9731 0.9587 0.9687", ("0.9732", "0.9731", "0.9687")),
       "model3": ("PointNet++(SSG)", "0.9579 0.9343 0.9579 0.9421 0.9579 0.9387 0.9578 0.9387 0.9508 0.9284", ("0.9387", "0.9387", "0.9284")),
       "model4": ("PointMLP", "0.9097 0.9827 0.931 0.9818 0.9301 0.9808 0.9294 0.9808 0.9181 0.9776", ("0.9808", "0.9808", "0.9776")),
       "model5": ("PointMLP-elite", "0.9467 0.9643 0.9562 0.9677 0.955 0.9655 0.955 0.9655 0.9473 0.9598", ("0.9655", "0.9655", "0.9598")),
       "model6": ("PointConv", "0.9432 0.9952 0.9507 0.9963 0.9505 0.9962 0.9505 0.9962 0.9423 0.9955", ("0.9962", "0.9962", "0.9955")),
       "model7": ("DGCNN", "0.9759 0.9614 0.9847 0.9648 0.9847 0.9647 0.9845 0.9647 0.9821 0.9588", ("0.9647", "0.9647", "0.9588")),
       "model8": ("PCT", "0.9321 0.9232 0.9343 0.9441 0.9343 0.9425 0.9342 0.9426 0.9234 0.9329", ("0.9425", "0.9426", "0.9329"))}
for item, (lab, vals, (oa, f, kp)) in T3V.items():
    ev = f"{lab} {vals}"
    st = "Table 3: test set (20 percent of each species), eight species"
    head = item == "model6"
    RID[item] = res()
    ROWS += result(RID[item], "oa", item, "dataset1", oa, 15, T3, ev, unit="fraction", setting=st, note="Re of the test set: the weighted average recall, which the paper defines as the overall accuracy (Acc)", **TS, primary=True, headline=head)
    ROWS += result(res(), "f1", item, "dataset1", f, 15, T3, ev, unit="fraction", setting=st, note="F of the test set (support-weighted average F-score)", **TS, primary=True)
    ROWS += result(res(), "kappa", item, "dataset1", kp, 15, T3, ev, unit="fraction", setting=st, note="kappa coefficient of the test set", **TS, primary=True)
ROWS += [R("limitations", "The PCT model did not reach a good accuracy, likely because of the small sample size, and the dataset covers eight species from three study areas of China", 1, "Abstract",
           "the PCT model did not achieve good accuracy in the tree species classification experiment, likely due to the small sample size"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "Data Availability Statement: Not applicable (p.20)")]
ROWS += [D("notes", "Table 3 (balanced accuracy, precision, recall, F and kappa of eight models on the training and test sets) is recorded for the test set: Re (the paper's overall accuracy), F and kappa; BAcc, Pr and the training set values are not. The text layer of Table 3 is garbled, so its values were read from the rendered page. The paper's Pr, Re and F are weighted averages over the species (Sec. 3.4). Table 4 repeats ModelNet40 results of the original papers with parameters and FLOPs, which are recorded as model sizes (literature values, not measured on TS8). The sampling experiment (NGFPS against FPS with PointNet++, Fig. 4) is not recorded. All metrics of PointConv, PointMLP and DGCNN are above 0.95 and PCT reaches 0.9425 recall on the test set.",
           "Sec. 3 to 4 and Tables 3 and 4 (pp. 8 to 16)")]
