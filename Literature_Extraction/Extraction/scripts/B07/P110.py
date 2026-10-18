import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T9 = "P110", "Table 9"
# Species classification of single-tree airborne multispectral LiDAR point clouds (AACB Transformer: dual-branch point network with cross-branch attention). Table 9 compares eight models on the species
# level (G1 sample group, test set of 337 trees, 256 input points), one run; OA and the macro F1 (mF1) are recorded, the precision and recall means are not. The genus and foliage type results of the proposed
# network (Tables 7 and 8) are not primary. The one-branch network is an ablation variant of the proposed model and is not primary.
TS = Tags(task="species_classification", level="tree", split="test", agg="single")
OBJ_EV = "This study therefore explores the feasibility of using a deep learning (DL) framework for processing low-density AML point clouds to enhance tree species classification"
DS_EV = "This study was tested with an average 9 points/m2 AML point cloud for 6 tree species"
SPLIT_EV = "randomly split into training and testing sets with a fixed ratio of 7:3,"
ACC_EV = "overall accuracies achieved 83.1 %, 85.8 %, and 95.3 % at species, genus, and leaftype levels, respectively"
BASE_EV = "PointNet (Qi et al., 2017a), PointNet++ (Qi et al., 2017b), PCT (Guo et al., 2021), and PT (Zhao et al., 2021a)"

ROWS = [
    R("task", "species_classification", 1, "Abstract", OBJ_EV, note="tree species, genus and foliage type classification of single trees"),
    R("output_unit", "tree_label", 1, "Abstract", "Cross-Branch Attention modules named Attribute-Aware Cross-Branch (AACB) Transformer"),
    R("tree_setting", "mixed", 3, "Sec. 2.1.1", "comprises a diverse mix of natural forest, plantation, and reforestation"),
    R("tree_taxa", "Populus tremuloides, Larix laricina, Acer saccharum, Picea abies, Pinus resinosa, Pinus strobus", 1, "Abstract", DS_EV),
    R("study_area", "York Regional Forest, York Region, Ontario, Canada", 3, "Sec. 2.1.1", "situated in York Region, Ontario, Canada"),
    D("biome", "temperate", "Ontario, Canada; coded from the study area"),
    R("platform", "airborne", 2, "Sec. 1", "Airborne Multispectral LiDAR"),
    R("sensors", "lidar", 3, "Table 1", "Wavelength (nm) 1550 1064 532", note="three-wavelength (multispectral) airborne laser scanning, Titan AML"),
    R("point_density", "9", 1, "Abstract", DS_EV, unit="pts/m2"),
    R("annotation_type", "image_label", 5, "Sec. 2.2.2.1", "manually delineate a set of individual-tree samples", note="one species label per manually delineated single-tree point cloud, from fieldwork records"),
    R("dataset_name", "own airborne multispectral LiDAR single trees of six species, York Regional Forest (1122 trees)", 1, "Abstract", DS_EV, item="dataset1"),
    R("dataset_key", "own_p110_york_aml", 1, "Abstract", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 2.2.2.3", SPLIT_EV, item="dataset1", note="G1 sample group: 1122 trees split into 785 training and 337 test trees; training samples augmented to 3140 (Table 3)"),
    R("dataset_public", "on_request", 15, "Data availability", "Data will be made available on request", item="dataset1"),
    R("n_trees", "1122", 6, "Sec. 2.2.2.3", "all 1122 samples were split into", item="dataset1", unit="trees", note="Table 2: 108 trembling aspen, 323 larch, 115 sugar maple, 109 Norway spruce, 320 red pine, 147 white pine"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the augmented G1 samples; initialisation not stated")
ROWS += model("model1", "AACB Transformer (Attribute-Aware Cross-Branch Transformer, proposed)", "proposed", "cnn_transformer_hybrid", "dual-branch point network (geometry branch and radiometric branch): stem MLPs, two GLocal modules (multi-scale grouping with graph convolution, global feature extraction) and Cross-Branch Attention",
              None, None, 7, "Sec. 2.2.3.1", "The overall architecture of our proposed network comprises", cat_note="point network with GLocal modules and cross-branch attention (Q, K, V exchanged between branches)",
              pre_status="not_reported", pre_note="not stated", strat_page=8, strat_loc="Sec. 2.2.3", strat_ev="we employed the Adam optimizer with a weight decay of 0.0001",
              strat_note="Adam, cosine annealing, at most 400 epochs, batch size 16; initialisation not stated")
ROWS += model("model2", "SVM", "other_baseline", "classical", "support vector machine on 29 selected geometric, radiometric and return features (linear kernel)", None, None, 8, "Sec. 2.2.4",
              "we compared it against two well-established and widely used ML methods, Support Vector Machine (SVM) and Random Forests (RF), in tree species classification", cat_note="support vector machine on hand-crafted features", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model3", "RF", "other_baseline", "classical", "random forest (165 trees, depth 13) on 29 selected features", None, None, 8, "Sec. 2.2.4",
              "we compared it against two well-established and widely used ML methods, Support Vector Machine (SVM) and Random Forests (RF), in tree species classification", cat_note="random forest on hand-crafted features", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model4", "PointNet", "other_baseline", "other", "PointNet", None, None, 9, "Sec. 2.2.4", BASE_EV, cat_note="MLP-based point network", **NPRE)
ROWS += model("model5", "PointNet++", "other_baseline", "other", "PointNet++ (multi-scale grouping)", None, None, 9, "Sec. 2.2.4", BASE_EV, cat_note="MLP-based point network", **NPRE)
ROWS += model("model6", "PCT", "transformer_baseline", "pure_vit", "Point Cloud Transformer", None, None, 9, "Sec. 2.2.4", "PCT and PT are the most widely used transformer-based networks", cat_note="transformer-based point network (global attention)", **NPRE)
ROWS += model("model7", "PT (Point Transformer)", "transformer_baseline", "pure_vit", "Point Transformer", None, None, 9, "Sec. 2.2.4", "PCT and PT are the most widely used transformer-based networks", cat_note="transformer-based point network (local attention)", **NPRE)
ROWS += model("model8", "One branch (AACB network without the secondary branch and the CBA modules)", "other_baseline", None, "the proposed network with all attributes input to the primary branch", None, None, 10, "Sec. 3.2", "the “One branch” method was also tested",
              cat_note="ablation of the proposed network; the paper does not say which attention blocks remain", **NPRE)
res = Counter("res")
LAB = {"model2": "SVM", "model3": "RF", "model4": "PointNet", "model5": "PointNet++", "model6": "PCT", "model7": "PT", "model8": "One branch", "model1": "Ours"}
RID = {}
for item in ("model2", "model3", "model4", "model5", "model6", "model7", "model8", "model1"):
    n, ev = row(PID, 10, rf"^\s*(?:ML-based methods\s+|DL-based methods\s+)?{LAB[item].replace('+', r'[+]')}\s+(?=\d)", 0)
    oa, mf1 = n[0], n[3]
    evs = f"{LAB[item]} {n[0]} {n[1]} {n[2]} {n[3]}"
    assert evs in ev, (evs, ev)
    prim = item != "model8"
    st = "Table 9: species level, test set (G1 samples with augmentation, 256 points), one run"
    RID[item] = res()
    ROWS += result(RID[item], "oa", item, "dataset1", oa, 10, T9, evs, unit="%", setting=st, note="overall accuracy of the six-species classification", **TS, primary=prim, headline=(item == "model1"))
    ROWS += result(res(), "f1", item, "dataset1", mf1, 10, T9, evs, unit="%", setting=st, note="mF1: F1 averaged over the six species (the paper's mF1)", **TS, primary=prim)
# genus and foliage type levels of the proposed network (Tables 7 and 8), not primary
ROWS += result(res(), "oa", "model1", "dataset1", "85.8", 1, "Abstract", ACC_EV, unit="%", setting="Table 7: genus level (five genera), test set", note="overall accuracy of the genus classifier (Table 7 gives 85.8)", **TS)
ROWS += result(res(), "f1", "model1", "dataset1", "81.7", 10, "Table 7", "Mean 82.3 81.6 81.7", unit="%", setting="Table 7: genus level (five genera), test set", note="mF1 of the genus classifier (mean over the five genera)", **TS)
ROWS += result(res(), "oa", "model1", "dataset1", "95.3", 1, "Abstract", ACC_EV, unit="%", setting="Table 8: foliage type level (broadleaf and needle leaf), test set", note="overall accuracy of the foliage type classifier (Table 8)", **TS)
ROWS += result(res(), "f1", "model1", "dataset1", "92.7", 10, "Sec. 3.1", "were 92.3 %, 93.2 %, and 92.7 %", unit="%", setting="Table 8: foliage type level (broadleaf and needle leaf), test set", note="mF1 of the foliage type classifier (92.3 mPre, 93.2 mRec)", **TS)
ROWS += [R("limitations", "Few training samples (ground truth in forests is hard to obtain), one study area and six species, and sparse point clouds (low point density)", 14, "Sec. 4.7",
           "The DL-based network usually requires a huge amount training samples, but acquiring ground truth in a forest environment is quite hard"),
         D("code_available", "no", "No code repository or availability statement for code in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "on_request", 15, "Data availability", "Data will be made available on request")]
ROWS += [D("notes", "Table 9 compares SVM, RF, PointNet, PointNet++, PCT, PT, the one-branch network and the AACB Transformer on the species level (OA, mPre, mRec, mF1 and their differences from the proposed model); OA and mF1 are recorded for all eight and the one-branch ablation is not primary. The species result of the proposed model is the same in Table 6 (confusion matrix of 337 test trees) and Table 9. The genus and foliage type results (Tables 7 and 8, OA 85.8 and 95.3) are recorded for the proposed network without primary. Tables 10 to 12 (channel merging, neighbour numbers, input fields, other sample groups; OA up to 87.9 in other settings) and the 10-fold cross-validation with 25 percent of the training samples of Fig. 10 (box plot) are not recorded.",
           "Sec. 3 to 4 and Tables 6 to 9 (pp. 9 to 10)")]
