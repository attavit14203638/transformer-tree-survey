import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T5, T6 = "P102", "Table V (image)", "Table VI (image)"
# Species classification of single-tree ground LiDAR point clouds (MFFTC-Net: BDPS sampling, umbrella RepSurf, set abstraction and a transformer feature interaction). Tables II and IV to XI are images. Table V (self-built
# dataset, first listed) is the main comparison and primary; Table VI (STPCTLS) is not. The paper repeats all experiments five times and averages the metrics. The kappa columns are Kappa x 100 (percent scale).
TS = Tags(task="species_classification", level="tree", split="test", agg="mean")
OBJ_EV = "Utilizing ground-based light detection and ranging (LiDAR) systems"
DS1_EV = "In the Saihanba (SHB) Forest, Zhangjiakou City, Hebei"
SPLIT1_EV = "divided into training and testing sets in a 7:3 ratio"
SPLIT2_EV = "divide the data into training and testing sets using an 8:2 ratio"
RUN_EV = "experiments five times under the same conditions and averaged the"

ROWS = [
    R("task", "species_classification", 1, "Abstract", "achieving the highest overall accuracy and kappa", note="tree species classification of single-tree point clouds"),
    R("output_unit", "tree_label", 1, "Abstract", "achieving the highest overall accuracy and kappa"),
    R("tree_setting", "mixed", 6, "Sec. III-A", "encompass naturally regenerated forests and", note="self-built plots: natural forests and managed plantations in China; STPCTLS: close-range scans of forests in Germany and the USA"),
    R("tree_taxa", "Schima superba, lychee, larch, birch (self-built); beech, Douglas fir, oak, ash, spruce, pine, red oak (STPCTLS)", 6, "Sec. III-A", "Our study area includes four different tree species:"),
    R("study_area", "Saihanba and Laohugou (Hebei) and Luotian (Shenzhen), China; STPCTLS: Germany and the USA", 6, "Sec. III-A", DS1_EV),
    D("biome", "multiple", "Hebei (temperate), Shenzhen (subtropical), Germany and USA forests (temperate); not stated in the paper"),
    D("platform", "multiple", "terrestrial laser scanning (Riegl VZ-1000, Faro F350), backpack (LiBackpack DGC50) and handheld (Lixel L1) laser scanning; STPCTLS: Faro Focus 3D 120 and Z+F Imager 5006"),
    R("sensors", "lidar", 6, "Sec. III-A", "We employed the Riegl VZ-1000, a terrestrial laser scanning (TLS), for its rapid"),
    R("annotation_type", "image_label", 6, "Sec. III-A", "Our study area includes four different tree species:", note="one species label per segmented single-tree point cloud"),
    R("dataset_name", "own self-built dataset of four species (Schima superba, lychee, larch, birch) from TLS, BLS and HLS, China", 6, "Sec. III-A", DS1_EV, item="dataset1"),
    R("dataset_key", "own_p102_china_four_species", 6, "Sec. III-A", DS1_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. III-A", SPLIT1_EV, item="dataset1", note="7:3 split; augmentation only in the training set (Table II: 728 original and 2380 augmented training trees, 312 test trees)"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_trees", "1040", 8, "Sec. III-C", "1040 samples among four tree categories", item="dataset1", unit="trees", note="Table II: 110 Schima superba, 30 lychee, 450 larch and 450 birch"),
    R("dataset_name", "STPCTLS single tree point clouds from terrestrial laser scanning (Seidel), seven species", 6, "Sec. III-A", "single-tree point clouds from terrestrial laser scanning", item="dataset2"),
    R("dataset_key", "stpctls", 6, "Sec. III-A", "single-tree point clouds from terrestrial laser scanning", item="dataset2"),
    R("dataset_role", "train_test", 7, "Sec. III-A", SPLIT2_EV, item="dataset2", note="8:2 split (Table II: 556 original and 2644 augmented training trees, 135 test trees)"),
    D("dataset_public", "yes", "public dataset of Seidel et al. cited in Sec. III-A", item="dataset2"),
    R("n_trees", "691", 7, "Table II (image)", "Total 691 556 2644 135", item="dataset2", unit="trees", note="total sample size of Table II (164 beech, 183 Douglas fir, 22 oak, 39 ash, 158 spruce, 25 pine, 100 red oak)"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained separately on each dataset (five runs averaged); initialisation not stated")
M = [  # item, name, role, category, backbone, evidence (Table IV image), note
    ("model1", "PointNet", "other_baseline", "other", "PointNet", "PointNet [30] CVPR(2017) It is the first deep learning method to directly extract global features from point clouds", "listed under MLP in Table IV"),
    ("model2", "PointNet++ (SSG)", "other_baseline", "other", "PointNet++ single-scale grouping", "PointNet++ [31] NIPS(2017) divided into single-scale (SSG) and multi-scale (MSG) feature aggregation strategies", "listed under MLP in Table IV"),
    ("model3", "PointNet++ (MSG)", "other_baseline", "other", "PointNet++ multi-scale grouping", "PointNet++ [31] NIPS(2017) divided into single-scale (SSG) and multi-scale (MSG) feature aggregation strategies", "listed under MLP in Table IV"),
    ("model4", "PointMLP", "other_baseline", "other", "residual point MLP blocks", "PointMLP [49] ICLR(2022) progressively extracts local features through residual point MLP blocks", "listed under MLP in Table IV"),
    ("model5", "RepSurf-U", "other_baseline", "other", "RepSurf umbrella surface features", "RepSurf-U [42] CVPR(2022) describes localized structures through the calculation of umbrella features", "listed under MLP in Table IV"),
    ("model6", "PCT", "transformer_baseline", "pure_vit", "offset-attention point cloud transformer", "PCT [50] CVPR(2021) an offset-attention module, which has inherent permutation invariance", "listed under Attention in Table IV"),
    ("model7", "Point Transformer (PT)", "transformer_baseline", "pure_vit", "vector attention point transformer", "PT [51] ICCV(2021) substituting scalar attention in the self-attention module with vector attention", "3-D Transformer network (Table IV)"),
    ("model8", "PT2 (dual-branch local-global attention network)", "transformer_baseline", "pure_vit", "dual-branch local and global feature extraction with local-global attention", "PT2 [52] Access(2021) linking these features through a local-global attention mechanism", "listed under Attention in Table IV"),
    ("model9", "PointConT", "transformer_baseline", "pure_vit", "Transformer module based on point cloud content", "PointConT [45] JAS(2023) Proposing a Transformer module based on point cloud content", "listed under Attention in Table IV"),
    ("model10", "PointConv", "cnn_baseline", "cnn", "PointConv operator (3-D continuous convolution)", "PointConv [53] CVPR(2019) achieving a Monte Carlo approximation of 3-D continuous convolution", "listed under Conv in Table IV"),
    ("model11", "DGCNN", "cnn_baseline", "cnn", "EdgeConv on a graph of the points", "DGCNN [54] TOG(2019) introducing a novel convolutional operator known as EdgeConv", "graph convolution (listed under Graph in Table IV; protocol 4.5: cnn)"),
]
for item, name, role, cat, bb, ev, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, 9, "Table IV (image)", ev, cat_note=note, **NPRE)
ROWS += model("model12", "MFFTC-Net (proposed)", "proposed", "cnn_transformer_hybrid", "boundary-driven point sampling, umbrella RepSurf, set abstraction and a transformer feature interaction (multi-pooling, content-based attention)", None, None, 1, "Abstract",
              "set abstraction for local information capture and transformer-based feature interaction", cat_note="set abstraction (MLP) with a transformer-based feature interaction module (TFI); built on RepSurf-U and PointConT attention", **NPRE)
ROWS += model("model13", "MFFTC-Net without data augmentation (ablation)", "proposed", "cnn_transformer_hybrid", "MFFTC-Net trained without data augmentation", None, None, 9, "Table V (image)", "MFFTC-Net (without data augmentation) 100.00 100.00 97.25 99.40 98.53 97.60",
              cat_note="ablation of MFFTC-Net", **NPRE)
res = Counter("res")
RID = {}
# Table V (self-built dataset, p.9) and Table VI (STPCTLS, p.10): OA and Kappa x 100 of 12 models and the variant without augmentation
V5 = {"model1": ("PointNet", "97.06 100.00 93.71 99.21", "96.47", "94.25"), "model2": ("PointNet++(SSG)", "100.00 96.00 97.05 97.33", "97.44", "95.82"), "model3": ("PointNet++(MSG)", "100.00 100.00 98.93 96.26", "97.88", "96.55"),
      "model4": ("PointMLP", "97.06 100.00 96.99 100.00", "98.33", "97.28"), "model5": ("RepSurf", "100.00 100.00 98.53 98.24", "98.59", "97.70"), "model6": ("PCT", "96.73 70.30 90.87 100.00", "94.23", "90.61"),
      "model7": ("PT1", "78.57 0.00 92.73 100.00", "93.72", "89.72"), "model8": ("PT2", "100.00 90.00 98.08 98.37", "98.14", "96.97"), "model9": ("PointConT", "86.84 100.00 91.84 100.00", "94.55", "91.10"),
      "model10": ("PointConv", "97.06 100.00 98.51 97.79", "98.08", "96.86"), "model11": ("DGCNN", "97.06 100.00 98.54 100.00", "99.04", "98.43"), "model12": ("MFFTC-Net(ours)", "100.00 100.00 98.54 100.00", "99.36", "98.96"),
      "model13": ("MFFTC-Net (without data augmentation)", "100.00 100.00 97.25 99.40", "98.53", "97.60")}
V6 = {"model1": ("PointNet", "92.31 100.00 12.00 22.22 84.00 15.38 75.00", "51.85", "44.45"), "model2": ("PointNet++(SSG)", "68.74 86.55 0.00 44.67 74.14 36.48 69.11", "69.04", "61.52"),
      "model3": ("PointNet++(MSG)", "66.82 93.97 100.00 62.62 91.67 80.00 81.06", "81.63", "76.64"), "model4": ("PointMLP", "82.69 92.44 63.33 61.51 92.95 67.00 79.87", "84.59", "80.60"),
      "model5": ("RepSurf-U", "79.63 88.61 32.00 70.76 87.00 46.11 86.39", "80.00", "74.83"), "model6": ("PCT", "42.04 96.92 0.00 28.33 49.53 25.00 41.48", "53.78", "42.00"),
      "model7": ("PT1", "92.60 88.24 26.83 39.06 72.44 32.00 66.59", "68.89", "61.76"), "model8": ("PT2", "74.86 88.30 53.33 31.87 74.69 52.57 58.31", "70.52", "63.15"),
      "model9": ("PointConT", "86.96 100.00 20.00 42.86 71.79 57.14 66.67", "74.81", "68.70"), "model10": ("PointConv", "92.86 94.44 100.00 40.00 90.63 37.50 89.47", "85.19", "81.43"),
      "model11": ("DGCNN", "84.85 100.00 28.57 66.67 93.94 75.00 93.33", "87.41", "84.18"), "model12": ("MFFTC-Net(ours)", "87.10 97.14 100.00 66.67 93.75 71.43 94.74", "90.37", "87.89"),
      "model13": ("MFFTC-Net (without data augmentation)", "83.33 89.47 50.00 80.00 86.32 50.00 73.68", "81.48", "76.40")}
for tbl, loc, pg, ds, dn, table in ((T5, T5, 9, "dataset1", "self-built dataset", V5), (T6, T6, 10, "dataset2", "STPCTLS", V6)):
    for item, (lab, prec, oa, kp) in table.items():
        ev = f"{lab} {prec} {oa} {kp}"
        prim = (ds == "dataset1" and item != "model13")
        st = f"{loc.split(' (')[0]}: seven or four species classification on the {dn}, test set, mean of five runs"
        ROWS += result(res(), "oa", item, ds, oa, pg, loc, ev, unit="%", setting=st, note="overall accuracy", **TS, primary=prim, headline=False)
        RID[(ds, item, "kappa")] = res()
        ROWS += result(RID[(ds, item, "kappa")], "kappa", item, ds, kp, pg, loc, ev, unit="%", setting=st, note="the table column is Kappa x 100 (the text gives kappa as 0.9896 and 0.8789 for MFFTC-Net)", **TS, primary=prim)
# Table XI (p.13, image): different training proportions (OA of MFFTC-Net)
d = Counter("de")
for ratio, share, a, b in (("5:5", "50", "98.62", "81.86"), ("6:4", "60", "98.46", "84.01"), ("7:3", "70", "99.36", "87.75"), ("8:2", "80", "99.42", "90.37"), ("9:1", "90", "99.81", "90.90")):
    for dsn, v in (("self-built dataset", a), ("STPCTLS dataset", b)):
        ev = f"{ratio} {a} {b}"
        x = d()
        ROWS += [R("label_model", "model12", 13, "Table XI (image)", ev, item=x), D("label_amount", share, f"training share of all samples from the train:test ratio {ratio} of Table XI (page 13)", item=x),
                 R("label_unit", "percent_of_training_set", 13, "Table XI (image)", ev, item=x, note=f"share of all samples used for training (train:test ratio {ratio}); the test set changes with the ratio; {dsn}"),
                 R("label_metric", "oa", 13, "Table XI (image)", ev, item=x), R("label_value", v, 13, "Table XI (image)", ev, item=x, unit="%", note=dsn)]
ROWS += conflict("cf1", "dataset2.n_trees", "690", 8, "Sec. III-C", "which comprises 690 samples across seven categories", "V1c",
                 "Table II (the dataset table) gives 691 samples for STPCTLS (the seven species sum to 691); the text of Sec. III-C gives 690", unit="trees")
ROWS += [R("limitations", "On STPCTLS the classes with few samples (ash and pine) are classified poorly by all models and beech and oak are confused; on the self-built data two birch trees were taken for larch, probably through seasonal changes", 10, "Sec. III-D", "sification performance for Ash and Pine species was suboptimal", note="STPCTLS: 32 ash and 20 pine trees in the original training sets (Table II)"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "partial", "STPCTLS is public; the self-built dataset has no availability statement")]
ROWS += [D("notes", "Tables II and IV to XI are images. Tables V (four species, self-built dataset) and VI (seven species, STPCTLS) give the class precision, OA and Kappa x 100 of 11 comparison models, MFFTC-Net and MFFTC-Net without data augmentation; OA and kappa are recorded for all, Table V is primary (the first listed dataset) and Table VI is not. All experiments are repeated five times and averaged. Table II prints 998 for the original training total of the self-built dataset, but its species rows (77, 21, 315, 315) sum to 728 (1040 minus 312 test trees); the total is not recorded. Tables VII to X (ablation, BDPS weights and intervals, receptive field K) and the time per epoch of Fig. 14 are not recorded; Table XI (different training proportions) is recorded as data-efficiency points.",
           "Sec. III and Tables II to VI and XI (pp. 7 to 13)")]
