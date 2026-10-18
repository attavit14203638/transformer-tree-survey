import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T1, T2 = "P078", "Table 1", "Table 2"
URL = "https://bxiang233.github.io/FF3D/"
ROWS = [
    R("task", "tree_segmentation_3d;component_segmentation", 1, "Abstract", "a new unified and end-to-end framework designed for precise individual tree and semantic segmentation",
      note="individual tree instances plus ground, wood and leaf classes"),
    R("output_unit", "tree_points;tree_component", 6, "Sec. 4.1", "Predicted tree instances are matched to GT trees based on the IoU of their respective point sets"),
    NR("tree_setting", note="forest types of the FOR-instanceV2 regions are given only in the supplementary statistics of the ICCV paper, not in the cited main text; the two test sets are natural broadleaved temperate forests (Sec. 4.2, p.7)"),
    NR("tree_taxa", note="species are given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
    NR("study_area", note="the regions are named in Sec. 4.1 and mapped in Fig. 3; their countries are given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
    R("biome", "multiple", 6, "Sec. 4.1", "data from tropical regions (Yuchen)", note="tropical (Yuchen) and temperate (the two test sets, Sec. 4.2, p.7); the other regions are given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
    R("platform", "multiple", 6, "Sec. 4.1 and 4.2", "including Mobile Laser Scanning (MLS) data (NIBIO MLS) collected via backpack LiDAR ... struggles with ULS data (e.g., NIBIO2, RMIT and Yuchen)",
      note="UAV and mobile (backpack) laser scanning named in the text; the sensors are given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
    R("sensors", "lidar", 6, "Sec. 4.1", "Mobile Laser Scanning (MLS) data (NIBIO MLS) collected via backpack LiDAR"),
    R("annotation_type", "point_label", 3, "Sec. 3", "The semantic segmentation involves three primary classes in forest environments: ground, wood and leaf",
      note="per-point semantic classes and tree instances"),
    R("dataset_name", "FOR-instanceV2", 1, "Abstract", "the newly introduced FOR-instanceV2 dataset, which spans diverse forest types and regions", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 4.2", "All baseline methods and our proposed ForestFormer3D were trained on the FOR-instanceV2 training set and evaluated on the corresponding test split", item="dataset1"),
    R("dataset_public", "yes", 1, "Abstract", f"The FOR-instanceV2 dataset and the ForestFormer3D code are publicly available at {URL}", item="dataset1"),
    R("dataset_url", URL, 1, "Abstract", f"publicly available at {URL}", item="dataset1"),
    NR("n_trees", item="dataset1", note="the main text gives instance counts only for BlueCat (6.3k), NIBIO2 (3k) and FOR-instance (1.1k) (Sec. 4.1); the total is given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
    R("dataset_name", "Wytham woods [8]", 7, "Sec. 4.2", "we test on Wytham woods [8] and LAUTx [1] datasets", item="dataset2"),
    R("dataset_role", "transfer_target", 7, "Sec. 4.2", "To evaluate the transferability of our model, we test on Wytham woods [8] and LAUTx [1] datasets", item="dataset2"),
    R("dataset_public", "yes", 7, "Sec. 4.2", "Wytham woods [8]", item="dataset2", note="published dataset"),
    NR("n_trees", item="dataset2", note="given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
    R("dataset_name", "LAUTx [1]", 7, "Sec. 4.2", "we test on Wytham woods [8] and LAUTx [1] datasets", item="dataset3"),
    R("dataset_role", "transfer_target", 7, "Sec. 4.2", "To evaluate the transferability of our model, we test on Wytham woods [8] and LAUTx [1] datasets", item="dataset3"),
    R("dataset_public", "yes", 7, "Sec. 4.2", "LAUTx [1]", item="dataset3", note="published dataset"),
    NR("n_trees", item="dataset3", note="given only in the supplementary statistics of the ICCV paper, not in the cited main text"),
]
TRN = dict(strat_page=6, strat_loc="Sec. 4.2", strat_ev="All baseline methods and our proposed ForestFormer3D were trained on the FOR-instanceV2 training set")
M = [  # item, Table 1 label, Table 2 label, name, role, category, backbone
    ("model1", r"ForestFormer3D \(ours\)", r"ForestFormer3D", "ForestFormer3D", "proposed", "cnn_transformer_hybrid", "3D sparse U-Net with transformer decoder"),
    ("model2", r"OneFormer3D \[30\]", r"OneFormer3D \[30\]", "OneFormer3D", "transformer_baseline", "cnn_transformer_hybrid", "3D sparse U-Net with transformer decoder"),
    ("model3", r"ForAINetV2 R16", r"ForAINetV2 R16", "ForAINetV2 (16 m input radius)", "cnn_baseline", "cnn", "3D sparse convolutional network"),
    ("model4", r"TreeLearn \[24\]", r"TreeLearn \[24\]", "TreeLearn", "cnn_baseline", "cnn", "3D sparse convolutional network"),
    ("model5", r"ForAINetV2 R8", None, "ForAINetV2 (8 m input radius)", "cnn_baseline", "cnn", "3D sparse convolutional network"),
]
res, tr = Counter("res"), Counter("tr")
for item, l1, l2, name, role, cat, bb in M:
    n1, e1 = row(PID, 7, rf"^{l1}\s+(?=\d)", 0, cols=(0, 78))
    ROWS += model(item, name, role, cat, bb, None, None, 7, T1, e1, pre_status="not_reported", pre_note="not stated",
                  strat_note="trained on FOR-instanceV2" + ("; 3000 epochs (p.10)" if item == "model1" else "") + "; initialisation not stated", **TRN)
    st = "FOR-instanceV2 test split"
    ROWS += result(res(), "f1", item, "dataset1", n1[2], 7, T1, e1, setting=st + "; individual trees, IoU > 0.5", headline=(item == "model1"))
    if len(n1) == 8:
        ROWS += result(res(), "miou", item, "dataset1", n1[7], 7, T1, e1, setting=st + "; semantic classes ground, wood, leaf")
    if l2:
        for k, (ds, tgt, shift) in enumerate((("dataset2", "Wytham woods (tree species absent from the FOR-instanceV2 training set)", "region;species"),
                                              ("dataset3", "LAUTx (collected using MLS)", "region;sensor"))):
            n2, e2 = row(PID, 7, rf"(?:Wytham|woods|LAUTx|\[8\]|\[1\])?\s*{l2}\s+(?=\d)", k, cols=(79, 200))
            ROWS += result(res(), "f1", item, ds, n2[2], 7, T2, e2, setting="unseen test data; individual trees")
            t = tr()
            ROWS += [R("transfer_source", "FOR-instanceV2 (training regions; in-domain value on its test split)", 6, "Sec. 4.2",
                       "were trained on the FOR-instanceV2 training set and evaluated on the corresponding test split", item=t),
                     R("transfer_target", tgt, 7, "Sec. 4.2", "Notably, the Wytham woods region includes tree species not present in the FOR-instanceV2 training set, while the LAUTx dataset, collected using MLS",
                       item=t),
                     R("shift_type", shift, 7, "Sec. 4.2", "the Wytham woods region includes tree species not present in the FOR-instanceV2 training set, while the LAUTx dataset, collected using MLS", item=t),
                     R("transfer_model", item, 7, T2, e2, item=t),
                     R("transfer_metric", "f1", 7, T2, "Individual Tree Seg. (%) Prec Rec F1", item=t),
                     R("value_in_domain", n1[2], 7, T1, e1, item=t, unit="%"),
                     R("value_out_of_domain", n2[2], 7, T2, e2, item=t, unit="%")]
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=f"The FOR-instanceV2 dataset and the ForestFormer3D code are publicly available at {URL}", data="yes",
              data_page=1, data_loc="Abstract", data_ev=f"The FOR-instanceV2 dataset and the ForestFormer3D code are publicly available at {URL}")
ROWS += [D("notes", "Methods quoted from the literature in Table 2 (SegmentAnyTree, Point2tree, TLS2trees) were not retrained by the authors and are not recorded. Semantic mIoU differs between Table 1 (ground, wood, leaf) and Table 2 (ground, tree), so transfer is recorded on F1 only.", "Sec. 4.2 and Tables 1 and 2 (pp. 6 to 7)")]
