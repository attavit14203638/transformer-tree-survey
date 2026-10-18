import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4 = "P067", "Table 4"
# Trunk and branch segmentation of single trees (component segmentation). The paper calls its mIoU instance-level but defines it over m classes (Sec. 2.8), and it does not say which trees the
# deep learning baselines are tested on, so the level and the split are not stated. ACRGS values are the average over 20 trees per species (Sec. 3.1); Table 4 does not say how the values of the baselines are aggregated, so all Table 4 values carry the same aggregation tag (not stated).
TA = Tags(task="component_segmentation", level="not_stated", split="not_stated", agg="mean")
TB = Tags(task="component_segmentation", level="not_stated", split="not_stated", agg="not_stated")
OBJ_EV = "the precise segmentation of trunks and branches remains a significant challenge"
SPEC_EV = "dormant Asentech Red Fuji apple trees (Malus domestica Borkh., 6 years old) and dormant Kordia cherry trees (Prunus avium L., 5 years old)"
N_EV = "For each species, 20 trees were randomly selected"
PUB_EV = "we have made publicly available both the original datasets and the associated source code"
NOTRAIN_EV = "ACRGS is a non-learning, rule-based algorithm that requires no annotated datasets, no pretraining"

ROWS = [
    R("task", "component_segmentation", 1, "Abstract", OBJ_EV),
    R("output_unit", "tree_component", 1, "Abstract", "The results show that the method enables high-precision and efficiency trunk–branch segmentation"),
    R("tree_setting", "orchard", 5, "Sec. 2.1", "TLS data collection was conducted in a commercial orchard located in Shunyi District, Beijing, China"),
    R("tree_taxa", "apple (Malus domestica) and cherry (Prunus avium)", 5, "Sec. 2.1", SPEC_EV, note="dormant trees"),
    R("study_area", "Shunyi District, Beijing, China", 5, "Sec. 2.1", "TLS data collection was conducted in a commercial orchard located in Shunyi District, Beijing, China"),
    D("biome", "temperate", "Beijing, China; coded from the study area"),
    R("platform", "terrestrial_lidar", 6, "Sec. 2.3", "The Trimble TX8 LiDAR was mounted on a stabilizing tripod"),
    R("sensors", "lidar", 6, "Sec. 2.2", "Performance parameters of Trimble TX8 LiDAR"),
    R("annotation_type", "point_label", 14, "Sec. 2.7.4", "each branch was assigned a unique instance ID", note="manual trunk and branch labels in CloudCompare, one operator and one reviewer"),
    R("dataset_name", "own TLS point clouds of dormant apple trees (20 trees)", 15, "Sec. 3.1", N_EV, item="dataset1"),
    R("dataset_key", "own_p067_orchard_tls", 15, "Sec. 3.1", N_EV, item="dataset1"),
    R("dataset_role", "train_test", 15, "Sec. 3.1", N_EV, item="dataset1", note="ACRGS needs no training; the training of the deep learning baselines is not described"),
    R("dataset_public", "yes", 4, "Sec. 1.2", PUB_EV, item="dataset1"),
    R("dataset_url", "https://github.com/henry3539/ACRGS", 5, "Sec. 1.2", "All materials have been uploaded to a public GitHub repository (https://github.com/henry3539/ACRGS", item="dataset1"),
    R("n_trees", "20", 15, "Sec. 3.1", N_EV, item="dataset1", unit="trees"),
    R("dataset_name", "own TLS point clouds of dormant cherry trees (20 trees)", 15, "Sec. 3.1", N_EV, item="dataset2"),
    R("dataset_key", "own_p067_orchard_tls", 15, "Sec. 3.1", N_EV, item="dataset2", note="same orchard campaign; cherry trees"),
    R("dataset_role", "train_test", 15, "Sec. 3.1", N_EV, item="dataset2", note="ACRGS needs no training; the training of the deep learning baselines is not described"),
    R("dataset_public", "yes", 4, "Sec. 1.2", PUB_EV, item="dataset2"),
    R("n_trees", "20", 15, "Sec. 3.1", N_EV, item="dataset2", unit="trees"),
]
ROWS += model("model1", "ACRGS (adaptive cuboid regional growth segmentation)", "proposed", "classical", "rule-based region growing with an adaptive growth vector and growth cuboid, reverse growth", "none", None, 1, "Abstract",
              "a novel adaptive cuboid regional growth segmentation algorithm is proposed in this study", cat_note="rule-based algorithm without a learned model (Sec. 4.2)",
              pre_page=20, pre_loc="Sec. 4.2", pre_ev=NOTRAIN_EV, pre_note="no pre-training", strat_note="a rule-based algorithm: nothing is trained")
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="re-implemented and tuned for the fruit tree point clouds (Sec. 4.2); the training split is not described")
ROWS += model("model2", "PointNet++", "other_baseline", "other", "PointNet++", None, None, 19, "Sec. 4.2", "PointNet++ [42], PointNeXt [43], and Point Transformer [44]", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model3", "PointNeXt", "other_baseline", "other", "PointNeXt", None, None, 19, "Sec. 4.2", "PointNet++ [42], PointNeXt [43], and Point Transformer [44]", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model4", "Point Transformer", "transformer_baseline", "pure_vit", "Point Transformer", None, None, 20, "Sec. 4.2", "Point Transformer further improves global reasoning", cat_note="self-attention point model (Sec. 4.2)", **NPRE)
res = Counter("res")
RES = {}
for ds, sp, lab in (("dataset1", "apple", "apple"), ("dataset2", "cherry", "cherry")):
    f1, evf = row(PID, 20, rf"^\s*F1\s*_{sp}\s+(?=\d)", 0)
    mi, evm = row(PID, 20, rf"^\s*mIoU_{sp}\s+(?=\d)", 0)
    assert len(f1) == 4 and len(mi) == 4
    for k, item in enumerate(("model2", "model3", "model4", "model1")):  # PointNet++, PointNeXt, Point Transformer, ACRGS
        tags = TB  # one aggregation for all compared models of the setting; Table 4 does not say how the values are aggregated
        P = ds == "dataset1"
        st = f"Table 4: {sp} trees"
        RES[(ds, item, "f1")] = res()
        ROWS += result(RES[(ds, item, "f1")], "f1", item, ds, f1[k], 20, T4, evf, unit="%", setting=st, note="F1 of the trunk and branch segmentation after post-processing", **tags, primary=P,
                       headline=(item == "model1" and P))
        RES[(ds, item, "miou")] = res()
        ROWS += result(RES[(ds, item, "miou")], "miou", item, ds, mi[k], 20, T4, evm, unit="fraction", setting=st, note="mIoU of the trunk and branch segmentation; defined over m classes (Sec. 2.8), called instance-level in the text", **tags, primary=P)
# Tables 2 and 3: the mean F1 of ACRGS for the branch and the trunk class over the 20 trees of each species
for ds, pg, tab, sp in (("dataset1", 17, "Table 2", "apple"), ("dataset2", 18, "Table 3", "cherry")):
    n, ev = row(PID, pg, r"^\s*F1\s+(?=\d)", 0)
    assert len(n) == 6, n  # branch min, max, mean; trunk min, max, mean
    ROWS += result(res(), "f1", "model1", ds, n[2], pg, tab, ev, unit="%", setting=f"{tab}: ACRGS branch class, mean over 20 {sp} trees", note="branch class F1, mean of the 20 trees", **TA)
    ROWS += result(res(), "f1", "model1", ds, n[5], pg, tab, ev, unit="%", setting=f"{tab}: ACRGS trunk class, mean over 20 {sp} trees", note="trunk class F1, mean of the 20 trees", **TA)
ROWS += avail(code_url="https://github.com/henry3539/ACRGS", page=5, loc="Sec. 1.2", ev="All materials have been uploaded to a public GitHub repository (https://github.com/henry3539/ACRGS",
              data="yes", data_page=23, data_loc="Data Availability Statement", data_ev="The original data presented in the study is openly available in GitHub", note_d="datasets and source code in the ACRGS GitHub repository")
ROWS += [R("limitations", "The study covers two fruit tree species and dormant trees; more species and severe structural overlap need further validation", 22, "Sec. 5", "expanding experimental validation across more")]
ROWS += [D("notes", "Table 4 compares ACRGS (a non-learning, rule-based algorithm) with PointNet++, PointNeXt and Point Transformer on apple and cherry trees (P, R, F1, mIoU, model size in MB); F1 and mIoU are recorded, and the model sizes (20.4, 11.6, 222.2 and 0.274 MB) are not parameter counts. Tables 2 and 3 give min, max and mean P, R and F1 of ACRGS for branches and trunks over the 20 trees of each species; the mean F1 is recorded. Primary results are the apple ones (first listed species). The text prints a cherry mIoU of 0.938 for ACRGS against 0.933 in Table 4 and the abstract.",
           "Sec. 3 to 4 and Tables 2 to 4 (pp. 15 to 20)")]
ROWS += conflict("cf1", f"{RES[('dataset2', 'model1', 'miou')]}.value", "0.938", 15, "Sec. 3.1", "while those for cherry trees were slightly higher at 96.52%, 95.91%, 96.21%, and 0.938", "V1a",
                 "Table 4, the comparison table, and the abstract give 0.933 for the ACRGS cherry mIoU; the text of Sec. 3.1 gives 0.938", unit="fraction")
