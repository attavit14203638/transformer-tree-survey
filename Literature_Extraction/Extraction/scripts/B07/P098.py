import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T5 = "P098", "Table 5 (image)"  # the text layer splits the decimals ("0. 741"), so the values are read from the rendered page
# TreegrowNet completes single-tree TLS point clouds (PointNet++ with Point Transformer and a Combined-Transformer). The completion scores (Chamfer distance, F-score; Tables 3, 4 and 6) measure the
# completion and are not recorded. Table 5 gives the RMSE of tree height, DBH and the two crown widths computed from the completed clouds against the values of the real (complete) clouds; it is
# the paper's tree-level comparison, so all four models are primary on the four attributes (the east-west crown width is not primary because the north-south width is the same attribute).
def tag(attr):
    return Tags(task="attribute_estimation", level="tree", split="not_stated", agg="single", attribute=attr)
OBJ_EV = "this paper proposes a single tree point cloud completion network TreegrowNet to complete the task"
DATA_EV = "we selected 150 clearly structured, visually intact trees for our experiments after segmentation"
SPLIT_EV = "Our training, testing and validation sets are divided in a 7:1:2 manner"
NOPRE_EV = "no pre-trained weights were used for any of the comparison networks"
REF_EV = "calculated the Root Mean Square Error (RMSE) between the complementary results and the real as the evaluation index"

ROWS = [
    R("task", "attribute_estimation", 2, "Abstract", "Root Mean Square Error (RMSE) in the measurement of tree height, diameter at breast height", note="single-tree point cloud completion; the evaluated tree outputs are the height, DBH and crown widths of the completed trees"),
    R("output_unit", "tree_points;tree_attribute", 2, "Abstract", OBJ_EV),
    D("tree_setting", "plantation", "even-aged poplar trees (7 to 12 years) on a sample plot of Hung-tse Lake wetland; the paper does not say plantation"),
    R("tree_taxa", "poplar (Populus); willow and ginkgo in the additional completion tests", 6, "Sec. 2.1.1", "The dominant tree species in the area is popular", note="printed 'popular'; the figure caption of the plot says poplar"),
    R("study_area", "Huai'an, Hung-tse Lake wetland, China", 6, "Sec. 2.1.1", "Hung-tse Lake wetland, the geographical location is shown in"),
    D("biome", "subtropical", "Huai'an, Jiangsu, China; coded from the study area"),
    R("platform", "terrestrial_lidar", 6, "Sec. 2.1.1", "was used to acquire point cloud data of trees", note="Trimble TX7 laser scanner, multi-station scanning (12 stations in the poplar plot)"),
    R("sensors", "lidar", 6, "Sec. 2.1.1", "was used to acquire point cloud data of trees"),
    R("annotation_type", "regression_target", 20, "Sec. 3.5", REF_EV, note="the complete point cloud of each tree is the target; the reference tree parameters are computed from it"),
    R("dataset_name", "own TLS point clouds of 150 poplar trees, Hung-tse Lake wetland (incomplete versions generated online)", 6, "Sec. 2.1.1", DATA_EV, item="dataset1"),
    R("dataset_key", "own_p098_hungtse_poplar", 6, "Sec. 2.1.1", DATA_EV, item="dataset1"),
    R("dataset_role", "train_test", 8, "Sec. 2.1.2", SPLIT_EV, item="dataset1", note="100 incomplete versions of each tree are generated online; difficulties easy, medium and hard remove 25, 50 and 75 percent of the points"),
    R("dataset_public", "no", 24, "Data availability statement", "have not been made available because the data are obtained by the authors and their institutions for a fee", item="dataset1"),
    R("n_trees", "150", 6, "Sec. 2.1.1", DATA_EV, item="dataset1", unit="trees"),
]
NPRE = dict(pre_page=15, pre_loc="Sec. 3.1", pre_ev=NOPRE_EV, pre_note="no pre-trained weights for any of the compared networks", strat_page=15, strat_loc="Sec. 3.1", strat_ev=NOPRE_EV,
            strat_note="trained from scratch on the 7:1:2 split")
ROWS += model("model1", "TreegrowNet (proposed)", "proposed", "cnn_transformer_hybrid", "PointNet++ set abstraction layers with Point Transformer, proxy generation, Combined-Transformer and point splitting modules", "none", "from_scratch", 9, "Sec. 2.2.2",
              "adopts the three layers of Set Abstraction from PointNet++", cat_note="PointNet++ with Point Transformer layers and a Combined-Transformer (self-attention with modified queries)", **NPRE)
ROWS += model("model2", "AdaPoinTr", "transformer_baseline", "cnn_transformer_hybrid", "PoinTr with an adaptive generative querying mechanism", "none", "from_scratch", 17, "Sec. 3.3",
              "network that introduces an adaptive generative querying mechanism on top of PoinTr", cat_note="built on PoinTr, whose proxies come from a lightweight DGCNN and feed a transformer (Sec. 2.2.3)", **NPRE)
ROWS += model("model3", "PoinTr", "transformer_baseline", "cnn_transformer_hybrid", "point proxies from a DGCNN fed to a transformer", "none", "from_scratch", 10, "Sec. 2.2.3",
              "In PoinTr (Yu et al. 2021), the concept of point proxies is proposed", cat_note="a lightweight DGCNN extracts the proxy features and a transformer processes the proxies (Sec. 2.2.3)", **NPRE)
ROWS += model("model4", "SnowflakeNet", "transformer_baseline", "cnn_transformer_hybrid", "point splitting decoder with Skip-Transformer", "none", "from_scratch", 20, "Sec. 3.6",
              "SnowFlakeNet’s Skip Transformer (Xiang et al. 2021)", cat_note="point-by-point splitting with Skip-Transformer modules (Sec. 3.6)", **NPRE)
res = Counter("res")
RID = {}
for item, k in (("model2", 0), ("model3", 1), ("model4", 2), ("model1", 3)):
    for attr, rx, lab, prim in (("height", r"RMSE\(TH\)", "tree height (TH)", True), ("dbh", r"RMSE\(DBH\)", "DBH", True), ("crown_diameter", r"RMSE\(NSCS\)", "north-south crown width (NSCS)", True), ("crown_diameter", r"RMSE\(EWCS\)", "east-west crown width (EWCS)", False)):
        n, ev = row(PID, 22, rf"^\s*{rx}\s+(?=\d)", 0)
        # the text layer prints "0. 741" (a space after the decimal point), so the numbers are re-read from the printed line
        vals = [x.replace(" ", "") for x in __import__("re").findall(r"\d\.\s*\d+", ev)]
        assert len(vals) == 4, (rx, vals, ev)
        ev_clean = f"{ev.split(' ')[0]} {' '.join(vals)}"
        ROWS += result(res(), "rmse", item, "dataset1", vals[k], 22, T5, ev_clean, unit="m", setting=f"Table 5: tree parameters of the completed trees against the real trees, {lab}",
                       note=f"RMSE of the {lab}; the table header gives metres; printed with a space after the decimal point", **tag(attr), primary=prim)
ROWS += [R("limitations", "Points that the network fails to predict can form noisy points, and only 150 clearly structured, intact trees of one plot were used", 23, "Sec. 4", "may still form noisy points"),
         D("code_available", "no", "No code repository or availability statement for code in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "no", 24, "Data availability statement", "have not been made available because the data are obtained by the authors and their institutions for a fee")]
ROWS += [D("notes", "TreegrowNet completes incomplete single-tree TLS point clouds. Table 3 (CD-L1, CD-L2 and F-score at 1 percent of seven networks), Table 4 (the same for 40 willow and 40 ginkgo trees) and Table 6 (ablation, CD-L1) score the completion and are not recorded. Table 5 gives the RMSE (in metres) of tree height, DBH and the north-south and east-west crown widths computed from the completed point clouds, for AdaPoinTr, PoinTr, SnowflakeNet and TreegrowNet; the reference values are computed the same way from the real (complete) point clouds, not measured in the field, and the split is not stated. The text layer prints decimals as '0. 741' (with a space).",
           "Sec. 3 and Tables 3 to 6 (pp. 16 to 22)")]
