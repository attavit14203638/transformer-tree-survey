import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T3 = "P149", "Table I", "Table III"
# One prediction per cropped tree image (health class); Table III is the only result table. The paper mentions an independent test set (10 percent, Table I).
T = Tags(task="health_classification", level="tree", split="test", agg="single")
TASK_EV = "diagnosing forest health using the SWIN Transformer model, which leverages UAVcaptured high-resolution images"
SPLIT = "45,023 36018 4502 4502"

ROWS = [
    R("task", "health_classification", 1, "Abstract", "healthy, lightly damaged, heavily damaged, and others", note="trees are classified into four health categories"),
    R("output_unit", "tree_label", 4, "Sec. 4.B", "individual tree images are cropped for further analysis"),
    R("tree_setting", "natural_forest", 1, "Abstract", TASK_EV, note="forest type is not described"),
    NR("tree_taxa", note="species not given"),
    NR("study_area", note="the location of the UAV images is not stated; Table IV cites high-resolution annotated forest images [3]"),
    NR("biome", note="not stated"),
    R("platform", "uav", 1, "Abstract", TASK_EV),
    D("sensors", "rgb", "imagery type not stated; thermal and multispectral data are named as future work (p.7)"),
    R("annotation_type", "bounding_box", 3, "Sec. 4.A", "each dataset includes corresponding XML annotation files, which contain essential information about the location of trees within the image through bounding boxes",
      note="each box carries one of four health classes"),
    R("dataset_name", "UAV forest images with XML bounding boxes and four health classes (source not stated)", 4, T1, SPLIT, item="dataset1"),
    R("dataset_key", "forest_health_uav_p149", 4, T1, SPLIT, item="dataset1"),
    R("dataset_role", "train_test", 4, T1, SPLIT, item="dataset1", note="80 percent training, 10 percent validation, 10 percent testing"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    D("n_images", "45022", "sum of the printed split counts: 36018 training, 4502 validation and 4502 test samples (Table I); the printed total is 45,023. Samples are cropped single-tree images; whether augmented copies are included is not stated", item="dataset1", unit="images"),
]
ROWS += model("model1", "SWIN Transformer", "proposed", "hierarchical_vit", "Swin Transformer (variant not stated)", "imagenet", "full_fine_tuning", 4, "Sec. 4.D",
              "the Swin Transformer was selected due to its advanced hierarchical design",
              pre_page=5, pre_loc="Sec. 4.E", pre_ev="This SWIN transformer model was initially trained on ImagENet data records", pre_note="initially trained on ImageNet (as printed)",
              strat_page=6, strat_loc="Table II", strat_ev="FINE TUNING MODEL PARAMETERS", strat_note="Table II: cross entropy loss, Adam, learning rate 1×10-3, batch size 64, 50 epochs",
              cat_note="Swin Transformer (shifted-window hierarchical transformer)")
nums, ev = row(PID, 6, r"^\s*SWIN Transformer\s+(?=95%)", 0)
assert len(nums) == 5, nums  # accuracy, precision, recall, F1, AUC
res = Counter("res")
ROWS += result("res1", "oa", "model1", "dataset1", nums[0], 6, T3, ev, unit="%", setting="Table III: SWIN Transformer on the four health classes", note="Accuracy column", **T, primary=True)
ROWS += result("res2", "f1", "model1", "dataset1", nums[3], 6, T3, ev, unit="fraction", setting="Table III: SWIN Transformer on the four health classes", note="F1 score column; precision, recall and AUC are in Table III", **T, primary=True, headline=True)
ROWS += avail()
ROWS += [D("notes", "Table III is the only result table (accuracy 95%, precision 0.95, recall 0.92, F1 0.93, AUC 97%). The text claims better results than U-Net, ResNet, CNNs and Vision Transformers but prints none of their results; Table IV compares with accuracies and AUCs that other studies report on other data (not recorded). The abstract and the text of Sec. 5 give other figures for the same results (cf items).",
           "Sec. 5 and Tables I to IV (pp. 4 to 7)")]
ROWS += conflict("cf1", "res1.value", "93", 1, "Abstract", "accurately finds and labels tree conditions 93% of the time", "V1a",
                 "Table III, the results table, gives an accuracy of 95%; the abstract's 93% equals the F1 score of Table III", unit="%")
ROWS += conflict("cf2", "res1.value", "85", 6, "Sec. 5", "the version done a median category accuracy of about 85%", "V1a",
                 "Table III, the results table, gives an accuracy of 95%; the text of Sec. 5 gives about 85%", unit="%")
ROWS += conflict("cf3", "dataset1.n_images", "45023", 4, T1, SPLIT, "V1c",
                 "The per-split counts of Table I add up to 45,022, which is recorded as a derived sum; the printed total is 45,023")
