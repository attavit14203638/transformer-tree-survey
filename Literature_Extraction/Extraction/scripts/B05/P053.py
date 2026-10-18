import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1 = "P053", "Table I (image)"
# The only score is the ratio of segmented olive trees to the true tree count of a plot, in percent: a count-based completeness (metric other, plot level, never primary). The plots are
# the evaluation data of a pipeline that is not trained, so the split is test; the paper does not say how the final accuracy rate is aggregated (it equals the mean of the five plot ratios).
TP = Tags(task="counting", level="plot", split="test", agg="single")
TA = Tags(task="counting", level="plot", split="test", agg="not_stated")
DS_EV = "For our tests, a set of five images is used, including five satellite images of olive tree plots located in Tunisia"
PERF_EV = "The performance of olive tree segmentation is evaluated by comparing the count of segmented trees with the true count of trees in each plot"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "This paper presents an innovative approach to olive tree segmentation from satellite images"),
    R("output_unit", "crown_mask", 2, "Sec. 3", "The third phase is dedicated to refining the segmentation of unsegmented olive trees with the SAM Predictor", note="one mask per tree; the evaluation counts masks"),
    R("tree_setting", "orchard", 1, "Sec. 1", "The Tunisian landscape, abundant with olive groves"),
    R("tree_taxa", "olive", 1, "Abstract", "olive tree segmentation from satellite images"),
    R("study_area", "Tunisia (plots at Nabeul, Monastir, Zghal and Benhlima)", 4, "Sec. 4.A", DS_EV),
    D("biome", "mediterranean", "olive groves in Tunisia; not stated in the paper"),
    R("platform", "satellite", 4, "Sec. 4.A", DS_EV),
    D("sensors", "rgb", "satellite image bands are not named; the SAM input is a three-channel image"),
    NR("gsd", note="no resolution is printed"),
    NR("annotation_type", note="the true tree count of each plot is the reference; how it was obtained is not stated"),
    R("dataset_name", "five satellite images of olive tree plots in Tunisia", 4, "Sec. 4.A", DS_EV, item="dataset1"),
    R("dataset_key", "own_p053_tunisia_olive", 4, "Sec. 4.A", DS_EV, item="dataset1"),
    R("dataset_role", "test_only", 4, "Sec. 4.A", DS_EV, item="dataset1", note="no training is involved"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    D("n_images", "5", "the paper says a set of five images is used (p.4)", item="dataset1", unit="images"),
    D("n_trees", "1343", "sum of the printed true counts of the five plots: 330, 71, 61, 373 and 508 (Table I)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "SAM automatic mask generator (phase 1, SAM alone)", "transformer_baseline", "foundation_model", "SAM automatic mask generator, default settings", "sa1b", "zero_shot", 2, "Sec. 2.A",
              "As a foundation model, SAM is pretrained on the SA-1B Dataset", cat_note="SAM used as a foundation model (Sec. 2.A)",
              strat_page=2, strat_loc="Sec. 3.A", strat_ev="we use the SAMG class with default settings", strat_note="no fine-tuning; very large plots are split into parts")
ROWS += model("model2", "multi-phase approach (SAM generator, tree row and column grid, SAM predictor refinement)", "proposed", "foundation_model", "SAM automatic mask generator and SAM predictor with a detected planting grid", "sa1b", "zero_shot", 2, "Sec. 2.A",
              "As a foundation model, SAM is pretrained on the SA-1B Dataset", cat_note="SAM prompted by tree positions from the detected rows and columns (Sec. 3)",
              strat_page=1, strat_loc="Abstract", strat_ev="methodology includes SAM segmentation and corrections based on trees alignement in the field", strat_note="no fine-tuning; the grid and shape filters use a priori knowledge")
# Table I (p.6, image): true count and the ratio of segmented trees after phase 1 (SAM alone) and after all phases, per plot
PLOTS = [("Nabeul", 330, "161", "48.78", "327", "99.09"), ("Monastir5_1parc", 71, "70", "98.59", "71", "100"), ("Monastir5_2parc", 61, "59", "96.72", "60", "98.36"),
         ("Zghal_1parc", 373, "294", "78.82", "369", "98.92"), ("Benhlima_1parc", 508, "466", "91.73", "490", "96.45")]
res = Counter("res")
for nm, true, n1, p1, n2, p2 in PLOTS:
    ROWS += result(res(), "other", "model1", "dataset1", p1, 6, T1, f"{nm} {true} {n1} {p1}%", unit="%", setting=f"Table I: plot {nm} ({n1} of {true} trees segmented by phase 1)", note="segmented tree count over true tree count", **TP)
    ROWS += result(res(), "other", "model2", "dataset1", p2, 6, T1, f"{nm} {true} {n2} {p2}%", unit="%", setting=f"Table I: plot {nm} ({n2} of {true} trees segmented after all phases)", note="segmented tree count over true tree count", **TP)
ROWS += result(res(), "other", "model1", "dataset1", "82.928", 6, T1, "Accuracy rate 82.928%", unit="%", setting="Table I: accuracy rate over the five plots, phase 1", note="the accuracy rate row; equals the mean of the five plot ratios", **TA)
ROWS += result(res(), "other", "model2", "dataset1", "98.564", 6, T1, "Accuracy rate 98.564%", unit="%", setting="Table I: accuracy rate over the five plots, all phases", note="the accuracy rate row; equals the mean of the five plot ratios", **TA, headline=True)
ROWS += [R("limitations", "The score counts segmented trees against the true count and not the overlap with reference crowns; only five plots were tested and fine-tuning SAM is left for future work", 5, "Sec. 5", "Exploring fine-tuning of the SAM could further enhance its performance"),
         D("code_available", "no", "No code repository or availability statement in the paper"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table I gives, for five olive plots, the true tree count, the trees segmented by SAM alone (phase 1), the predicted tree positions (phase 2), the extra trees from the two phase 3 methods, and the final count with its ratio to the true count. The paper reports no IoU, F1 or AP: the only score is this count ratio, recorded as metric other for the SAM-only and the final result per plot and as the accuracy rate row (82.928 and 98.564).",
           "Sec. 4 and Table I (pp. 4 to 6)")]
