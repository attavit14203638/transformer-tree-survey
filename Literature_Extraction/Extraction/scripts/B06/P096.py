import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T7 = "P096", "Table 7 (image)"
# Individual tree segmentation of ALS plots matched to reference trees by the 3D distance of the treetops (a segmented tree matched to a reference tree: instance_3d). Table 7 gives the four plots U1, U2, G1, G2 and an Overall
# row (pooled counts) for five methods; the Overall F-score is primary, the per-plot F-scores are not. The text layer of the tables of this PDF is garbled, so the table values are typed from the rendered page.
T = Tags(task="tree_segmentation_3d", level="instance_3d", split="test", agg="single")
OBJ_EV = "a top-down segmentation strategy is adopted to propose an adaptive segmentation method based on extreme offset deep learning"
DS_EV = "two coniferous forest plots (U1, U2) in the Blue Ridge area of Washington, USA, and two mixed forest plots (G1, G2) in Bretten, Germany"
PUB_EV = "Data are available on"

ROWS = [
    R("task", "tree_segmentation_3d", 1, "Abstract", OBJ_EV),
    R("output_unit", "tree_instance", 1, "Abstract", OBJ_EV, note="segmented tree point clouds; accuracy is scored by matching treetops"),
    R("tree_setting", "natural_forest", 1, "Abstract", DS_EV),
    R("tree_taxa", "coniferous forest (U1, U2) and mixed forest (G1, G2)", 1, "Abstract", DS_EV),
    R("study_area", "Blue Ridge area, Washington, USA and Bretten, Germany", 1, "Abstract", DS_EV),
    D("biome", "temperate", "Washington state, USA and Baden-Wuerttemberg, Germany; coded from the study areas"),
    R("platform", "airborne", 1, "Abstract", "individual tree segmentation (ITS) algorithms based on ALS point cloud"),
    R("sensors", "lidar", 1, "Abstract", "individual tree segmentation (ITS) algorithms based on ALS point cloud"),
    R("annotation_type", "point", 1, "Abstract", "the regional point cloud and its treetop are marked, and the offset network is trained", note="treetop points and tree labels of the reference trees"),
    R("dataset_name", "ALS test plots U1, U2 (Blue Ridge, USA) and G1, G2 (Bretten, Germany): 687 reference trees", 1, "Abstract", DS_EV, item="dataset1"),
    R("dataset_key", "als_four_plots_p096", 1, "Abstract", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 1, "Abstract", "the regional point cloud and its treetop are marked, and the offset network is trained", item="dataset1", note="25 m sub-plots; the four plots of Table 1 are the test set"),
    R("dataset_public", "yes", 18, "Data Availability Statement", PUB_EV, item="dataset1", note="PANGAEA (Germany) and FUSION of the University of Washington (USA)"),
    R("dataset_url", "https://doi.pangaea.de/10.1594/PANGAEA.942856", 18, "Data Availability Statement", PUB_EV, item="dataset1"),
    D("n_trees", "687", "sum of the reference trees of U1, U2, G1 and G2 (132, 186, 108 and 261; Tables 1 and 7)", item="dataset1", unit="trees"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on marked sub-plots with the extreme offset loss (learning rate 0.001, 900 sampled points per sub-plot, Sec. 2)")
NA = dict(pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model1", "Extreme offset network (Point Transformer) with dynamic-bandwidth mean shift and ITS set aggregation (proposed)", "proposed", "pure_vit", "Point Transformer offset network with an extreme offset loss, adaptive mean shift and gradient-based aggregation", None, None, 5, "Sec. 2.2",
              "Point Transformer is a transformer-based deep network.", cat_note="Point Transformer (self-attention) predicts the offsets of the points to the treetops (Sec. 2.2)", **NPRE)
ROWS += model("model2", "SHDR (spatial horizontal distance rule)", "other_baseline", "classical", "top-down rule based on horizontal distances", None, None, 15, "Sec. 5.1", "SHDR method [12] in the four", cat_note="rule-based top-down method without a neural network", **NA)
ROWS += model("model3", "DK (DBSCAN with K-means)", "other_baseline", "classical", "bottom-up clustering from stem points", None, None, 16, "Sec. 5.1", "DBSCAN with K-means (DK) is a bottom-up method.", cat_note="clustering without a neural network", **NA)
ROWS += model("model4", "Improved DK (DK with the extreme offset network)", "transformer_baseline", "pure_vit", "DK clustering applied to the points offset by the Point Transformer network", None, None, 15, "Sec. 5.1", "Improved DK method", cat_note="the authors' improvement of DK that integrates the extreme offset deep learning (Sec. 5.1)", **NPRE)
ROWS += model("model5", "HAIS (hierarchical aggregation for instance segmentation)", "other_baseline", None, "HAIS deep learning instance segmentation applied to the plots", None, None, 15, "Sec. 5.1", "HAIS method [25]", cat_note="named as a deep learning method for instance segmentation; no architecture is described", **NPRE)
res = Counter("res")
# Table 7 (p.15, image): per plot and Overall counts, precision p, recall r, F-score F and time (min)
T7V = {  # model: {plot: (p, r, F, time)}
    "model2": {"U1": ("0.88350", "0.78448", "0.83105", "4.5"), "U2": ("0.84431", "0.86503", "0.85455", "4.7"), "G1": ("0.83929", "0.87850", "0.85845", "6.6"), "G2": ("0.82203", "0.87783", "0.84902", "7.5"), "Overall": ("0.84728", "0.85146", "0.84826", "23.3")},
    "model3": {"U1": ("0.59574", "0.58333", "0.58947", "1.9"), "U2": ("0.47436", "0.60656", "0.53237", "2.2"), "G1": ("0.77215", "0.63542", "0.69714", "3.9"), "G2": ("0.83750", "0.56303", "0.67337", "4.5"), "Overall": ("0.66993", "0.59708", "0.62308", "12.5")},
    "model4": {"U1": ("0.86441", "0.93578", "0.89868", "2.7"), "U2": ("0.86420", "0.88050", "0.87227", "2.6"), "G1": ("0.76768", "0.84444", "0.80423", "4.1"), "G2": ("0.84404", "0.81416", "0.82883", "5.5"), "Overall": ("0.83508", "0.86872", "0.85084", "14.9")},
    "model5": {"U1": ("0.76106", "0.81905", "0.78899", "2.2"), "U2": ("0.87500", "0.65882", "0.75168", "2.6"), "G1": ("0.78750", "0.77778", "0.78261", "4.1"), "G2": ("0.76344", "0.65741", "0.70647", "4.8"), "Overall": ("0.79675", "0.72826", "0.75743", "13.7")},
    "model1": {"U1": ("0.88000", "0.92437", "0.90164", "2.1"), "U2": ("0.91329", "0.92398", "0.91860", "2.5"), "G1": ("0.94565", "0.87931", "0.90667", "4.0"), "G2": ("0.93981", "0.81526", "0.87312", "4.9"), "Overall": ("0.91968", "0.88573", "0.90000", "13.5")},
}
NAMES = {"model1": "The proposed method", "model2": "SHDR method", "model3": "DK method", "model4": "Improved DK method", "model5": "HAIS method"}
for item in ("model2", "model3", "model4", "model5", "model1"):
    for plot, (p_, r_, f_, tm) in T7V[item].items():
        ov = plot == "Overall"
        st = f"Table 7: individual tree segmentation, {'overall of the four plots' if ov else 'plot ' + plot}"
        ROWS += result(res(), "f1", item, "dataset1", f_, 15, T7, f"{NAMES[item]} {plot} {p_} {r_} {f_} {tm}", unit="fraction", setting=st, note="F-score of the segmented trees matched by treetop distance" + ("; counts pooled over the four plots" if ov else ""), **T, primary=ov, headline=(ov and item == "model1"))
    ROWS += [R("inference_speed", f"{T7V[item]['Overall'][3]} min for the four test plots", 15, T7, f"{NAMES[item]} Overall {' '.join(T7V[item]['Overall'])}", item=item)]
ROWS += [R("limitations", "The accuracy depends on the canopy density of the stand (the gain over the improved DK is small in the sparse plot U1), and the method needs marked treetops for training", 16, "Sec. 5.1", "Improved DK method", note="the F-score of the proposed method is only slightly higher than the improved DK in U1 with low canopy density (Sec. 5.1)"),
         D("code_available", "no", "No code repository or availability statement for code in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "yes", 18, "Data Availability Statement", PUB_EV)]
ROWS += [D("notes", "Table 7 compares SHDR, DK, the improved DK, HAIS and the proposed method on the four ALS plots U1, U2 (coniferous, Blue Ridge) and G1, G2 (mixed, Bretten) with counts, precision, recall, F-score and time; the F-score of the Overall row (primary) and of each plot (not primary) and the Overall time are recorded; precision and recall are not. Tables 2 to 6 (the proposed method per plot, the ablation of the extreme offset, bandwidth and set aggregation, and the offset comparison) are not recorded. The page text of this PDF is garbled (overlaid layers), so the values of Table 7 are typed from the rendered page and checked by OCR.",
           "Sec. 4 to 5 and Table 7 (pp. 12 to 15)")]
