import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P081"
# 3DPS-Net (SAM-inspired promptable 3D instance segmentation, KPConv backbone with a prompt encoder of six self-attention layers). ForestSemantic: three models trained on Plot 1, 3 or 5 and run in the
# automated mode on all six plots (training plots included), so the split is not stated; the Mean row is primary. FOR-instance: per-plot values only (not primary). "Mean accuracy" is
# 2 n_matched / (n_reference + n_predicted) (Sec. 2.4), the F-score, so it is recorded as f1; completeness and correctness (recall and precision) are not recorded.
TF = Tags(task="tree_segmentation_3d", level="instance_3d", split="not_stated", agg="mean")
TO = Tags(task="tree_segmentation_3d", level="instance_3d", split="test", agg="single")
OBJ_EV = "we propose a novel approach to forest point cloud instance segmentation, called the 3D Promptable Segmentation Network"
SA = "passed through six self-\u00adattention layers to obtain the encoding vector"
URL_DATA = "https://zenodo.org/records/15080627"
URL_CODE = "https://doi.org/10.5281/zenodo.15245851"
DATA_EV = "The labelled dataset, referred to as the ForestSemantic dataset, is obtainable from"
SIZE_EV = "which have 51, 133 and 104 individual trees"

ROWS = [
    R("task", "tree_segmentation_3d", 1, "Abstract", OBJ_EV),
    R("output_unit", "tree_points", 1, "Abstract", "This network generates object masks using prompt points", note="a 3D mask of the points of one tree for each prompt point"),
    R("tree_setting", "natural_forest", 3, "Sec. 2.1", "terrestrial laser scanner within a southern boreal forest situated in", note="ForestSemantic: boreal forest plots; FOR-instance: five UAV laser scanning forest collections"),
    R("tree_taxa", "Scots pine, Norway spruce and birches (ForestSemantic); spruce, pine, deciduous, eucalypt and radiata pine collections (FOR-instance)", 4, "Table 1", "Scots pine (Pinus sylvestris L.)"),
    R("study_area", "Evo, Finland (ForestSemantic); Norway, Czech Republic, Austria, New Zealand and Australia (FOR-instance)", 3, "Sec. 2.1", "terrestrial laser scanner within a southern boreal forest situated in"),
    D("biome", "multiple", "boreal forest in Finland and forest collections of five countries (Table 1)"),
    D("platform", "multiple", "terrestrial laser scanning (ForestSemantic, Leica HDS6100) and UAV laser scanning (FOR-instance)"),
    R("sensors", "lidar", 3, "Sec. 2.1", "conducted in the summer of 2014 and employed the Leica HDS6100"),
    R("annotation_type", "point_label", 3, "Sec. 2.1", "the plot has a unique identifier.", note="tree instance identifiers per point; only the instance annotations are used"),
    R("dataset_name", "ForestSemantic (Evo, Finland): six TLS plots of 32 m by 32 m, Plots 1, 3 and 5 annotated", 3, "Sec. 2.1", "plots, each measuring 32 meters by 32 meters. The predominant tree", item="dataset1"),
    R("dataset_key", "forestsemantic_evo", 3, "Sec. 2.1", "plots, each measuring 32 meters by 32 meters. The predominant tree", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.1", SIZE_EV, item="dataset1", note="one model per training plot (1, 3 or 5), tested on all six plots"),
    R("dataset_public", "yes", 13, "Data availability", DATA_EV, item="dataset1"),
    R("dataset_url", URL_DATA, 13, "Data availability", DATA_EV, item="dataset1"),
    D("n_trees", "288", "sum of the printed counts of Plots 1, 3 and 5: 51, 133 and 104 individual trees (Sec. 2.1)", item="dataset1", unit="trees"),
    R("dataset_name", "FOR-instance (UAV laser scanning, five collections: NIBIO, CULS, TU_WIEN, RMIT, SCION)", 3, "Sec. 2.1", "detailed in Table 1. The dataset is divided into training, testing and", item="dataset2"),
    R("dataset_key", "for_instance_v1", 3, "Sec. 2.1", "detailed in Table 1. The dataset is divided into training, testing and", item="dataset2"),
    R("dataset_role", "train_test", 3, "Sec. 2.1", "detailed in Table 1. The dataset is divided into training, testing and", item="dataset2"),
    D("dataset_public", "yes", "FOR-instance is the public benchmark of Puliti et al. (2023), cited in Sec. 2.1", item="dataset2"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the annotated plots (the network needs training samples to work, Sec. 1)")
BB = "KPConv (KP-FCNN) point encoder and decoder with a prompt encoder of six self-attention layers"
CATN = "KPConv backbone with a prompt encoder of six self-attention layers (Sec. 2.2.2)"
for item, nm in (("model1", "3DPS-Net trained on Plot 1 (Model 1)"), ("model2", "3DPS-Net trained on Plot 3 (Model 3)"), ("model3", "3DPS-Net trained on Plot 5 (Model 5)"), ("model4", "3DPS-Net trained on the FOR-instance training plots")):
    ROWS += model(item, nm, "proposed", "cnn_transformer_hybrid", BB, None, None, 5, "Sec. 2.2.2", SA, cat_note=CATN, **NPRE)
ROWS += [R("inference_speed", "1.5 s per 3D individual tree mask from a prompt point", 1, "Abstract", "It requires only 1.5 seconds to predict a 3D individual tree mask from a prompt point", item="model1")]
res = Counter("res")
# Table 2 (p.8, image): Mean row, ForestSemantic automated testing; completeness, correctness and mean accuracy of Models 1, 3 and 5
T2 = [("model1", "57.8"), ("model2", "51.2"), ("model3", "49.4")]
for item, v in T2:
    ROWS += result(res(), "f1", item, "dataset1", v, 8, "Table 2 (image)", "Mean 61.9 57.1 53.3 57.2 48.3 49.5 57.8 51.2 49.4", unit="%", setting="Table 2: ForestSemantic, automated testing, mean over the six plots",
                   note="mean accuracy, 2 n_matched / (n_reference + n_predicted), i.e. the F-score; the Mean row of the six plots", **TF, primary=True)
# Table 4 (p.10): FOR-instance automated testing, one row per test plot (mean accuracy, fractions)
PLOTS = ["NIBIO_plot_1", "NIBIO_plot_5", "NIBIO_plot_17", "NIBIO_plot_18", "NIBIO_plot_22", "NIBIO_plot_23", "NIBIO2_plot1", "NIBIO2_plot3", "NIBIO2_plot6", "NIBIO2_plot10", "NIBIO2_plot15", "NIBIO2_plot27",
         "NIBIO2_plot32", "NIBIO2_plot34", "NIBIO2_plot35", "NIBIO2_plot48", "NIBIO2_plot49", "NIBIO2_plot52", "NIBIO2_plot53", "NIBIO2_plot58", "NIBIO2_plot60", "CULS_plot_2", "RMIT", "SCION_plot_31", "SCION_plot_61", "TUWIEN"]
for pl in PLOTS:
    n, ev = row(PID, 10, rf"^\s*{pl}\s+(?=\d)", 0)
    assert len(n) == 6, (pl, n)  # reference, matched, predicted, completeness, correctness, mean accuracy
    ROWS += result(res(), "f1", "model4", "dataset2", n[5], 10, "Table 4", ev, unit="fraction", setting=f"Table 4: FOR-instance automated testing, plot {pl}", note="mean accuracy of one test plot (the F-score); no overall value is printed", **TO)
ROWS += [R("limitations", "Automated testing finds fewer trees in dense multi-species forests with understory trees, and the workflow has no automatic selection of masks (such as non-maximum suppression)", 12, "Sec. 4", "automated selection mechanisms (e.g. classical approaches"),
         R("code_available", "yes", 13, "Data availability", "is available via the following link", note=URL_CODE),
         R("code_url", URL_CODE, 13, "Data availability", "is available via the following link"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "yes", 13, "Data availability", DATA_EV)]
ROWS += [D("notes", "Models 1, 3 and 5 are the same network trained on Plot 1, 3 or 5 of ForestSemantic; Table 2 (an image) gives completeness, correctness and mean accuracy per plot and on average, and the Mean row of the mean accuracy (the F-score) is recorded and primary. The prompt-mode results (Fig. 3, per-tree IoU; Table 3, mIoU of dominant and understory trees per FOR-instance plot) are not recorded. For FOR-instance, Table 4 gives the automated results per test plot (26 plots, mean accuracy recorded as f1, not primary because no overall value is printed), and Table 5 compares matched and predicted tree counts with ForAINet for 11 plots; those counts are not recorded as results. The NIBIO2 plots contain many unannotated understory trees.",
           "Sec. 3 and Tables 2 to 5 (pp. 7 to 12)")]
