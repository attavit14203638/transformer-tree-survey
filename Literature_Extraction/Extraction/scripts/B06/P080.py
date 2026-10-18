import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P080", "Table 3"
# Individual-tree aboveground biomass (AGB) from single-scan TLS: CoAtNet on 3D-to-2D projections (ensemble of twelve predictions) against Random Forest and Point Transformer on 60 test trees.
# The reference AGB is derived from field DBH with generalized allometric equations (Chojnacky et al. 2014), not measured.
T = Tags(task="attribute_estimation", level="tree", split="test", agg="single", attribute="biomass")
OBJ_EV = "estimating individual tree AGB from single-scan TLS data by integrating a deep learning network (CoAtNet)"
DS_EV = "As a result, 616 segmented tree point clouds were retained with assigned reference AGB values"
SPLIT_EV = "This procedure resulted in 496 samples for training, 60 for validation, and 60 for testing"
AREA_EV = "Martell Forest, located in Indiana, USA"

ROWS = [
    R("task", "attribute_estimation", 1, "Abstract", "This study presents a novel and effective framework for estimating individual tree AGB from single-scan TLS data"),
    R("output_unit", "tree_attribute", 1, "Abstract", "This study presents a novel and effective framework for estimating individual tree AGB from single-scan TLS data"),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "northern mixed hardwood forest, Martell Forest, located in Indiana, USA"),
    R("tree_taxa", "37 hardwood species, mostly sugar maple (Acer saccharum) and white oak (Quercus alba)", 5, "Sec. 2.1", "A total of 37 hardwood species were recorded, with Sugar Maple (Acer saccharum) being the most dominant"),
    R("study_area", "Martell Forest, Indiana, USA", 4, "Sec. 2.1", AREA_EV),
    D("biome", "temperate", "Indiana, USA; coded from the study area"),
    R("platform", "terrestrial_lidar", 6, "Sec. 2.2", "using a FARO Focus3D X330 laser scanner", note="one scan at each plot centre"),
    R("sensors", "lidar", 6, "Sec. 2.2", "using a FARO Focus3D X330 laser scanner"),
    R("annotation_type", "regression_target", 6, "Sec. 2.2", "the generalized AGB models for North American species presented by Chojnacky et al. (2014) were adopted to calculate individual tree AGB", note="reference AGB computed from field DBH by allometric equations, not measured"),
    R("dataset_name", "Martell Forest single-scan TLS plots (61 plots) with 616 segmented trees and reference AGB", 9, "Sec. 2.4", DS_EV, item="dataset1"),
    R("dataset_key", "martell_forest_tls", 9, "Sec. 2.4", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 2.4", SPLIT_EV, item="dataset1", note="stratified 8:1:1 split by AGB bins"),
    R("dataset_public", "yes", 24, "Data Availability", "The single-scan TLS data and the corresponding tree inventory data are publicly available through the Data-to-Science STAC", item="dataset1"),
    R("dataset_url", "https://stac.d2s.org/collections/martell-forest-benchmark-data", 24, "Data Availability", "The single-scan TLS data and the corresponding tree inventory data are publicly available through the Data-to-Science STAC", item="dataset1"),
    R("n_images", "61", 6, "Sec. 2.2", "TLS data were successfully collected for 61 plots", item="dataset1", unit="plots"),
    R("n_trees", "616", 9, "Sec. 2.4", DS_EV, item="dataset1", unit="trees", note="496 training, 60 validation and 60 test trees"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on 496 trees with hyperparameters of Table 2 (AdamW, 200 epochs, RMSE loss)")
ROWS += model("model1", "CoAtNet-based framework (3D-to-2D projection, augmentation and ensemble)", "proposed", "cnn_transformer_hybrid", "CoAtNet on projected mean height images: two MBConv stages and two transformer stages", None, None, 12, "Sec. 2.5",
              "The CoAtNet architecture combines depthwise convolutional blocks", cat_note="convolutional (MBConv) stages followed by self-attention transformer stages (Sec. 2.5)", **NPRE)
ROWS += model("model2", "Random Forest on lidar features", "other_baseline", "classical", "random forest on manually extracted lidar features", None, None, 17, "Sec. 3.1", "Random Forest relies solely", cat_note="classical regression on hand-crafted features",
              pre_status="not_reported", pre_note="not applicable", strat_note="no data augmentation; trained on the training trees")
ROWS += model("model3", "Point Transformer", "transformer_baseline", "pure_vit", "Point Transformer on the segmented tree point clouds", None, None, 15, "Sec. 2.5", "deep learning model, the Point Transformer (Zhao et al. 2021)",
              cat_note="self-attention point network", **NPRE)
ROWS += [R("flops", "4.484", 19, "Sec. 3.1", "The proposed CoAtNet model required 4.484 Giga-FLOPs with 32.8 million parameters", item="model1", unit="GFLOPs", note="per inference"),
         R("total_params", "32.8", 19, "Sec. 3.1", "The proposed CoAtNet model required 4.484 Giga-FLOPs with 32.8 million parameters", item="model1", unit="M"),
         R("flops", "9.418", 19, "Sec. 3.1", "the Point Transformer model required 9.418 Giga-FLOPs with 3.166 million parameters", item="model3", unit="GFLOPs", note="per inference"),
         R("total_params", "3.166", 19, "Sec. 3.1", "the Point Transformer model required 9.418 Giga-FLOPs with 3.166 million parameters", item="model3", unit="M")]
res = Counter("res")
for metric, rx, unit, nm in (("r2", r"R2", "fraction", "R2"), ("rmse", r"RMSE \(kg\)", "kg", "RMSE"), ("mape", r"MAPE \(%\)\s+Overall", "%", "MAPE over all test trees"), ("other", r"MedAPE \(%\)\s+Overall", "%", "median absolute percentage error over all test trees")):
    n, ev = row(PID, 17, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(n) == 3, n  # proposed, Random Forest, Point Transformer
    for item, v in zip(("model1", "model2", "model3"), n):
        ROWS += result(res(), metric, item, "dataset1", v, 17, T3, ev, unit=unit, setting="Table 3: test dataset (60 trees)", note=nm, **T, primary=(metric != "other"), headline=(metric == "r2" and item == "model1"))
ROWS += [R("limitations", "Segmentation errors of occluded single-scan data produce large errors for small trees, and the reference AGB comes from generalized allometric equations", 19, "Sec. 4", "The exceptionally large errors were caused by segmentation errors, particularly for small trees"),
         R("code_available", "yes", 13, "Sec. 2.5", "The Python code for the proposed CoAtNet implementation is available via a", note="https://github.com/gdslab/agbm_tls_dl"),
         R("code_url", "https://github.com/gdslab/agbm_tls_dl", 13, "Sec. 2.5", "The Python code for the proposed CoAtNet implementation is available via a"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "yes", 24, "Data Availability", "The single-scan TLS data and the corresponding tree inventory data are publicly available through the Data-to-Science STAC")]
ROWS += [D("notes", "Table 3 compares the proposed CoAtNet framework with Random Forest and Point Transformer on the 60 test trees (R2, RMSE, mean and median bias, MPE, MedPE, MAPE and MedAPE overall and by AGB group); R2, RMSE and the overall MAPE are primary, the overall MedAPE is recorded as `other`. The reference AGB is computed from field DBH with the generalized equations of Chojnacky et al. (2014), so the target is itself an estimate. The final prediction of the proposed model is the mean of twelve predictions per tree (ensemble). The ablations (no ensemble, H_3 model) are not recorded.",
           "Sec. 3 and Table 3 (pp. 17 to 19)")]
