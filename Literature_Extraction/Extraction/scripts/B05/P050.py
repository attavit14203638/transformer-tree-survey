import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P050", "Table 3"
# SAM (GeoSAM) on three-channel versions of NEON hyperspectral plots. Masks are scored as boxes with the NEON TreeEvaluation package (F1 at IoU 0.4), so the results are detection at object
# level; the F1 values of Table 3 are averages over the 21 sites. All 192 plots are evaluation data (SAM is not trained): test split.
T = Tags(task="detection", level="object", split="test", agg="mean")
DS_EV = "In total, 192 annotated hyperspectral images, each covering 40m x 40m"
SITES_EV = "The 21 chosen test sites for this study reach from Florida to Alaska, encompassing forests dominated by conifers, broadleaves, or a combination of both"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "This study investigates the use of hyperspectral imagery and dimensionality reduction methods for individual tree-crown (ITC) segmentation"),
    R("output_unit", "crown_mask", 3, "Sec. 3.2", "GeoSAM segmented each image into multiple masks"),
    R("tree_setting", "natural_forest", 4, "Sec. 4.1", SITES_EV, note="NEON forest sites"),
    R("tree_taxa", "conifer and broadleaf forests of the NEON sites; species not given", 4, "Sec. 4.1", SITES_EV),
    R("study_area", "21 NEON sites from Florida to Alaska, USA", 4, "Sec. 4.1", SITES_EV),
    D("biome", "multiple", "21 NEON sites from Florida to Alaska; not stated in the paper"),
    R("platform", "airborne", 4, "Sec. 4.1", "hyperspectral data acquired using aerial imagery, at 0.1 m and 1 m"),
    R("sensors", "rgb;hyperspectral", 4, "Sec. 4.1", "The data includes RGB data and hyperspectral data acquired using aerial imagery", note="hyperspectral bands reduced to three channels; RGB only for comparison"),
    R("gsd", "1", 4, "Sec. 4.1", "hyperspectral data acquired using aerial imagery, at 0.1 m and 1 m", unit="m", note="hyperspectral 1 m (used as input); the RGB data are 0.1 m and were down-sampled to 1 m"),
    D("annotation_type", "bounding_box", "the NEON benchmark crowns are scored as bounding boxes at IoU 0.4 (Sec. 3.3); annotated by one observer on the RGB images"),
    R("dataset_name", "NEON benchmark: 192 annotated hyperspectral plots (40 m × 40 m) of 21 sites", 4, "Sec. 4.1", DS_EV, item="dataset1"),
    R("dataset_key", "neon_treeevaluation", 4, "Sec. 4.1", DS_EV, item="dataset1", note="NeonTreeEvaluation benchmark (Weinstein et al., 2020)"),
    R("dataset_role", "test_only", 4, "Sec. 4.1", DS_EV, item="dataset1", note="SAM is used without training"),
    D("dataset_public", "yes", "Public NEON benchmark (Weinstein et al., 2020); no link printed in this paper", item="dataset1"),
    R("n_images", "192", 4, "Sec. 4.1", DS_EV, item="dataset1", unit="images", note="more than 6,000 annotated crowns in the benchmark evaluation data"),
]
M = [  # item, column index in the Average row, name, role, evidence
    ("model1", 0, "GeoSAM on factor analysis of the hyperspectral image (local)", "proposed", "FA applied to HSI locally and semi-globally"),
    ("model2", 1, "GeoSAM on factor analysis of the hyperspectral image (semi-global)", "proposed", "FA applied to HSI locally and semi-globally"),
    ("model3", 2, "GeoSAM on factor analysis of hyperspectral vegetation indices (local)", "proposed", "FA, PCA, and UMAP applied to HVIs locally"),
    ("model4", 3, "GeoSAM on factor analysis of hyperspectral vegetation indices (semi-global)", "proposed", "FA and UMAP applied to HVIs semi-globally"),
    ("model5", 4, "GeoSAM on PCA of hyperspectral vegetation indices (local)", "proposed", "FA, PCA, and UMAP applied to HVIs locally"),
    ("model6", 5, "GeoSAM on UMAP of hyperspectral vegetation indices (local)", "proposed", "FA, PCA, and UMAP applied to HVIs locally"),
    ("model7", 6, "GeoSAM on UMAP of hyperspectral vegetation indices (semi-global)", "proposed", "FA and UMAP applied to HVIs semi-globally"),
    ("model8", 7, "GeoSAM on RGB images (down-sampled to 1 m)", "transformer_baseline", "RGB images (downscaled to match HSI resolution)"),
]
nums, ev = row(PID, 5, r"^\s*Average:\s+(?=\d)", 0)
assert len(nums) == 9, nums  # seven reductions, RGB, then the mean over the methods
res = Counter("res")
for item, k, name, role, mev in M:
    ROWS += model(item, name, role, "foundation_model", "SAM through GeoSAM, default configuration, with a three-channel input", None, "zero_shot", 4 if item != "model8" else 4, "Sec. 4.2", mev,
                  cat_note="SAM (GeoSAM) prompt-free automatic mask generation; only the three-channel input differs between the rows (Sec. 3.2)",
                  pre_status="not_reported", pre_note="pre-trained SAM; the pre-training data are not named", strat_page=3, strat_loc="Sec. 3.2", strat_ev="The model was used in its default configuration", strat_note="no training")
    ROWS += result(res(), "f1", item, "dataset1", nums[k], 5, T3, ev, unit="fraction", setting="Table 3: average F1 over the 21 sites (boxes at IoU 0.4)", note="F1 of the NEON TreeEvaluation package; mean of the site values", **T, primary=True)
ROWS += [R("limitations", "Overall accuracy stays poor and varies strongly between sites with forest density and tree type; dense forests merge trees into large background objects", 1, "Abstract",
           "however the overall segmentation accuracy remains poor"),
         D("code_available", "no", "No code repository or availability statement in the paper"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table 3 gives the F1 of GeoSAM for seven dimensionality reduction inputs and the down-sampled RGB image at each of the 21 NEON sites; the Average row is recorded for the eight inputs (primary), the Average column over the methods (0.261) is not a model. Precision and recall appear in the text only for the local UMAP input (p.5). Site-level values are not recorded.",
           "Sec. 4 to 5 and Table 3 (pp. 4 to 5)")]
