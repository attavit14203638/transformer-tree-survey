import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P075", "Table II (image)"
# DBH of 43 trees from 360 degree video: trunks are segmented zero-shot with Grounded SAM and projected onto the photogrammetric cloud, then three curve fits (Fourier, ellipse, B-spline) give the DBH.
# There is no overall score: Table II splits the 61 acquisitions by observability, platform, sequence length and sensing modality. The photogrammetry row (the headline subset: 2 to 4 points above LiDAR) is primary.
TA = Tags(task="attribute_estimation", level="tree", split="test", agg="mean", attribute="dbh")
TN = TA(agg="not_stated")
OBJ_EV = "semantic trunk segmentation by projecting Grounded Segment Anything (SAM) masks onto the 3D cloud"
DS_EV = "dataset comprising 61 samples of 43 trees located in Stadium"

ROWS = [
    R("task", "attribute_estimation", 1, "Abstract", "a robust RANSAC-based technique to estimate cross section shape and DBH", note="trunk segmentation with Grounded SAM, then DBH from the trunk cross section"),
    R("output_unit", "tree_attribute", 1, "Abstract", "a robust RANSAC-based technique to estimate cross section shape and DBH"),
    R("tree_setting", "natural_forest", 1, "Title", "Estimating the Diameter at Breast Height of Trees in a Forest", note="woodlot (Stadium Woods) on a university campus"),
    R("tree_taxa", "mixed hardwoods: white, black, red and sugar maple, oaks and cherries (Quercus, Acer, Prunus)", 5, "Sec. 4.A", "The trees also cover several", note="Quercus alba, Q. velutina, Prunus avium, P. serotina, Acer saccharum, A. rubrum, A. platanoides"),
    R("study_area", "Stadium Woods, Virginia Tech, Blacksburg, Virginia, USA", 5, "Sec. 4.A", DS_EV),
    D("biome", "temperate", "Virginia, USA; coded from the study area"),
    D("platform", "multiple", "360 degree video from a drone and from a backpack-mounted tower (Sec. 3.B); LiDAR-inertial mapping on the same trajectories"),
    R("sensors", "rgb;lidar", 1, "Abstract", "a consumer-grade 360", note="consumer 360 degree video camera for the photogrammetry; a LiDAR on the same trajectory only for the comparison"),
    R("annotation_type", "regression_target", 5, "Sec. 4.A", "Ground truth DBH for each of the 43 trees was measured manually using a standard diameter tape", note="DBH measured at 1.3 m with a diameter tape"),
    R("dataset_name", "own 360 degree camera and LiDAR scans of 43 trees (61 acquisitions), Stadium Woods", 5, "Sec. 4.A", DS_EV, item="dataset1"),
    R("dataset_key", "own_p075_stadium_woods", 5, "Sec. 4.A", DS_EV, item="dataset1"),
    R("dataset_role", "test_only", 5, "Sec. 4.A", "The proposed system was evaluated on a", item="dataset1", note="no training: Grounded SAM is used with a text prompt"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_trees", "43", 5, "Sec. 4.A", DS_EV, item="dataset1", unit="trees", note="61 acquisitions of 43 trees"),
]
PRE = dict(pre_status="not_reported", pre_note="Grounded SAM is used as released; the pre-training data are not stated", strat_page=5, strat_loc="Sec. 3.D", strat_ev="the model produces segmentation masks identifying potential tree trunks in equirectangular images", strat_note="zero-shot with the text prompt 'tree trunk'; no fine-tuning")
FIT = [("model1", "Fourier series (degree 2)", "Truncated Fourier series (degree 2) [50]"), ("model2", "ellipse", "Least-squares ellipse fitting [53]"), ("model3", "periodic B-spline", "Periodic B-spline fitting [50]")]
for item, nm, ev in FIT:
    ROWS += model(item, f"Grounded SAM trunk segmentation with {nm} DBH fitting" + (" (proposed pipeline)" if item == "model1" else " (variant)"), "proposed" if item == "model1" else "other_baseline", "foundation_model",
                  "Grounded SAM (text-prompted detector and SAM mask decoder) on equirectangular frames, 3D projection, RANSAC cross-section fit", None, None, 5, "Sec. 3.D", "We leverage Grounded SAM [23], a large vision foundation model, for initial segmentation of the trunks",
                  cat_note=f"a large vision foundation model segments the trunks; the DBH fit ({nm}) is a classical step: " + ev, **PRE)
res = Counter("res")
# Table II (p.7, image): R2 of the DBH estimate per setting and fitting method (printed as shown)
T2R = [("fully observed cross sections", "0.97", "0.94", "0.92", TA), ("partially observed cross sections", "0.86", "0.72", "0.75", TA), ("backpack platform", "0.84", "0.7", "0.73", TA), ("drone platform", "0.92", "0.79", "0.83", TA),
       ("short sequences", "0.69", "0.36", "0.53", TA), ("long sequences", "0.89", "0.78", "0.78", TA), ("photogrammetry (short sequences)", "0.69", "0.36", "0.53", TN), ("LiDAR-inertial odometry (labels transferred)", "0.82", "0.65", "0.52", TN)]
for lab, f, e, b, tg in T2R:
    prim = lab.startswith("photogrammetry")
    for item, v, mname in (("model1", f, "Fourier"), ("model2", e, "Ellipse"), ("model3", b, "BSpline")):
        ROWS += result(res(), "r2", item, "dataset1", v, 7, T2, f"{lab} {mname} R2 {v}", unit="fraction", setting=f"Table II: {lab}", note="R2 of the estimated against the manually measured DBH; the comparisons are averaged over the photogrammetry datasets of the category (Sec. 4.C)",
                       **tg, primary=prim)
# Median absolute relative error of the Fourier fit (text of Sec. 4.C)
REL = [("fully observed cross sections", "4.8", 7, "Fourier fitting yields a median absolute relative error of 4.8%"), ("partially observed cross sections", "8.1", 7, "median errors rise to 8.1% (Fourier)"),
       ("backpack platform", "6.2", 7, "produces median errors of 6.2% for Fourier"), ("drone platform", "11.5", 7, "roughly double the median errors: 11.5% (Fourier)"),
       ("short sequences", "6.2", 7, "yield median errors of 6.2% (Fourier)"), ("long sequences", "8.1", 8, "median error rises modestly to 8.1%"),
       ("photogrammetry (short sequences)", "6.2", 8, "median errors are 6.2% (Fourier)"), ("LiDAR-inertial odometry (labels transferred)", "8.9", 8, "Fourier degrades to a 8.9% median error")]
for lab, v, pg, ev in REL:
    ROWS += result(res(), "other", "model1", "dataset1", v, pg, "Sec. 4.C", ev, unit="%", setting=f"Fig. 5: {lab}", note="median absolute relative error of the DBH (percent), Fourier fitting", **(TN if "photogrammetry" in lab or "LiDAR" in lab else TA))
ROWS += [R("limitations", "Partially observed trunks, drone motion blur and long trajectories (drift) raise the DBH error; the pipeline is semi-automated", 7, "Sec. 4.C", "From the scatter plot, we see that all methods have the tendency to underestimate when the tree trunks are partially observable", note="also: semi-automated pipeline (abstract)"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table II (an image) gives median bias, MAD, rCV and R2 of the DBH estimate for three curve fits (Fourier, ellipse, B-spline) in eight settings (fully and partially observed cross sections, backpack and drone, short and long sequences, photogrammetry and LiDAR-inertial odometry); R2 is recorded for all 24 cells and the median absolute relative error of the Fourier fit from the text of Sec. 4.C (also Fig. 5). Median bias, MAD and rCV are not recorded. The short-sequence and photogrammetry rows are identical in the table. There is no overall score for the 61 acquisitions; the photogrammetry row (the headline comparison with LiDAR) is primary. The paper reports no RMSE or MAE.",
           "Sec. 4 and Table II (pp. 6 to 8)")]
