import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P084", "Table 2"
# Zero-shot SAM (ViT-H, automatic mask generator) on DSM and DTM imagery of ornamental maple nurseries: tree count, height (24 trees per dataset against tape measures), canopy area and volume (D3 only).
# No training is involved, so every comparison is evaluation data (split test). The averages are means over the 24 trees. D1 (first listed) is the primary dataset for the height.
TH = Tags(task="attribute_estimation", level="tree", split="test", agg="mean", attribute="height")
TA = Tags(task="attribute_estimation", level="tree", split="test", agg="mean", attribute="crown_area")
TV = Tags(task="attribute_estimation", level="tree", split="test", agg="mean", attribute="other")
TC = Tags(task="counting", level="plot", split="test", agg="single")
OBJ_EV = "A self-supervised zero-shot segmentation model, Segment Anything Model (SAM), was applied for tree segmentation and counting"
GSD_EV = "sample distance (GSD) of 0.025 m per pixel"

ROWS = [
    R("task", "counting;attribute_estimation;crown_segmentation", 1, "Abstract", OBJ_EV, note="tree count, tree height, canopy area and canopy volume from SAM masks"),
    R("output_unit", "tree_count;crown_mask;tree_attribute", 1, "Abstract", OBJ_EV),
    R("tree_setting", "plantation", 3, "Sec. 2.1", "The first experimental site was located on Smith Farm, McMinnville, TN", note="ornamental tree nurseries (red maples in rows)"),
    R("tree_taxa", "maple (red maple, Acer rubrum)", 3, "Sec. 2.1", "featuring more than 350 strategically planted red maple trees spaced in rows"),
    R("study_area", "McMinnville, Tennessee and Athens, Georgia, USA", 3, "Sec. 2.1", "The second experimental site was located at Southeastern Growers Inc., Athens, GA"),
    D("biome", "temperate", "Tennessee and Georgia, USA; coded from the study area"),
    R("platform", "uav", 3, "Sec. 2.1", "height measurements using DJI Matrice 300 RTK", note="DJI Matrice 300 RTK (D1, D2) and Matrice 350 RTK (D3) with a Zenmuse H20 RGB camera"),
    R("sensors", "rgb;dsm", 1, "Abstract", "mounted on a drone captured three aerial image datasets", note="RGB images stitched to an orthomosaic, DSM and DTM"),
    R("gsd", "0.025", 1, "Abstract", GSD_EV, unit="m"),
    R("annotation_type", "regression_target", 3, "Sec. 2.2", "the height of 24 trees (randomly selected) was measured using a measuring tape", note="tape-measured heights of 24 trees per dataset; manual crown area and volume for D3; manual tree counts from the orthomosaic"),
    R("dataset_name", "own UAV images of red maples, Smith Farm, McMinnville (D1, 6 July 2023)", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset1"),
    R("dataset_key", "own_p084_d1", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset1"),
    R("dataset_role", "test_only", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset1", note="no training: SAM is used zero-shot"),
    R("dataset_public", "on_request", 16, "Data availability", "Data will be made available on request", item="dataset1"),
    R("n_trees", "373", 10, "Sec. 3.1", "revealing 373 trees in D1", item="dataset1", unit="trees", note="manual count; 24 trees measured for height"),
    R("dataset_name", "own UAV images of red maples, Smith Farm, McMinnville (D2, 24 July 2023)", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset2"),
    R("dataset_key", "own_p084_d2", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset2"),
    R("dataset_role", "test_only", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset2", note="no training: SAM is used zero-shot"),
    R("dataset_public", "on_request", 16, "Data availability", "Data will be made available on request", item="dataset2"),
    R("n_trees", "361", 10, "Sec. 3.1", "361 trees in D2", item="dataset2", unit="trees", note="manual count; 24 trees measured for height"),
    R("dataset_name", "own UAV images of maples, Southeastern Growers, Athens (D3, 28 May 2024)", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset3"),
    R("dataset_key", "own_p084_d3", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset3"),
    R("dataset_role", "test_only", 3, "Sec. 2.2", "On July 6, 2023, July 24, 2023, and May 28, 2024", item="dataset3", note="no training: SAM is used zero-shot"),
    R("dataset_public", "on_request", 16, "Data availability", "Data will be made available on request", item="dataset3"),
    R("n_trees", "423", 10, "Sec. 3.1", "and 423 trees in D3, averaging 71 trees per row", item="dataset3", unit="trees", note="manual count; 24 trees measured for height, area and volume"),
]
ROWS += model("model1", "SAM ViT-H automatic mask generator on DSM-DTM imagery (zero-shot)", "proposed", "foundation_model", "Segment Anything Model with the ViT-H image encoder, automatic mask generator with filtering of masks", "sa1b", "zero_shot", 4, "Sec. 2.4",
              "The choice of the ViT-H model for this project was driven by its", cat_note="SAM, a transformer-based segmentation foundation model, used without training", pre_page=2, pre_loc="Sec. 1", pre_ev="million images and 1 billion masks, the fast and adaptable SAM demonstrates",
              pre_note="SAM pre-trained on SA-1B (1 billion masks) by Meta AI", strat_page=1, strat_loc="Abstract", strat_ev=OBJ_EV, strat_note="zero-shot; mask filtering thresholds set by trial and error")
res = Counter("res")
HID = {}
n, ev = row(PID, 12, r"^\s*Avg\.\s+(?=\d)", 0)
assert len(n) == 12, n  # D1: manual, UAV, absolute error (m), (%); D2: same; D3: same
for ds, k, rmse, rev in (("dataset1", 0, "0.13", "The MAE and RMSE obtained for D1 were 0.09 m and 0.13 m, respectively"), ("dataset2", 4, "0.25", "For D2, they were 0.23 m and 0.25 m, respectively"), ("dataset3", 8, None, None)):
    P = ds == "dataset1"
    st = f"Table 2: height of 24 trees, drone against tape, D{ds[-1]}"
    ROWS += result(res(), "mape", "model1", ds, n[k + 3], 12, T2, ev, unit="%", setting=st, note="average absolute percentage error of the tree height (Avg. row)", **TH, primary=P, headline=P)
    HID[(ds, "mae")] = res()
    ROWS += result(HID[(ds, "mae")], "mae", "model1", ds, n[k + 2], 12, T2, ev, unit="m", setting=st, note="average absolute error of the tree height (Avg. row)", **TH, primary=P)
    if rmse:
        HID[(ds, "rmse")] = res()
        ROWS += result(HID[(ds, "rmse")], "rmse", "model1", ds, rmse, 12, "Sec. 3.3", rev, unit="m", setting=st, note="RMSE of the tree height as printed in the text", **TH, primary=P)
# Canopy area and volume of D3 (RMSE and MAE of the text, which equal the abstract and the mean of the printed per-tree errors)
AREA_EV = "The RMSE and MAE were calculated as 0.18 m2 and 0.15 m2"
VOL_EV = "The RMSE and MAE for volume measurements were calculated as 0.33 m3 and 0.26 m3"
RID = {}
for tg, metric, v, u, ev_, nm in ((TA, "rmse", "0.18", "m2", AREA_EV, "canopy area"), (TA, "mae", "0.15", "m2", AREA_EV, "canopy area"), (TV, "rmse", "0.33", "m3", VOL_EV, "canopy volume"), (TV, "mae", "0.26", "m3", VOL_EV, "canopy volume")):
    RID[(nm, metric)] = res()
    ROWS += result(RID[(nm, metric)], metric, "model1", "dataset3", v, 13, "Sec. 3.3" if nm == "canopy area" else "Sec. 3.4", ev_, unit=u, setting=f"Table 3: {nm} of 24 trees, drone against manual, D3", note=f"{metric.upper()} of the {nm} (text); manual volume from the cylinder formula" if nm == "canopy volume" else f"{metric.upper()} of the {nm} (text)", **tg, primary=True)
# Tree counts (SAM count; the manual count is the reference)
for ds, v, ref, ev_ in (("dataset1", "370", "373", "The SAM identified 370 trees in D1"), ("dataset2", "367", "361", "367 trees in D2"), ("dataset3", "420", "423", "420 trees in D3")):
    ROWS += result(res(), "other", "model1", ds, v, 10, "Sec. 3.1", ev_, unit="trees", setting=f"Sec. 3.1: SAM tree count, D{ds[-1]}", note=f"number of trees found by SAM (manual count {ref}); the paper gives no counting metric", **TC)
ROWS += [R("limitations", "Tall grass and potholes cause overcounting and a poorer height from the terrain model, and the height error grows from D1 to D3", 10, "Sec. 3.1", "SAM encountered challenges distinguishing between grass and potholes"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "on_request", 16, "Data availability", "Data will be made available on request")]
ROWS += [D("notes", "Table 2 gives the manual and drone height, the absolute error in m and in percent of 24 trees in each of D1, D2 and D3 with an Avg. row (recorded: percent and m error of the Avg. row, and the RMSE of the text for D1 and D2). Table 3 gives manual and drone area and volume of the 24 trees of D3; the RMSE and MAE of the text (which equal the abstract and the mean of the printed per-tree errors) are recorded. The abstract and the text of Sec. 3.3 print height MAE and RMSE pairs that differ from each other and from Table 2 (cf items); the D3 RMSE of the text (0.21 m) is below the Table 2 MAE (0.37 m) and is not recorded. SAM tree counts against manual counts are recorded as metric `other`.",
           "Sec. 3 and Tables 2 and 3 (pp. 10 to 14)")]
ROWS += conflict("cf1", f"{RID[('canopy area', 'mae')]}.value", "0.16", 14, "Table 3", "Average 1.18 1.27 0.16 14.94 1.64 1.52 0.13 17.58", "V2",
                 "The Average row of Table 3 prints an area error of 0.16; the text and the abstract give an MAE of 0.15 m2, which equals the mean of the 24 printed per-tree area errors (0.155)", unit="m2")
ROWS += conflict("cf2", f"{RID[('canopy volume', 'mae')]}.value", "0.13", 14, "Table 3", "Average 1.18 1.27 0.16 14.94 1.64 1.52 0.13 17.58", "V2",
                 "The Average row of Table 3 prints a volume error of 0.13; the text and the abstract give an MAE of 0.26 m3, which equals the mean of the 24 printed per-tree volume errors (0.263)", unit="m3")
ROWS += conflict("cf3", f"{HID[('dataset1', 'rmse')]}.value", "0.23", 1, "Abstract", "0.09 m and 0.23 m in D1", "V1a",
                 "Sec. 3.3 gives an RMSE of 0.13 m for D1; the abstract prints 0.23 m, which is the D2 MAE of Table 2 and Sec. 3.3 (the abstract exchanges the D1 RMSE and the D2 MAE)", unit="m")
ROWS += conflict("cf4", f"{HID[('dataset2', 'mae')]}.value", "0.13", 1, "Abstract", "0.13 m and 0.25 m in D2", "V1a",
                 "The Avg. row of Table 2 and Sec. 3.3 give 0.23 m for the D2 MAE (the mean of the 24 printed errors is 0.228); the abstract prints 0.13 m, which is the D1 RMSE of Sec. 3.3", unit="m")
ROWS += conflict("cf5", f"{HID[('dataset3', 'mae')]}.value", "0.16", 1, "Abstract", "0.16 m and 0.21 m in D3, respectively", "V1a",
                 "The Avg. row of Table 2 gives 0.37 m for the D3 MAE (the mean of the 24 printed errors is 0.3675); the abstract and Sec. 3.3 print 0.16 m and an RMSE of 0.21 m, an RMSE below the Table 2 MAE, so the D3 RMSE is not recorded", unit="m")
