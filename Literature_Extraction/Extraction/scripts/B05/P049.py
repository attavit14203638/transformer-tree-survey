import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1 = "P049", "Table 1"
# Crown masks of individual ash trees against manual crown polygons: crown segmentation at instance-mask level; Table 1 averages over 12 trees (three per vitality class). Neither method
# is trained (SAM is pre-trained and prompted, the fine mask is per-image k-means), so the evaluation sample is treated as a test set as for other zero-shot SAM studies.
T = Tags(task="crown_segmentation", level="instance_mask", split="test", agg="mean")
SUB_EV = "only a representative subset of twelve trees in total, three typical trees per vitality class, was selected for validation"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "both fine and coarse crown segmentation methods were applied to close-range multispectral UAV imagery"),
    R("output_unit", "crown_mask", 4, "Sec. 2", "creating a coarse and a fine mask for each tree crown"),
    R("tree_setting", "plantation", 3, "Sec. 2.1", "this plantation was established in 1995 with an initial layout of 228 ash trees planted in a 10 m × 10 m grid", note="two ash seed orchards"),
    R("tree_taxa", "Fraxinus excelsior (common ash)", 1, "Abstract", "Fraxinus excelsior L. trees affected by ash dieback"),
    R("study_area", "Emmendingen (near Freiburg) and Schorndorf (near Stuttgart), Baden-Wuerttemberg, Germany", 3, "Sec. 2.1", "is situated about 15 km north of Freiburg ... lies east of Stuttgart"),
    D("biome", "temperate", "Baden-Wuerttemberg, Germany; not stated in the paper"),
    R("platform", "uav", 3, "Sec. 2.2", "recorded using the UAV system DJI Mavic 3 Multispectral (Mavic 3 M)"),
    R("sensors", "rgb;multispectral", 3, "Sec. 2.2", "The Mavic 3 M is equipped with a 20 MP RGB camera and four 5 MP multispectral sensors", note="the segmentation uses the NIR and red bands (NDVI)"),
    NR("gsd", note="close-up images were taken about 8 m above the crowns and no GSD is printed; the 80 m survey orthophoto has 2.77 cm"),
    R("annotation_type", "polygon_mask", 5, "Sec. 2.4", "a tree crown polygon was generated for all three mask types", note="manual crown polygons for twelve trees (ArcGIS Pro)"),
    R("dataset_name", "close-up multispectral UAV images of ash tree crowns, two seed orchards (30 trees each)", 8, "Sec. 3", "Based on the 174 individual images of analysed ash trees", item="dataset1"),
    R("dataset_key", "own_p049_ash_orchards", 8, "Sec. 3", "Based on the 174 individual images of analysed ash trees", item="dataset1"),
    R("dataset_role", "test_only", 5, "Sec. 2.4", SUB_EV, item="dataset1", note="no training data are used"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_images", "174", 8, "Sec. 3", "Based on the 174 individual images of analysed ash trees", item="dataset1", unit="images", note="close-up images of 60 trees in June, July and October 2023, after six trees were excluded"),
    D("n_trees", "60", "30 ash trees were selected at each of the two sites (p.3)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "Fine segmentation (k-means on a blended NIR and NDVI image with texture and edge features)", "proposed", "classical", "unsupervised k-means clustering (k = 2) of a NIR-NDVI hybrid image with GLCM texture and Sobel edge features", None, None, 4, "Sec. 2.3",
              "subjected to k-means clustering with k = 2", cat_note="unsupervised clustering without a neural network (Sec. 2.3)", pre_status="not_reported", pre_note="no training", strat_note="unsupervised clustering per image; no annotated training data (p.10)")
ROWS += model("model2", "Coarse segmentation (pre-trained SAM, one point prompt)", "proposed", "foundation_model", "SAM image encoder (size not stated), prompt encoder and mask decoder", None, "zero_shot", 4, "Sec. 2.3",
              "we applied the pre-trained Segment Anything Model (SAM)", cat_note="pre-trained SAM prompted by one foreground point and manual selection of the best of several masks (Sec. 2.3)",
              pre_status="not_reported", pre_note="pre-trained SAM; the pre-training data are not named", strat_page=4, strat_loc="Sec. 2.3", strat_ev="we applied the pre-trained Segment Anything Model (SAM)", strat_note="used without training")
nums, ev = row(PID, 7, r"^\s*Overall mean\s+(?=\d)", 0)
assert len(nums) == 8, nums  # fine mask IoU, precision, recall, F1; coarse mask IoU, precision, recall, F1
res = Counter("res")
for item, k0, nm in (("model1", 0, "fine mask"), ("model2", 4, "coarse mask")):
    st = f"Table 1: {nm} against manual crown polygons, overall mean of twelve trees"
    ROWS += result(res(), "iou", item, "dataset1", nums[k0], 7, T1, ev, unit="fraction", setting=st, note="IoU, overall mean", **T, primary=True)
    ROWS += result(res(), "f1", item, "dataset1", nums[k0 + 3], 7, T1, ev, unit="fraction", setting=st, note="F1, overall mean", **T, primary=True, headline=(item == "model1"))
ROWS += [R("limitations", "The fine method needs isolated trees without overlapping foliage; touching crowns are merged into one object", 9, "Sec. 4", "segmentation often merged them into a single object"),
         D("code_available", "no", "No code repository or availability statement in the paper"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table 1 compares the fine and the coarse mask with manual segmentation of twelve trees (three per vitality class, classes 1 to 4): IoU, precision, recall and F1 per class and overall; the overall means are recorded (IoU and F1; precision and recall are in the table). The orthophoto test of the fine method (80 m, 2.77 cm GSD) and the NDVI statistics are qualitative or not tree extraction accuracy and are not recorded.",
           "Sec. 2.4 to 3 and Table 1 (pp. 5 to 7)")]
