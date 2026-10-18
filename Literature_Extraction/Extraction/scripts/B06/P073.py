import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P073", "Table 3 (image)"
# Twelve TLS plots of Jasper National Park. A 3D voxel SegFormer classifies tree, stem and downed woody log points (binary classifiers, scored by mIoU over the class and its complement, which includes
# non-tree points), stems and trees are then isolated and measured. Individual-tree clustering mIoU is the tree segmentation result (primary); height, DBH and LCBH are validated against field trees.
TP = Tags(task="component_segmentation", level="point", split="test", agg="mean")
TS = Tags(task="tree_segmentation_3d", level="instance_3d", split="test", agg="mean")
TA = Tags(task="attribute_estimation", level="tree", split="not_stated", agg="single")
OBJ_EV = "using a transformer-based deep learning classifier"
FLD_EV = "partially validating the results through field measurements of tree height"
SPEC_EV = "A total of 572 trees were surveyed including six major species"

ROWS = [
    R("task", "component_segmentation;tree_segmentation_3d;attribute_estimation", 1, "Abstract", OBJ_EV, note="tree, stem and downed log point classes, individual-tree clustering, tree height, DBH and live crown base height"),
    R("output_unit", "tree_component;tree_points;tree_attribute", 1, "Abstract", FLD_EV),
    R("tree_setting", "natural_forest", 1, "Abstract", "sampled within the mountain pine beetle-impacted forests of Jasper National Park, Canada"),
    R("tree_taxa", "lodgepole pine (Pinus contorta) mainly; white and black spruce, trembling aspen, Douglas fir, subalpine fir", 5, "Sec. 2.2", SPEC_EV),
    R("study_area", "Jasper National Park, Alberta, Canada", 1, "Abstract", "sampled within the mountain pine beetle-impacted forests of Jasper National Park, Canada"),
    D("biome", "temperate", "montane forests of the Canadian Rockies; coded from the study area"),
    R("platform", "terrestrial_lidar", 4, "Sec. 2.2", "A Teledyne Optech Polaris TLS station (1550 nm) was", note="one central scan and four corner scans per plot"),
    R("sensors", "lidar", 4, "Sec. 2.2", "A Teledyne Optech Polaris TLS station (1550 nm) was"),
    R("annotation_type", "point_label;regression_target", 8, "Fig. 5", "labeled tree (red) and non-tree (blue) points of JP1", note="manually labelled tree, stem and log points; field-measured height, DBH and live crown base height as regression targets"),
    R("dataset_name", "own TLS point clouds of 12 plots in Jasper National Park with 572 field-surveyed trees", 4, "Sec. 2.2", "Twelve adjacent TLS plots coincident with the field data were collected at six locations", item="dataset1"),
    R("dataset_key", "own_p073_jasper_tls", 4, "Sec. 2.2", "Twelve adjacent TLS plots coincident with the field data were collected at six locations", item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 3.2", "used as the training sample (Figure 5b) with another single plot used as the independent", item="dataset1", note="tree classifier: 4 training plots and 1 test plot; stem: 10 training and 2 test clips; downed logs: 21 training and 4 test clips"),
    R("dataset_public", "yes", 22, "Data Availability Statement", "The reference TLS dataset is currently being published to the Federated", item="dataset1", note="Federated Research Data Repository, doi 10.20383/103.0814"),
    R("dataset_url", "https://doi.org/10.20383/103.0814", 22, "Data Availability Statement", "Research Data Repository (FRDR) and can be accessed via https://doi.org/10.20383/103.0814", item="dataset1"),
    D("n_images", "12", "twelve TLS plots (JP1 to JP12 in Table 1; abstract); plot scans of 25 m radius at 5 mm resolution", item="dataset1", unit="plots"),
    R("n_trees", "572", 5, "Sec. 2.2", SPEC_EV, item="dataset1", unit="trees", note="field-surveyed trees within 11.3 m of the plot centres; 269 were matched to isolated TLS trees"),
]
ROWS += model("model1", "SegFormer (3D voxel version) with stem isolation, tree clustering and QSM", "proposed", "hierarchical_vit", "3D voxel SegFormer (MiT-b0 style, 128 voxel blocks) for tree, stem and downed log points", None, None, 7, "Sec. 3.2",
              "layers with self-attention-based transformers, surpassing well-recognized semantic classifiers",
              cat_note="SegFormer replaces CNN layers with self-attention transformers (Sec. 3.2); the later steps (stem isolation, clustering, QSM) are not learned", pre_status="not_reported", pre_note="not stated",
              strat_note="SegFormer parameters at their defaults, trained on the labelled plots and clips (Table 2); initial weights are not stated")
res = Counter("res")
TB = "Table 3: average testing accuracy of the processing steps"
for metric, v, ev, pg, loc, tags, prim, note in (
        ("miou", "0.91", "yielding a high mIoU accuracy of 0.91 from four testing plots", 12, "Sec. 4.1", TP, False, "tree point classification (binary: tree against non-tree, so the mIoU includes the non-tree class)"),
        ("miou", "0.94", "the stem classification over all testing samples was 0.94", 12, "Sec. 4.1", TP, False, "stem point classification (binary: stem against non-stem, so the mIoU includes non-tree points)"),
        ("miou", "0.81", "the mIoU of woody log classification was 0.81", 12, "Sec. 4.1", TP, False, "downed woody log point classification (not a tree component)")):
    ROWS += result(res(), metric, "model1", "dataset1", v, pg, loc, ev, unit="fraction", setting=TB, note=note, **tags, primary=prim)
ROWS += result(res(), "miou", "model1", "dataset1", "0.86", 15, "Sec. 4.1", "individual tree clustering was 0.73 and and 0.86, respectively", unit="fraction", setting=TB, note="individual-tree clustering mIoU (after stem isolation, which scores 0.73 together with downed logs)",
               **TS, primary=True, headline=True)
ROWS += result(res(), "miou", "model1", "dataset1", "0.73", 15, "Sec. 4.1", "The mIoU of stem isolation and individual tree clustering was 0.73 and", unit="fraction", setting=TB, note="isolation of stems and downed woody logs (includes non-tree logs)", **TS)
for attr, r2, rmse, ru, tag_ev, n_note in (("height", "0.95", "1.3", "m", "tree height", "269 trees"), ("dbh", "0.96", "1.3", "cm", "DBH", "269 trees"), ("other", "0.60", "2.6", "m", "LCBH", "84 trees with live crowns")):
    tg = TA(attribute=attr)
    st = f"Sec. 4.2: TLS extraction against field measurement of {tag_ev}"
    ROWS += result(res(), "r2", "model1", "dataset1", r2, 18, "Sec. 4.2", "The r2 between the field measurements and TLS extraction was 0.95 for tree height, 0.96 for DBH, and 0.60 for LCBH", unit="fraction", setting=st, note=f"R2 of {tag_ev}; {n_note}" + ("; attribute is live crown base height" if attr == "other" else ""), **tg, primary=True)
    ROWS += result(res(), "rmse", "model1", "dataset1", rmse, 18, "Sec. 4.2", "RMSE (RMSE%) was 1.3 m (9.0%) for tree height, 1.3 cm (8.7%) for DBH, and 2.6 m (39.6%) for LCBH", unit=ru, setting=st, note=f"RMSE of {tag_ev}; {n_note}", **tg, primary=True)
ROWS += [R("limitations", "The stem isolation assumes one stem per tree, and SegFormer is used only as a binary classifier", 21, "Sec. 5", "one stem per tree was the underlying assumption of our stem isolation algorithm"),
         D("code_available", "no", "No code repository in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "yes", 22, "Data Availability Statement", "The reference TLS dataset is currently being published to the Federated")]
ROWS += [D("notes", "Table 3 gives the average testing mIoU of the processing steps (tree, stem and downed log point classification with the SegFormer, stem isolation, individual-tree clustering, branch filtering and QSM); the SegFormer classifiers and the two isolation steps are recorded, branch filtering (0.62) is a rule-based step and is not. The mIoU of the binary classifiers includes the complement class (non-tree points), so only the individual-tree clustering mIoU is primary. Field validation (Sec. 4.2): 269 field trees for height and DBH, 84 live-crown trees for LCBH (R2 and RMSE recorded; the biases are not). The QSM volume comparisons (Fig. 14) are not recorded. The page text of this PDF is partly duplicated.",
           "Sec. 4 and Table 3 (pp. 12 to 18)")]
