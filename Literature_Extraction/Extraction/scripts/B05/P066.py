import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P066"
# Three LiDAR benchmarks (UAV, TLS, ALS) with manually isolated trees; the Average rows of Tables 6 to 8 are means over the test plots. Final tree segmentation: 3D tree instances (instance_3d),
# mean IoU over trees (mIoUt) and F1 at an IoU above 0.5 (Sec. 2.4). Stem base or tree top finding is detection at object level (F1). StemCls is a point-level classification of stem points.
# Primary: the UAV plots (first listed dataset), the hands-off TreeisoNet (fully automatic) and ForAINet, for the tree segmentation and the stem base detection.
TS = Tags(task="tree_segmentation_3d", level="instance_3d", split="test", agg="mean")
TD = Tags(task="detection", level="object", split="test", agg="mean")
TC = Tags(task="component_segmentation", level="point", split="test", agg="mean")
OBJ_EV = "This study introduces TreeisoNet, a suite of supervised deep neural networks tailored for robust 3D tree segmentation across natural forest environments"
LOC_EV = "Finland and Canada (Liang et al., 2018; Xi and Hopkinson, 2022) Europe and Oceania (Puliti et al., 2023) Canada (Xi et al., 2024)"
SPLIT_EV = "approximately two-thirds of the total plots were used to train the DL models, with the remaining reserved for testing"
PUB_EV = "All datasets are open-sourced and have previously been used in benchmarking other algorithms"
DS = {"dataset1": ("UAV", "FOR-instance UAV laser scans (Riegl (mini) VUX-1), 11 test plots", "for_instance_v1", 6, 11), "dataset2": ("TLS", "TLS plots (Teledyne Optech Ilris), 6 test plots", "xi_hopkinson_tls_2022", 7, 6),
      "dataset3": ("ALS", "ALS plots (Teledyne Optech Titan), 4 test plots", "xi_2024_als_canada", 8, 4)}

ROWS = [
    R("task", "tree_segmentation_3d;detection;component_segmentation", 1, "Abstract", "These networks are specifically designed to identify tree locations, stem components (if available), and crown clusters"),
    R("output_unit", "tree_points;tree_instance;tree_component", 1, "Abstract", "These networks are specifically designed to identify tree locations, stem components (if available), and crown clusters"),
    R("tree_setting", "natural_forest", 1, "Abstract", OBJ_EV),
    R("tree_taxa", "deciduous and coniferous forest plots (pines, spruce, poplar, aspen and others)", 10, "Sec. 4.1", "A strong allometric difference between deciduous and coniferous plots is evident"),
    R("study_area", "Finland and Canada (TLS), Europe and Oceania (UAV), Canada (ALS)", 3, "Sec. 2.1", LOC_EV),
    D("biome", "multiple", "boreal, temperate and other forests of Europe, Oceania and Canada; not coded in the paper"),
    D("platform", "multiple", "terrestrial, UAV and airborne laser scanning (Sec. 2.1)"),
    R("sensors", "lidar", 3, "Sec. 2.1", "Teledyne Optech Ilris: TLS Riegl (mini) VUX-1: UAV Teledyne Optech Titan: ALS"),
    R("annotation_type", "point_label", 3, "Sec. 2.1", "three datasets were used, as detailed in Table 2, including manually isolated trees", note="manually isolated reference trees at plot level"),
]
for ds, (nm, desc, key, pg, nplots) in DS.items():
    ROWS += [R("dataset_name", f"{nm}: {desc}", 3, "Sec. 2.1", LOC_EV, item=ds), R("dataset_key", key, 3, "Sec. 2.1", LOC_EV, item=ds),
             R("dataset_role", "train_test", 4, "Sec. 2.2", SPLIT_EV, item=ds, note=f"{nplots} test plots (Table {pg})"), R("dataset_public", "yes", 3, "Sec. 2.1", PUB_EV, item=ds)]
ROWS += [NR("point_density", note="the paper gives point spacings, not densities: 2.0 cm for TLS, 4.3 cm for UAV and 28 cm for ALS (Sec. 2.1)")]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="")
ROWS += model("model1", "TreeisoNet (hands-off: 3D-SegFormer and 3D U-Net modules in sequence)", "proposed", "cnn_transformer_hybrid", "3D voxel SegFormer (MiT-b0 style) for stem classification and stem base detection, 3D U-Net for stem and crown clustering", None, None, 3, "Sec. 2.3",
              "stem points are first classified from a plot scan using a 3D voxel-based SegFormer", cat_note="3D-SegFormer modules with multi-head attention plus 3D U-Net modules (Sec. 2.3, Table 4)", pre_status="not_reported", pre_note="not stated",
              strat_note="AdamW, learning rate 0.001, batch size 1, training until no gain for an 8 h tolerance (Sec. 2.4); initial weights are not stated")
ROWS += model("model2", "TreeisoNet with manual intervention between the modules (intervened)", "proposed", "cnn_transformer_hybrid", "same modules; each module is fed manually refined input", None, None, 6, "Sec. 2.5",
              "focused on the separate accuracy of each individual DL network using manually refined references as inputs", cat_note="the same modules as the hands-off TreeisoNet (Sec. 2.5)",
              pre_status="not_reported", pre_note="not stated", strat_note="same training as the hands-off TreeisoNet")
ROWS += model("model3", "ForAINet (UAV benchmark)", "cnn_baseline", "cnn", "sparse 3D convolutional network (ForAINet)", None, None, 6, "Table 6", "Accuracy data sourced from Xiang et al. (2024)", cat_note="ForAINet is a sparse convolutional network (protocol 4.5); its numbers are copied from Xiang et al. (2024)", **NPRE)
ROWS += model("model4", "Treeiso (TLS benchmark)", "other_baseline", "classical", "unsupervised isolation of trees from point clouds", None, None, 9, "Sec. 3.2",
              "When compared to the unsupervised Treeiso (Xi and Hopkinson, 2022), which was fine-tuned for the same TLS datasets", cat_note="unsupervised, no neural network", **NPRE)
ROWS += model("model5", "lidR-itcSegment (ALS benchmark)", "other_baseline", "classical", "local maximum and region growing on the canopy height model", None, None, 9, "Sec. 3.3",
              "Table 8 also benchmarks the accuracies of lidR-itcSegment (Dalponte and Coomes, 2016), Mask R-CNN, and AMS3D", cat_note="unsupervised, no neural network", **NPRE)
ROWS += model("model6", "Mask R-CNN (ALS benchmark)", "cnn_baseline", "cnn", "Mask R-CNN on rasterised ALS data", None, None, 9, "Sec. 3.3",
              "Table 8 also benchmarks the accuracies of lidR-itcSegment (Dalponte and Coomes, 2016), Mask R-CNN, and AMS3D", cat_note="convolutional instance segmentation, re-trained on the plot split of this study", **NPRE)
ROWS += model("model7", "AMS3D (ALS benchmark)", "other_baseline", "classical", "adaptive mean shift in 3D with crown shape assumptions", None, None, 9, "Sec. 3.3",
              "Table 8 also benchmarks the accuracies of lidR-itcSegment (Dalponte and Coomes, 2016), Mask R-CNN, and AMS3D", cat_note="unsupervised mean shift, no neural network", **NPRE)
ROWS += model("model8", "Watershed3D crown clustering from known seed points", "other_baseline", "classical", "3D watershed", None, None, 9, "Sec. 3.5",
              "The accuracies for the watershed and shortest path methods are reported as mIoUs of 0.68 and 0.79, respectively", cat_note="classical clustering with known seeds (Table 10)", **NPRE)
ROWS += model("model9", "ShortestPath crown clustering from known seed points", "other_baseline", "classical", "shortest path clustering", None, None, 9, "Sec. 3.5",
              "The accuracies for the watershed and shortest path methods are reported as mIoUs of 0.68 and 0.79, respectively", cat_note="classical clustering with known seeds (Table 10)", **NPRE)
ROWS += model("model10", "Local maximum (LM) tree top detection", "other_baseline", "classical", "local maximum of the point cloud heights", None, None, 10, "Table 10", "LM is the local maximum method", cat_note="classical, no neural network", **NPRE)

res = Counter("res")
RES = {}
def add(key, item, ds, metric, val, page, loc, ev, tags, setting, note, prim=False, headline=False, copied=None):
    global ROWS
    RES[key] = res()
    ROWS += result(RES[key], metric, item, ds, val, page, loc, ev, unit="fraction", setting=setting, note=note, **tags, primary=prim, headline=headline, copied_from=copied)

# Tables 6 to 8: Average rows (columns as in the table headers; the UAV and TLS tables have 15 numbers, the ALS table 10)
u, evu = row(PID, 6, r"^\s*Average\s+(?=\d)", 0)
t, evt = row(PID, 7, r"^\s*Average\s+(?=\d)", 0)
a, eva = row(PID, 8, r"^\s*Average\s+(?=\d)", 0)
assert len(u) == 15 and len(t) == 15 and len(a) == 10
for ds, nums, ev, pg, tab in (("dataset1", u, evu, 6, "Table 6"), ("dataset2", t, evt, 7, "Table 7")):
    nm = DS[ds][0]
    P = ds == "dataset1"
    st_h, st_i = f"{tab}: {nm} test plots, hands-off TreeisoNet", f"{tab}: {nm} test plots, intervened TreeisoNet"
    add(f"{ds}_h_miou", "model1", ds, "miou", nums[11], pg, tab, ev, TS, st_h, "mean IoU over trees (mIoUt) of the final CrownOff3D segmentation, average over the test plots", prim=P, headline=P)
    add(f"{ds}_h_f1", "model1", ds, "f1", nums[12], pg, tab, ev, TS, st_h, "instance F1 of the final CrownOff3D segmentation at an IoU above 0.5, average over the test plots", prim=P)
    add(f"{ds}_i_miou", "model2", ds, "miou", nums[5], pg, tab, ev, TS, st_i, "mean IoU over trees (mIoUt) of CrownOff3D with manually refined inputs, average over the test plots")
    add(f"{ds}_i_f1", "model2", ds, "f1", nums[6], pg, tab, ev, TS, st_i, "instance F1 of CrownOff3D with manually refined inputs at an IoU above 0.5, average over the test plots")
    add(f"{ds}_h_base", "model1", ds, "f1", nums[7], pg, tab, ev, TD, st_h, "F1 of the stem base detection (TreeBase), average over the test plots", prim=P)
    add(f"{ds}_i_base", "model2", ds, "f1", nums[1], pg, tab, ev, TD, st_i, "F1 of the stem base detection (TreeBase) with manually cleaned stem points, average over the test plots")
    add(f"{ds}_stem", "model1", ds, "miou", nums[0], pg, tab, ev, TC, f"{tab}: {nm} test plots, StemCls (shared by both versions)", "class mean IoU (mIoUc) of the stem point classification (3D-SegFormer), average over the test plots")
add("dataset1_forainet_miou", "model3", "dataset1", "miou", u[13], 6, "Table 6", evu, TS, "Table 6: UAV test plots, ForAINet", "mean IoU over trees (mIoUt), average over the test plots; copied from Xiang et al. (2024)", prim=True, copied="Xiang et al. (2024)")
add("dataset1_forainet_f1", "model3", "dataset1", "f1", u[14], 6, "Table 6", evu, TS, "Table 6: UAV test plots, ForAINet", "instance F1, average over the test plots; copied from Xiang et al. (2024)", prim=True, copied="Xiang et al. (2024)")
add("dataset2_treeiso_miou", "model4", "dataset2", "miou", t[13], 7, "Table 7", evt, TS, "Table 7: TLS test plots, Treeiso", "mean IoU over trees (mIoUt), average over the test plots")
add("dataset2_treeiso_f1", "model4", "dataset2", "f1", t[14], 7, "Table 7", evt, TS, "Table 7: TLS test plots, Treeiso", "instance F1, average over the test plots")
# ALS: the first stage (tree top finding) has no manual input, so TreeBase is listed once; TreeOff2D clusters the crowns of the ALS scans
st = "Table 8: ALS test plots"
add("dataset3_h_miou", "model1", "dataset3", "miou", a[5], 8, "Table 8", eva, TS, st + ", hands-off TreeisoNet", "mean IoU over trees (mIoUt) of the final TreeOff2D segmentation, average over the test plots")
add("dataset3_h_f1", "model1", "dataset3", "f1", a[6], 8, "Table 8", eva, TS, st + ", hands-off TreeisoNet", "instance F1 of the final TreeOff2D segmentation, average over the test plots")
add("dataset3_i_miou", "model2", "dataset3", "miou", a[3], 8, "Table 8", eva, TS, st + ", intervened TreeisoNet", "mean IoU over trees (mIoUt) of TreeOff2D with manually refined inputs, average over the test plots")
add("dataset3_i_f1", "model2", "dataset3", "f1", a[4], 8, "Table 8", eva, TS, st + ", intervened TreeisoNet", "instance F1 of TreeOff2D with manually refined inputs, average over the test plots")
add("dataset3_base", "model1", "dataset3", "f1", a[0], 8, "Table 8", eva, TD, st + ", TreeBase (tree top finding)", "F1 of the tree top detection (TreeBase, listed under the intervened header; no manual input at this first stage), average over the test plots")
for key, item, idx in (("dataset3_itc_miou", "model5", 7), ("dataset3_mrcnn_miou", "model6", 8), ("dataset3_ams3d_miou", "model7", 9)):
    add(key, item, "dataset3", "miou", a[idx], 8, "Table 8", eva, TS, st + ", benchmark", "mean IoU over trees (mIoUt), average over the test plots")
# Table 10: crown clustering and stem detection benchmarks with known seed points (UAV rows, then TLS rows)
for ds, n, pg in (("dataset1", 0, 10), ("dataset2", 1, 10)):
    b, evb = row(PID, pg, r"^\s*Average\s+(?=\d)", n)
    assert len(b) == 12, b  # CrownOff3D mIoUt, F1; Watershed3D mIoUt, F1; ShortestPath mIoUt, F1; TreeBase F1, P, R; LM F1, P, R
    nm = DS[ds][0]
    st = f"Table 10: {nm} test plots, benchmark with known seed points"
    add(f"{ds}_ws_miou", "model8", ds, "miou", b[2], pg, "Table 10", evb, TS, st, "mean IoU over trees (mIoUt) of 3D watershed clustering, average over the test plots")
    add(f"{ds}_ws_f1", "model8", ds, "f1", b[3], pg, "Table 10", evb, TS, st, "instance F1 of 3D watershed clustering, average over the test plots")
    add(f"{ds}_sp_miou", "model9", ds, "miou", b[4], pg, "Table 10", evb, TS, st, "mean IoU over trees (mIoUt) of shortest path clustering, average over the test plots")
    add(f"{ds}_sp_f1", "model9", ds, "f1", b[5], pg, "Table 10", evb, TS, st, "instance F1 of shortest path clustering, average over the test plots")
    add(f"{ds}_lm_f1", "model10", ds, "f1", b[9], pg, "Table 10", evb, TD, st, "F1 of local maximum tree top detection, average over the test plots")
# Transfer pairs (V8): UAV-trained and TLS-trained TreeisoNet swapped (final CrownOff3D mIoU, text of Sec. 4.2)
XFER_EV = "the final CrownOff3D mIoU is 0.80 when using the TLS model for UAV dataset testing and 0.84 when using the UAV model for TLS dataset testing"
tr = Counter("tr")
for src, tgt, vin, evin, pgin, vout in (("UAV", "TLS", u[11], evu, 6, "0.84"), ("TLS", "UAV", t[11], evt, 7, "0.80")):
    k = tr()
    tab_in = "Table 6" if src == "UAV" else "Table 7"
    ROWS += [R("transfer_source", f"{src} laser scans (the plots used to train the TreeisoNet modules)", 11, "Sec. 4.2", "the final CrownOff3D mIoU is 0.80 when using the TLS model for UAV dataset testing", item=k),
             R("transfer_target", f"{tgt} laser scans (test plots of the other sensor)", 11, "Sec. 4.2", "Despite differences in forest type, sensors, and scanning perspectives between the TLS and UAV datasets", item=k),
             R("shift_type", "species;sensor;platform", 11, "Sec. 4.2", "Despite differences in forest type, sensors, and scanning perspectives between the TLS and UAV datasets", item=k),
             R("transfer_model", "model1", 11, "Sec. 4.2", XFER_EV, item=k),
             R("transfer_metric", "miou", 11, "Sec. 4.2", XFER_EV, item=k, note="mean IoU over trees of the final CrownOff3D segmentation"),
             R("value_in_domain", vin, pgin, tab_in, evin, item=k, unit="fraction", note=f"{src} test plots, hands-off TreeisoNet (Table average)"),
             R("value_out_of_domain", vout, 11, "Sec. 4.2", XFER_EV, item=k, unit="fraction", note=f"{tgt} test plots with the {src}-trained model"),
             R("transfer_level", "instance_3d", 11, "Sec. 4.2", XFER_EV, item=k)]
ROWS += avail(code_url="https://github.com/truebelief/cc-TreeAIBox-plugin", page=3, loc="Sec. 1", ev="The source code is accessible as a module that can be integrated into the 3D editing software CloudCompare",
              data="yes", data_page=3, data_loc="Sec. 2.1", data_ev=PUB_EV, note_d="the three benchmark datasets are open-sourced")
ROWS += [R("limitations", "TreeisoNet stays below the accuracy of human interpretation and cannot adequately handle low-resolution data", 11, "Sec. 5", "it still falls short of the maximum accuracy achievable through human interpretation")]
ROWS += [D("notes", "Tables 6 to 8 give the plot-wise and average accuracies of the four TreeisoNet modules (StemCls, TreeBase, TreeOff2D, CrownOff3D) for the intervened and the hands-off versions and for benchmark methods (ForAINet for UAV with numbers from Xiang et al. 2024, Treeiso for TLS, lidR-itcSegment, Mask R-CNN and AMS3D for ALS); Table 10 benchmarks crown clustering and tree top finding with known seeds. Recorded are the Average rows: final tree segmentation (mIoUt and F1), stem base or tree top F1 and the StemCls mIoUc; precision and recall of TreeBase and the TreeOff2D stem clustering are not recorded. The transfer pairs swap the UAV-trained and the TLS-trained models (Sec. 4.2, Fig. 7). Text and table disagree for some averages (cf items).",
           "Sec. 3 and Tables 6 to 8 and 10 (pp. 6 to 10)")]
# Text against table (V1a: the tables are used)
ROWS += conflict("cf1", f"{RES['dataset1_stem']}.value", "0.77", 7, "Sec. 3.1", "The intervened TreeisoNet averaged an mIoUc of 0.77 for StemCls", "V1a",
                 "The Average row of Table 6 gives 0.74 for the StemCls mIoUc; the text of Sec. 3.1 gives 0.77", unit="fraction")
ROWS += conflict("cf2", f"{RES['dataset2_i_base']}.value", "0.92", 9, "Sec. 3.2", "an F1 of 0.92 for TreeBase", "V1a",
                 "The Average row of Table 7 gives 0.95 for the intervened TreeBase F1; the text of Sec. 3.2 gives 0.92", unit="fraction")
ROWS += conflict("cf3", f"{RES['dataset2_i_miou']}.value", "0.87", 9, "Sec. 3.2", "and 0.87 for CrownOff3D", "V1a",
                 "The Average row of Table 7 and the abstract give 0.86 for the intervened CrownOff3D mIoUt; the text of Sec. 3.2 gives 0.87", unit="fraction")
ROWS += conflict("cf4", f"{RES['dataset3_h_miou']}.value", "0.57", 9, "Sec. 3.3", "reported average F1 scores of 0.68 and mIoUt values of 0.80 and 0.57", "V1a",
                 "The Average row of Table 8 and the abstract give 0.59 for the hands-off TreeOff2D mIoUt; the text of Sec. 3.3 gives 0.57 (the F1 of that column)", unit="fraction")
