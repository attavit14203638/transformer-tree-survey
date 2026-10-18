import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1 = "P079", "Table 1"
# Part segmentation (trunk, branch, leaf) of isolated synthetic trees, scored per point on the 10 percent test split; pIoU is the part IoU averaged over the three classes (primary), the class-wise F1 is not.
T = Tags(task="component_segmentation", level="point", split="test", agg="single")
OBJ_EV = "synthetic tree point cloud datasets for automated tree part segmentation"
DS_EV = "1000 stochastic generations were exported for each species composing a total of 5000 unique tree pointclouds"
MOD_EV = "PointNet [11], PointNet++ [12], ShellNet [22], Dynamic Graph CNN (DGCNN) [17], and Point-Voxel Transformer (PVT) [21]"

ROWS = [
    R("task", "component_segmentation", 1, "Abstract", OBJ_EV),
    R("output_unit", "tree_component", 1, "Abstract", "We assess these models based on their ability to segment tree"),
    R("tree_setting", "synthetic", 1, "Abstract", "we generate a diverse array of synthetic trees"),
    R("tree_taxa", "quaking aspen, Douglas fir, western white pine, black oak, western juniper (synthetic SpeedTree models)", 4, "Sec. 3", "western white pine, black oak, western juniper"),
    NR("study_area", note="synthetic trees of five Pacific Northwest species; no study area"),
    D("biome", "temperate", "species of the Pacific Northwest (USA); synthetic models without a study area"),
    D("platform", "synthetic", "point clouds generated from SpeedTree procedural tree models (Sec. 3)"),
    R("sensors", "lidar", 1, "Abstract", "synthetic tree point cloud datasets", note="synthetic point clouds from tree meshes, not a sensor"),
    R("annotation_type", "point_label", 1, "Abstract", "complete with detailed, per-point segmentational annotations"),
    R("dataset_name", "own synthetic SpeedTree point clouds, 5000 trees of five species (trunk, branch, leaf labels)", 6, "Sec. 3", DS_EV, item="dataset1"),
    R("dataset_key", "own_p079_speedtree_synthetic", 6, "Sec. 3", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 3", "80/10/10 split on a perspecies basis for training, validation, and testing", item="dataset1"),
    D("dataset_public", "no", "only the code that turns SpeedTree models into labelled point clouds is released (GitHub); the dataset is not stated as released", item="dataset1"),
    R("n_trees", "5000", 6, "Sec. 3", DS_EV, item="dataset1", unit="trees", note="1000 per species, 80/10/10 split per species; subsampled to 2048 points"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the synthetic training split with 2048-point subsamples; hyperparameters by Bayesian search (Sec. 4)")
ROWS += model("model1", "PointNet", "other_baseline", "other", "PointNet", None, None, 3, "Sec. 2", "Advanced models like PointNet and PointNet++ have revolutionized point cloud processing", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model2", "PointNet++", "other_baseline", "other", "PointNet++", None, None, 9, "Sec. 5.1", "PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set", cat_note="MLP-based point network (protocol 4.5)", **NPRE)
ROWS += model("model3", "ShellNet", "cnn_baseline", "cnn", "ShellConv blocks on concentric spherical shells", None, None, 9, "Sec. 5.1", "Using ShellConv blocks instead of PointNet blocks", cat_note="convolution on concentric shells (ShellConv)", **NPRE)
ROWS += model("model4", "DGCNN", "cnn_baseline", "cnn", "Edge Conv blocks on k-nearest-neighbour graphs", None, None, 9, "Sec. 5.1", "Edge Conv blocks seem to perform well on the cylindrical structure of the tree", cat_note="graph convolution (EdgeConv, protocol 4.5)", **NPRE)
ROWS += model("model5", "Point-Voxel Transformer (PVT)", "transformer_baseline", "cnn_transformer_hybrid", "point-voxel hybrid blocks with a U-Net-like architecture", None, None, 9, "Sec. 5.1",
              "hybrid-based PVT blocks that simultaneously learn from varying scales of the raw pointcloud and voxelized representations", cat_note="voxel branch combined with attention (Point-Voxel Transformer)", **NPRE)
res = Counter("res")
for item, rx in (("model1", "PointNet"), ("model2", r"PointNet\+\+"), ("model3", "ShellNet"), ("model4", "DGCNN"), ("model5", "PVT")):
    n, ev = row(PID, 9, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(n) == 11, (item, n)  # F1 trunk, branch, leaf; precision t, b, l; recall t, b, l; pIoU; latency
    ROWS += result(res(), "miou", item, "dataset1", n[9], 9, T1, ev, unit="fraction", setting="Table 1: test set of the synthetic dataset", note="pIoU: part IoU averaged over the trunk, branch and leaf classes (Sec. 4)", **T, primary=True, headline=(item == "model5"))
    for cls, v in zip(("trunk", "branch", "leaf"), n[0:3]):
        ROWS += result(res(), "f1", item, "dataset1", v, 9, T1, ev, unit="fraction", setting=f"Table 1: test set, {cls} class", note=f"F1 of the {cls} class", **T)
    ROWS += [R("inference_speed", f"{n[10]} ms per tree", 9, T1, ev, item=item, note="latency for a single tree of 2048 points")]
ROWS += [R("limitations", "The models are trained and tested on synthetic trees only; the real TLS example is qualitative because it has no point-wise labels", 10, "Sec. 5.2", "While quantitative performance metrics for the real tree are unavailable due to the lack of point-wise ground truth labels"),
         R("code_available", "yes", 2, "Sec. 1", "are available on our GitHub repository", note="code that transforms SpeedTree models into labelled point clouds; the training code is not mentioned"),
         D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "the dataset is not stated as released; only the generation code is")]
ROWS += [D("notes", "Table 1 compares PointNet, PointNet++, ShellNet, DGCNN and PVT on the synthetic test split (class-wise F1, precision and recall, pIoU and latency); pIoU and the class-wise F1 are recorded, and the latency of each model. Precision and recall are not recorded. The only real tree (a Douglas fir TLS scan, Fig. 3) is a qualitative demonstration with the PVT, so there is no transfer pair.",
           "Sec. 5 and Table 1 (pp. 8 to 10)")]
