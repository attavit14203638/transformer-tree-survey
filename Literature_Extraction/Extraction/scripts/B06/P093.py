import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P093"
# Woody and foliage separation of single-tree point clouds (component segmentation, point level; mIoU over the wood and foliage classes). Tables III to VI are images. Table IV (main comparison) gives four
# methods on three datasets; the first listed dataset (tropical, also the headline) is primary. Table V (backbone replaced inside SPTNet) is an ablation and not primary. 70 percent of each dataset trains
# the network and the rest is used as validation and test data without saying which, so the split is not stated.
T = Tags(task="component_segmentation", level="point", split="not_stated", agg="single")
OBJ_EV = "a network that combines sparse convolution (SpConv) and transform blocks is proposed for the separation of woody and foliage components in tree point clouds called SPTNet"
HEAD_EV = "SPTNet obtains an overall classification accuracy (OA) of 94.69% and 89.96% mean of intersection-over-union (mIoU) on the large tropical dataset"
SPLIT_EV = "70% of each of these datasets are"
TI = "Table I (image)"

ROWS = [
    R("task", "component_segmentation", 1, "Abstract", OBJ_EV),
    R("output_unit", "tree_component", 1, "Abstract", OBJ_EV),
    R("tree_setting", "mixed", 8, "Sec. 4.A", "The first six datasets", note="TLS plots of six temperate species, a simulated larch set and tropical trees"),
    R("tree_taxa", "spruce, aspen, poplar, birch, pine, maple, larch (simulated) and 15 tropical species", 8, "Sec. 4.A", "The first six datasets", note="Table I; the tropical dataset has 15 species (abstract)"),
    R("study_area", "Canada, Finland and Cameroon", 8, "Sec. 4.A", "collected manually from Canada, Finland, and Cameroon,"),
    D("biome", "multiple", "boreal and temperate stands of Canada and Finland, tropical forest of Cameroon (Table I)"),
    D("platform", "multiple", "terrestrial laser scanning (Optech ILRIS and Leica scanners) and simulated trees (Table I)"),
    R("sensors", "lidar", 8, "Sec. 4.A", "collected manually from Canada, Finland, and Cameroon,", note="TLS point clouds of single trees; one dataset is simulated"),
    R("annotation_type", "point_label", 8, "Sec. 4.A", "manually labeled", note="wood and foliage labels of single trees"),
]
DS = [("dataset1", "lewos", "tropical tree dataset, Cameroon (LeWoS, Momo Takoudjou et al.)", "61", "collected manually from Canada, Finland, and Cameroon,", "yes"),
      ("dataset2", "stpctls", "Birch dataset, Finland (STPCTLS of Xi et al.)", "100", "The first six datasets", "yes"),
      ("dataset3", "own_p093_larch_sim", "simulated larch dataset (OnyxTree)", "100", "The Larch", "no"),
      ("dataset4", "stpctls", "Spruce dataset, Canada (STPCTLS of Xi et al.)", "174", "The first six datasets", "yes"),
      ("dataset5", "stpctls", "Aspen dataset, Canada (STPCTLS of Xi et al.)", "162", "The first six datasets", "yes"),
      ("dataset6", "stpctls", "Poplar dataset, Canada (STPCTLS of Xi et al.)", "80", "The first six datasets", "yes"),
      ("dataset7", "stpctls", "Pine dataset, Canada and Finland (STPCTLS of Xi et al.)", "220", "The first six datasets", "yes"),
      ("dataset8", "stpctls", "Maple dataset, Canada (STPCTLS of Xi et al.)", "169", "The first six datasets", "yes")]
for ds, key, nm, n, ev, pub in DS:
    ROWS += [R("dataset_name", nm, 8, "Sec. 4.A", ev, item=ds), R("dataset_key", key, 8, "Sec. 4.A", ev, item=ds),
             R("dataset_role", "train_test", 11, "Sec. 4.C", SPLIT_EV, item=ds, note="70 percent training; the rest for validation and test"),
             R("n_trees", n, 8, TI, f"Quantities {n}", item=ds, unit="trees")]
    ROWS += [D("dataset_public", "yes", "public data of Xi et al. (2020) or of the LeWoS benchmark, cited in Table I", item=ds)] if pub == "yes" else [D("dataset_public", "no", "simulated with OnyxTree; no release stated", item=ds)]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on 70 percent of each dataset (50 epochs on Pine, 100 on the others for SPTNet; Sec. 4.C)")
ROWS += model("model1", "SPTNet (sparse convolution and transformer blocks, proposed)", "proposed", "cnn_transformer_hybrid", "sparse convolution blocks and five transformer blocks with multi-head attention on voxels with MDC and NDO point features", None, None, 6, "Sec. 3",
              "SpConv blocks and five transformer blocks.", cat_note="SpConv blocks for local features and transformer blocks (multi-head attention) for global features (abstract, Fig. 5)", **NPRE)
ROWS += model("model2", "LWCLF (2018)", "other_baseline", "classical", "traditional feature extraction with threshold segmentation", None, None, 13, "Sec. 4.C", "LWCLF and LeWoS are two methods", cat_note="traditional feature extraction and threshold segmentation (Sec. 4.C)", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model3", "LeWoS (2020)", "other_baseline", "classical", "traditional feature extraction with threshold segmentation", None, None, 13, "Sec. 4.C", "based on traditional feature extraction threshold segmentation", cat_note="traditional feature extraction and threshold segmentation (Sec. 4.C)", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model4", "FWCNN (2020)", "other_baseline", "other", "PointNet-based wood and foliage separation combined with physical mechanisms", None, None, 13, "Sec. 4.C", "PointNet. At the same time, FWCNN is based on the com-", cat_note="described as a method based on PointNet (Sec. 4.C)", **NPRE)
ROWS += model("model5", "PointNet++ (2017) backbone", "other_baseline", "other", "PointNet++", None, None, 13, "Sec. 4.D", "PointNet++ is a point-based convolution method;", cat_note="MLP-based point network (protocol 4.5); the paper calls it a point-based convolution method", **NPRE)
ROWS += model("model6", "DGCNN (2019) backbone", "cnn_baseline", "cnn", "DGCNN", None, None, 13, "Sec. 4.D", "DGCNN is a graph convolution-based method, and Point", cat_note="graph convolution (Sec. 4.D)", **NPRE)
ROWS += model("model7", "Point Transformer (2021) backbone", "transformer_baseline", "pure_vit", "Point Transformer", None, None, 14, "Sec. 4.D", "Transformer and PointBert are transformer-based methods that perform state-of-the-art in each task", cat_note="transformer-based (Sec. 4.D)", **NPRE)
ROWS += model("model8", "PointBert (2022) backbone", "transformer_baseline", "pure_vit", "Point-BERT", None, None, 14, "Sec. 4.D", "Transformer and PointBert are transformer-based methods that perform state-of-the-art in each task", cat_note="transformer-based (Sec. 4.D)", **NPRE)
ROWS += model("model9", "PointMLP (2022) backbone", "other_baseline", "other", "PointMLP", None, None, 14, "Sec. 4.D", "PointMLP is a method based on the pure MLP structure", cat_note="pure MLP structure (Sec. 4.D)", **NPRE)
res = Counter("res")
RID = {}
# Table IV (p.13, image): mIoU of four methods on the tropical, Birch and Larch datasets
T4 = [("model2", "LWCLF (2018)", ("68.10", "84.11", "48.59")), ("model3", "LeWoS (2020)", ("63.89", "87.34", "60.29")), ("model4", "FWCNN (2020)", ("89.24", "95.89", "77.95")), ("model1", "SPTNet(Ours)", ("89.96", "96.92", "78.21"))]
for item, lab, vals in T4:
    for (ds, dn), v in zip((("dataset1", "tropical"), ("dataset2", "Birch"), ("dataset3", "Larch")), vals):
        ROWS += result(res(), "miou", item, ds, v, 13, "Table IV (image)", f"{lab} {dn} mIoU(%) {v}", unit="%", setting=f"Table IV: comparison of methods, {dn} dataset", note="mIoU of the woody and foliage classes", **T, primary=(ds == "dataset1"), headline=(item == "model1" and ds == "dataset1"))
# Table III (p.11, image): SPTNet on the five other datasets (the tropical, Birch and Larch rows repeat Table IV)
for ds, dn, v in (("dataset4", "Spruce", "93.70"), ("dataset5", "Aspen", "92.95"), ("dataset6", "Poplar", "85.70"), ("dataset7", "Pine", "92.20"), ("dataset8", "Maple", "94.82")):
    ROWS += result(res(), "miou", "model1", ds, v, 11, "Table III (image)", f"{dn} mIoU(%) {v}", unit="%", setting=f"Table III: SPTNet, {dn} dataset", note="mIoU of the woody and foliage classes", **T)
# Table V (p.14, image): backbone replaced in the pipeline (ablation, not primary)
T5 = [("model5", "PointNet++ (2017)", ("83.81", "88.77", "65.51")), ("model6", "DGCNN (2019)", ("78.23", "82.61", "57.03")), ("model7", "Point Transformer (2021)", ("88.02", "90.69", "67.72")),
      ("model8", "PointBert (2022)", ("83.97", "89.89", "67.57")), ("model9", "PointMLP (2022)", ("73.88", "81.24", "47.06"))]
for item, lab, vals in T5:
    for (ds, dn), v in zip((("dataset1", "Tropical"), ("dataset2", "Birch"), ("dataset3", "Larch")), vals):
        RID[(item, dn)] = res()
        ROWS += result(RID[(item, dn)], "miou", item, ds, v, 14, "Table V (image)", f"{lab} {dn} mIoU(%) {v}", unit="%", setting=f"Table V: backbone comparison inside the SPTNet pipeline, {dn} dataset", note="mIoU of the woody and foliage classes; ablation of the backbone", **T)
# Table VI (p.14, image): model weight size and inference time of one tree of about 50,000 points
T6 = [("model5", "PointNet++ (2017) 16s*245 95.23M 10950ms", "95.23", "10950"), ("model6", "DGCNN (2019) 15s*161 - 10113ms", None, "10113"), ("model7", "Point Transformer(2021) 45s*90 222.91M 25314ms", "222.91", "25314"),
      ("model8", "PointBert (2022) 17s*102 526.2M 60650ms", "526.2", "60650"), ("model9", "PointMLP (2022) 10s*198 192.28M 22109ms", "192.28", "22109"), ("model1", "SPTNet(Ours) 8s*48 29.49M 3391ms", "29.49", "3391")]
for item, ev, mb, ms in T6:
    if mb:
        ROWS += [R("total_params", mb, 14, "Table VI (image)", ev, item=item, unit="M", note="model weight parameter size as printed")]
    ROWS += [R("inference_speed", f"{ms} ms per tree of about 50,000 points", 14, "Table VI (image)", ev, item=item)]
ROWS += conflict("cf1", f"{RID[('model5', 'Larch')]}.value", "65.15", 14, "Sec. 4.D", "89.80% OA and 65.15 mIoU", "V1a",
                 "Table V gives 65.51 for the PointNet++ mIoU on the Larch dataset; the text of Sec. 4.D gives 65.15", unit="%")
ROWS += [R("limitations", "The method is sensitive to the voxelisation resolution, which also drives the computation, and it is trained per dataset on single trees only", 17, "Sec. 5", "parameter. Different voxel resolutions can lead to fluctuations"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "partial", "the TLS datasets and the tropical dataset are published by others (Table I); the simulated larch data are not stated as released")]
ROWS += [D("notes", "Tables III to VI are images. Table IV compares LWCLF, LeWoS, FWCNN and SPTNet (OA, FA, WA, mIoU) on the tropical, Birch and Larch datasets; only mIoU is recorded, and the tropical column is primary (it is also the abstract's headline, OA 94.69 and mIoU 89.96). Table III gives SPTNet on all eight datasets (OA, FA, WA, F1, mIoU; mIoU recorded for the five datasets not in Table IV). Table V replaces the backbone inside the SPTNet pipeline (PointNet++, DGCNN, Point Transformer, PointBert, PointMLP; OA and mIoU, mIoU recorded; ablation, not primary). Table VI gives model weight sizes and inference times. The split of the results is not stated (70 percent training, the rest validation and test). The STPCTLS data of Xi et al. (2020) and the LeWoS tropical data are shared with other papers of the corpus.",
           "Sec. 4 and Tables III to VI (pp. 11 to 14)")]
