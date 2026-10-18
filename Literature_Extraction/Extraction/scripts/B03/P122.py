import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1 = "Table I (image)"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "this article proposes a tree species classification network, VSLNet, based on multimodal data fusion"),
    R("output_unit", "tree_label", 5, "Sec. III-E", "For each tree on the list, GPS is utilized to query the fused features, obtaining classification information for the corresponding pixels, which is then employed as the category identifier for the trees"),
    R("tree_setting", "mixed", 5, "Sec. IV-A", "The data collection encompassed an area approximately five meters around the OHTL", note="vegetation along overhead transmission line corridors"),
    R("tree_taxa", "camphor; pine; bamboo; Chinese fir", 6, T1, "Camphor Pine Bamboo Chinese", note="bamboo is one of the four classes"),
    R("study_area", "Chenzhou City and Zhangjiajie City / Xiangxi Prefecture, Hunan Province, China", 5, "Sec. IV-A",
      "the first being Chenzhou City situated in the southeast of Hunan Province, China ... the second region encompasses Zhangjiajie City and Xiangxi Prefecture in the northwest of Hunan Province"),
    R("biome", "subtropical", 5, "Sec. IV-A", "by a subtropical warm and humid monsoon climate"),
    R("platform", "uav", 5, "Sec. IV-A", "The DJI Matrice 300 RTK UAV is equipped with Zenuse L1 camera and AQ600 Pro camera to collect LiDAR and multispectral data"),
    R("sensors", "rgb;multispectral;lidar", 5, "Sec. IV-A", "to collect LiDAR and multispectral data, respectively"),
    R("annotation_type", "image_label", 6, "Sec. IV-B", "The accurate annotation of tree species categories combined both onsite and expert online annotations", note="pixel-level species labels; trees from LiDAR clustering"),
    R("dataset_name", "own UAV LiDAR and multispectral data along transmission-line corridors, Hunan", 5, "Sec. IV-A", "Data collection was carried out from September to December 2022", item="dataset1"),
    NR("dataset_role", item="dataset1", note="training and test split not described"),
    D("dataset_public", "no", "No dataset link; coordinates withheld under confidentiality agreements (p.6)", item="dataset1"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="not stated")
VS = "VSLNet framework (multi-head self-attention branches, spectral index blocks, MLP decoder) with"
M = [  # item, name, role, category, backbone, AP, AR, evidence, note
    ("model1", "VSLNet with SSC fusion", "proposed", "cnn_transformer_hybrid", f"{VS} SSC fusion", "72.27", "75.75", "VSLNet+SSC V+S+L ... AP 72.27 AR 75.75", "proposed fusion module"),
    ("model2", "VSLNet with CMX fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} CMX fusion", "72.96", "71.11", "VSLNet+CMX [29] V+S+L ... AP 72.96 AR 71.11", "fusion module swapped"),
    ("model3", "VSLNet with GMNet fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} GMNet fusion", "70.91", "71.40", "VSLNet+GMNet [30] V+S+L ... AP 70.91 AR 71.40", "fusion module swapped"),
    ("model4", "VSLNet with RTFNet fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} RTFNet fusion", "70.15", "71.50", "VSLNet+RTFNet [33] V+S+L ... AP 70.15 AR 71.50", "fusion module swapped"),
    ("model5", "VSLNet with FuseSeg fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} FuseSeg fusion", "67.96", "69.60", "VSLNet+FuseSeg [32] V+S+L ... AP 67.96 AR 69.60", "fusion module swapped"),
    ("model6", "VSLNet with AFNet fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} AFNet fusion", "67.71", "68.58", "VSLNet+AFNet [35] V+S+L ... AP 67.71 AR 68.58", "fusion module swapped"),
    ("model7", "VSLNet with ABMDRNet fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} ABMDRNet fusion", "66.88", "68.83", "VSLNet+ABMDRNet [36] V+S+L ... AP 66.88 AR 68.83", "fusion module swapped"),
    ("model8", "VSLNet with DHFNet fusion", "transformer_baseline", "cnn_transformer_hybrid", f"{VS} DHFNet fusion", "66.99", "67.79", "VSLNet+DHFNet [31] V+S+L ... AP 66.99 AR 67.79", "fusion module swapped"),
    ("model9", "Pointformer (LiDAR only)", "transformer_baseline", "pure_vit", "Pointformer", "18.06", "19.56", "Pointformer [26] L ... AP 18.06 AR 19.56", "point-cloud transformer"),
    ("model10", "PointNet++ (LiDAR only)", "other_baseline", "other", "PointNet++", "22.20", "20.20", "PointNet++ [21] L ... AP 22.20 AR 20.20", "MLP-based point network; best non-transformer"),
]
res = Counter("res")
for item, name, role, cat, bb, ap, ar, ev, note in M:
    ROWS += model(item, name, role, cat, bb, None, None, 6, T1, ev, cat_note=note, **NP)
    st = "test data; mean over four species" + ("; LiDAR only" if item in ("model9", "model10") else "; visible, spectral and LiDAR")
    ROWS += result(res(), "precision", item, "dataset1", ap, 6, T1, ev, setting=st, headline=(item == "model1"),
                   note="'AP' in Table I = average of per-class precision; the paper reports precision and recall only")
    ROWS += result(res(), "recall", item, "dataset1", ar, 6, T1, ev, setting=st, note="'AR' in Table I = average of per-class recall")
ROWS += avail()
ROWS += [D("notes", "Table I lists 11 models; following protocol 4.1, recorded: every model built on the VSLNet transformer framework (different fusion modules), Pointformer, and PointNet++ (best non-transformer). KCNet (AP 19.72) is in Table I. Generalisation tests on the MFNet urban RGB-T dataset (Table III) are not tree data.", "Sec. IV-B and Table I (pp. 6 to 7)")]
