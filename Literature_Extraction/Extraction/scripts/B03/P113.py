import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T3 = "Table 3 (rotated, image)"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "To effectively classify tree species within datasets characterized by limited samples, we introduced a novel approach named DenseNetBL"),
    R("output_unit", "tree_label", 7, "Sec. 2.5.4", "This study utilized SLIC to pre-segment individual tree crowns on images", note="SLIC crown segments classified by species, then aggregated into species maps"),
    R("tree_setting", "plantation", 3, "Sec. 2.1", "The typical woodland A ... consists of planted forests", note="mostly planted forest with some natural forest"),
    R("tree_taxa", "Ligustrum lucidum; Photinia beauverdiana; Cupressus; Osmanthus fragrans; Cinnamomum camphora", 4, "Table 1",
      "Ligustrum lucidum Ait Photinia beauverdiana Cupressus Osmanthus fragrans Cinnamomum camphora Bare land"),
    R("study_area", "Langya Mountain, Chuzhou City, Anhui Province, China (two 100 m x 100 m woodlands)", 2, "Sec. 2.1", "in Chuzhou City, Anhui Province, China’s Langya Mountain"),
    R("biome", "subtropical", 3, "Sec. 2.1", "the transition from the northern subtropical zone to the warm temperate zone", note="transition to warm temperate"),
    R("platform", "uav", 3, "Sec. 2.2", "The UAV high-resolution photos from Dajiang UAV (spirit p4r)"),
    R("sensors", "rgb", 3, "Sec. 2.2", "Red, green, and blue bands make up the three ba"),
    R("gsd", "0.05", 3, "Sec. 2.2", "with a spatial resolution of 0.05 m", unit="m"),
    R("annotation_type", "image_label", 3, "Sec. 2.3", "Field surveying refers to identifying the unlabeled samples obtained from SLIC segmentation", note="one label per crown segment, checked in the field"),
    R("dataset_name", "own UAV crown-segment dataset, Langya Mountain (five species plus bare land)", 3, "Sec. 2.3", "leaving 759 photographs of tree species with identified categories", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.4", "60% of the data is randomly chosen as classifier training data ... A further 40% of the data is used to verify the accuracy", item="dataset1",
      note="Table 1 lists 609 training and 150 validation samples"),
    R("dataset_public", "on_request", 16, "Data availability statement", "Data will be made available on request.", item="dataset1"),
    R("n_images", "759", 3, "Sec. 2.3", "leaving 759 photographs of tree species with identified categories", item="dataset1", unit="images", note="including 90 bare-land samples"),
]
SCR = dict(pre_page=1, pre_loc="Abstract", pre_ev="in scenarios devoid of pre-trained weights, DenseNetBL consistently outperformed its DenseNet counterpart",
           strat_page=16, strat_loc="Sec. 5", strat_ev="in instances where no pre-trained weights were employed", pre_note="no pre-trained weights")
M = [  # item, name, role, category, backbone, OA, kappa, note
    ("model1", "DenseNet33BL", "proposed", "cnn", "DenseNet33 with bottleneck-layer dense blocks", "0.901", "0.892", "best of the proposed DenseNetBL variants"),
    ("model2", "Swin Transformer (small)", "transformer_baseline", "hierarchical_vit", "Swin-S", "0.845", "0.805", ""),
    ("model3", "Vision Transformer (base, patch 16)", "transformer_baseline", "pure_vit", "ViT-B/16", "0.767", "0.708", ""),
    ("model4", "DenseNet33", "cnn_baseline", "cnn", "DenseNet33", "0.877", "0.845", "same-depth counterpart and best non-proposed CNN"),
]
EV = {"model1": "DenseNet33BL ... OA 0.901 Kappa 0.892", "model2": "Swintransformer ... OA 0.845 Kappa 0.805", "model3": "Visiontransformer ... OA 0.767 Kappa 0.708",
      "model4": "DenseNet33 ... OA 0.877 Kappa 0.845"}
res = Counter("res")
for item, name, role, cat, bb, oa, kp, note in M:
    ROWS += model(item, name, role, cat, bb, "none", "from_scratch", 10, T3, EV[item], cat_note=note, **SCR)
    st = "validation set (150 samples; five species and bare land); best hyperparameters"
    ROWS += result(res(), "oa", item, "dataset1", oa, 10, T3, EV[item], unit="fraction", setting=st, headline=(item == "model1"))
    ROWS += result(res(), "kappa", item, "dataset1", kp, 10, T3, EV[item], unit="fraction", setting=st)
ROWS += avail(data="on_request", data_page=16, data_loc="Data availability statement", data_ev="Data will be made available on request.")
ROWS += [D("notes", "Table 3 compares 13 classifiers; following protocol 4.1, recorded: both transformers, the best proposed model (DenseNet33BL) and DenseNet33 (counterpart and best other CNN). The other DenseNet/DenseNetBL depths and ResNet16 (OA 0.865) are in Table 3. A second experiment on a non-tree scene dataset (Table 6) is not recorded.", "Sec. 3.2 and Table 3 (pp. 8 to 10)")]
