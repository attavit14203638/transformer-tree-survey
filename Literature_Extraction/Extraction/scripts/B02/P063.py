import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T5, T6 = "P063", "Table 5", "Table 6 (rotated)"
URL = "https://github.com/3DOM-FBK/3D-forest-semantic-segmentation"
DATA = "https://doi.org/10.5281/zenodo.17172162"
ROWS = [
    R("task", "component_segmentation", 1, "Abstract", "fine-grained forest semantic segmentation into six components (ground, low vegetation, trunk, branches, foliage, and woody debris)"),
    R("output_unit", "tree_points;tree_component", 1, "Abstract", "forest semantic segmentation into six components (ground, low vegetation, trunk, branches, foliage, and woody debris)",
      note="per-point classes, including trunk, branch and foliage points"),
    R("tree_setting", "natural_forest", 5, "Sec. 3", "biodiverse boreal natural forests located in Espoonlahti in southern Finland"),
    R("tree_taxa", "mixed boreal species (pine, spruce, birch, maple, aspen, rowan, oak, lime, alder)", 5, "Sec. 3",
      "close to 20 different tree species available, the most common ones are pine, spruce, birch, maple, aspen, rowan, oak, lime, and alder"),
    R("study_area", "Espoonlahti, southern Finland", 5, "Sec. 3", "biodiverse boreal natural forests located in Espoonlahti in southern Finland"),
    R("biome", "boreal", 5, "Sec. 3", "biodiverse boreal natural forests"),
    R("platform", "airborne", 5, "Sec. 3", "acquired in July 2023 in leaf-on condition by the helicopter-mounted HeliALS multispectral LiDAR system"),
    R("sensors", "lidar", 6, "Table 2", "Wavelength (nm) 1550 905 532", note="three-wavelength (multispectral) airborne laser scanning"),
    R("point_density", "1198", 6, "Sec. 3.1", "1,198 points/m2, approximately twice as high as the densest mono-scanner", unit="pts/m2", note="merged multispectral point cloud"),
    R("annotation_type", "point_label", 6, "Table 3", "Distribution of forest components in the ForestSemantic-MS dataset across training, testing, and all plots"),
    R("dataset_name", "ForestSemantic-MS", 5, "Sec. 3", "This dataset is publicly available as the first fine-grained MS LiDAR forestry benchmark dataset", item="dataset1"),
    R("dataset_role", "train_test", 6, "Table 3", "Data ... Training ... Testing ... All", item="dataset1", note="two test plots"),
    R("dataset_public", "yes", 5, "Sec. 3", "This dataset is publicly available as the first fine-grained MS LiDAR forestry benchmark dataset", item="dataset1"),
    R("dataset_url", DATA, 21, "References", f"ForestSemantic-MS Dataset ... [dataset] {DATA}", item="dataset1"),
]
TRN = "trained on ForestSemantic-MS (e.g. PTv3 3,000 epochs with AdamW, p.11); initialisation not stated"
M = [("model1", "KPConv", "KPConv", "cnn_baseline", "cnn", "kernel point convolution network", "point-convolution network"),
     ("model2", "SPT", "Superpoint Transformer (SPT)", "transformer_baseline", "hierarchical_vit", "Superpoint Transformer", "attention on superpoint graphs"),
     ("model3", "PTv3", "Point Transformer V3 (PTv3)", "proposed", "pure_vit", "Point Transformer V3", "point-cloud transformer; best model in the study"),
     ("model4", "RF", "Random forest", "other_baseline", "classical", "random forest on point features", "classical machine learning")]
res = Counter("res")
for item, lab, name, role, cat, bb, cnote in M:
    nums, ev = row(PID, 14, rf"^{lab}\s+(?=\d)")
    assert len(nums) == 10, (lab, nums)
    ROWS += model(item, name, role, cat, bb, None, None, 14, T5, ev, pre_status="not_reported", pre_note="not stated", strat_note=TRN if item != "model4" else "classical model",
                  cat_note=cnote)
    st = "two test plots; input coordinates + SWIR + NIR + green"
    ROWS += result(res(), "miou", item, "dataset1", nums[6], 14, T5, ev, setting=st, note="six classes")
    ROWS += result(res(), "oa", item, "dataset1", nums[9], 14, T5, ev, setting=st)
for v_miou, v_oa, setting, ev, hl in (("69.1", "94.9", "two test plots; input coordinates + SWIR + green (best spectral configuration)",
                                       "+SWIR + green 93.2 66.5 71.5 26.1 95.1 62.2 69.1 67.0 80.0 94.9", True),
                                      ("66.7", "94.4", "two test plots; input coordinates only", "Coordinates 93.8 69.5 69.0 21.6 94.2 52.1 66.7 63.7 77.2 94.4", False)):
    ROWS += result(res(), "miou", "model3", "dataset1", v_miou, 15, T6, ev, setting=setting, headline=hl, note="six classes; read from the rotated table")
    ROWS += result(res(), "oa", "model3", "dataset1", v_oa, 15, T6, ev, setting=setting)
ROWS += avail(code_url=URL, page=9, loc="Sec. 4", ev=f"The source code and pretrained weights of the best semantic segmentation model are publicly available at {URL}", weights="yes",
              data="yes", data_page=5, data_loc="Sec. 3", data_ev="This dataset is publicly available as the first fine-grained MS LiDAR forestry benchmark dataset")
ROWS += [D("notes", "Table 5 compares models on the three-wavelength input; Table 6 varies the input channels for PTv3 (the abstract's 69.1% mIoU is the coordinates + SWIR + green configuration). Training time: KPConv 17.5 h, SPT 1.1 h, PTv3 5.1 h, RF 0.2 h (Table 7).", "Sec. 5 to 6 and Tables 5 to 7 (pp. 14 to 16)")]
