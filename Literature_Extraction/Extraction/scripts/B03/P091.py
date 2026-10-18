import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T5 = "P091", "Table 5"
SP = {"MON": "Quercus mongolica", "JAP": "Sophora japonica", "TAB": "Pinus tabulaeformis", "DAV": "Populus davidiana"}
ROWS = [
    R("task", "component_segmentation;attribute_estimation", 1, "Abstract",
      "The framework integrates the SegFormer deep-learning model, trunk-skeleton extraction, an adaptive curvature-segmentation algorithm, and segment-wise 3-D reconstruction",
      note="trunk segmentation in single photographs, then stem volume and tree height"),
    R("output_unit", "tree_component;tree_attribute", 11, "Sec. 3.4", "stem volumes derived with the PMPE method were compared with destructive (felled-tree) measurements"),
    R("tree_setting", "natural_forest", 2, "Sec. 2.1", "The study area comprises three forest sites in Beijing: Badaling Forest Farm, Magezhuang Forest Farm, and Gongqing Forest Farm",
      note="forest farms, including a public welfare forest; stand densities 130 to 6528 trees/ha"),
    R("tree_taxa", "Pinus tabulaeformis; Sophora japonica; Quercus mongolica; Populus davidiana", 3, "Sec. 2.2",
      "including Pinus tabulaeformis (TAB), Sophora japonica (JAP), Quercus mongolica (MON), and Populus davidiana (DAV)"),
    R("study_area", "Badaling, Magezhuang and Gongqing forest farms, Beijing, China", 2, "Sec. 2.1",
      "The study area comprises three forest sites in Beijing: Badaling Forest Farm, Magezhuang Forest Farm, and Gongqing Forest Farm"),
    R("biome", "temperate", 2, "Sec. 2.1", "This site represents the warm temperate deciduous broad-leaved forests"),
    R("platform", "ground_camera", 3, "Sec. 2.2.3", "Image data were captured using a Huawei Mate 50 smartphone camera", note="handheld LiDAR and felled trees used as references"),
    R("sensors", "rgb", 3, "Sec. 2.2.3", "Image data were captured using a Huawei Mate 50 smartphone camera"),
    R("annotation_type", "polygon_mask;regression_target", 5, "Sec. 2.3.2", "All training images were manually annotated at the pixel level, and trunk regions were delineated individually",
      note="trunk masks; destructive measurements of volume and height as reference"),
    R("dataset_name", "own smartphone trunk images (Beijing, four species) with destructive measurements of 141 trees", 5, "Sec. 2.3.2",
      "this study acquired 3013 images covering four tree species", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.3.2", "we allocated 60 %, 20 %, and 20 % of the images for each species to the training, validation, and test sets", item="dataset1",
      note="test set 147 JAP, 146 DAV, 158 TAB and 151 MON images (p.7)"),
    R("dataset_public", "on_request", 14, "Data availability", "Data will be made available on request", item="dataset1"),
    R("n_images", "3013", 5, "Sec. 2.3.2", "this study acquired 3013 images covering four tree species", item="dataset1", unit="images"),
    R("n_trees", "141", 1, "Abstract", "conducted destructive measurements on 141 trees of various species for comparative evaluation", item="dataset1", unit="trees",
      note="trees with destructive reference measurements"),
]
ROWS += model("model1", "SegFormer trunk segmentation within the PMPE framework", "proposed", "hierarchical_vit", "SegFormer (MiT variant not stated)", None, None, 5,
              "Sec. 2.3.3", "We employed the SegFormer semantic-segmentation network to delineate stem regions", pre_status="not_reported", pre_note="not stated",
              strat_note="trained on the annotated trunk images, at most 100 epochs (p.5); initialisation not stated", cat_note="no CNN comparison in the paper")
res = Counter("res")
for n, sp in enumerate(("MON", "JAP", "TAB", "DAV")):
    nums, ev = row(PID, 8, rf"(?:Segformer\s+)?{sp}\s+(?=\d)", 0, cols=(80, 200))
    assert len(nums) == 4, (sp, nums)
    st = f"test images of {SP[sp]} ({sp}); trunk masks"
    ROWS += result(res(), "iou", "model1", "dataset1", nums[0], 8, T5, ev, setting=st, headline=(sp == "DAV"))
    ROWS += result(res(), "f1", "model1", "dataset1", nums[1], 8, T5, ev, setting=st, note="Dice coefficient")
VOL = "For the PMPE method, all four species exhibited a good fit to the reference data. MON showed the highest accuracy, with R2 = 1.00, rBias = 0.71 %, and rRMSE = 2.01 %. TAB and DAV each achieved R2 = 0.99; their rBias values were 0.86 % and 1.98 %, and rRMSE values were 7.68 % and 4.94 %, respectively. JAP yielded a slightly higher bias with R2 = 0.99, rBias = 2.42 %, and rRMSE = 6.85 %"
HGT = "Under PMPE, TAB and JAP exhibited the highest fit (R2 = 0.95); however, their rBias values (2.75 % and 3.34 %) and rRMSE values (5.18 % and 5.38 %) were slightly higher than those of the other species. By contrast, MON (R2 = 0.86) and DAV (R2 = 0.87) showed lower rBias (1.80 % and 2.18 %) and acceptable rRMSE (5.53 % and 4.97 %)"
for sp, r2, rr in (("MON", "1.00", "2.01"), ("TAB", "0.99", "7.68"), ("DAV", "0.99", "4.94"), ("JAP", "0.99", "6.85")):
    st = f"stem volume of {SP[sp]} ({sp}) against destructive measurement"
    ROWS += result(res(), "r2", "model1", "dataset1", r2, 11, "Sec. 3.4", VOL, unit="fraction", setting=st)
    ROWS += result(res(), "other", "model1", "dataset1", rr, 11, "Sec. 3.4", VOL, setting=st, note="rRMSE (relative RMSE), %")
for sp, r2, rr in (("TAB", "0.95", "5.18"), ("JAP", "0.95", "5.38"), ("MON", "0.86", "5.53"), ("DAV", "0.87", "4.97")):
    st = f"tree height of {SP[sp]} ({sp}) against destructive measurement"
    ROWS += result(res(), "r2", "model1", "dataset1", r2, 11, "Sec. 3.4", HGT, unit="fraction", setting=st)
    ROWS += result(res(), "other", "model1", "dataset1", rr, 11, "Sec. 3.4", HGT, setting=st, note="rRMSE (relative RMSE), %")
ROWS += avail(data="on_request", data_page=14, data_loc="Data availability", data_ev="Data will be made available on request")
ROWS += [D("notes", "Single-photograph measurement framework built on SegFormer trunk masks; handheld LiDAR (HMLS) is the non-learning comparison (volume rRMSE 3.41% to 5.46%). The dataset has 3013 images of four species and 141 felled trees.", "Sec. 2 to 3.4 (pp. 2 to 11)")]
