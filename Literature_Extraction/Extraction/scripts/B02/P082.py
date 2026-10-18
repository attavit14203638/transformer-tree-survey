import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2 = "P082", "Table 2"
ROWS = [
    R("task", "crown_segmentation", 2, "Sec. 3.1", "The research proposes a methodology based on conducting ITC segmentation from high-resolution CHM derived from Geiger Mode LiDAR data"),
    R("output_unit", "crown_mask", 3, "Sec. 4.1", "Only one class is included in this study, i.e., crowns"),
    R("tree_setting", "natural_forest", 1, "Abstract", "Inundation mapping in forest and dense vegetated areas"),
    NR("tree_taxa", note="species not given"),
    R("study_area", "Payette River near Crouch, Idaho, United States", 2, "Sec. 2.1", "Geiger Mode LiDAR data was collected over a portion of Payette River, near Crouch Idaho on June 25, 2022"),
    D("biome", "temperate", "Idaho, United States; not stated in the paper"),
    R("platform", "airborne", 2, "Sec. 2.1", "with average acquisition height at 3,787 m above ground level"),
    R("sensors", "lidar", 2, "Sec. 2.1", "Geiger Mode LiDAR data was collected", note="Geiger-mode LiDAR; model input is a canopy height model"),
    R("gsd", "50", 2, "Sec. 2.2", "A 50 cm CHM based on ground and vegetation points was created", unit="cm", note="CHM raster"),
    R("point_density", "50", 2, "Sec. 2.1", "produces a uniform point cloud distribution of 50 points per meter squared", unit="pts/m2"),
    R("annotation_type", "polygon_mask", 3, "Sec. 3.2", "We propose a semi-automated data labeling process incorporating local maxima-based tree detection and watershed-based crown segmentation"),
    R("dataset_name", "own Geiger-mode LiDAR CHM dataset, Payette River (4 tiles)", 2, "Sec. 2.2", "A total of 4 tiles are used in this study", item="dataset1"),
    R("dataset_role", "train_test", 3, "Table 1", "Number of trees 5,705 1,211", item="dataset1", note="evaluation on 300 manually digitised crowns in one tile"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_trees", "6916", "5,705 training + 1,211 validation trees (Table 1, p.3)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "Mask2Former with Swin Transformer backbone", "proposed", "hierarchical_vit", "Swin (size not stated)", "imagenet", "full_fine_tuning", 3, "Sec. 3.3",
              "Swin Transformer model (Liu et al. 2021) is used in this study, which is a transformer-based backbone", pre_page=3, pre_loc="Sec. 4.1",
              pre_ev="we initialize the Swin Transformer backbone using the pretrained weights obtained from the imagenet-1k dataset", strat_page=3, strat_loc="Sec. 4.1",
              strat_ev="The total iteration of the training process is set to 68,750")
ROWS += model("model2", "Watershed segmentation (marker-controlled, CHM)", "other_baseline", "classical", "local maxima and watershed (no learning)", "none", None, 4, "Sec. 4.2",
              "For further comparison, the watershed segmentation results are also quantified by the same indices", strat_note="not a trained model")
res = Counter("res")
for lab, key in (("Rate of correct detection", "detection"), ("Rate of correct segmentation", "segmentation")):
    nums, ev = row(PID, 4, rf"^\s*{lab}\s+(?=\d)", 0, cols=(70, 200))
    assert len(nums) == 2, (lab, nums)
    note = ("digitised crowns intersecting any predicted crown / digitised crowns (Eq. 1)" if key == "detection"
            else "predicted crowns with IoU > 0.5 / correctly detected crowns (Eq. 2)")
    for item, v in (("model1", nums[0]), ("model2", nums[1])):
        ROWS += result(res(), "other", item, "dataset1", v, 4, T2, ev, unit="fraction", setting=f"300 manually digitised crowns; rate of correct {key}",
                       headline=(item == "model1" and key == "segmentation"), note=note)
ROWS += avail()
ROWS += [D("notes", "Counts behind the rates: Mask2Former 261 of 300 crowns detected and 167 correctly segmented; watershed 300 and 162 (Table 2). Validation mAP is only plotted (Fig. 8).", "Sec. 4.2 to 5 and Table 2 (p.4)")]
