import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T1 = "P062", "Table 1"
URL = "https://github.com/open-forest-observatory/tree-detection-framework"
ROWS = [
    R("task", "detection;crown_segmentation", 1, "Abstract", "using a state-of-the-art image segmentation model, Segment Anything Model 2 (SAM2), in a zero-shot manner for individual tree detection and segmentation",
      note="quantitative evaluation is on bounding boxes"),
    R("output_unit", "tree_instance;crown_mask", 2, "Sec. 2", "After mask generation, bounding box predictions can be calculated for each mask to obtain tree detection results"),
    R("tree_setting", "natural_forest", 2, "Sec. 3", "the National Ecological Observatory Network (NEON) TreeEvaluation dataset ... two tropical field sites in Malaysia"),
    NR("tree_taxa", note="all trees; species not given"),
    R("study_area", "22 NEON sites, United States; Danum and Sepilok, Malaysia; Emerald Bay State Park, California (qualitative)", 2, "Sec. 3",
      "30,975 manually-annotated tree bounding boxes in 22 sites across the United States ... two tropical field sites in Malaysia"),
    R("biome", "multiple", 2, "Sec. 3", "three datasets spanning two continents", note="temperate and boreal US sites and tropical Malaysian sites"),
    R("platform", "multiple", 2, "Sec. 3", "aerial photographs captured by a quadcopter (referred to as Emerald Point)", note="NEON airborne imagery, Detectree2 aerial imagery and UAV imagery (Emerald Point)"),
    R("sensors", "rgb", 4, "Fig. 3", "Tree crown detection results using RGB images from the NEON (Row 1) and Detectree2 (Row 2) datasets"),
    R("annotation_type", "bounding_box;polygon_mask", 2, "Sec. 3", "30,975 manually-annotated tree bounding boxes ... approximately 1,000 labeled crowns", note="boxes (NEON) and crown polygons (Detectree2)"),
    R("dataset_name", "NEON TreeEvaluation", 2, "Sec. 3", "the National Ecological Observatory Network (NEON) TreeEvaluation dataset", item="dataset1"),
    R("dataset_role", "test_only", 3, "Sec. 3", "To the best of our knowledge, the data we evaluate all methods on was not used during training", item="dataset1",
      note="DeepForest was trained on other NEON data (in-distribution for DeepForest)"),
    R("dataset_public", "yes", 2, "Sec. 3", "the National Ecological Observatory Network (NEON) TreeEvaluation dataset (Weinstein et al., 2019; 2021)", item="dataset1", note="published benchmark"),
    R("n_trees", "30975", 2, "Sec. 3", "includes 30,975 manually-annotated tree bounding boxes in 22 sites", item="dataset1", unit="trees"),
    R("dataset_name", "Detectree2 released data (Danum, Sepilok East, Sepilok West)", 2, "Sec. 3", "the Danum, Sepilok East, and Sepilok West datasets released as part of Detectree2", item="dataset2"),
    R("dataset_role", "test_only", 3, "Sec. 3", "To the best of our knowledge, the data we evaluate all methods on was not used during training", item="dataset2",
      note="Detectree2 was trained on other data from the same sites (in-distribution for Detectree2)"),
    R("dataset_public", "yes", 2, "Sec. 3", "the released Detectree2 dataset (Ball et al., 2023)", item="dataset2"),
    R("n_trees", "1000", 2, "Sec. 3", "which includes approximately 1,000 labeled crowns", item="dataset2", unit="trees", note="approximately"),
]
ROWS += model("model1", "SAM2 (automatic mask generator, zero-shot)", "transformer_baseline", "foundation_model", "SAM2 Hiera-L", "other", "zero_shot", 2, "Sec. 2",
              "we use a pre-trained SAM2 model with the Hiera-L image encoder and refer to this zero-shot prediction approach as 'SAM2'",
              pre_note="released SAM2 weights; image encoder pre-trained with MAE (p.2)", cat_note="the method studied; not trained on any tree data")
ROWS += model("model2", "DeepForest 1.5.0", "cnn_baseline", "cnn", "RetinaNet (ResNet-50)", "domain_specific", None, 3, "Sec. 3",
              "The specific models we use are the DeepForest 1.5.0 release and the Detectree2 2.0.1 release, both with default parameters",
              pre_page=3, pre_loc="Sec. 3", pre_ev="DeepForest was trained on NEON data", pre_note="released model trained on NEON data", strat_note="released model used as is")
ROWS += model("model3", "Detectree2 2.0.1", "cnn_baseline", "cnn", "Mask R-CNN", "domain_specific", None, 3, "Sec. 3",
              "The specific models we use are the DeepForest 1.5.0 release and the Detectree2 2.0.1 release, both with default parameters",
              pre_page=3, pre_loc="Sec. 3", pre_ev="Detectree2 was trained on the Danum, Sepilok, and Paracou sites in their collected dataset",
              pre_note="released model trained on Danum, Sepilok and Paracou data", strat_note="released model used as is")
res = Counter("res")
VALS = {}
for item, lab in (("model2", "DeepForest"), ("model3", "Detectree2"), ("model1", "SAM2")):
    nums, ev = row(PID, 4, rf"^\s*{lab}\s+(?=\d)")
    assert len(nums) == 4, (lab, nums)
    VALS[item] = (nums, ev)
    for ds, (p, r) in (("dataset1", (nums[0], nums[1])), ("dataset2", (nums[2], nums[3]))):
        st = f"{'NEON' if ds == 'dataset1' else 'Detectree2'} benchmark; boxes at IoU 0.4"
        ROWS += result(res(), "precision", item, ds, p, 4, T1, ev, unit="fraction", setting=st, note="the paper reports precision and recall only")
        ROWS += result(res(), "recall", item, ds, r, 4, T1, ev, unit="fraction", setting=st, headline=(item == "model1" and ds == "dataset2"))
tr = Counter("tr")
for item, src, tgt, idx_in, idx_out in (("model2", "NEON (training data of DeepForest)", "Detectree2 tropical sites (Malaysia)", 1, 3),
                                        ("model3", "Detectree2 sites (training data of Detectree2)", "NEON sites (United States)", 3, 1)):
    nums, ev = VALS[item]
    t = tr()
    ROWS += [R("transfer_source", src, 3, "Sec. 3", "DeepForest was trained on NEON data, and Detectree2 was trained on the Danum, Sepilok, and Paracou sites", item=t),
             R("transfer_target", tgt, 3, "Sec. 3", "both of which are performing in-distribution inference for their corresponding datasets (NEON and Detectree2)", item=t),
             R("shift_type", "region;biome", 2, "Sec. 3", "three datasets spanning two continents", item=t),
             R("transfer_model", item, 4, T1, ev, item=t),
             R("transfer_metric", "recall", 4, T1, ev, item=t, note="boxes at IoU 0.4"),
             R("value_in_domain", nums[idx_in], 4, T1, ev, item=t, unit="fraction"),
             R("value_out_of_domain", nums[idx_out], 4, T1, ev, item=t, unit="fraction")]
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=f"We make our code available at: {URL}", data="yes", note_d="public benchmark datasets")
ROWS += [D("notes", "Zero-shot study: SAM2 is not trained on tree data, so its scores are cross-domain by construction. Box-prompted SAM2 (DeepForest + SAM2) is shown only qualitatively (Fig. 3).", "Sec. 2 to 3 and Table 1 (pp. 2 to 4)")]
