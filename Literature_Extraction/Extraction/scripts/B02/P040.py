import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T1 = "P040", "Table 1"
PAR = "The total number of parameters for each backbone is as follows: DAViT has 28.36 M parameters, Swin Transformer has 28.29 M, GLIP has 27.521 M, and EVA02 has 5.759 M"
ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "Mask2Former, a semantic segmentation architecture utilizing a dual-attention vision transformer backbone, was implemented to segment the Acacia trees"),
    R("output_unit", "crown_mask", 3, "Sec. 2.5", "mIoU = 1/2 (IoUbackgrounds + IoUAcacia tree)", note="binary Acacia crown versus background masks"),
    R("tree_setting", "mixed", 2, "Sec. 2.1", "The study area covers multiple urban and farmlands across the Fujairah and Sharjah emirates"),
    R("tree_taxa", "Acacia tortilis", 1, "Abstract", "The Acacia tortilis tree, locally known as Al Samr"),
    R("study_area", "Fujairah and Sharjah emirates (Fujairah City, Kalba), United Arab Emirates; 25 km2", 2, "Sec. 2.1", "including Fujairah City and Kalba, with a total area of 25 km²"),
    R("biome", "arid", 1, "Abstract", "one of the native trees in arid and semi-arid ecosystems"),
    R("platform", "uav", 2, "Sec. 2.1", "The dataset for this study was collected using the senseFly eBee X, a fixed-wing, survey-grade UAV system"),
    R("sensors", "rgb", 2, "Sec. 2.1", "equipped with a 20 MP S.O.D.A. (Sensor Optimized for Drone Applications) ultra-compact digital camera", note="RGB camera"),
    R("gsd", "2.5", 2, "Sec. 2.1", "with a ground sampling distance of 2.5 cm per pixel", unit="cm"),
    R("annotation_type", "polygon_mask", 2, "Sec. 2.3", "the automatically generated boundaries from the fine-tuned model are refined and improved", note="SAM-assisted crown delineation, refined by experts"),
    R("dataset_name", "own UAV Acacia dataset (Fujairah and Sharjah, 1024 x 1024 image-mask pairs)", 2, "Sec. 2.3", "Each zone was further subdivided into 1024×1024 image-mask pairs", item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 2.3", "A total of 11,067 pairs were allocated for training, 1010 for validation, and 800 for testing the models", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_images", "12877", "11,067 training + 1,010 validation + 800 test image-mask pairs (p.2, Sec. 2.3)", item="dataset1", unit="images"),
    R("n_trees", "9100", 2, "Sec. 2.2", "A total of 9100 Acacia trees were collected and widely distributed across the study area", item="dataset1", unit="trees",
      note="field-recorded trees; labels were expanded by image interpretation (p.4)"),
]
M = [  # item, label in Table 1, name, role, category, backbone, pretraining, pre note, params
    ("model1", "DAViT", "Mask2Former with DaViT-tiny backbone", "proposed", "hierarchical_vit", "DaViT-T", None, "not stated", "28.36"),
    ("model2", "Swin transformer", "Mask2Former with Swin-tiny backbone", "transformer_baseline", "hierarchical_vit", "Swin-T", None, "not stated", "28.29"),
    ("model3", "GLIP", "Mask2Former with GLIP-tiny backbone", "transformer_baseline", "hierarchical_vit", "GLIP-T image backbone", "other",
     "Grounded Language-Image Pre-training (GLIP) backbone", "27.521"),
    ("model4", "EVA-2", "Mask2Former with EVA02-tiny backbone", "transformer_baseline", "pure_vit", "EVA02-T", "other", "EVA02 pre-trained visual representation", "5.759"),
]
res = Counter("res")
for item, lab, name, role, cat, bb, pre, pnote, par in M:
    nums, ev = row(PID, 4, rf"^\s*{lab}\b")
    assert len(nums) == 4, (lab, nums)
    if pre:
        ROWS += model(item, name, role, cat, bb, pre, None, 4, T1, ev, pre_page=1, pre_loc="Abstract",
                      pre_ev="Swin Transformer, Grounded Language-Image Pre-training, and EVA02, a transformer-based visual representation pre-trained model",
                      pre_note=pnote, strat_note="trained with AdamW, batch size 2, learning rate 0.0001 (p.3); initialisation not stated")
    else:
        ROWS += model(item, name, role, cat, bb, None, None, 4, T1, ev, pre_status="not_reported", pre_note=pnote,
                      strat_note="trained with AdamW, batch size 2, learning rate 0.0001 (p.3); initialisation not stated")
    ROWS += [R("total_params", par, 3, "Sec. 3", PAR, item=item, unit="M", note="backbone parameters")]
    ROWS += result(res(), "miou", item, "dataset1", nums[2], 4, T1, ev, setting="test set", headline=(item == "model1"),
                   note="mean of background and Acacia IoU")
    ROWS += result(res(), "f1", item, "dataset1", nums[3], 4, T1, ev, setting="test set", note="mean F-score of background and Acacia")
ROWS += avail()
ROWS += [D("notes", "All four models are transformer-based (Mask2Former with different transformer backbones), so the paper has no CNN comparison. Training times: DaViT 40.89 h, EVA02 36.3 h, GLIP 24.1 h, Swin 19.4 h (p.3).", "Sec. 3 and Table 1 (pp. 3 to 4)")]
