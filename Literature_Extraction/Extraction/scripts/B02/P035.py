import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID = "P035"
ROWS = [
    R("task", "crown_segmentation;species_classification", 4, "Sec. 4", "We extensively study the performance of SAM and the informativeness of the DSM for tree crown instance segmentation",
      note="instance segmentation with species, genus or family classes"),
    R("output_unit", "crown_mask;tree_label", 5, "Sec. 4.2", "We evaluate our instance segmentation models with mean Average Precision (mAP)"),
    R("tree_setting", "mixed", 3, "Sec. 3", "three datasets representing different realistic application contexts: boreal plantations, temperate forests and tropical forests"),
    R("tree_taxa", "Quebec plantation conifers and hardwoods (9 classes); temperate forest species (18 classes); tropical families (31 classes)", 4, "Sec. 3",
      "we consider 18 classes of interest ... leaving 31 classes of interest", note="the 9 Quebec Plantations classes are on p.3"),
    R("study_area", "15 plantation sites in Quebec, Canada; Station de Biologie des Laurentides, Quebec, Canada; Barro Colorado Island, Panama", 4, "Sec. 3",
      "Station de Biologie de Laurentides ... the 2022 imagery of the Barro Colorado Island crown maps dataset"),
    R("biome", "multiple", 3, "Sec. 3", "boreal plantations, temperate forests and tropical forests"),
    R("platform", "uav", 3, "Sec. 3", "plantation sites from the UAV Canadian (Quebec) Plantations dataset"),
    R("sensors", "rgb;dsm", 3, "Sec. 3", "We use RGB orthomosaics, photogrammetry digital surface models (DSMs)"),
    R("gsd", "5", 3, "Sec. 3", "The imagery has a resolution of 5 mm/pixel", unit="mm", note="Quebec Plantations 5 mm; SBL 1.9 cm; BCI 4 cm"),
    R("annotation_type", "polygon_mask", 3, "Sec. 3", "tree crown delineation and species labels"),
    R("dataset_name", "Quebec Plantations (UAV Canadian (Quebec) Plantations dataset)", 3, "Sec. 3", "plantation sites from the UAV Canadian (Quebec) Plantations dataset [21]", item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 3", "We split the data spatially into training, validation and test sets", item="dataset1"),
    R("dataset_public", "yes", 3, "Sec. 3", "the UAV Canadian (Quebec) Plantations dataset [21]", item="dataset1", note="published dataset [21]"),
    R("n_trees", "19500", 16, "Table 5", "Total 5379 925 992 9079 763 419 23 78 1842 19500", item="dataset1", unit="trees"),
    R("dataset_name", "SBL (Quebec Trees dataset, Station de Biologie des Laurentides, 2021-09-02)", 4, "Sec. 3",
      "We consider the Quebec Trees dataset [22] which covers a temperate forest site and use the RGB imagery and corresponding DSM from date 2021-09-02", item="dataset2"),
    R("dataset_role", "train_test", 4, "Sec. 3", "We use the AOIs defined in Ramesh et al. [63] for training, validation and testing", item="dataset2"),
    R("dataset_public", "yes", 4, "Sec. 3", "the Quebec Trees dataset [22]", item="dataset2", note="published dataset [22]"),
    R("n_trees", "22933", 4, "Sec. 3", "for which 22, 933 tree crowns were manually labelled", item="dataset2", unit="trees"),
    R("dataset_name", "BCI (Barro Colorado Island crown maps, 2022)", 4, "Sec. 3", "We use the 2022 imagery of the Barro Colorado Island crown maps dataset [23]", item="dataset3"),
    R("dataset_role", "train_test", 4, "Sec. 3", "the spatial split to avoid geospatial auto-correlation", item="dataset3"),
    R("dataset_public", "yes", 4, "Sec. 3", "the Barro Colorado Island crown maps dataset [23]", item="dataset3", note="published dataset [23]"),
    R("n_trees", "2280", 4, "Sec. 3", "This version contains 112 species with 2, 280 tree crown delineations", item="dataset3", unit="trees"),
]
SAMPRE = dict(pre_page=6, pre_loc="Sec. 4.3", pre_ev="In all experiments, we use the ViT-Huge version of SAM", pre_note="SAM weights (SA-1B) used as released")
INPRE = dict(pre_page=6, pre_loc="Table 1", pre_ev="The column Pre-trained refers to ImageNet pre-training for the backbones of the Mask R-CNN, Faster R-CNN and Mask2Former models")
M2FPRE = dict(pre_page=5, pre_loc="Sec. 4.1", pre_ev="We fine-tune Mask2Former models that were pre-trained on the COCO instance segmentation dataset",
              pre_note="COCO instance segmentation weights (Swin backbones)")
M = {  # key: (item, name, role, category, backbone, pretraining, strategy, strategy note, pre kwargs)
    "SAM100": ("model1", "SAM automatic mode (100 points per side)", "transformer_baseline", "foundation_model", "SAM ViT-H", "sa1b", "zero_shot", "", SAMPRE),
    "SAM10": ("model2", "SAM automatic mode (10 points per side)", "transformer_baseline", "foundation_model", "SAM ViT-H", "sa1b", "zero_shot", "", SAMPRE),
    "SAMp": ("model3", "SAM with DSM local-maximum point prompts", "transformer_baseline", "foundation_model", "SAM ViT-H", "sa1b", "zero_shot", "", SAMPRE),
    "MR_scr": ("model4", "Mask R-CNN (from scratch)", "cnn_baseline", "cnn", "ResNet-50", "none", "from_scratch", "", INPRE),
    "MR_IN": ("model5", "Mask R-CNN (ImageNet)", "cnn_baseline", "cnn", "ResNet-50", "imagenet", "full_fine_tuning", "", INPRE),
    "MR_DSM": ("model6", "Mask R-CNN with DSM as a fourth input channel (ImageNet)", "cnn_baseline", "cnn", "ResNet-50 (RGB + DSM input)", "imagenet", "full_fine_tuning", "", INPRE),
    "FR_scr": ("model7", "Faster R-CNN box prompts to SAM (detector from scratch)", "transformer_baseline", "foundation_model", "SAM ViT-H; Faster R-CNN ResNet-50", "sa1b", "zero_shot",
               "SAM not tuned; Faster R-CNN trained from scratch", SAMPRE),
    "FR_IN": ("model8", "Faster R-CNN box prompts to SAM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Faster R-CNN ResNet-50", "sa1b", "zero_shot",
              "SAM not tuned; Faster R-CNN fine-tuned from ImageNet", SAMPRE),
    "FR_DSM": ("model9", "Faster R-CNN box prompts to SAM, with DSM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Faster R-CNN ResNet-50 (RGB + DSM)",
               "sa1b", "zero_shot", "SAM not tuned", SAMPRE),
    "MRS_IN": ("model10", "Mask R-CNN box and mask prompts to SAM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Mask R-CNN ResNet-50", "sa1b",
               "zero_shot", "SAM not tuned", SAMPRE),
    "MRS_DSM": ("model11", "Mask R-CNN box and mask prompts to SAM, with DSM (ImageNet detector)", "transformer_baseline", "foundation_model",
                "SAM ViT-H; Mask R-CNN ResNet-50 (RGB + DSM)", "sa1b", "zero_shot", "SAM not tuned", SAMPRE),
    "MR_enc": ("model12", "Mask R-CNN with DSM encoder (ImageNet)", "cnn_baseline", "cnn", "ResNet-50 with DSM encoder", "imagenet", "full_fine_tuning", "", INPRE),
    "RSP": ("model13", "RSPrompter (anchor)", "transformer_baseline", "foundation_model", "SAM ViT-H with learnable prompter", "sa1b", "peft_prompt",
            "SAM image encoder and mask decoder frozen; prompter trained", SAMPRE),
    "Bal": ("model14", "BalSAM", "proposed", "foundation_model", "SAM ViT-H with learnable prompter and DSM encoder", "sa1b", "peft_prompt",
            "builds on RSPrompter with a trainable DSM encoder", SAMPRE),
    "M2F_B": ("model15", "Mask2Former (Swin-base)", "transformer_baseline", "hierarchical_vit", "Swin-base", "other", "full_fine_tuning",
              "fine-tuned from COCO instance segmentation weights", M2FPRE),
    "M2F_B_DSM": ("model16", "Mask2Former (Swin-base) with DSM as a fourth input channel", "transformer_baseline", "hierarchical_vit", "Swin-base (RGB + DSM input)",
                  "other", "full_fine_tuning", "fine-tuned from COCO instance segmentation weights", M2FPRE),
    "M2F_L": ("model17", "Mask2Former (Swin-L)", "transformer_baseline", "hierarchical_vit", "Swin-L", "other", "full_fine_tuning",
              "fine-tuned from COCO instance segmentation weights", M2FPRE),
    "M2F_L_DSM": ("model18", "Mask2Former (Swin-L) with DSM as a fourth input channel", "transformer_baseline", "hierarchical_vit", "Swin-L (RGB + DSM input)",
                  "other", "full_fine_tuning", "fine-tuned from COCO instance segmentation weights", M2FPRE),
}
for key, (item, name, role, cat, bb, pre, strat, snote, prekw) in M.items():
    ROWS += model(item, name, role, cat, bb, pre, strat, 5, "Sec. 4.1", name, strat_note=snote, **prekw)
TABLES = [  # page, table label, dataset item, [(key, regex, occurrence)]
    (6, "Table 1", "dataset1", [("SAM100", r"SAM \(100 pps\)", 0), ("SAM10", r"SAM \(10 pps\)", 0), ("SAMp", r"SAM\s+✓\(prompts\)", 0),
                                ("MR_scr", r"^\s+✗\s+✗\s+(?=\d)", 0), ("MR_IN", r"Mask R-CNN\s+✗\s+✓", 0), ("MR_DSM", r"^\s+✓\s+✓\s+(?=\d)", 0),
                                ("FR_scr", r"^\s+✗\s+✗\s+(?=\d)", 1), ("FR_IN", r"Faster R-CNN\+SAM\s+✗\s+✓", 0), ("FR_DSM", r"^\s+✓\s+✓\s+(?=\d)", 1),
                                ("MRS_IN", r"^\s+✗\s+✓\s+(?=\d)", 0), ("MRS_DSM", r"^\s+✓\s+✓\s+(?=\d)", 2), ("M2F_B", r"^\s+✗\s+✓\s+(?=\d)", 1),
                                ("M2F_B_DSM", r"^\s+✓\s+✓\s+(?=\d)", 3), ("M2F_L", r"^\s+✗\s+✓\s+(?=\d)", 2), ("M2F_L_DSM", r"^\s+✓\s+✓\s+(?=\d)", 4),
                                ("RSP", r"RSPrompter", 0), ("Bal", r"BalSAM", 0)]),
    (7, "Table 2", "dataset2", [("SAM100", r"SAM \(100 pps\)", 0), ("SAM10", r"SAM \(10 pps\)", 0), ("SAMp", r"SAM\s+✓\(prompts\)", 0),
                                ("MR_scr", r"^\s+✗\s+✗\s+(?=\d)", 0), ("MR_IN", r"Mask R-CNN\s+✗\s+✓", 0), ("MR_DSM", r"^\s+✓\s+✓\s+(?=\d)", 0),
                                ("FR_IN", r"^\s+✗\s+✓\s+(?=\d)", 0), ("FR_DSM", r"^\s+✓\s+✓\s+(?=\d)", 1), ("MRS_IN", r"^\s+✗\s+✓\s+(?=\d)", 1),
                                ("MRS_DSM", r"^\s+✓\s+✓\s+(?=\d)", 2), ("RSP", r"RSPrompter", 0), ("Bal", r"BalSAM", 0)]),
    (7, "Table 3", "dataset3", [("SAM100", r"SAM \(100 pps\)", 1), ("SAM10", r"SAM \(10 pps\)", 1), ("SAMp", r"SAM\s+✓\(prompts\)", 1),
                                ("MR_IN", r"^\s+✗\s+✓\s+(?=\d)", 2), ("MR_DSM", r"^\s+✓\s+✓\s+(?=\d)", 3), ("MR_enc", r"Mask R-CNN \+ DSM encoder", 0),
                                ("RSP", r"RSPrompter", 1), ("Bal", r"BalSAM", 1)]),
]
DSN = {"dataset1": "Quebec Plantations", "dataset2": "SBL", "dataset3": "BCI"}
res = Counter("res")
for page, tab, ds, specs in TABLES:
    for key, rx, n in specs:
        nums, ev = row(PID, page, rx, n)
        if key == "M2F_L_DSM":  # the standard error of mIoU is printed as "0.17 3"; the stray digit is not a value
            assert nums[2] == "3", nums
            nums = [nums[0], nums[1], nums[3], nums[4]]
        item = M[key][0]
        assert len(nums) in (2, 4), (tab, key, nums)
        st = f"{DSN[ds]} test set; single class 'tree'"
        ROWS += result(res(), "map", item, ds, nums[0], page, tab, ev, setting=st, note="mean over 3 seeds; COCO-style mAP",
                       headline=(key == "Bal" and ds == "dataset1"))
        ROWS += result(res(), "miou", item, ds, nums[1], page, tab, ev, setting=st, note="best-match IoU per ground-truth tree, averaged over trees")
        if len(nums) == 4:
            ROWS += result(res(), "other", item, ds, nums[3], page, tab, ev, setting=f"{DSN[ds]} test set; multi-class",
                           note="class-weighted mAP (wmAP), weights = class proportions in the test set")
for item, v, ev in (("model13", "117", "RSPrompter 1.00 117M"), ("model14", "126", "BalSAM 1.04 126M"),
                    ("model15", "106.9", "Mask2Former Swin-base∗ 0.27 106.9M"), ("model17", "215.5", "Mask2Former Swin-large∗ 0.30 215.5M")):
    ROWS.append(R("trainable_params", v, 23, "Table 10", ev, item=item, unit="M",
                  note="trainable parameters including the 9-class final layer; Mask2Former used 384 x 384 inputs, the other models 1024 x 1024"))
ROWS += avail(note_code="Code sample in the supplementary material; full code promised 'upon paper decision' (p.6)")
ROWS += [D("notes", "Report of the study group teng_2025 (one study with two reports, counted once); this report adds the SBL and BCI datasets and calls the model BalSAM. The published (NeurIPS) version adds Mask2Former baselines (Swin-base, Swin-L; RGB and RGB + DSM) to Table 1 and trainable parameters in Table 10 (p.23). Values are x100 with standard errors over 3 seeds; multi-class unweighted mAP is also in Tables 1 to 3.", "Sec. 3 to 5 and Tables 1 to 3 (pp. 3 to 7)")]
