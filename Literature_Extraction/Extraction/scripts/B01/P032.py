import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1 = "Table 1"
ROWS = [
    R("task", "crown_segmentation;species_classification", 1, "Abstract", "we compare SAM methods for the task of automatic tree crown instance segmentation in high resolution drone imagery",
      note="multi-class instance segmentation over 9 species classes"),
    R("output_unit", "crown_mask;tree_label", 3, "Sec. 3.1", "We evaluate our instance segmentation models with mean Average Precision (mAP)"),
    R("tree_setting", "plantation", 1, "Abstract", "high resolution drone imagery of young tree plantations"),
    R("tree_taxa", "Pinus banksiana; Picea mariana; Pinus strobus; Picea glauca; Thuja occidentalis; Ulmus americana; Betula alleghaniensis; Acer saccharum; other", 8, "Table 2",
      "piba Pinus banksiana / pima Picea mariana / pist Pinus strobus / pigl Picea glauca / thoc Thuya occidentalis / ulam Ulmus americana / beal Betula allegnaniensis / acsa Acer Saccharum",
      note="'Other' groups species with 20 trees or fewer"),
    R("study_area", "15 plantation sites in Quebec, Canada", 2, "Sec. 2", "tree crown delineation and species labels in plantation sites from the UAV Canadian (Quebec) Plantations dataset ... kept 15 sites of interest"),
    NR("biome", note="not stated; the 15 plantation sites in Quebec may lie in the temperate or the boreal zone"),
    R("platform", "uav", 2, "Sec. 2", "RGB orthomosaics, photogrammetry digital surface models (DSMs) ... from the UAV Canadian (Quebec) Plantations dataset"),
    R("sensors", "rgb;dsm", 2, "Sec. 2", "We use RGB orthomosaics, photogrammetry digital surface models (DSMs)"),
    R("gsd", "5", 2, "Sec. 2", "The imagery has a resolution of 5mm/pixel", unit="mm"),
    R("annotation_type", "polygon_mask", 2, "Sec. 2", "tree crown delineation and species labels"),
    R("dataset_name", "UAV Canadian (Quebec) Plantations dataset (Lefebvre and Laliberté 2024)", 2, "Sec. 2",
      "the UAV Canadian (Quebec) Plantations dataset (Lefebvre & Laliberté, 2024)", item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 2", "the training, validation and test sets contain respectively 15,742, 6,691 and 3,995 tiles", item="dataset1",
      note="split by geographical blocks"),
    R("dataset_public", "yes", 5, "References", "I. Lefebvre and E. Laliberté. UAV LiDAR, UAV Imagery, tree segmentations and ground measurements for estimating tree biomass in Canadian (Quebec) plantations, 2024. URL",
      item="dataset1", note="published dataset with a URL in the references"),
    R("n_images", "26428", 2, "Sec. 2", "is composed 26,428 tiles", item="dataset1", unit="images", note="1024 x 1024 tiles with 50% overlap"),
]
SAM_PRE = dict(pre_page=8, pre_loc="Appendix C", pre_ev="In all experiments, we use the ViT-Huge version of SAM", pre_note="SAM weights (SA-1B) used as released")
M = [  # item, name, role, category, backbone, pretraining, strategy, note, label, single (mAP, mIoU), wmAP
    ("model1", "SAM automatic mode (100 points per side)", "transformer_baseline", "foundation_model", "SAM ViT-H", "sa1b", "zero_shot", "", "SAM (100 pps)", ("8.05", "35.06"), None),
    ("model2", "SAM automatic mode (10 points per side)", "transformer_baseline", "foundation_model", "SAM ViT-H", "sa1b", "zero_shot", "", "SAM (10 pps)", ("10.11", "34.01"), None),
    ("model3", "SAM with DSM local-maximum point prompts", "transformer_baseline", "foundation_model", "SAM ViT-H", "sa1b", "zero_shot", "", "SAM ✓(prompts)", ("9.28", "46.15"), None),
    ("model4", "Mask R-CNN (from scratch)", "cnn_baseline", "cnn", "ResNet-50", "none", "from_scratch", "", "Mask R-CNN ✗ ✗", ("59.36", "79.63"), "55.75"),
    ("model5", "Mask R-CNN (ImageNet)", "cnn_baseline", "cnn", "ResNet-50", "imagenet", "full_fine_tuning", "", "Mask R-CNN ✗ ✓", ("63.65", "81.82"), "58.30"),
    ("model6", "Mask R-CNN with DSM (ImageNet)", "cnn_baseline", "cnn", "ResNet-50 (RGB + DSM input)", "imagenet", "full_fine_tuning", "", "Mask R-CNN ✓ ✓", ("64.64", "81.89"), "60.32"),
    ("model7", "Faster R-CNN box prompts to SAM (detector from scratch)", "transformer_baseline", "foundation_model", "SAM ViT-H; Faster R-CNN ResNet-50", "sa1b", "zero_shot",
     "SAM not tuned; Faster R-CNN trained from scratch", "Faster R-CNN+SAM ✗ ✗", ("53.56", "76.22"), "45.79"),
    ("model8", "Faster R-CNN box prompts to SAM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Faster R-CNN ResNet-50", "sa1b", "zero_shot",
     "SAM not tuned; Faster R-CNN fine-tuned from ImageNet", "Faster R-CNN+SAM ✗ ✓", ("57.85", "78.0"), "50.30"),
    ("model9", "Faster R-CNN box prompts to SAM, with DSM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Faster R-CNN ResNet-50 (RGB + DSM)", "sa1b",
     "zero_shot", "SAM not tuned", "Faster R-CNN+SAM ✓ ✓", ("58.0", "78.27"), "52.08"),
    ("model10", "Mask R-CNN box and mask prompts to SAM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Mask R-CNN ResNet-50", "sa1b", "zero_shot",
     "SAM not tuned", "Mask R-CNN+SAM ✗ ✓", ("57.6", "78.18"), "50.46"),
    ("model11", "Mask R-CNN box and mask prompts to SAM, with DSM (ImageNet detector)", "transformer_baseline", "foundation_model", "SAM ViT-H; Mask R-CNN ResNet-50 (RGB + DSM)", "sa1b",
     "zero_shot", "SAM not tuned", "Mask R-CNN+SAM ✓ ✓", ("57.83", "77.65"), "51.33"),
    ("model12", "RSPrompter (anchor)", "transformer_baseline", "foundation_model", "SAM ViT-H with learnable prompter", "sa1b", "peft_prompt",
     "image encoder and mask decoder frozen; prompter trained", "RSPrompter ✗", ("66.37", "82.58"), "62.37"),
    ("model13", "DSMPrompter", "proposed", "foundation_model", "SAM ViT-H with learnable prompter and DSM encoder", "sa1b", "peft_prompt",
     "SAM frozen; prompter and DSM encoder trained", "DSMPrompter ✓", ("65.03", "83.24"), "64.84"),
]
res = Counter("res")
SD = {"model4": ("0.21", "0.53", "1.50"), "model5": ("0.43", "0.36", "1.23"), "model6": ("0.69", "0.60", "0.73"), "model7": ("0.20", "0.20", "0.67"),
      "model8": ("0.66", "0.56", "1.51"), "model9": ("0.25", "0.74", "1.73"), "model10": ("0.19", "0.31", "0.52"), "model11": ("0.11", "0.5", "0.85"),
      "model12": ("0.91", "1.63", "2.45"), "model13": ("1.76", "0.41", "1.49")}
for item, name, role, cat, bb, pre, strat, note, lab, (mAP, miou), wmap in M:
    if cat == "cnn":
        ROWS += model(item, name, role, cat, bb, pre, strat, 4, T1, lab, pre_page=4, pre_loc=T1, pre_ev="The column Pre-trained refers to ImageNet pretraining for the backbones",
                      strat_note=note)
    else:
        ROWS += model(item, name, role, cat, bb, pre, strat, 4, T1, lab, strat_note=note, **SAM_PRE)
    sd = SD.get(item)
    n1 = "mean over 3 seeds" + (f" (± {sd[0]})" if sd else "")
    ROWS += result(res(), "map", item, "dataset1", mAP, 4, T1, f"{lab} single-class mAP {mAP}", setting="test set; single class 'tree'", headline=(item == "model13"),
                   note=n1 + "; threshold convention not stated")
    ROWS += result(res(), "miou", item, "dataset1", miou, 4, T1, f"{lab} single-class mIoU {miou}", setting="test set; single class 'tree'",
                   note="best-match IoU per ground-truth tree, averaged; ignores false positives" + (f"; ± {sd[1]}" if sd else ""))
    if wmap:
        ROWS += result(res(), "other", item, "dataset1", wmap, 4, T1, f"{lab} multi-class wmAP {wmap}", setting="test set; 9 species classes",
                       note=f"class-weighted mAP (wmAP) over species; ± {sd[2]}")
ROWS += avail()
ROWS += [D("notes", "Part of the same study as the other Teng 2025 report (study group teng_2025). Multi-class (unweighted) mAP is also in Table 1; per-class mAP in Table 5. Out-of-the-box SAM performed poorly (single-class mAP 8 to 10); the tuned SAM methods (RSPrompter, DSMPrompter) outperformed Mask R-CNN on wmAP.", "Sec. 4 and Table 1 (pp. 3 to 4)")]
