import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T3 = "P058", "Table III"
ROWS = [
    R("task", "crown_segmentation;species_classification", 3, "Sec. IV-A", "The Tree Species Instance Segmentation Dataset comprises 4060 RGB images of various tree species",
      note="instance segmentation with 37 species classes"),
    R("output_unit", "crown_mask;tree_label", 3, "Sec. IV-A", "A total of 37 classes of instances are annotated in the dataset"),
    NR("tree_setting", note="setting of the images not described"),
    R("tree_taxa", "37 classes including Masson pine, lemon-scented gum and bracken fern, plus a withered-tree class", 3, "Sec. IV-A",
      "A total of 37 classes of instances are annotated in the dataset, including Masson pine, lemon-scented gum, bracken fern, etc."),
    NR("study_area", note="not stated; the dataset is built on [22] (Lin et al. 2021)"),
    NR("biome", note="not stated"),
    D("platform", "multiple", "not stated; the dataset is created based on [22], 'Aerial and Optical Images-Based Plant Species Segmentation' (p.3)"),
    R("sensors", "rgb", 3, "Sec. IV-A", "The Tree Species Instance Segmentation Dataset comprises 4060 RGB images"),
    R("annotation_type", "polygon_mask", 3, "Sec. IV-A", "A total of 37 classes of instances are annotated in the dataset"),
    R("dataset_name", "Tree Species Instance Segmentation Dataset (based on [22])", 3, "Sec. IV-A", "The Tree Species Instance Segmentation Dataset comprises 4060 RGB images of various tree species, created based on [22]", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. IV-A", "The dataset is divided into 80% for training and 20% for validation and testing", item="dataset1"),
    NR("dataset_public", item="dataset1", note="availability not stated"),
    R("n_images", "4060", 3, "Sec. IV-A", "comprises 4060 RGB images", item="dataset1", unit="images"),
]
M = [  # item, Table III label regex, name, role, category, params as printed
    ("model1", r"Spec-Mask2former \(our\)", "Spec-Mask2former", "proposed", "cnn_transformer_hybrid"),
    ("model2", r"Mask2Former \[3\]", "Mask2Former", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model3", r"Mask2Former-DQ \[6\]", "Mask2Former-DQ", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model4", r"Detr \[1\]", "DETR", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model5", r"deformable detr \[21\]", "Deformable DETR", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model6", r"conditional-de tr\[15\]", "Conditional DETR", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model7", r"dab-detr \[12\]", "DAB-DETR", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model8", r"Mask Transfiner \[10\]", "Mask Transfiner", "transformer_baseline", "cnn_transformer_hybrid"),
    ("model9", r"mask rcnn \[9\]", "Mask R-CNN", "cnn_baseline", "cnn"),
    ("model10", r"Condinst \[16\]", "CondInst", "cnn_baseline", "cnn"),
    ("model11", r"BoxSnack \[19\]", "BoxSnake", "cnn_baseline", "cnn"),
]
NOTE = {"model1": "Mask2Former with a wavelet-based query generator and position enhance block", "model9": "same-backbone convolutional counterpart",
        "model10": "best CNN on AP in Table III", "model11": "best CNN on mAP50 in Table III (printed 'BoxSnack')"}
res = Counter("res")
for item, rx, name, role, cat in M:
    nums, ev = row(PID, 4, rx)
    assert len(nums) == 7, (item, nums)
    par = re.search(r"([\d.]+)M\b", ev).group(1)
    if item == "model1":
        ROWS += model(item, name, role, cat, "ResNet-50", "imagenet", "full_fine_tuning", 4, T3, ev, pre_page=3, pre_loc="Sec. IV-A",
                      pre_ev="We chose ResNet-50 as the backbone network and utilized weights pre-trained on ImageNet", cat_note=NOTE[item])
    else:
        ROWS += model(item, name, role, cat, "ResNet-50", None, None, 4, T3, ev, pre_status="not_reported", pre_note="not stated", strat_note="not stated",
                      cat_note=NOTE.get(item, "") or "ResNet-50 for all compared models (p.3)")
    ROWS += [R("total_params", par, 4, T3, ev, item=item, unit="M"), R("flops", nums[0], 4, T3, ev, item=item, unit="GFLOPs")]
    ROWS += result(res(), "ap50_95", item, "dataset1", nums[1], 4, T3, ev, setting="validation/test split; masks", note="COCO-style AP")
    ROWS += result(res(), "ap50", item, "dataset1", nums[2], 4, T3, ev, setting="validation/test split; masks", headline=(item == "model1"))
ROWS += avail()
ROWS += [D("notes", "Table III lists 14 methods; following protocol 4.1, recorded: every transformer-based method, Mask R-CNN (same backbone), CondInst (best CNN on AP) and BoxSnake (best CNN on mAP50). SOLOv2, SparseInst and DiffusionInst are in Table III (p.4).", "Sec. IV-B and Table III (pp. 3 to 4)")]
