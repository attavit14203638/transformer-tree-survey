import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T3 = "P145", "Table 2", "Table 3"
# Declining and dead ash trees (emerald ash borer) in drone images: instance segmentation with Mask2Former (Swin-S) against RetinaNet, YOLACT and Mask R-CNN (Swin-T) (Table 2, the main comparison; box and mask
# AP50, AP75 and mAP) and Mask2Former with four backbones (Table 3, an ablation). The abstract gives the mask values of Mask2Former Swin-S "on the validation dataset" (0.789, 0.617, 0.542), the same numbers as
# Table 2, so the tables are scored on the validation set. mAP is the COCO mAP (the paper uses the MS COCO metrics), recorded as ap50_95; AP75 has no metric code and is not recorded. Box results are level object and
# mask results are level instance_mask; Table 2 is primary for both levels. Table 3 (the rows other than the repeated Swin-S one) and the in-situ field validation (recall at two confidence thresholds, mean over nine
# sites, a different data source) are recorded without primary.
TB = Tags(task="health_classification", level="object", split="validation", agg="single")
TM = TB(level="instance_mask")
TASK_EV = "Aerial surveys with drones were conducted to obtain 6174 aerial images including ash decline in the deciduous forests in West Virginia and Pennsylvania, USA"
ABS_EV = "Our comprehensive analyses with instance segmentation models showed that Mask2former was the most effective model for detecting declining and dead ash trees with 0.789, 0.617, and 0.542 for AP50, AP75, and mAP, respectively, on the validation dataset"
SITES_EV = "A total of nine sites were selected for in-situ validation study"
ROWS = [
    R("task", "health_classification", 1, "Abstract", "This study was conducted to develop and evaluate a new tool for safe, efficient, and precise detection and assessment of ash decline and death caused by emerald ash borer", note="one class: declining or dead ash trees (declining and dead are not separated)"),
    R("output_unit", "tree_instance;crown_mask", 1, "Abstract", ABS_EV, note="instance segmentation: a mask and a box for each declining or dead ash tree"),
    R("tree_setting", "natural_forest", 1, "Abstract", TASK_EV, note="deciduous forests; forest type otherwise not described"),
    R("tree_taxa", "ash (Fraxinus spp.)", 1, "Abstract", "Emerald ash borer (Agrilus planipennis) is an invasive pest that has killed millions of ash trees (Fraxinus spp.)"),
    R("study_area", "Warren County, Pennsylvania (training data); nine in-situ validation sites in Pennsylvania and northern West Virginia, USA", 11, "Sec. 4.1", "This study was conducted in Pennsylvania and Northern West Virginia, USA"),
    D("biome", "temperate", "Pennsylvania and West Virginia, USA; not stated in the paper"),
    R("platform", "uav", 12, "Sec. 4.2", "rotary-wing", note="two rotary-wing drones (DJI Mavic 2 Enterprise Advanced and Mavic 2 Pro) flown 50 to 75 m above the ground, 4K video"),
    R("sensors", "rgb", 12, "Sec. 4.2", "RGB camera", note="the Mavic 2 Enterprise Advanced also has a thermal camera, which is not used"),
    R("annotation_type", "polygon_mask", 13, "Sec. 4.3", "binary masks", note="ash trees outlined by hand in 4K frames (BeFunky Photo Editor, about 2 minutes per image) and converted into binary masks"),
    R("dataset_name", "own drone dataset of declining and dead ash trees, Warren County, Pennsylvania (640 x 720 patches of 343 annotated 4K frames)", 1, "Abstract", TASK_EV, item="dataset1"),
    R("dataset_key", "own_p145_warren_ash", 1, "Abstract", TASK_EV, item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.1", "A total of 6174 images were annotated for generating training (70% of the images), validating (20% of the images), and testing (10% of the images) image data for deep learning", item="dataset1",
      note="Table 1: 4357 training, 1200 validation and 617 testing patches; the testing patches are only shown in Fig. 3 (no score)"),
    D("dataset_public", "on_request", "Data Availability Statement (p.17): The data that support the findings of this study will be available upon request", item="dataset1"),
    R("n_images", "6174", 13, "Sec. 4.3", "6174 images in total for deep learning", item="dataset1", unit="patches", note="640 x 720 pixel patches (18 per frame) of 343 annotated 4K frames; many patches have no ash tree"),
    R("dataset_name", "own drone in-situ validation images of declining and dead ash trees, nine sites in Pennsylvania and northern West Virginia", 16, "Sec. 4.6", SITES_EV, item="dataset2"),
    R("dataset_key", "own_p145_insitu_sites", 16, "Sec. 4.6", SITES_EV, item="dataset2"),
    R("dataset_role", "transfer_target", 7, "Sec. 2.6", "nine sites", item="dataset2", note="field images dissimilar in lighting and flight altitude to the training data; compared with ground survey of the ash trees"),
    D("dataset_public", "on_request", "Data Availability Statement (p.17): The data that support the findings of this study will be available upon request", item="dataset2"),
]
PRE = dict(pre_page=5, pre_loc="Sec. 2.3", pre_ev="ImageNet-1K [33] weights for all of the backbones")
NP = dict(strat_note="ImageNet-1K weights for the backbones, AdamW with learning rate 0.0001 and weight decay 0.05, batch size 2, large-scale jittering, single RTX 3090 (Sec. 4.5); training schedule not stated")
M = [  # row text prefix on the table, name, role, category, backbone, category note
    ("RetinaNet ResNet-18", "RetinaNet", "cnn_baseline", "cnn", "ResNet-18", "convolutional one-stage detector"),
    ("Yolact ResNet-101", "YOLACT", "cnn_baseline", "cnn", "ResNet-101", "convolutional instance segmentation model"),
    ("Mask R-CNN SWIN\u2014T", "Mask R-CNN with Swin-T backbone", "transformer_baseline", "hierarchical_vit", "Swin-T", "Mask R-CNN with a Swin transformer backbone (protocol 4.5: hierarchical_vit)"),
    ("Mask2former SWIN\u2014S", "Mask2Former with Swin-S backbone", "proposed", "hierarchical_vit", "Swin-S", "Mask2Former with a Swin-S backbone; off-the-shelf architecture applied by the authors (protocol 4.5: hierarchical_vit)"),
]
res = Counter("res")
for k, (lab, name, role, cat, bb, cnote) in enumerate(M, start=1):
    item = f"model{k}"
    pat = lab.replace(" ", r"\s+")
    nums, line = row(PID, 5, rf"^\s*{pat}\s+ImageNet-1K\s+", 0)
    assert len(nums) in (3, 6), (name, nums)  # box AP50, AP75, mAP and (not RetinaNet) mask AP50, AP75, mAP
    ROWS += model(item, name, role, cat, bb, "imagenet", "full_fine_tuning", 5, T2, line, cat_note=cnote, pre_note="ImageNet-1K weights for the backbone", **PRE, **NP)
    for metric, v, tag, lvl in (("ap50", nums[0], TB, "box"), ("ap50_95", nums[2], TB, "box")) + ((("ap50", nums[3], TM, "mask"), ("ap50_95", nums[5], TM, "mask")) if len(nums) == 6 else ()):
        nm = "AP50" if metric == "ap50" else "mAP (COCO)"
        ROWS += result(res(), metric, item, "dataset1", v, 5, T2, line, unit="fraction", setting=f"Table 2: {lvl} {nm}; validation set", note=f"{lvl} {nm}; AP75 of the row is not recorded", **tag, primary=True,
                       headline=(item == "model4" and lvl == "mask" and metric == "ap50"))
# Table 3: Mask2Former with other backbones (the Swin-S row repeats Table 2), not primary
B3 = [("CNNs ResNet-50", "Mask2Former with ResNet-50 backbone", "cnn_transformer_hybrid", "ResNet-50", "Mask2Former with a CNN backbone (ResNet-50); a transformer decoder on a CNN backbone (protocol 4.5)"),
      ("CNNs ResNet-101", "Mask2Former with ResNet-101 backbone", "cnn_transformer_hybrid", "ResNet-101", "Mask2Former with a CNN backbone (ResNet-101); a transformer decoder on a CNN backbone (protocol 4.5)"),
      ("Transformers SWIN\u2014T", "Mask2Former with Swin-T backbone", "hierarchical_vit", "Swin-T", "Mask2Former with a Swin-T backbone (protocol 4.5: hierarchical_vit)")]
for j, (lab, name, cat, bb, cnote) in enumerate(B3):
    item = f"model{len(M) + 1 + j}"
    pat = lab.replace(" ", r"\s+")
    nums, line = row(PID, 5, rf"^\s*{pat}\s+ImageNet-1K\s+", 0)
    assert len(nums) == 6, (name, nums)
    ROWS += model(item, name, "other_baseline", cat, bb, "imagenet", "full_fine_tuning", 5, T3, line, cat_note=cnote, pre_note="ImageNet-1K weights for the backbone", **PRE, **NP)
    for metric, v, tag, lvl in (("ap50", nums[0], TB, "box"), ("ap50_95", nums[2], TB, "box"), ("ap50", nums[3], TM, "mask"), ("ap50_95", nums[5], TM, "mask")):
        nm = "AP50" if metric == "ap50" else "mAP (COCO)"
        ROWS += result(res(), metric, item, "dataset1", v, 5, T3, line, unit="fraction", setting=f"Table 3: backbone ablation of Mask2Former, {lvl} {nm}; validation set", note=f"{lvl} {nm}; AP75 of the row is not recorded", **tag)
# in-situ field validation: Mask2Former (Swin-S) at two confidence thresholds, mean over the nine sites
TF = Tags(task="health_classification", level="object", split="test", agg="mean")
ROWS += result(res(), "recall", "model4", "dataset2", "81", 7, "Sec. 2.6", "81% ± 10.7", unit="%", setting="in-situ field validation, nine sites; confidence threshold 0.1",
               note="share of the declining or dead ash trees of the ground survey that were detected (81 percent, SD 10.7, per-site range 59 to 100); the type II error is 19 percent", **TF)
ROWS += result(res(), "recall", "model4", "dataset2", "27", 7, "Sec. 2.6", "27% ± 13.6", unit="%", setting="in-situ field validation, nine sites; confidence threshold 0.5",
               note="share of the declining or dead ash trees of the ground survey that were detected (27 percent, SD 13.6); the type II error is 71 percent (p.10)", **TF)
ROWS += [R("limitations", "Many declining or dead ash trees are missed in the field validation (type II error 71 percent at confidence 0.5, 19 percent at 0.1), and the model failed on autumn images", 10, "Sec. 3",
           "At the confidence level of 0.5, the type II error was higher (71%)")]
ROWS += avail(code="no", weights="not_stated", data="on_request", data_page=17, data_loc="Data Availability Statement", data_ev="The data that support the findings of this study will be available upon request")
ROWS += [D("notes", "Table 2 compares RetinaNet, YOLACT, Mask R-CNN (Swin-T) and Mask2Former (Swin-S) on the validation patches (box and mask AP50 and COCO mAP; AP75 not recorded); Mask2Former has mask AP50 0.789 and mAP 0.542, the abstract's headline. Table 3 gives Mask2Former with ResNet-50, ResNet-101, Swin-T and Swin-S (not primary; the Swin-S row repeats Table 2): the ResNet-101 backbone has the best box mAP (0.579) and Swin-S the best mask mAP (0.542). The testing patches are only shown in Fig. 3. The in-situ validation at nine sites (different lighting and flight altitude) gives the share of ground-surveyed declining or dead ash trees that Mask2Former detected: 81 percent at confidence 0.1 and 27 percent at 0.5 (type II errors 19 and 71 percent) and false detections of 30 and 6 percent; only the detection shares are recorded (as recall, mean over the nine sites), the false detection shares have no stated denominator. The class is declining or dead ash trees (not separated), so the task is health_classification. The abstract and Table 2 give the same mask values (0.789, 0.617, 0.542).",
           "Sec. 2 and 4 and Tables 1 to 3 (pp. 3 to 16)")]
