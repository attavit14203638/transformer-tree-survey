import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2 = "P051", "Table 2"
ROWS = [
    R("task", "crown_segmentation", 1, "Title", "Large-Scale Date Palm Tree Segmentation from Multiscale UAV-Based and Aerial Images Using Deep Vision Transformers"),
    R("output_unit", "crown_mask", 6, "Sec. 3.2", "The revised multiscale vector layers of the date palm trees were converted into raster layers (binary masks)"),
    R("tree_setting", "mixed", 3, "Sec. 2.1", "The date palm trees in the study area are planted in agricultural and urban landscapes"),
    R("tree_taxa", "date palm (Phoenix dactylifera)", 3, "Sec. 2.1", "The date palm trees in the study area are planted in agricultural and urban landscapes"),
    R("study_area", "eastern Ajman Emirate, United Arab Emirates; 30 km2", 3, "Sec. 2.1", "positioned in the eastern region of Ajman Emirate, United Arab Emirates (UAE). The study site covers 30 km2"),
    R("biome", "arid", 3, "Sec. 2.1", "The climatic conditions in the area range from arid to hyper-arid"),
    R("platform", "multiple", 4, "Sec. 2.2", "The datasets consist of large-scale UAV orthomosaic images and two very-high-resolution aerial photographs"),
    R("sensors", "rgb", 4, "Sec. 2.2", "This study utilized multisource and multiscale RGB images"),
    R("gsd", "5", 6, "Table 1 (image)", "Spatial resolution 5 cm 20 cm 15 cm", unit="cm", note="UAV 5 cm; orthophotos 20 cm (2017) and 15 cm (2015)"),
    R("annotation_type", "polygon_mask", 6, "Fig. 3 (image)", "Illustrations of image–mask pairs", note="binary crown masks"),
    R("dataset_name", "own UAV date palm dataset (Ajman, 2019, 5 cm)", 6, "Table 1 (image)", "UAV Data 2019 5 cm 21,045 3784 3491", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 3.2", "three distinct regions ... were dedicated to training, validating, and testing deep vision transformer models", item="dataset1"),
    D("dataset_public", "no", "No dataset link; the Data Availability Statement reads 'Not applicable' (p.21)", item="dataset1"),
    D("n_images", "28320", "21,045 training + 3,784 validation + 3,491 test tiles of 512 x 512 (Table 1, p.6)", item="dataset1", unit="images"),
    R("dataset_name", "multisource dataset (UAV 2019 at 5 cm with orthophotos of 2017 at 20 cm and 2015 at 15 cm)", 6, "Table 1 (image)",
      "Training samples 21,045 5687 16,531 43,263", item="dataset2"),
    R("dataset_role", "train_test", 12, "Sec. 4.2", "The deep vision transformers were trained and evaluated on a comprehensive multidate and multisource remotely sensed data", item="dataset2"),
    D("dataset_public", "no", "No dataset link; the Data Availability Statement reads 'Not applicable' (p.21)", item="dataset2"),
    D("n_images", "52316", "43,263 training + 4,687 validation + 4,366 test tiles (Table 1 totals, p.6)", item="dataset2", unit="images"),
    R("dataset_name", "Al Ain public UAV date palm dataset [81]", 11, "Sec. 3.7", "The independent UAV dataset was acquired in Al Ain City with a spatial resolution of 10 cm", item="dataset3"),
    R("dataset_role", "transfer_target", 11, "Sec. 3.7", "the performance of the models was scrutinized on an independent public dataset [81]", item="dataset3"),
    R("dataset_public", "yes", 11, "Sec. 3.7", "an independent public dataset [81]", item="dataset3"),
    R("n_images", "1078", 11, "Sec. 3.7", "A total of 1078 image tiles (containing 9883 date palm trees)", item="dataset3", unit="images"),
    R("n_trees", "9883", 11, "Sec. 3.7", "A total of 1078 image tiles (containing 9883 date palm trees)", item="dataset3", unit="trees"),
]
ARCH = [  # Table 2 label, Figure 7 label, name, category, backbone, approximate in-domain mIoU (Figure 7), bar end measured against the axis labels
    ("Segformer_MiT-b2", "Segformer_MiT-b2", "SegFormer", "hierarchical_vit", "MiT-B2", "86.3", "86.41"),
    ("Segformer_MiT-b3", "Segformer_MiT-b3", "SegFormer", "hierarchical_vit", "MiT-B3", "86.3", "86.45"),
    ("Segformer_MiT-b4", "Segformer_MiT-b4", "SegFormer", "hierarchical_vit", "MiT-B4", "86.3", "86.39"),
    ("Segmenter-ViT-tiny", "Segmentor-ViT-tiny", "Segmenter", "pure_vit", "ViT-T", "85.0", "85.11"),
    ("Segmenter-ViT-small", "Segmentor-ViT-small", "Segmenter", "pure_vit", "ViT-S", "85.6", "85.66"),
    ("Segmenter-ViT-base", "Segmentor-ViT-base", "Segmenter", "pure_vit", "ViT-B", "85.2", "85.33"),
    ("UperNet-Swin-tiny", "UperNet-Swin-tiny", "UperNet", "hierarchical_vit", "Swin-T", "86.2", "86.32"),
    ("UperNet-Swin-small", "UperNet-Swin-small", "UperNet", "hierarchical_vit", "Swin-S", "85.5", "85.62"),
    ("UperNet-Swin-base", "UperNet-Swin-base", "UperNet", "hierarchical_vit", "Swin-B", "86.1", "86.20"),
    ("DPT-base", "DPT-base", "DPT", "pure_vit", "ViT-B", "85.7", "85.81"),
]
PRE = dict(pre_page=10, pre_loc="Sec. 3.5", pre_ev="The encoder of all of the models was initialized using ImageNet pre-trained weights",
           strat_page=11, strat_loc="Sec. 3.5", strat_ev="The models were trained for 50,000 iterations")
res, tr = Counter("res"), Counter("tr")
APPROX = ("read from the bar lengths in Figure 7 (no data labels): measured {} against the axis labels, then shifted by -0.11 so that the lowest and highest bars "
          "match the range printed in the text (85 to 86.3, p.11); about ±0.2")
for i, (lab, flab, name, cat, bb, ind, meas) in enumerate(ARCH, start=1):
    nums, ev = row(PID, 16, rf"^\s*{lab}\b")
    assert len(nums) == 6, (lab, nums)
    mu, mg = f"model{i}", f"model{i + 10}"
    ROWS += model(mu, f"{name} ({bb}) trained on UAV data", "transformer_baseline", cat, bb, "imagenet", "full_fine_tuning", 16, T2, ev,
                  cat_note="comparison of transformer models only; no CNN baseline", **PRE)
    ROWS += model(mg, f"{name} ({bb}) trained on multisource data (generic model)", "transformer_baseline", cat, bb, "imagenet", "full_fine_tuning", 16, T2, ev, **PRE)
    ROWS += result(res(), "miou", mu, "dataset1", ind, 12, "Fig. 7 (image)", f"{flab}: mIoU bar (UAV testing data)", setting="UAV test set (5 cm)",
                   note=APPROX.format(meas), status="approximate")
    t = tr()
    ROWS += [R("transfer_source", "Ajman UAV dataset (5 cm), training region; in-domain value on its test region", 6, "Sec. 3.2", "three distinct regions ... were dedicated to training, validating, and testing", item=t),
             R("transfer_target", "Al Ain public UAV dataset (10 cm), different city and UAV system", 11, "Sec. 3.7", "The independent UAV dataset was acquired in Al Ain City with a spatial resolution of 10 cm", item=t),
             R("shift_type", "site;sensor;resolution", 11, "Sec. 3.7", "an independent public dataset ... acquired in Al Ain City with a spatial resolution of 10 cm", item=t),
             R("transfer_model", mu, 16, T2, ev, item=t),
             R("transfer_metric", "miou", 16, T2, "UAV-Based Models mIoU", item=t),
             R("value_in_domain", ind, 12, "Fig. 7 (image)", f"{flab}: mIoU bar (UAV testing data)", item=t, unit="%", status="approximate", note=APPROX.format(meas)),
             R("value_out_of_domain", nums[0], 16, T2, ev, item=t, unit="%")]
    ROWS += result(res(), "miou", mg, "dataset3", nums[3], 16, T2, ev, setting="unseen Al Ain dataset; model trained on multisource data")
    ROWS += result(res(), "f1", mg, "dataset3", nums[4], 16, T2, ev, setting="unseen Al Ain dataset; model trained on multisource data", note="mean F-score")
for mu, ind, meas, v, ev in (("model3", "86.3", "86.39", "80.74", "the Segformer-MiT-b4 and the UperNet-Swin-small outperformed the rest of the models and scored an mIoU of 80.74% and 80.09%, respectively"),
                       ("model8", "85.5", "85.62", "80.09", "the Segformer-MiT-b4 and the UperNet-Swin-small outperformed the rest of the models and scored an mIoU of 80.74% and 80.09%, respectively")):
    t = tr()
    ROWS += [R("transfer_source", "Ajman UAV test set at the training resolution (5 cm)", 11, "Sec. 3.6", "the evaluated deep vision transformers were first trained on UAVs with a ground space distance of 5 cm", item=t),
             R("transfer_target", "a testing UAV dataset with a GSD of 15 cm", 12, "Sec. 4.1.2", "When the models were applied to a testing dataset with a GSD of 15 cm", item=t, note="the paper does not say how this dataset was made or whether it covers the same site"),
             R("shift_type", "resolution", 11, "Sec. 3.6", "assessed on different testing UAV datasets with various s (e.g., 10, 15, 20, 25, and 30 cm)", item=t),
             R("transfer_model", mu, 12, "Sec. 4.1.2", ev, item=t),
             R("transfer_metric", "miou", 12, "Sec. 4.1.2", ev, item=t),
             R("value_in_domain", ind, 12, "Fig. 7 (image)", "mIoU bar (UAV testing data)", item=t, unit="%", status="approximate", note=APPROX.format(meas)),
             R("value_out_of_domain", v, 12, "Sec. 4.1.2", ev, item=t, unit="%")]
ROWS += avail()
ROWS += [D("notes", "Comparison of ten transformer segmentation models, each trained on UAV data and on multisource data; no CNN baseline. In-domain test results are given only as bar charts (Figures 7 and 10); the text gives their ranges (UAV: mIoU 85 to 86.3, mF-score 91.6 to 92.4). Exact values are in Table 2 (unseen Al Ain dataset).", "Sec. 4 and Table 2 (pp. 11 to 16)")]
