import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T4, T5 = "P118", "Table 4", "Table 5"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "we focus on tree species classification by using five state-of-the-art deep learning models on UAV-based RGB images"),
    R("output_unit", "tree_label", 1, "Abstract", "we captured 8799 crown images of eight species"),
    R("tree_setting", "plantation", 4, "Sec. 2.1", "We focused on eight ecologically and economically important tree species from plantations (Table 3)"),
    R("tree_taxa", "Prunus serotina; Juglans cinerea; Castanea dentata; Quercus rubra; Pinus resinosa; Juglans nigra; Quercus alba; Pinus strobus", 4, "Sec. 2.1",
      "Black cherry (Prunus serotina), northern red oak (Quercus rubra), red pine (Pinus resinosa), black walnut (Juglans nigra), and white oak (Q. alba) ... butternut (J. cinerea), American chestnut (Castanea dentata), and white pine (P. strobus)"),
    R("study_area", "Martell Forest, Tippecanoe County, Indiana, USA", 4, "Sec. 2.1", "The study area, Martell Forest, located in Tippecanoe County, Indiana, USA"),
    R("biome", "temperate", 1, "Abstract", "which is the optimal season for tree species classification in a temperate forest"),
    R("platform", "uav", 4, "Sec. 2.1", "We collected UAV images using the DJI ZENMUSE P1 (35 mm) camera"),
    R("sensors", "rgb", 1, "Abstract", "five state-of-the-art deep learning models on UAV-based RGB images"),
    R("gsd", "1.5", 4, "Sec. 2.1", "resulting in a GSD of approximately 1.5 cm to 2.01 cm", unit="cm", note="1.5 to 2.01 cm"),
    R("annotation_type", "image_label", 6, "Sec. 2.2", "We generated 8799 labeled crown images for these eight tree species from the two seasons", note="cropped crown images with species labels"),
    R("dataset_name", "own UAV crown images, Martell Forest (summer and fall 2021)", 6, "Sec. 2.2", "We generated 8799 labeled crown images for these eight tree species from the two seasons", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 2.4", "the tree images dataset was divided into a training set comprising 80% of the data and a testing set comprising the remaining 20%", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    R("n_images", "8799", 6, "Sec. 2.2", "We generated 8799 labeled crown images for these eight tree species from the two seasons", item="dataset1", unit="images"),
]
FT = dict(pre_page=6, pre_loc="Sec. 2.4", pre_ev="ResNet18, DenseNet, EfficientNet, and ViT were pre-trained with ImageNet [47] dataset",
          strat_page=6, strat_loc="Sec. 2.4", strat_ev="we chose to fine-tune the pre-trained open-sourced models [46]")
avg = {}
for n, season in ((0, "summer"), (1, "fall")):
    nums, ev = row(PID, 8, r"^\s*Average\s+(?=\d)", n)
    assert len(nums) == 16, nums
    avg[season] = (nums, ev)
M = [  # item, name, role, category, backbone, column index of F1 in the Average row, pretraining
    ("model1", "Vision Transformer (ViT)", "transformer_baseline", "pure_vit", "ViT (size not stated)", 11, "imagenet"),
    ("model2", "EfficientNet-B0", "cnn_baseline", "cnn", "EfficientNet-B0", 8, "imagenet"),
    ("model3", "YOLOv5 (classification)", "cnn_baseline", "cnn", "YOLOv5", 14, "other"),
    ("model4", "DenseNet", "cnn_baseline", "cnn", "DenseNet (depth not stated)", 5, "imagenet"),
    ("model5", "ResNet18", "cnn_baseline", "cnn", "ResNet-18", 2, "imagenet"),
]
res = Counter("res")
for item, name, role, cat, bb, k, pre in M:
    if pre == "other":
        ROWS += model(item, name, role, cat, bb, pre, "full_fine_tuning", 8, T4, avg["summer"][1], pre_page=6, pre_loc="Sec. 2.4",
                      pre_ev="YOLOv5 was pre-trained by another benchmark dataset(COCO)", pre_note="COCO", strat_page=6, strat_loc="Sec. 2.4",
                      strat_ev="we chose to fine-tune the pre-trained open-sourced models [46]")
    else:
        ROWS += model(item, name, role, cat, bb, pre, "full_fine_tuning", 8, T4, avg["summer"][1], cat_note=("ResNet18 is the paper's baseline model" if item == "model5" else ""), **FT)
    for season in ("summer", "fall"):
        nums, ev = avg[season]
        ROWS += result(res(), "f1", item, "dataset1", nums[k], 8, T4, f"{season.capitalize()} ... {ev}", unit="fraction",
                       setting=f"{season} test images (20%); average over eight species",
                       note="columns: Prec, Rec, F1 for ResNet18, DenseNet, EfficientNet-B0, ViT, YOLOv5, then average F1")
ROWS += result(res(), "f1", "model1", "dataset1", "0.986", 7, "Sec. 3.1", "ResNet18, as our baseline model, achieved the lowest average F1-scores of 0.87, and, in contrast, ViT model achieved the highest average F1-score of 0.986 (Table 4)",
               unit="fraction", setting="average over both seasons and eight species", headline=True)
ROWS += result(res(), "f1", "model5", "dataset1", "0.87", 7, "Sec. 3.1", "ResNet18, as our baseline model, achieved the lowest average F1-scores of 0.87",
               unit="fraction", setting="average over both seasons and eight species")
tr = Counter("tr")
T5ROWS = {("model5", "summer"): (r"^\s*ResNet18\s+Summer\s+(?=[-\d])", 0), ("model5", "fall"): (r"^\s*Fall\s+(?=[-\d])", 0),
          ("model1", "summer"): (r"^\s*ViT\s+Summer\s+(?=[-\d])", 0), ("model1", "fall"): (r"^\s*Fall\s+(?=[-\d])", 1)}
for item, lab in (("model5", "ResNet18"), ("model1", "ViT")):
    for src, tgt in (("summer", "fall"), ("fall", "summer")):
        rx, occ = T5ROWS[(item, src)]
        nums, ev = row(PID, 9, rx, occ)
        assert len(nums) == 1, (item, src, nums, ev)
        val = nums[0]
        t = tr()
        ROWS += [R("transfer_source", f"{src} images (training season)", 7, "Sec. 2.5", "Our key approach was to train a model using one seasonal", item=t),
                 R("transfer_target", f"{tgt} images", 9, "Sec. 3.2", "models trained on one season can hardly be transferred to another season", item=t),
                 R("shift_type", "season", 9, "Sec. 3.2", "models trained on one season can hardly be transferred to another season", item=t),
                 R("transfer_model", item, 9, T5, ev, item=t),
                 R("transfer_metric", "oa", 9, T5, "Overall accuracy on transferability of models across seasons on ResNet18 and ViT", item=t),
                 NR("value_in_domain", item=t, note="same-season results are reported only as F1 (Table 4), not as overall accuracy"),
                 R("value_out_of_domain", val, 9, T5, ev, item=t, unit="fraction")]
ROWS += avail()
ROWS += [D("notes", "ViT has the highest average F1 in both seasons (1.00 summer, 0.97 fall). Cross-season overall accuracy is below 0.5 for both ResNet18 and ViT (Table 5); per-species transfer for ResNet18 is in Table 6.", "Sec. 3 and Tables 4 to 6 (pp. 7 to 9)")]
