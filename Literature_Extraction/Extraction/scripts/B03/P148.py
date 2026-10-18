import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2, T3, T4 = "P148", "Table 2", "Table 3", "Table 4"
S = "Sec. 2.1.1 (image)"
ROWS = [
    R("task", "detection;health_classification", 3, S, "The task of the orchard patrol is to detect the anomaly in orchard, including two types of abnormal targets, one is yellowing plants suspected of disease in the citrus canopy, and the other is uneven planting or felled areas in the orchard",
      note="one class is individual trees (yellowing plants), the other planting gaps; mAP covers both"),
    R("output_unit", "tree_instance", 3, S, "one is yellowing plants suspected of disease in the citrus canopy"),
    R("tree_setting", "orchard", 3, S, "The test of UAV patrol orchard was conducted in a citrus orchard in Boluo County"),
    R("tree_taxa", "citrus", 3, S, "There were 324 citrus trees in the orchard"),
    R("study_area", "Boluo County, Huizhou City, Guangdong Province, China", 3, S, "a citrus orchard in Boluo County (23°29′56.74″N–114°28′4.11″E), Huizhou City, Guangdong Province, China"),
    D("biome", "subtropical", "Huizhou, Guangdong; not stated in the paper"),
    R("platform", "uav", 3, S, "The DJI PHANTOM 4 UAV, as shown in Figure 2"),
    R("sensors", "rgb", 3, S, "was used to collect RGB images of the above orchard"),
    R("annotation_type", "bounding_box", 3, S, "The open source software LabelImg was used to label images, which generates an XML file containing labels and location information"),
    R("dataset_name", "own UAV citrus orchard patrol dataset, Boluo County (640 x 640 crops)", 3, S, "The original dataset consists of 1007 cropped and filtered images", item="dataset1"),
    R("dataset_role", "train_test", 3, S, "The labeled dataset was randomly allocated to train set, valid set and test set according to 8:1:1, corresponding to 806, 101 and 100 images", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
    R("n_images", "1007", 3, S, "The original dataset consists of 1007 cropped and filtered images", item="dataset1", unit="images", note="3725 after augmentation"),
    R("n_trees", "324", 3, S, "There were 324 citrus trees in the orchard", item="dataset1", unit="trees", note="trees in the orchard, not annotations"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_page=10, strat_loc="Sec. 3.2", strat_ev="Training epochs was set to 300 and batch size was set to 16",
          strat_note="initialisation not stated")
res = Counter("res")
for item, lab, name, role, cat, bb in (("model1", r"Swin-T YOLOX\(proposed\)", "Swin-T YOLOX", "proposed", "hierarchical_vit", "Swin-T backbone with YOLOX head"),
                                        ("model2", r"DarkNet53 \+ YOLOX-L", "YOLOX-L (DarkNet53)", "cnn_baseline", "cnn", "DarkNet53"),
                                        ("model3", r"DarkNet53 \+ YOLOX-M", "YOLOX-M (DarkNet53)", "cnn_baseline", "cnn", "DarkNet53")):
    nums, ev = row(PID, 11, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 3, (item, nums, ev)
    ROWS += model(item, name, role, cat, bb, None, None, 11, T2, ev, **NP)
    if item != "model1":
        ROWS += [R("total_params", nums[0], 11, T2, ev, item=item, unit="M")]
    ROWS += result(res(), "ap50", item, "dataset1", nums[1], 11, T2, ev, setting="test set; two anomaly classes; five training runs, combination not stated", note="mAP@0.5; mAP@[0.5:0.95] also in Table 2")
    ROWS += result(res(), "ap50_95", item, "dataset1", nums[2], 11, T2, ev, setting="test set; two anomaly classes; five training runs, combination not stated")
full, ev_full = row(PID, 11, r"^\s*None\s+(?=\d)", 0)
ROWS += [R("total_params", full[1], 11, T3, "The Number of Pruning Layers mAP@0.5 Model Parameters (M) Calculation (G) " + ev_full, item="model1", unit="M",
           note="Table 2 lists 29.0 M for this model; the text (p.10) compares Swin-T with YOLOX-M at about 25 M, so 29.0 M appears to be the Swin-T backbone"),
         R("flops", full[2], 11, T3, "Calculation (G) " + ev_full, item="model1", unit="GFLOPs")]
lw, ev_lw = row(PID, 11, r"^\s*10~12 layer\s+(?=\d)", 0)
ROWS += model("model4", "Lightweight Swin-T YOLOX (layers 10 to 12 of Swin-T pruned)", "proposed", "hierarchical_vit", "pruned Swin-T backbone with YOLOX head", None, None, 11, T3, ev_lw,
              cat_note="deployed model", **NP)
ROWS += [R("total_params", lw[1], 11, T3, ev_lw, item="model4", unit="M"), R("flops", lw[2], 11, T3, ev_lw, item="model4", unit="GFLOPs"),
         R("inference_speed", "40 FPS on Jetson Xavier NX with TensorRT; 48 FPS on an RTX 3090 GPU server", 11, T4,
           "Jetson + TensorRT Lightweigh Swin-T YOLOX 40 94.0% ... GPU server Lightweigh Swin-T YOLOX 48 94.0%", item="model4"),
         R("inference_speed", "22 FPS on an RTX 3090 GPU server; 24 FPS on Jetson Xavier NX with TensorRT", 11, T4,
           "GPU server Swin-T YOLOX 22 93.3% ... Jetson + TensorRT Swin-T YOLOX 24 93.2%", item="model1")]
ROWS += result(res(), "ap50", "model4", "dataset1", lw[0], 11, T3, ev_lw, setting="test set; two anomaly classes", headline=True)
ROWS += avail()
ROWS += [D("notes", "The headline 94.0% is the pruned (lightweight) model; the unpruned Swin-T YOLOX gives 93.3% mAP@0.5 against 93.0% (YOLOX-L) and 92.1% (YOLOX-M). Accuracy in Table 4 is computed on 50 test images per deployment platform.", "Sec. 3.2 to 3.4 and Tables 2 to 4 (pp. 10 to 12)")]
