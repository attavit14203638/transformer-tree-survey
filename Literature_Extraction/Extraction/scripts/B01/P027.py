import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, T2 = "Table 1", "Table 2"
ROWS = [
    R("task", "detection;health_classification", 1, "Abstract", "on remote sensing image datasets of pine wilt disease", note="single class: PWD-infected (dead) pines"),
    R("output_unit", "tree_instance", 9, "Sec. 4.2", "Each infected tree was labeled as a rectangular bounding box and saved in an XML file"),
    R("tree_setting", "natural_forest", 9, "Sec. 4.2", "The main vegetation consists of horsetail pine forest and broad-leaved evergreen forest"),
    R("tree_taxa", "horsetail pine (Pinus massoniana)", 9, "Sec. 4.2", "with horsetail pine accounting for a higher percentage"),
    R("study_area", "Longyou County, Quzhou City, Zhejiang Province, China; four towns, about 8 km2", 9, "Sec. 4.2",
      "The dataset for this study was collected in Longyou County, Quzhou City, Zhejiang Province ... an area of about 8 square kilometers"),
    R("biome", "subtropical", 9, "Sec. 4.2", "It has a subtropical monsoon climate with four distinct seasons"),
    R("platform", "uav", 1, "Title", "Security of target recognition for UAV forestry remote sensing"),
    R("sensors", "rgb", 9, "Sec. 4.2", "PWD-infected pines exhibiting reddish-brown symptoms that clearly distinguish them from broadleaf trees", note="camera not described; colour imagery"),
    R("annotation_type", "bounding_box", 9, "Sec. 4.2", "Each infected tree was labeled as a rectangular bounding box"),
    R("dataset_name", "own UAV pine wilt disease dataset (Longyou County, seven sample plots)", 9, "Fig. 9 (image)", "7 sample plots ... 1160 Images", item="dataset1"),
    R("dataset_role", "train_test", 9, T1, "Training set 1,2,3,6 3488 / Test set 7 743 / Validation set 4,5 755", item="dataset1", note="split by sample plot"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    R("n_images", "1160", 9, "Fig. 9 (image)", "1160 Images", item="dataset1", unit="images"),
    R("n_trees", "4986", 9, T1, "Total 1,2,3,4,5,6,7 4986", item="dataset1", unit="trees"),
]
M = [("model1", "SC-RTDETR", "proposed", "cnn_transformer_hybrid", "RT-DETR with residual shrinkage network and cascaded group attention", "91.7", "20", "54.6"),
     ("model2", "RT-DETR", "transformer_baseline", "cnn_transformer_hybrid", "not stated", "91.5", "20", "60"),
     ("model3", "YOLOv8s", "cnn_baseline", "cnn", "YOLOv8s", "86.2", "11.2", "28.3"),
     ("model4", "YOLOv5s", "cnn_baseline", "cnn", "YOLOv5s", "92.1", "7.2", "16.5")]
res = Counter("res")
for item, name, role, cat, bb, v, par, fl in M:
    lab = "RTDETR" if name == "RT-DETR" else name
    if item in ("model1", "model2"):
        ROWS += model(item, name, role, cat, bb, None, "full_fine_tuning", 9, T2, f"{lab} {v} {par} {fl}", pre_status="not_reported",
                      pre_note="trained from a pre-trained model; source not named (p.9)", strat_page=9, strat_loc="Sec. 4.3",
                      strat_ev="A transfer learning fine-tuning strategy was used ... Both the RTDETR and the improved SC-RTDETR model were trained based on the pre-trained model")
    else:
        ROWS += model(item, name, role, cat, bb, None, None, 9, T2, f"{lab} {v} {par} {fl}", pre_status="not_reported", pre_note="not stated", strat_note="not stated")
    ROWS += [R("total_params", par, 9, T2, f"{lab} {v} {par} {fl}", item=item, unit="M"), R("flops", fl, 9, T2, f"{lab} {v} {par} {fl}", item=item, unit="GFLOPs")]
    ROWS += result(res(), "map", item, "dataset1", v, 9, T2, f"{lab} mAP {v}", setting="test plot 7; no adversarial perturbation", headline=(item == "model1"),
                   note="mAP; the IoU threshold is given as 0.65 for the evaluation (p.9)")
ROWS += avail()
ROWS += [D("notes", "The study's focus is robustness to adversarial attacks (FGSM, BIM, MIM, PGD; Tables 3 to 4): at FGSM ε = 8, SC-RTDETR scored 61.4 mAP versus 52.3 for RT-DETR. These perturbation tests are not domain shifts and are not recorded as transfer. An ablation table (p.11) reports RT-DETR at 89.5 mAP, which differs from Table 2 (91.5).", "Sec. 4.4 to 4.6 and Tables 2 to 5 (pp. 9 to 11)")]
