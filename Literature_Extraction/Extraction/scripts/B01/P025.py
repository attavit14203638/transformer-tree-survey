import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, T2 = "Table 1", "Table 2"
ZEN = "https://doi.org/10.5281/zenodo.8148479"
ROWS = [
    R("task", "detection;species_classification", 1, "Title", "Parameter-Efficient Fine-Tuning for Individual Tree Crown Detection and Species Classification"),
    R("output_unit", "tree_instance;tree_label", 5, "Sec. 2", "transformed them into COCO-format bounding box annotations for object detection", note="species-labelled crown boxes"),
    R("tree_setting", "natural_forest", 4, "Sec. 2", "Predominantly covered by unmanaged mixed temperate forest"),
    R("tree_taxa", "Abies balsamea; Betula papyrifera; Acer rubrum", 5, "Sec. 2", "we selected 300 images representing three tree species: Abies balsamea, Betula papyrifera, and Acer rubrum"),
    R("study_area", "Université de Montréal research station, Saint-Hippolyte, Quebec, Canada", 4, "Sec. 2", "within the research station of Université de Montréal in Saint-Hippolyte"),
    R("biome", "temperate", 4, "Sec. 2", "The study area is a mixed temperate forest located in Quebec, Canada"),
    R("platform", "uav", 4, "Sec. 2", "The imagery was captured using a DJI Phantom 4 RTK UAV"),
    R("sensors", "rgb", 4, "Sec. 2", "a DJI Phantom 4 RTK UAV ... equipped with a real-time kinematic (RTK) GNSS receiver integrated with an inertial measurement unit (IMU) and camera",
      note="RGB camera of the Phantom 4 RTK"),
    R("gsd", "1.81", 5, "Sec. 2", "a ground sampling distance ranging from 1.81 to 2.02 cm", unit="cm", note="1.81 to 2.02 cm"),
    R("annotation_type", "bounding_box", 5, "Sec. 2", "converted the original polygon annotations into a format suitable for object detection", note="boxes derived from crown polygons with species labels"),
    R("dataset_name", "Quebec temperate forest UAV dataset (Cloutier et al.), 2 September 2021 subset with three species", 5, "Sec. 2",
      "we selected the data acquired on 2 September 2021", item="dataset1"),
    R("dataset_role", "train_test", 11, "Sec. 4.1", "80% of the images are allocated to the training set, and the remaining 20% are designated as the testing set", item="dataset1"),
    R("dataset_public", "yes", 17, "Data Availability Statement", f"The raw dataset can be available at {ZEN}", item="dataset1"),
    R("dataset_url", ZEN, 17, "Data Availability Statement", f"The raw dataset can be available at {ZEN}", item="dataset1"),
    R("n_images", "300", 5, "Sec. 2", "we selected 300 images representing three tree species", item="dataset1", unit="images"),
    D("n_trees", "2610", "1045 Acer rubrum + 1003 Betula papyrifera + 562 Abies balsamea boxes (p.5, Sec. 2)", item="dataset1", unit="trees"),
]
GD = dict(pre_page=11, pre_loc="Sec. 4.1", pre_ev="The Swin-T backbone from Grounding-DINO is used as the base model", pre_note="Grounding-DINO pre-trained weights")
PEFT = [  # label, strategy, trainable %, mAP, AP50, note
    ("FULL", "full_fine_tuning", "100.00", "33.5", "60.7", "all parameters updated"),
    ("FIXED", "head_only", "0.00", "31.2", "60.0", "backbone frozen"),
    ("BITFIT", "peft_other", "0.28", "31.3", "59.3", "bias terms only"),
    ("NORM-TUNING", "peft_other", "0.09", "31.8", "60.5", "normalisation layers only"),
    ("LORA", "peft_lora", "4.11", "31.5", "59.9", "low-rank matrices in the Swin transformer blocks"),
    ("VPT", "peft_prompt", "0.26", "31.8", "58.7", "visual prompt tokens"),
    ("EVP2", "peft_prompt", "19.83", "32.2", "59.9", "explicit visual prompting"),
    ("SCT", "peft_other", "0.01", "30.9", "57.7", "salient channel tuning"),
    ("Adapter", "peft_adapter", "4.15", "31.7", "60.4", "MLP adapters in each transformer layer"),
    ("OURS", "peft_other", "0.28", "32.4", "60.6", "adaptive salient channel selection with bias tuning (ASCS)"),
]
res = Counter("res")
for i, (lab, strat, par, mAP, ap50, note) in enumerate(PEFT, start=1):
    item = f"model{i}"
    ROWS += model(item, f"Grounding-DINO (Swin-T) with {lab} fine-tuning" if lab != "OURS" else "Grounding-DINO (Swin-T) with ASCS (proposed PEFT)",
                  "proposed" if lab == "OURS" else "transformer_baseline", "vision_language_model", "Swin-T", "other", strat, 12, T1, f"{lab} {par}% {mAP} {ap50}",
                  strat_note=note, **GD)
    ROWS += [R("trainable_params", par, 12, T1, f"{lab} {par}%", item=item, unit="%", note="share of parameters updated (Params column)")]
    ROWS += result(res(), "map", item, "dataset1", mAP, 12, T1, f"{lab} mAP {mAP}", setting="test split (20%)", note="mAP as printed; the paper averages precision over several IoU thresholds without stating them (Sec. 4.1.1, Eq. 10)")
    ROWS += result(res(), "ap50", item, "dataset1", ap50, 12, T1, f"{lab} AP50 {ap50}", setting="test split (20%)", headline=(lab == "OURS"))
OTHER = [("model11", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8", "18.0", None), ("model12", "YOLOv10", "cnn_baseline", "cnn", "YOLOv10", "22.6", None),
         ("model13", "YOLOv11", "cnn_baseline", "cnn", "YOLOv11", "21.2", None),
         ("model14", "Grounding-DINO (zero-shot)", "transformer_baseline", "vision_language_model", "Swin-T", "2.4", "zero_shot"),
         ("model15", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "not stated", "57.5", None)]
for item, name, role, cat, bb, v, strat in OTHER:
    if strat == "zero_shot":
        ROWS += model(item, name, role, cat, bb, "other", "zero_shot", 14, "Sec. 4.1.3", "directly applying the original Grounding-DINO yields a low AP50 of only 2.4", **GD)
    else:
        ROWS += model(item, name, role, cat, bb, None, None, 14, T2, f"{name} {v}", pre_status="not_reported", pre_note="not stated", strat_note="not stated")
    ROWS += result(res(), "ap50", item, "dataset1", v, 14, T2, f"{name.split(' (')[0]} {v}", setting="test split (20%)")
ROWS += avail(data="yes", data_page=17, data_loc="Data Availability Statement", data_ev=f"The raw dataset can be available at {ZEN}")
ROWS += [D("notes", "Full fine-tuning gave the highest mAP (33.5); the proposed ASCS reached 32.4 mAP and 60.6 AP50 while updating 0.28% of parameters. Ablations (Tables 3 to 6) vary the channel selection, bias tuning, insertion position and scaling factor.", "Tables 1 to 6 (pp. 12 to 16)")]
