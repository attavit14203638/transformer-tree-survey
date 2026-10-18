import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4 = "P138", "Table IV"
# YOLOv11m against RF-DETR-base for two-class (healthy, unhealthy) oil palm detection in UAV tiles of MOPAD. A benchmark of two detectors, no proposed model. Healthy and unhealthy are the health classes of the palms,
# so mAP over both classes is health_classification (a judgement call, see notes). The paper does not state the split of Table IV and the mAP values: the unhealthy palms of the YOLOv11m confusion matrix (406 + 2) equal the validation count of
# Table III (408 unhealthy palms) and that of RF-DETR is close to the test count (201), so split is not_stated. F1 is given per class only (the Overall row of Table IV has no F1), so the
# F1 of the first listed class (Healthy) is primary (protocol 6.3 item 5); mAP50 and mAP50:95 are the overall values and primary. Headline: the YOLOv11m mAP50 of the abstract.
T = Tags(task="health_classification", level="object", split="not_stated", agg="single")
TASK_EV = "This study compares YOLOv11 (You Only Look Once v11) and RF-DETR (Roboflow Detection Transformer) for detecting healthy and unhealthy oil palm trees from Unmanned Aerial Vehicle (UAV) imagery"
SPLIT_EV = "The dataset was divided into 70% (1,612 images) allocated to the training dataset, 20% (461 images) to the validation dataset, and 10% (230 images) to the testing dataset"
SITE_EV = "covering a total area of approximately 11.45 km2 in Papua"
COUNT_EV = "comprising 114,842 healthy and 1,884 unhealthy palm cases as shown in Table I"
ROWS = [
    R("task", "health_classification", 1, "Abstract", TASK_EV, note="two-class detection of healthy and unhealthy oil palms (canopy colour, leaf consistency and crown density)"),
    R("output_unit", "tree_instance", 3, "Sec. II.A", "The assessment used bounding box predictions and COCO-format annotations"),
    R("tree_setting", "plantation", 1, "Abstract", "oil palm trees from Unmanned Aerial Vehicle (UAV) imagery", note="oil palm plantation monitoring"),
    R("tree_taxa", "oil palm", 1, "Abstract", "oil palm trees from Unmanned Aerial Vehicle (UAV) imagery"),
    R("study_area", "Papua, Indonesia (about 11.45 km2)", 3, "Sec. II.A", SITE_EV),
    D("biome", "tropical", "Papua, Indonesia (6 57 S); not stated in the paper"),
    R("platform", "uav", 3, "Sec. II.B", "Following the collection of RGB images via UAV"),
    R("sensors", "rgb", 3, "Sec. II.B", "Following the collection of RGB images via UAV"),
    R("annotation_type", "bounding_box", 3, "Sec. II.A", "The assessment used bounding box predictions and COCO-format annotations", note="two classes (healthy, unhealthy) following the MOPAD taxonomy"),
    R("dataset_name", "MOPAD multi-class oil palm UAV dataset (Papua), two classes healthy and unhealthy, 1024 x 1024 tiles", 3, "Sec. II.A", "The dataset can be accessed via https://github.com/rs-dl/MOPAD", item="dataset1",
      note="the paper says the dataset was custom-curated and annotated based on the MOPAD taxonomy (p.2)"),
    R("dataset_key", "mopad", 3, "Sec. II.A", "The dataset can be accessed via https://github.com/rs-dl/MOPAD", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. II.B", SPLIT_EV, item="dataset1", note="70, 20 and 10 percent of the tiles for training, validation and testing"),
    R("dataset_public", "yes", 3, "Sec. II.A", "The dataset can be accessed via https://github.com/rs-dl/MOPAD", item="dataset1"),
    R("dataset_url", "https://github.com/rs-dl/MOPAD", 3, "Sec. II.A", "The dataset can be accessed via https://github.com/rs-dl/MOPAD", item="dataset1"),
    R("n_images", "2303", 3, "Sec. II.B", "(total: 2,303)", item="dataset1", unit="tiles", note="non-overlapping 1024 x 1024 tiles, resized to 640 x 640; 1,612 training, 461 validation and 230 test tiles"),
    D("n_trees", "116726", "sum of the printed class counts of Table I: 114,842 healthy and 1,884 unhealthy palms (Table III splits: 80,590 + 22,810 + 11,442 and 1,275 + 408 + 201)", item="dataset1", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained with the same protocol and COCO-format annotations (p.2); 307 epochs in Ultralytics; initialisation not stated")
ROWS += model("model1", "YOLOv11m", "cnn_baseline", "cnn", "CNN backbone with C3K2 modules, SPPF and attention-based C2PSA neck, decoupled heads at three scales", None, None, 4, "Sec. II.C",
              "a CNNbased backbone with C3K2 modules for hierarchical feature extraction", cat_note="the paper calls YOLOv11 CNN-based; the C2PSA attention modules of the neck are not described as self-attention or transformer blocks", **NP)
ROWS += model("model2", "RF-DETR-base", "transformer_baseline", "cnn_transformer_hybrid", "pre-trained DINOv2 backbone (also described as a CNN backbone) with a Transformer encoder and decoder",
              "self_supervised", None, 4, "Sec. II.C", "The model features a CNN backbone for spatial feature extraction, followed by a Transformer encoder with positional encodings and a Transformer decoder with learnable object queries",
              cat_note="the paper names a pre-trained DINOv2 backbone (abstract, p.1 and Sec. II.C) and also a CNN backbone followed by a transformer encoder and decoder (Sec. II.C); recorded as a DETR detector with a CNN backbone (cnn_transformer_hybrid, protocol 4.5)",
              pre_page=4, pre_loc="Sec. II.C", pre_ev="integrating a pre-trained DINOv2 backbone for enhanced cross-domain adaptability through self-supervised learning", pre_note="DINOv2 backbone pre-trained by self-supervised learning",
              strat_note="trained on Roboflow with early stopping after 73 epochs, same protocol as YOLOv11m (p.2 and 6); initialisation of the detector not stated")
ROWS.append(R("total_params", "29", 1, "Introduction", "the Base version features 29 million parameters", item="model2", unit="M", note="stated in the description of RF-DETR (Base), not measured in the study"))
res = Counter("res")
SETM = "mAP over the healthy and unhealthy classes; split not stated"
ROWS += result(res(), "ap50", "model1", "dataset1", "99.3", 6, "Sec. III.B", "YOLOv11m achieves a mAP-50 of 99.3% and a mAP-50:95 of 97.5%", unit="%", setting=SETM, **T, primary=True, headline=True)
ROWS += result(res(), "ap50_95", "model1", "dataset1", "97.5", 6, "Sec. III.B", "YOLOv11m achieves a mAP-50 of 99.3% and a mAP-50:95 of 97.5%", unit="%", setting=SETM, **T, primary=True)
ROWS += result(res(), "ap50", "model2", "dataset1", "99.1", 6, "Sec. III.B", "DETR trained on Roboflow reaches a mAP-50 of 99.1%", unit="%", setting=SETM, **T, primary=True)
ROWS += result(res(), "ap50_95", "model2", "dataset1", "89.8", 6, "Sec. III.B", "a mAP-50:95 of 89.8%", unit="%", setting=SETM, **T, primary=True)
for item, cls, ev, f1, prim in (("model1", "Healthy", "Healthy 0.9864 0.9988 0.9926 98.56%", "0.9926", True), ("model1", "Unhealthy", "YOLOv11m Unhealthy 0.9783 0.9951 0.9867 99.95%", "0.9867", False),
                                ("model2", "Healthy", "Healthy 0.9889 0.9904 0.9896 98.00%", "0.9896", True), ("model2", "Unhealthy", "RF- DETR-base Unhealthy 0.9740 0.9350 0.9535 99.85%", "0.9535", False)):
    ROWS += result(res(), "f1", item, "dataset1", f1, 6, T4, ev, unit="fraction", setting=f"Table IV: {cls} class; split not stated", note=f"F1 score of the {cls} class (precision, recall and accuracy of the row are not recorded); no overall F1 is printed",
                   **T, primary=prim)
ROWS += [R("limitations", "Strong class imbalance (114,842 healthy against 1,884 unhealthy palms), early stopping of the RF-DETR training, and occlusion and lighting variation in the UAV imagery", 6, "Sec. III.C",
           "Limitations include the class imbalance (Healthy: 114,842 vs. Unhealthy: 1,884)")]
ROWS += conflict("cf1", "dataset1.n_trees", "276883", 1, "Abstract", "A custom dataset with 276,883 COCOformat annotations was used", "V1c",
                 "Table I and Table III give 114,842 healthy and 1,884 unhealthy palms (Table III splits add up to the same counts); the abstract's 276,883 annotations match neither the class counts nor their sum", unit="trees")
ROWS += avail(code="no", weights="not_stated", data="yes", data_page=3, data_loc="Sec. II.A", data_ev="The dataset can be accessed via https://github.com/rs-dl/MOPAD", note_d="the MOPAD dataset is public; the paper's own two-class annotations are not stated to be released")
ROWS += [D("notes", "A benchmark of YOLOv11m and RF-DETR-base for healthy and unhealthy oil palm detection. Recorded: mAP50 and mAP50:95 from the text (YOLOv11m 99.3 and 97.5; RF-DETR 99.1 and 89.8; Fig. 11 repeats them as a chart) and the per-class F1 of Table IV (YOLOv11m 0.9926 healthy, 0.9867 unhealthy; RF-DETR 0.9896 and 0.9535). The Overall row of Table IV (99.3 and 98.8 for YOLOv11m; 99.1 and 98 for RF-DETR) sits under the precision and recall columns but the first value equals the mAP50 of the text, so it is not recorded. Healthy and unhealthy palms are health classes of one detector, so the mAP is health_classification. The paper does not state the split of the results: the unhealthy palms of the YOLOv11m confusion matrix (406 true positives and 2 misclassified, 408 in all) equal the validation count of Table III (408), whereas those of RF-DETR (187 and 9, 196 in all) are close to the test count (201), so the two models may not be scored on the same split. The COCO and RF100-VL benchmark values and the latencies of Table II (5.7 and 6.0 ms on a T4) are copied from the model benchmarks, not measured here, and are not recorded.",
           "Sec. III and Tables II to IV (pp. 4 to 6)")]
