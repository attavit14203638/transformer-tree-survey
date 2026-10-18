import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T2 = "P026", "Table I", "Table II"
# Oil palm detection with boxes. The paper says which split the tables are computed on nowhere (70/15/15 split only), so split is not_stated.
T = Tags(task="detection", level="object", split="not_stated", agg="single")
TC = T(task="counting", level="not_stated")
LOC = "Annotated datasets from plantations in Pahang, Malaysia, and Riau, Indonesia, are used to train and assess RTDETR-Palm"

ROWS = [
    R("task", "detection;counting", 2, "Sec. 1", "specifically adapted for oil palm tree detection and counting from UAV imagery"),
    R("output_unit", "tree_instance", 3, "Sec. 3", "produce bounding boxes around each detected oil palm crown"),
    R("tree_setting", "plantation", 1, "Abstract", "Accurate and efficient detection of oil palm trees is essential for plantation auditing"),
    R("tree_taxa", "oil palm", 1, "Abstract", "Accurate and efficient detection of oil palm trees is essential for plantation auditing"),
    R("study_area", "Pahang, Malaysia, and Riau, Indonesia", 1, "Abstract", LOC),
    D("biome", "tropical", "Pahang (Malaysia) and Riau (Indonesia); not stated in the paper"),
    R("platform", "uav", 4, "Sec. 4", "high-resolution RGB images captured using rotary-wing UAVs equipped with nadir-facing cameras"),
    R("sensors", "rgb", 4, "Sec. 4", "high-resolution RGB images captured using rotary-wing UAVs equipped with nadir-facing cameras"),
    R("annotation_type", "bounding_box", 4, "Sec. 4", "Each image was manually annotated with bounding boxes around individual oil palm tree crowns"),
    R("dataset_name", "own UAV oil palm dataset (Pahang, Malaysia and Riau, Indonesia)", 1, "Abstract", LOC, item="dataset1"),
    R("dataset_key", "own_p026_oil_palm", 1, "Abstract", LOC, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 4", "The dataset was split into 70% for training, 15% for validation, and 15% for testing", item="dataset1", note="image and tree counts are not printed; images resized to 640 × 640"),
    D("dataset_public", "no", "No dataset link or availability statement found in the paper", item="dataset1"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="initialisation not stated; AdamW, learning rate 1e-4, 300 epochs with early stopping (Sec. 4)")
ROWS += model("model1", "RT-DETR-Palm", "proposed", "cnn_transformer_hybrid", "ResNet-50 with RT-DETR transformer encoder-decoder", None, None, 2, "Sec. 1",
              "This paper presents RT-DETR-Palm, a deep learning framework based on RT-DETR ... The model uses ResNet-50 as the backbone network", cat_note="RT-DETR with a ResNet-50 backbone (protocol 4.5)", **NP)
ROWS += model("model2", "Faster R-CNN", "cnn_baseline", "cnn", "backbone not stated", None, None, 5, "Sec. 5.B",
              "we compared RTDETR-Palm against the widely adopted Faster R-CNN, using identical datasets and training settings", cat_note="region-based CNN", **NP)
res = Counter("res")
for item, lab in (("model1", "RT-DETR-Palm"), ("model2", "Faster R-CNN")):
    n1, ev1 = row(PID, 5, rf"^\s*{lab}\s+(?=\d)", 0, cols=(0, 86))
    n2, ev2 = row(PID, 5, rf"^\s*{lab}\s+(?=\d)", 1, cols=(0, 86))
    assert len(n1) == 3 and len(n2) == 3, (lab, n1, n2)  # Table I: precision, recall, F1; Table II: overall accuracy, average IoU, counting error
    ROWS += result(res(), "f1", item, "dataset1", n1[2], 5, T1, ev1, setting="Table I: Faster R-CNN against RT-DETR-Palm, same data and settings", note="F1-score column (Table I also gives precision and recall)", **T, primary=True)
    ROWS += result(res(), "other", item, "dataset1", n2[0], 5, T2, ev2, unit="%", setting="Table II", note="OA, overall accuracy for detection; the paper gives no definition", **T,
                   headline=(item == "model1"))
    ROWS += result(res(), "other", item, "dataset1", n2[1], 5, T2, ev2, unit="fraction", setting="Table II", note="average IoU of the predicted and reference boxes; the paper gives no definition", **T)
    ROWS += result(res(), "other", item, "dataset1", n2[2], 5, T2, ev2, unit="%", setting="Table II", note="counting error (percent); the paper gives no definition", **TC)
ROWS += avail()
ROWS += [D("notes", "Table I gives precision, recall and F1 (Faster R-CNN 91.2, 83.5, 87.2; RT-DETR-Palm 89.7, 91.5, 90.6); Table II gives overall accuracy, average IoU and counting error (Faster R-CNN 82.6, 0.72, 7.9; RT-DETR-Palm 86.4, 0.78, 4.0), recorded as metric other because the paper defines none of them. The text claims real-time inference more than twice as fast as Faster R-CNN but prints no speed. mAP is announced in Sec. 4 but no mAP value is printed.",
           "Sec. 4 to 6 and Tables I to II (pp. 4 to 5)")]
