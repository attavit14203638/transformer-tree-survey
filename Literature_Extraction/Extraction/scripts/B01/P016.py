import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T2, T4, T6 = "Table 2 (image)", "Table 4 (image)", "Table 6 (image)"
ROWS = [
    R("task", "detection", 1, "Abstract", "we proposed an improved faster region-based convolutional neural network (Faster R-CNN) with Swin transformer method"),
    R("output_unit", "tree_instance", 5, "Sec. III-D", "the ratio between the union and the intersection of predicted box and ground truth box", note="bounding boxes"),
    R("tree_setting", "urban", 1, "Abstract", "Individual Tree Detection Based on High-Resolution RGB Images for Urban Forestry Applications"),
    NR("tree_taxa", note="urban trees; species not given"),
    R("study_area", "Campo Grande, Brazil (training data); Guangzhou, Guangdong, China (three transfer sites)", 3, "Sec. II-B", "The three sites were located in Guangzhou, Guangdong province, China", note="Campo Grande dataset from [21]"),
    D("biome", "multiple", "Campo Grande (tropical) and Guangzhou (subtropical); not stated in the paper"),
    R("platform", "multiple", 2, "Sec. II-A", "two RGB high-resolution, airborne orthoimages ... of the Campo Grande urban area", note="airborne (Campo Grande) and UAV (Guangzhou, DJI Phantom 4 Pro, p.3)"),
    R("sensors", "rgb", 3, "Sec. II-B", "UAV based RGB imagery was acquired in April-October 2020 using a DJI Phantom 4 Pro"),
    R("gsd", "0.1", 3, "Sec. II-B", "The orthoimages with a GSD equal to 0.1m were used", unit="m", note="both datasets 0.1 m"),
    R("annotation_type", "bounding_box", 2, "Fig. 1", "An annotated patch of Campo Grande dataset. The bounding ..."),
    R("dataset_name", "Campo Grande urban tree dataset [21]", 2, "Sec. II-A", "The Campo Grande dataset [21] includes two RGB high-resolution, airborne orthoimages", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. IV-A", "The training, validation, and testing sets comprising 60%, 20% and 20% of the available images", item="dataset1"),
    R("dataset_public", "yes", 2, "Sec. II", "one public and three private datasets", item="dataset1"),
    R("n_images", "220", 3, "Sec. II-A", "split into 220 non-overlapping patches of 512 × 512 pixels", item="dataset1", unit="images"),
    R("n_trees", "3382", 3, "Sec. II-A", "A total of 3382 trees were identified as ground truth", item="dataset1", unit="trees"),
    R("dataset_name", "Guangzhou street trees (UAV)", 7, "Sec. IV-C", "We annotated a small amount of street trees", item="dataset2"),
    R("dataset_role", "transfer_target", 5, "Fig. 8", "Once trained, the network was applied to detect trees in Guangzhou dataset", item="dataset2"),
    R("dataset_public", "no", 2, "Sec. II", "one public and three private datasets", item="dataset2"),
    D("n_trees", "235", "TP + FN for the proposed model in Table 4 (226 + 9)", item="dataset2", unit="trees"),
]
ROWS += model("model1", "Faster R-CNN with Swin-T backbone", "proposed", "hierarchical_vit", "Swin-T", None, "full_fine_tuning", 6, T2, "Faster R-CNN (Swin-T) 0.699",
              pre_status="not_reported", pre_note="pre-trained weights not specified", strat_page=5, strat_loc="Sec. III-C",
              strat_ev="a semi-supervised approach was used to fine tune the base model further", strat_note="trained on Campo Grande, then retrained on high-confidence pseudo-labels of the Guangzhou images")
ROWS += [R("inference_speed", "0.103 s (±0.062) per 512 x 512 image on a GeForce RTX 2080 Super", 9, T6, "Faster R-CNN (Swin-T) 0.103(±0.062)", item="model1")]
ROWS += model("model2", "Faster R-CNN with ResNet-50 backbone", "cnn_baseline", "cnn", "ResNet-50", None, None, 6, T2, "Faster R-CNN (Resnet50) 0.652",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated")
ROWS += [R("inference_speed", "0.122 s (±0.067) per 512 x 512 image on a GeForce RTX 2080 Super", 9, T6, "Faster R-CNN (Resnet50) 0.122(±0.067)", item="model2")]
for item, lab, bb in (("model3", "Faster R-CNN (Resnet101FPN) [21]", "ResNet-101-FPN"), ("model4", "RetinaNet [21]", "not stated"), ("model5", "FSAF [21]", "not stated")):
    ROWS += model(item, lab.replace(" [21]", ""), "cnn_baseline", "cnn", bb, None, None, 6, T2, lab, pre_status="not_reported",
                  pre_note="value reported from [21]", strat_note="value reported from [21]")
res = Counter("res")
for m, lab, v in (("model1", "Faster R-CNN (Swin-T)", "0.699"), ("model2", "Faster R-CNN (Resnet50)", "0.652"), ("model3", "Faster R-CNN (Resnet101FPN) [21]", "0.660"),
                  ("model4", "RetinaNet [21]", "0.650"), ("model5", "FSAF [21]", "0.701")):
    ROWS += result(res(), "ap50", m, "dataset1", v, 6, T2, f"{lab} {v}", unit="fraction", setting="test set (20%)", headline=(m == "model1"),
                   note="" if m in ("model1", "model2") else "value reported from [21], not re-run")
for m, lab, f1 in (("model1", "Faster R-CNN (Swin-T) TP 226 FP 16 FN 9 P 0.934 R 0.962", "0.948"), ("model2", "Faster R-CNN (Resnet50) TP 218 FP 33 FN 17 P 0.869 R 0.928", "0.898")):
    ROWS += result(res(), "f1", m, "dataset2", f1, 8, T4, f"{lab} F1 {f1}", unit="fraction", setting="street trees in Guangzhou after pseudo-label retraining")
ROWS += avail()
ROWS += [D("notes", "The Guangzhou evaluation follows training on Campo Grande and retraining on pseudo-labels without manual target labels; in-domain (AP50) and target (F1) use different metrics, so no transfer pair is recorded. Training time per epoch: Swin-T 35.760 s, ResNet-50 40.527 s (Table 5, p.9). All tables are images.", "Sec. III-C, IV and Tables 2 to 6 (pp. 5 to 9)")]
