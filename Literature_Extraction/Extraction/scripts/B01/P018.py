import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, T2 = "Table 1", "Table 2"
ROWS = [
    R("task", "detection;crown_segmentation;health_classification", 1, "Abstract", "identify, countify, monitor, and evaluate the overall well-being of palm trees using extensive UAV imagery"),
    R("output_unit", "tree_instance;crown_mask;tree_label", 10, T1, "Bounding box ... Segmentation", note="healthy or unhealthy label per tree after fine-tuning"),
    R("tree_setting", "mixed", 4, "Sec. 3", "The study region encompasses a wide variety of agricultural farms and urban landscapes"),
    R("tree_taxa", "date palm (Phoenix dactylifera)", 1, "Introduction", "The date palm (Phoenix dactylifera L.)"),
    R("study_area", "Ajman Emirate, United Arab Emirates; about 30 km2", 4, "Sec. 3", "The experimental site, located in Ajman Emirate, United Arab Emirates (UAE), covers an area of 30 km2"),
    R("biome", "arid", 4, "Sec. 3", "hyper-arid climate"),
    R("platform", "uav", 4, "Sec. 3", "The dataset in this study was acquired using a SenseFly eBee-Plus UAV platform"),
    R("sensors", "rgb", 4, "Sec. 3", "fitted with a 20 MP digital camera ... that acquires visible-color spectral (RGB) images"),
    R("annotation_type", "polygon_mask", 5, "Sec. 4", "the date palm trees in the image were manually delineated", note="health labels (healthy or unhealthy) assigned in LabelMe"),
    R("dataset_name", "own UAV date palm dataset (Ajman)", 5, "Sec. 4.2.1", "89,069 images formatted in COCO style were used in training", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 4.2.1", "a testing dataset was established, encompassing a sum of 3645 images that contained 30,385 instances", item="dataset1"),
    R("dataset_public", "no", 17, "Data availability", "The authors do not have permission to share data", item="dataset1"),
    NR("n_images", item="dataset1", note="89,069 training images (augmented), 3,500 validation and 3,645 test images (p.5)"),
    NR("n_trees", item="dataset1", note="664,689 training instances (augmented); 27,636 validation and 30,385 test instances (p.5)"),
    R("dataset_name", "Al Ain date palm dataset (Al-Saad et al. 2022)", 10, "Sec. 4.7", "a publicly available UAV dataset (Al-Saad et al., 2022)", item="dataset2"),
    R("dataset_role", "transfer_target", 10, "Sec. 4.7", "Evaluating the transferability of the proposed model to a new dataset collected using a different UAV system in a distinct geographical region", item="dataset2"),
    R("dataset_public", "yes", 10, "Sec. 4.7", "a publicly available UAV dataset (Al-Saad et al., 2022)", item="dataset2"),
    R("n_images", "1209", 10, "Sec. 4.7", "The dataset comprised 1209 image tiles", item="dataset2", unit="images"),
    R("n_trees", "10975", 10, "Sec. 4.7", "1209 image tiles, encompassing 10,975 date palm trees", item="dataset2", unit="trees"),
    R("dataset_name", "own date palm health dataset (healthy and unhealthy palms, Ajman)", 7, "Sec. 4.2.2", "The training dataset comprised 29,040 annotations, including 16,095 healthy date palm trees and 12,945 unhealthy ones", item="dataset3"),
    R("dataset_role", "train_test", 7, "Sec. 4.2.2", "the testing dataset comprises 27,467 annotations, featuring 9126 healthy and 18,341 unhealthy date palm tree instances", item="dataset3"),
    R("dataset_public", "no", 17, "Data availability", "The authors do not have permission to share data", item="dataset3"),
    D("n_trees", "78368", "29,040 training + 21,861 validation + 27,467 test annotations (p.7, Sec. 4.2.2)", item="dataset3", unit="trees"),
]
TAB = [  # method, backbone, category, test box mAP50, box F, seg mAP50, seg F; independent box mAP50, seg mAP50
    ("Mask R-CNN", "ResNet-50-FPN", "cnn", "89.8", "90.9", "88.808", "90.1", "98.39", "97.4"),
    ("Mask R-CNN", "ResNet-101-FPN", "cnn", "88.9", "90.38", "88.892", "90", "98.40", "97.4"),
    ("Mask R-CNN", "MViTv2-tiny", "hierarchical_vit", "92.5", "94.33", "92.442", "94.20", "98.426", "98.1"),
    ("Cascade Mask R-CNN", "ResNet-50-FPN", "cnn", "88.1", "90.20", "87.2", "89.4", "94.9", "95.0"),
    ("Cascade Mask R-CNN", "ResNet-101-FPN", "cnn", "87.1", "89.48", "86.9", "89.07", "93.8", "93.7"),
    ("Cascade Mask R-CNN", "MViTv2-tiny", "hierarchical_vit", "92.22", "93.97", "89.25", "91.08", "98.55", "96.401"),
    ("Cascade Mask R-CNN", "MViTv2-small", "hierarchical_vit", "92.12", "93.77", "91.31", "92.88", "98.54", "97.441"),
    ("PointRend", "ResNet-50", "cnn", "90.6", "92.7", "90.4", "92.33", "97.1", "96.9"),
    ("YOLACT", "ResNet-50-FPN", "cnn", "89.6", "91.35", "88.4", "90.2", "97.2", "96.1"),
    ("YOLACT", "ResNet-101-FPN", "cnn", "89.7", "91.6", "88.6", "90.51", "97", "95.6"),
]
res, tr = Counter("res"), Counter("tr")
for i, (meth, bb, cat, bm, bf, sm, sf, ibm, ism) in enumerate(TAB, start=1):
    item, lab = f"model{i}", f"{meth} {bb}"
    proposed = meth == "Mask R-CNN" and bb == "MViTv2-tiny"
    role = "proposed" if proposed else ("transformer_baseline" if cat != "cnn" else "cnn_baseline")
    ROWS += model(item, f"{meth} with {bb} backbone" + (" (improved MViTv2 with FPN)" if proposed else ""), role, cat, bb, "imagenet", "full_fine_tuning",
                  10, T1, lab, pre_page=9, pre_loc="Sec. 4.5", pre_ev="All instance segmentation models were initialized using ImageNet pre-trained weights and trained for 100,000 iterations")
    ROWS += result(res(), "ap50", item, "dataset1", bm, 10, T1, f"{lab} bounding box mAP50 {bm}", setting="test set; detection (box)", headline=proposed)
    ROWS += result(res(), "f1", item, "dataset1", bf, 10, T1, f"{lab} bounding box F-score {bf}", setting="test set; detection (box)")
    ROWS += result(res(), "ap50", item, "dataset1", sm, 10, T1, f"{lab} segmentation mAP50 {sm}", setting="test set; segmentation (mask)")
    ROWS += result(res(), "f1", item, "dataset1", sf, 10, T1, f"{lab} segmentation F-score {sf}", setting="test set; segmentation (mask)")
    for kind, ind, outd in (("box", bm, ibm), ("mask", sm, ism)):
        t = tr()
        ROWS += [R("transfer_source", "Ajman UAV dataset (training region; test region for the in-domain value)", 5, "Sec. 4.2.1", "The study site was divided into three distinct regions", item=t),
                 R("transfer_target", "Al Ain UAV dataset (different city, different UAV system, 0.1 m GSD)", 10, "Sec. 4.7", "a new dataset collected using a different UAV system in a distinct geographical region", item=t),
                 R("shift_type", "site;sensor", 10, "Sec. 4.7", "collected using a different UAV system in a distinct geographical region", item=t),
                 R("transfer_model", item, 13, T2, lab, item=t),
                 R("transfer_metric", "ap50", 13, T2, f"{'Bounding box' if kind == 'box' else 'Segmentation'} mAP50", item=t, note=f"{kind}"),
                 R("value_in_domain", ind, 10, T1, f"{lab} {'bounding box' if kind == 'box' else 'segmentation'} mAP50 {ind}", item=t, unit="%"),
                 R("value_out_of_domain", outd, 13, T2, f"{lab} {'bounding box' if kind == 'box' else 'segmentation'} mAP50 {outd}", item=t, unit="%")]
for kind, ind, outd, ev in (("box", "94.33", "91", "the model achieved an F-score of 91% in the detection"), ("mask", "94.20", "90", "and an F-score of 90% in segmenting the palm trees")):
    t = tr()
    ROWS += [R("transfer_source", "Ajman test set at the original resolution", 11, "Sec. 5.3", "These multiscale testing datasets were generated by downsampling the original UAV data", item=t),
             R("transfer_target", "Ajman test set downsampled to 15 cm GSD", 11, "Sec. 5.3", "when evaluated on a dataset with a GSD of 15 cm", item=t),
             R("shift_type", "resolution", 11, "Sec. 5.3", "multiscale testing datasets comprising different spatial resolutions, specifically 10, 15, 20, 25, and 30 cm", item=t),
             R("transfer_model", "model3", 11, "Sec. 5.3", "The proposed model", item=t),
             R("transfer_metric", "f1", 11, "Sec. 5.3", ev, item=t, note=kind),
             R("value_in_domain", ind, 10, T1, f"Mask R-CNN MViTv2-tiny {'bounding box' if kind == 'box' else 'segmentation'} F-score {ind}", item=t, unit="%"),
             R("value_out_of_domain", outd, 11, "Sec. 5.3", ev, item=t, unit="%")]
ROWS += model("model11", "Mask R-CNN with MViTv2-tiny fine-tuned for palm health (healthy or unhealthy)", "proposed", "hierarchical_vit", "MViTv2-tiny", "domain_specific",
              "full_fine_tuning", 7, "Sec. 4.2.2", "the proposed model was fine-tuned using the dataset in Section 4.2.2 to explore the potential health of the date palm trees",
              pre_note="initialised from the palm detection model trained on dataset1")
ROWS += result(res(), "f1", "model11", "dataset3", "88.4", 16, "Sec. 6", "precision, recall, and F-scores of 81.66%, 96.6%, and 88.4%, respectively",
               setting="test set; healthy versus unhealthy palms", note="abstract gives the same F-score (88.4%)")
ROWS += avail(data="no", data_page=17, data_loc="Data availability", data_ev="The authors do not have permission to share data")
ROWS += [D("notes", "This paper does not state the GSD. Transfer to Al Ain (Table 2) raised scores for all models (box mAP50 93.8 to 98.55). Accuracy fell by about 20% at 25 cm GSD (Sec. 5.3).", "Sec. 4 to 6 and Tables 1 to 2 (pp. 5 to 16)")]
