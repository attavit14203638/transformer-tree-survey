import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1 = "Table 1"
ROWS = [
    R("task", "detection;crown_segmentation", 2, "Abstract", "examines the efficacy of various instance segmentation models ... in detecting and delineating individual date palm trees"),
    R("output_unit", "tree_instance;crown_mask", 19, T1, "mAP50 and F-measure metrics for the bounding box and segmentation"),
    R("tree_setting", "mixed", 9, "Sec. 3.1.4", "These regions were mostly covered by varied agricultural and urban landscapes"),
    R("tree_taxa", "date palm (Phoenix dactylifera)", 2, "Abstract", "Timely and reliable mapping of individual date palm trees"),
    R("study_area", "eastern Ajman Emirate, United Arab Emirates", 6, "Sec. 3.1.1", "The study area is situated in the eastern part Ajman Emirate, United Arab Emirates"),
    R("biome", "arid", 6, "Sec. 3.1.1", "The climate in the region varies from arid to hyper-arid"),
    R("platform", "uav", 6, "Sec. 3.1.2", "a SenseFly eBee-Plus, a commercially available fixed-wing UAV"),
    R("sensors", "rgb", 7, "Sec. 3.1.2", "S.O.D.A. camera acquires visible-colour images (red [660 nm], green [520 nm], blue [450 nm])"),
    R("gsd", "5", 7, "Sec. 3.1.2", "image of the entire study area with an average ground sampling distance (GSD) of 5 cm", unit="cm"),
    R("annotation_type", "polygon_mask", 8, "Sec. 3.1.3", "extract the coordinates of the boundary of each ... detected region (i.e. palm tree)"),
    R("dataset_name", "own UAV date palm dataset (Ajman, 512 x 512 tiles)", 9, "Sec. 3.1.4", "three distinct geographical regions from the orthomosaic of the study area were chosen to train, evaluate, and test", item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 3.1.4", "3645 images (containing 30,385 date palm tree instances) were used as a testing dataset", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_images", "30383", "23,238 training tiles (before 4x rotation augmentation to 89,069) + 3,500 validation + 3,645 test tiles (p.9, Sec. 3.1.4)", item="dataset1", unit="images"),
    NR("n_trees", item="dataset1", note="664,689 training instances after augmentation; 27,636 validation and 30,385 test instances (p.9); unaugmented training count not given"),
    R("dataset_name", "Al Ain date palm dataset (Al-Saad et al. 2022)", 16, "Sec. 3.4", "an independent dataset (Al-Saad et al. 2022) ... UAV-based images acquired in Al Ain City with a spatial resolution of 10 cm", item="dataset2"),
    R("dataset_role", "transfer_target", 16, "Sec. 3.4", "used to evaluate the transferability of different instance segmentation models", item="dataset2"),
    NR("dataset_public", item="dataset2", note="third-party dataset; availability not stated"),
    R("n_images", "1078", 16, "Sec. 3.4", "a total of 1078 image tiles (containing 9883 date palm trees) were selected", item="dataset2", unit="images"),
    R("n_trees", "9883", 16, "Sec. 3.4", "1078 image tiles (containing 9883 date palm trees)", item="dataset2", unit="trees"),
]
# model rows from Table 1 (testing columns): name, backbone, category, box mAP, box F, seg mAP, seg F, inference column
TAB = [
    ("Mask R-CNN", "ResNet-101-FPN", "cnn", "88.94", "0.902", "88.91", "0.9", "0.056"),
    ("Mask R-CNN", "ResNet-50-FPN", "cnn", "88.99", "0.9", "88.02", "0.89", "0.045"),
    ("Mask R-CNN", "Swin-tiny-FPN", "hierarchical_vit", "91.5", "0.936", "91.5", "0.94", "0.11"),
    ("Mask R-CNN", "Swin-small-FPN", "hierarchical_vit", "92.1", "0.94", "91.5", "0.94", "0.137"),
    ("Mask Scoring R-CNN", "ResNet-50-FPN", "cnn", "91.9", "0.937", "91.7", "0.93", "0.11"),
    ("SOLOv2", "ResNet-50", "cnn", None, None, "92.7", "0.95", "0.12"),
    ("SOLOv2", "ResNet-101", "cnn", None, None, "92.4", "0.95", "0.138"),
]
res = Counter("res")
for i, (arch, bb, cat, bm, bf, sm, sf, inf) in enumerate(TAB, start=1):
    item, lab = f"model{i}", f"{arch} {bb}"
    role = "transformer_baseline" if cat != "cnn" else "cnn_baseline"
    ROWS += model(item, f"{arch} with {bb} backbone", role, cat, bb, "imagenet", "full_fine_tuning", 19, T1, lab,
                  pre_page=17, pre_loc="Sec. 4.2.1", pre_ev="trained ... for 100,000 iterations and initialized with ImageNet weights",
                  strat_page=17, strat_loc="Sec. 4.2.1", strat_ev="trained on Titan RTX ... for 100,000 iterations and initialized with ImageNet weights")
    ROWS += [R("inference_speed", f"{inf} (column headed 'Inference time (fps)')", 19, T1, f"{lab} ... {inf}", item=item,
               note="values below 1 suggest seconds per image rather than frames per second; recorded as printed")]
    hl = bb == "Swin-small-FPN"
    if bm:
        ROWS += result(res(), "ap50", item, "dataset1", bm, 19, T1, f"Testing, {lab}: bounding box mAP {bm}", setting="test set; detection (box)", headline=hl)
        ROWS += result(res(), "f1", item, "dataset1", bf, 19, T1, f"Testing, {lab}: bounding box F-measure {bf}", unit="fraction", setting="test set; detection (box)")
    ROWS += result(res(), "ap50", item, "dataset1", sm, 19, T1, f"Testing, {lab}: segmentation mAP {sm}", setting="test set; segmentation (mask)")
    ROWS += result(res(), "f1", item, "dataset1", sf, 19, T1, f"Testing, {lab}: segmentation F-measure {sf}", unit="fraction", setting="test set; segmentation (mask)")
SWIN_S = next(f"model{i}" for i, t in enumerate(TAB, start=1) if t[1] == "Swin-small-FPN")
tr = Counter("tr")
for gsd, v in (("15", "85"), ("20", "73")):
    i = tr()
    ROWS += [R("transfer_source", "test set at the training resolution (5 cm GSD)", 16, "Sec. 3.3", "different datasets with various GSDs (i.e. 10, 15, 20, 25, and 30 cm) were generated from the testing dataset", item=i),
             R("transfer_target", f"same test set resampled to {gsd} cm GSD", 22, "Sec. 4.3", "achieved mAP50 values of 85% and 73% when the spatial resolution was decreased by a third (GSD = 15 cm) and a quarter (GSD = 20 cm)", item=i),
             R("shift_type", "resolution", 16, "Sec. 3.3", "different datasets with various GSDs ... were generated from the testing dataset", item=i),
             R("transfer_model", SWIN_S, 22, "Sec. 4.3", "The Mask R-CNN based on Swin-small-FPN", item=i),
             R("transfer_metric", "ap50", 22, "Sec. 4.3", "achieved mAP50 values of 85% and 73%", item=i, note="bounding box (detection level)"),
             R("value_in_domain", "92.1", 19, T1, "Testing, Mask R-CNN Swin-small-FPN: bounding box mAP 92.1", item=i, unit="%"),
             R("value_out_of_domain", v, 22, "Sec. 4.3", f"achieved mAP50 values of 85% and 73% ... (GSD = {gsd} cm)", item=i, unit="%")]
ROWS += avail()
ROWS += [D("notes", "Comparative study of 17 instance segmentation configurations (15 convolutional, 2 Swin). Following protocol 4.1 for tables with more than ten models, recorded: both Swin models; Mask R-CNN with ResNet-50-FPN and ResNet-101-FPN (same-framework counterparts of the Swin-FPN backbones); Mask Scoring R-CNN ResNet-50-FPN (best CNN on box mAP and box F-measure); SOLOv2 ResNet-50 and ResNet-101 (best CNN on mask mAP; tied best on mask F-measure). The other ten rows are in Table 1 (p.19). Figure 13 gives box and mask mAP50 for all models at 10 to 30 cm GSD (figure only). On the unseen Al Ain dataset (Sec. 4.4) all models reached mask mAP50 of 95.6% to 98.2%, without per-model values. Training times: Swin-small-FPN 35.7 h, Swin-tiny-FPN 24 h (Sec. 4.2.1).", "Sec. 4.2 to 4.4 (pp. 17 to 22)")]
