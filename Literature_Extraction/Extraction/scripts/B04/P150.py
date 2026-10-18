import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4, T6, T7, T9 = "P150", "Table 4", "Table 6", "Table 7", "Table 9"
# Single-step detection of healthy and unhealthy apple trees (boxes with a health class): health classification at object level, test set, results averaged over repeated runs.
T = Tags(task="health_classification", level="object", split="test", agg="mean")
BACKBONES = ("ResNet-50", "ResNet-101", "ResNeXt-101", "Swin Transformer")
BB_EV = "In this study, four backbones, i.e., ResNet-50, ResNet-101, ResNeXt-101, and Swin Transformer, were compared"
GSD_EV = "The ground sampling distance (GSD) of the image is 7 cm"
ROWS = [
    R("task", "detection;health_classification", 1, "Abstract", "a deep learning-based object detection single-step method for detecting tree health on multispectral UAV imagery"),
    R("output_unit", "tree_instance", 8, "Sec. 2.2.1", "draws bounding boxes around each tree and labels each box according to the tree’s health status (healthy or unhealthy)"),
    R("tree_setting", "orchard", 6, "Sec. 2.1", "The UAV multispectral images were acquired over an apple orchard in Souris, Prince Edward Island, Canada"),
    R("tree_taxa", "apple (18 cultivars)", 6, "Sec. 2.1", "The surveyed orchard included 18 different apple tree cultivars"),
    R("study_area", "Souris, Prince Edward Island, Canada", 6, "Sec. 2.1", "an apple orchard in Souris, Prince Edward Island, Canada"),
    D("biome", "temperate", "Prince Edward Island, Canada; not stated in the paper"),
    R("platform", "uav", 6, "Sec. 2.1", "The multispectral camera was mounted on a DJI Matrice 100 multirotor UAV platform"),
    R("sensors", "multispectral", 6, "Sec. 2.1", "MicaSense multispectral camera ... equipped with five sensors including Blue (central wavelength: 475 nm), Green (560 nm), Red (668 nm), Red-Edge (717 nm), and Near-Infrared (840 nm)"),
    R("gsd", "7", 6, "Sec. 2.1", GSD_EV, unit="cm"),
    R("annotation_type", "bounding_box", 8, "Sec. 2.2.1", "draws bounding boxes around each tree and labels each box according to the tree’s health status (healthy or unhealthy)"),
    R("dataset_name", "own UAV multispectral apple orchard dataset, Souris (orchards A to D, 1024 × 1024 patches)", 6, "Sec. 2.1", "The UAV multispectral images were acquired over an apple orchard in Souris, Prince Edward Island, Canada", item="dataset1"),
    R("dataset_key", "own_p150_souris_apple", 6, "Sec. 2.1", "The UAV multispectral images were acquired over an apple orchard in Souris, Prince Edward Island, Canada", item="dataset1"),
    R("dataset_role", "train_test", 10, "Sec. 2.2.1", "The test dataset used in the study was extracted from another portion of the UAV imagery to guarantee it is truly unseen", item="dataset1",
      note="about 70/15/15 training, validation and test split"),
    R("dataset_public", "on_request", 23, "Data Availability Statement", "Further inquiries can be directed to the corresponding author(s)", item="dataset1"),
    R("n_images", "4122", 10, "Sec. 2.2.1", "This process yielded a total of 4122 image patches", item="dataset1", unit="tiles", note="1024 × 1024 patches around each tree, about 71.7 m × 71.7 m"),
    R("n_trees", "4122", 10, "Sec. 2.2.1", "which included about 4122 individual apple trees", item="dataset1", unit="trees",
      note="approximately; 2752 healthy and 785 unhealthy trees in the training and validation sets, 527 healthy and 58 unhealthy trees in the test set"),
]
PRE = dict(pre_page=13, pre_loc="Sec. 2.2.2", pre_ev="The ImageNet conv1 weight tensor (64, 3, 7, 7) was replaced with (64, 5, 7, 7) so the backbone directly ingests 5 channels", pre_note="ImageNet-pretrained conv1 weights (RGB slices copied, the two extra bands start at zero)",
           strat_page=13, strat_loc="Sec. 2.2.2", strat_ev="During finetuning, the network learns band-specific filters for these channels", strat_note="fine-tuned with early stopping on the validation loss (Sec. 3)")
M = [  # item, backbone, name, role, category, T4 label, note
    ("model1", "ResNet-50", "Mask R-CNN with ResNet-50", "cnn_baseline", "cnn", r"ResNet-50", "mask branch removed (boxes and health class only); ResNet-50 is the original backbone"),
    ("model2", "ResNet-101", "Mask R-CNN with ResNet-101", "cnn_baseline", "cnn", r"ResNet-101", "mask branch removed"),
    ("model3", "ResNeXt-101", "Mask R-CNN with ResNeXt-101 (32 × 8d)", "proposed", "cnn", r"ResNeXt-101 \(32 × 8d\)", "mask branch removed; the best configuration of the paper"),
    ("model4", "Swin Transformer", "Mask R-CNN with Swin Transformer (Base)", "transformer_baseline", "hierarchical_vit", r"Swin Transformer \(Base\)", "mask branch removed; Swin Transformer backbone"),
]
res = Counter("res")
T4V = {}
for item, bb, name, role, cat, lab, note in M:
    nums, ev = row(PID, 12, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(nums) == 3, (item, nums)  # parameters with 3 bands, parameters with 5 bands, GFLOPs
    T4V[item] = (nums, ev)
    ROWS += model(item, name, role, cat, bb, "imagenet", "full_fine_tuning", 11, "Sec. 2.2.2", BB_EV, cat_note=note, **PRE)
    ROWS += [R("total_params", nums[1], 12, T4, ev, item=item, unit="M", note="5-band whole detector (Table 4 column 2)" + ("; the printed value equals the GFLOPs value beside it and the 3-band count is 88 M" if item == "model4" else "")),
             R("flops", nums[2], 12, T4, ev, item=item, unit="GFLOPs", note="per 1024 × 1024 pixel image")]
INPUTS = ("RGB", "Multispectral (5 bands)", "3PCs")
RES = {}
for tab, tl, cond, page in ((6, T6, "no class weights", 18), (7, T7, "class weights", 18)):
    for k, inp in enumerate(INPUTS):
        for item, bb, name, role, cat, lab, note in M:
            rx = {"ResNet-50": r"Resnet-50", "ResNet-101": r"Resnet-101", "ResNeXt-101": r"ResneXt-101", "Swin Transformer": r"Swin Transformer"}[bb]
            nums, ev = row(PID, page, rf"^\s*{rx}\s+(?=\d)", (tab - 6) * 3 + k)
            assert len(nums) == 4, (tab, inp, bb, nums)  # precision, recall, F1, mIoU
            prim = (tab == 6 and k == 1)
            st = f"{tl}: {inp}, {cond}; mean of repeated runs"
            RES[(tab, k, item)] = res()
            ROWS += result(RES[(tab, k, item)], "f1", item, "dataset1", nums[2], page, tl, ev, unit="%", setting=st, note="average F1-score over the test samples", **T, primary=prim,
                           headline=(tab == 7 and k == 1 and item == "model3"))
            ROWS += result(res(), "miou", item, "dataset1", nums[3], page, tl, ev, unit="%", setting=st, note="mean IoU of the predicted and annotated boxes", **T, primary=prim)
# Table 9: ResNeXt-101 on the 5-band images with class weights, cross entropy against focal loss
for lab, lname in (("Cross Entropy", "cross entropy"), ("Focal", "focal loss")):
    nums, ev = row(PID, 19, rf"^\s*{lab}\s+(?=\d)", 1)  # the second hit: the first is the accuracy line of Table 8
    assert len(nums) == 6, (lab, nums)  # AP healthy, AP unhealthy, mAP@50, average F1, macro F1, weighted F1
    st = f"Table 9: ResNeXt-101, 5 bands, class weights and {lname}"
    ROWS += result(res(), "ap50", "model3", "dataset1", nums[2], 19, T9, ev, unit="%", setting=st, note="mAP@50: mean of the healthy and unhealthy class APs at IoU 0.5", **T)
    if lab == "Focal":
        ROWS += result(res(), "f1", "model3", "dataset1", nums[3], 19, T9, ev, unit="%", setting=st, note="average F1-score over the test samples", **T)
    ROWS += result(res(), "f1", "model3", "dataset1", nums[4], 19, T9, ev, unit="%", setting=st, note="macro F1-score of the healthy and unhealthy classes (aggregated confusion matrix)", **T)
    ROWS += result(res(), "f1", "model3", "dataset1", nums[5], 19, T9, ev, unit="%", setting=st, note="weighted average F1-score of the two classes", **T)
ROWS += [R("limitations", "A single orchard imaged in 2018 (pilot study), so generalisation across orchards, seasons and species is untested", 22, "Sec. 4", "The dataset used in this study is limited to a single orchard in 2018")]
ROWS += avail(data="on_request", data_page=23, data_loc="Data Availability Statement", data_ev="Further inquiries can be directed to the corresponding author(s)")
ROWS += [D("notes", "Tables 6 and 7 give precision, recall, F1 and mIoU of Mask R-CNN (mask branch removed) with four backbones on RGB, 5-band and 3PCs inputs, without and with class weights; F1 and mIoU are recorded for all 24 configurations. The primary setting is Table 6 (no class weights) on the 5-band images, the headline is the best Table 7 result (ResNeXt-101, 5 bands, class weights). Table 9 compares cross entropy and focal loss for ResNeXt-101 (the cross-entropy average F1 equals Table 7). Text-only variants of the Swin Transformer (progressive layer-wise fine-tuning: AP 69.83, F1 32.70) and of SMOTE oversampling (p.21 to p.22) are not recorded.",
           "Sec. 3 and Tables 4 to 9 (pp. 12 to 22)")]
# Value choices (protocol 6.1)
ROWS += conflict("cf1", f"{RES[(7, 1, 'model3')]}.value", "85.68", 1, "Abstract", "an F1-score of 85.68% and a mean Intersection over Union (mIoU) of 92.85%", "V1a",
                 "Table 7, the results table, gives 85.70 for ResNeXt-101 on the 5-band images with class weights (the highlights on p.1 also say 85.70); the abstract and the conclusion give 85.68", unit="%")
for i, (item, bb, name, role, cat, lab, note) in enumerate(M, start=2):
    nums, ev = T4V[item]
    ROWS += conflict(f"cf{i}", f"{item}.total_params", nums[0], 12, T4, ev, "V1b",
                     "Table 4 gives a second parameter count for 3-band images (backbone-sized); the 5-band count is the whole detector that goes with the printed GFLOPs", unit="M")
