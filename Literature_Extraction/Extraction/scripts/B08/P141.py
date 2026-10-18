import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T8, T5, T9, T10 = "P141", "Table 8", "Table 5", "Table 9", "Table 10"
# FCL-RTDETR: RT-DETR with a FasterNet backbone, a parameter-free SPA block in the fusion module and the MPDIoU loss, trained with FGSM adversarial examples mixed into the data, for infected pines in UAV RGB images of Sanmen
# County. Table 8 is the main comparison (11 models, mAP50, parameters, FLOPs, FPS): every row is recorded (mAP50 only, precision and recall are not recorded because AP is given) and primary; the headline is FCL-RTDETR.
# The paper does not say which split Table 8 uses, so split is not_stated (the P, R and mAP50 of FCL-RTDETR equal the Sanmen test-plot row of Table 10, 199 images and 755 trees, which suggests the test plot of Table 1).
# Table 5 (RT-DETR with other backbones, the r18 row repeats the RT-DETR of Table 8), Table 9 (hybrid training) and Table 10 (external validation in Longyou County) are recorded without primary. Tables 4, 6 and 7 (ablations of
# the modules, attention variants and IoU losses of the proposed model) and the FGSM robustness curves are not recorded.
TH = Tags(task="health_classification", level="object", split="not_stated", agg="single")
TASK_EV = "we propose FCL-RTDETR, a robust real-time detection framework integrating Transformer-based with adversarial hybrid training"
BOX_EV = "For each infected tree in the image, a corresponding rectangular bounding box was drawn"
DS_EV = "resulting in a total of 1542 images with a pixel size of 300"
EXT_EV = "External validation utilized 419 acquired UAV images covering three representative sample sites: Guiguangyan Village, Xihaizhai Village, and Baiheqiao Village"
ROWS = [
    R("task", "health_classification", 1, "Abstract", TASK_EV, note="detection of pine wilt disease infected pines (discoloured trees) in UAV RGB images"),
    R("output_unit", "tree_instance", 3, "Sec. 2.1", BOX_EV),
    R("tree_setting", "natural_forest", 3, "Sec. 2.1", "The main types of pine trees are Masson pine (Pinus massoniana Lamb) and broad-leaved evergreen forests", note="mountainous county with 55.9 percent forest cover; forest type otherwise not described"),
    R("tree_taxa", "Masson pine (Pinus massoniana)", 3, "Sec. 2.1", "The main types of pine trees are Masson pine (Pinus massoniana Lamb) and broad-leaved evergreen forests"),
    R("study_area", "Sanmen County, Taizhou City, Zhejiang Province, China (training and test plots); external validation in Longyou County, Zhejiang", 2, "Sec. 2.1",
      "The study area is located in Sanmen County, Taizhou City, which is situated in the southeastern part of Zhejiang Province"),
    R("biome", "subtropical", 2, "Sec. 2.1", "The region has a typical subtropical monsoon climate"),
    R("platform", "uav", 3, "Sec. 2.1", "The experimental aerial platform for image collection is the Crossbeam CW-15", note="61 million pixel camera with 35 mm focal length; flights October 18 to 24, 2023"),
    R("sensors", "rgb", 3, "Sec. 2.1", "RGB images of sample plots in Zhennao, Tingbian, Hengdu, Huaqiao, and Pubaigang townships were acquired by researchers from October 18th to 24th, 2023"),
    R("annotation_type", "bounding_box", 3, "Sec. 2.1", "the sample was labeled using the open source labeling tool LabelImg", note="one rectangular box per infected tree, checked by on-site experts"),
    R("dataset_name", "own UAV RGB images of PWD-infected pines, Sanmen County (five sample plots, 300 x 300 slices)", 3, "Sec. 2.1", DS_EV, item="dataset1"),
    R("dataset_key", "own_p141_sanmen_pwd", 3, "Sec. 2.1", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Table 1", "Test set 5 755", item="dataset1", note="Table 1 splits by sample plot: plots 1, 2 and 4 for training (4793 trees), plot 3 for validation (990), plot 5 for testing (755)"),
    D("dataset_public", "on_request", "Data availability statement (p.16): Data will be made available on request", item="dataset1"),
    R("n_images", "1542", 3, "Sec. 2.1", DS_EV, item="dataset1", unit="images", note="300 x 300 pixel slices of five orthophotos, slices without infected trees removed"),
    R("n_trees", "6538", 4, "Table 1", "Total 1,2,3,4,5 6538", item="dataset1", unit="trees", note="infected pine trees"),
    R("dataset_name", "own UAV RGB images of PWD-infected pines, Xihaizhai and Baiheqiao villages, Longyou County (external validation)", 14, "Table 10", "Xihaizhai and Baiheqiao 300 819 76.8 74 80.6", item="dataset2"),
    R("dataset_key", "own_p141_longyou_xihaizhai_baiheqiao", 14, "Table 10", "Xihaizhai and Baiheqiao 300 819 76.8 74 80.6", item="dataset2"),
    R("dataset_role", "transfer_target", 13, "Sec. 3.10", EXT_EV, item="dataset2", note="independent dataset of another ecological region, used only for testing the model trained on Sanmen County"),
    D("dataset_public", "on_request", "Data availability statement (p.16): Data will be made available on request", item="dataset2"),
    R("n_images", "300", 14, "Table 10", "Xihaizhai and Baiheqiao 300 819 76.8 74 80.6", item="dataset2", unit="images"),
    R("n_trees", "819", 14, "Table 10", "Xihaizhai and Baiheqiao 300 819 76.8 74 80.6", item="dataset2", unit="trees", note="printed as instances"),
    R("dataset_name", "own UAV RGB images of PWD-infected pines, Guiguangyan village, Longyou County (external validation)", 14, "Table 10", "Guiguangyan village 119 224 89.2 74.7 87.5", item="dataset3"),
    R("dataset_key", "own_p141_longyou_guiguangyan", 14, "Table 10", "Guiguangyan village 119 224 89.2 74.7 87.5", item="dataset3"),
    R("dataset_role", "transfer_target", 13, "Sec. 3.10", EXT_EV, item="dataset3", note="independent dataset of another ecological region, used only for testing the model trained on Sanmen County"),
    D("dataset_public", "on_request", "Data availability statement (p.16): Data will be made available on request", item="dataset3"),
    R("n_images", "119", 14, "Table 10", "Guiguangyan village 119 224 89.2 74.7 87.5", item="dataset3", unit="images"),
    R("n_trees", "224", 14, "Table 10", "Guiguangyan village 119 224 89.2 74.7 87.5", item="dataset3", unit="trees", note="printed as instances"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="all models trained and validated on the same dataset; the FCL-RTDETR setup is 300 x 300 inputs, 150 epochs, batch size 32, learning rate 0.0001, Adam, weight decay 0.0005 (Sec. 2.4); initialisation not stated")
M8 = [  # row regex, name, role, category, backbone, category note
    (r"Faster R-CNN", "Faster R-CNN", "cnn_baseline", "cnn", "Faster R-CNN", "convolutional two-stage detector"),
    (r"SSD \(VGG-16\)", "SSD (VGG-16)", "cnn_baseline", "cnn", "VGG-16", "convolutional one-stage detector"),
    (r"YOLOv3-tiny", "YOLOv3-tiny", "cnn_baseline", "cnn", "YOLOv3-tiny", "convolutional detector"),
    (r"Yolov5s", "YOLOv5s", "cnn_baseline", "cnn", "YOLOv5s", "convolutional detector"),
    (r"Yolov5m", "YOLOv5m", "cnn_baseline", "cnn", "YOLOv5m", "convolutional detector"),
    (r"Yolov8s", "YOLOv8s", "cnn_baseline", "cnn", "YOLOv8s", "convolutional detector"),
    (r"Yolov8m", "YOLOv8m", "cnn_baseline", "cnn", "YOLOv8m", "convolutional detector"),
    (r"YOLOv11m", "YOLOv11m", "cnn_baseline", "cnn", "YOLOv11m", "convolutional detector"),
    (r"Swin-", "Swin-Transformer (detector, head not stated)", "transformer_baseline", "hierarchical_vit", "Swin Transformer", "named Swin-Transformer in Table 8 without a description of the detector (protocol 4.5: Swin is hierarchical_vit)"),
    (r"RT-DETR", "RT-DETR", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-18 (RTDETR-r18 in Table 5)", "RT-DETR with a CNN backbone and a transformer decoder (protocol 4.5)"),
    (r"FCL-RTDETR", "FCL-RTDETR (our)", "proposed", "cnn_transformer_hybrid", "FasterNet (partial convolution) backbone; RT-DETR hybrid encoder with parameter-free SPA blocks in the fusion module; MPDIoU loss",
     "RT-DETR (transformer decoder) with a lightweight CNN backbone"),
]
res = Counter("res")
RID = {}
for k, (rx, name, role, cat, bb, cnote) in enumerate(M8, start=1):
    item = f"model{k}"
    nums, line = row(PID, 11, rf"^\s*{rx}\s+(?=\d)", 0, cols=(87, 230))
    assert len(nums) == 6, (name, nums)  # mAP50, P, R, params (M), FLOPs (G), FPS
    ev = line + " ... Transformer" if rx == "Swin-" else line
    ROWS += model(item, name, role, cat, bb, None, None, 11, T8, ev, cat_note=cnote, **NP)
    ROWS += [R("total_params", nums[3], 11, T8, ev, item=item, unit="M"), R("flops", nums[4], 11, T8, ev, item=item, unit="GFLOPs"),
             R("inference_speed", f"{nums[5]} FPS (experimental environment: NVIDIA RTX 4090, 300 x 300 inputs)", 11, T8, ev, item=item)]
    RID[item] = res()
    ROWS += result(RID[item], "ap50", item, "dataset1", nums[0], 11, T8, ev, unit="%", setting="Table 8: Sanmen County dataset; split not stated", note="mAP50; precision and recall of the row are not recorded", **TH, primary=True, headline=(rx == "FCL-RTDETR"))
# Table 5: RT-DETR with other backbones (the RTDETR-r18 row repeats the RT-DETR of Table 8), not primary
BB = [(r"RTDETR-Repvit", "RT-DETR with RepViT backbone", "cnn_transformer_hybrid", "RepViT", "RT-DETR with a RepViT backbone; the paper lists structural reparameterisation and parameter sharing"),
      (r"RTDETR-Mobilenetv4", "RT-DETR with MobileNetV4 backbone", "cnn_transformer_hybrid", "MobileNetV4", "RT-DETR with a MobileNetV4 backbone"),
      (r"RTDETR-EfficientViT", "RT-DETR with EfficientViT backbone", "cnn_transformer_hybrid", "EfficientViT", "RT-DETR with an EfficientViT backbone (cascaded group attention); EfficientViT inside a CNN detector is hybrid (protocol 4.5)"),
      (r"RTDETR-", "RT-DETR with Swin Transformer backbone", "hierarchical_vit", "Swin Transformer", "RT-DETR with a Swin Transformer backbone (sliding window attention, hierarchical feature pyramid)"),
      (r"RTDETR-FasterNet", "RT-DETR with FasterNet backbone", "cnn_transformer_hybrid", "FasterNet", "RT-DETR with the FasterNet backbone only (the backbone of FCL-RTDETR without SPA and MPDIoU)")]
for j, (rx, name, cat, bb, cnote) in enumerate(BB):
    item = f"model{len(M8) + 1 + j}"
    nums, line = row(PID, 10, rf"^\s*{rx}\s+(?=\d)", 0, cols=(0, 120))
    assert len(nums) == 7, (name, nums)  # mAP50, F1, P, R, FLOPs (G), FPS, params (M)
    ev = line + " ... SwinTransformer" if rx == "RTDETR-" else line
    ROWS += model(item, name, "transformer_baseline" if cat == "hierarchical_vit" else "other_baseline", cat, bb, None, None, 10, T5, ev, cat_note=cnote, **NP)
    ROWS += [R("total_params", nums[6], 10, T5, ev, item=item, unit="M"), R("flops", nums[4], 10, T5, ev, item=item, unit="GFLOPs"),
             R("inference_speed", f"{nums[5]} FPS (experimental environment: NVIDIA RTX 4090, 300 x 300 inputs)", 10, T5, ev, item=item)]
    for metric, v in (("ap50", nums[0]), ("f1", nums[1])):
        ROWS += result(res(), metric, item, "dataset1", v, 10, T5, ev, unit="%", setting="Table 5: backbone ablation in RT-DETR, Sanmen County dataset; split not stated", note=("mAP50" if metric == "ap50" else "F1 score"), **TH)
# Table 9: after hybrid training with adversarial examples (clean samples), not primary
MIDX = {"Yolov5s": 4, "Yolov5m": 5, "Yolov8s": 6, "Yolov8m": 7, "RTDETR": 10, "FCL-RTDETR": 11}
for rx, idx in MIDX.items():
    nums, line = row(PID, 12, rf"^\s*{rx}\s+(?=\d)", 0, cols=(0, 78))
    assert len(nums) == 3, (rx, nums)  # mAP50, P, R
    ROWS += result(res(), "ap50", f"model{idx}", "dataset1", nums[0], 12, T9, line, unit="%", setting="Table 9: accuracy on clean samples after hybrid training with FGSM adversarial examples; split not stated", note="mAP50; precision and recall of the row are not recorded", **TH)
# Table 10: external validation in Longyou County (mAP50 of FCL-RTDETR trained on Sanmen County)
IN = "Our (Sanmen county) 199 755 83.6 82.6 91.2"
TR = Tags(task="health_classification", level="object", split="test", agg="single")
tr = Counter("tr")
for rx, lab, ds in ((r"Xihaizhai and Baiheqiao", "Xihaizhai and Baiheqiao villages", "dataset2"), (r"Guiguangyan village", "Guiguangyan village", "dataset3")):
    nums, line = row(PID, 14, rf"^\s*{rx}\s+(?=\d)", 0, cols=(0, 75))
    assert len(nums) == 5, (lab, nums)  # images, instances, P, R, mAP50
    ROWS += result(res(), "ap50", "model11", ds, nums[4], 14, T10, line, unit="%", setting=f"Table 10: cross-site test in {lab}, Longyou County (model trained on Sanmen County)", note="mAP50; precision and recall of the row are not recorded", **TR)
    t = tr()
    ROWS += [R("transfer_source", "Sanmen County (five sample plots; test plot 5)", 14, T10, IN, item=t),
             R("transfer_target", f"{lab}, Longyou County (independent dataset of another ecological region)", 13, "Sec. 3.10", EXT_EV, item=t),
             R("shift_type", "region", 13, "Sec. 3.10", EXT_EV, item=t, note="another county with other forest types, rainfall pattern and elevation (Sec. 3.10)"),
             R("transfer_model", "model11", 14, T10, line, item=t),
             R("transfer_metric", "ap50", 14, T10, line, item=t),
             R("value_in_domain", "91.2", 14, T10, IN, item=t, unit="%", note="mAP50 on the Sanmen test row (199 images, 755 trees), the same value as in Table 8"),
             R("value_out_of_domain", nums[4], 14, T10, line, item=t, unit="%"),
             R("transfer_level", "object", 14, T10, line, item=t)]
ROWS += [R("limitations", "The sample collection has a single data source and low flight altitudes, and background objects with the colour of diseased trees (brown mud or stone) cause false positives", 14, "Sec. 4",
           "existing sample collection has the limitations of single data source and low UAV flight altitudes")]
ROWS += conflict("cf1", "model10.total_params", "20.7", 10, "Table 5", "RTDETR-r18 83.8 77.9 77.7 76.2 56.9 63 20.7", "V1a",
                 "Table 8 is the main comparison table and prints 20.75 M for RT-DETR; the ablation Table 5 prints 20.7 M for the same model (RTDETR-r18), a difference of rounding", unit="M")
ROWS += conflict("cf2", "model10.total_params", "20.8", 10, "Sec. 3.6", "achieving a 47.8 % parameter reduction (from 20.8 MB to 10.8 MB)", "V1a",
                 "the text rounds the RT-DETR size to 20.8 (printed as MB) while Table 8, the main comparison table, prints 20.75 M", unit="M")
ROWS += avail(code="no", weights="not_stated", data="on_request", data_page=16, data_loc="Data availability", data_ev="Data will be made available on request")
ROWS += [D("notes", "Table 8 compares eleven detectors on the Sanmen County data (mAP50, parameters, FLOPs, FPS): FCL-RTDETR 91.2 against YOLOv11m 90.7 (best CNN), Faster R-CNN 88.9 and RT-DETR 83.8. All rows are recorded and primary; the abstract's headline is the 7.4 point gain of FCL-RTDETR over RT-DETR. Precision and recall are not recorded (AP is given); the recall of YOLOv3-tiny is printed as 715. The split of Table 8 is not stated: FCL-RTDETR has the same P, R and mAP50 as the Sanmen test row of Table 10 (199 images, 755 trees), so the test plot (Table 1) is likely, but split is recorded as not_stated. Table 5 (RT-DETR with RepViT, MobileNetV4, EfficientViT, Swin Transformer and FasterNet backbones, not primary), Table 9 (mAP50 of six models after hybrid training with FGSM adversarial examples, not primary) and Table 10 (FCL-RTDETR in two external Longyou County datasets, two transfer pairs, shift region) are recorded. The model is trained on Sanmen data. Tables 4, 6 and 7 and the adversarial robustness values (Fig. 10) are not recorded.",
           "Sec. 3 and Tables 1 and 4 to 10 (pp. 4 to 14)")]
