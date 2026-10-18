import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T4, T6 = "P124", "Table 3", "Table 4", "Table 6"
# Light-ViTeYOLO (YOLOv5 with an EfficientViT backbone with lightweight multi-scale attention, the CACSNet neck and the EIoU loss) detects pine wilt disease trees (one diseased class) in UAV RGB patches. Table 3 is the main
# comparison (seven models, recall, mAP@0.5, mAP@0.5:0.95; primary), Table 4 gives parameters, GFLOPs and FPS. Table 6 compares backbones inside YOLOv5 (transformer and lightweight CNN backbones) and is not primary.
# The patches of the 1200-patch validation set are the evaluation data. Recall is not recorded because AP is given.
TH = Tags(task="health_classification", level="object", split="validation", agg="single")
OBJ_EV = "this paper proposes Light-ViTeYOLO, a lightweight pine wilt disease detection method based on Vision Transformer-enhanced YOLO"
DS_EV = "The image data for this study were obtained from the forest field of Zhuanshanzi,"
ROWS = [
    R("task", "health_classification", 1, "Abstract", OBJ_EV, note="detection of trees with pine wilt disease (one diseased class)"),
    R("output_unit", "tree_instance", 9, "Sec. 4.1", "annotate the diseased trees in the images under the guidance of forestry experts", note="boxes of diseased trees"),
    R("tree_setting", "natural_forest", 8, "Sec. 4.1", DS_EV, note="forest type is not described"),
    R("tree_taxa", "pine trees infected by pine wilt disease (species not named)", 1, "Abstract", "Pine wilt disease (PWD) is a forest disease characterized by rapid spread and extremely high lethality"),
    R("study_area", "forest field of Zhuanshanzi, Tai'an City, Shandong Province, China", 8, "Sec. 4.1", "Tai'an City, Shandong Province, China", note="the paper prints latitude 31 14 N, longitude 117 01 E"),
    D("biome", "temperate", "Shandong, China; coded from the study area"),
    D("platform", "uav", "DJI Mavic Air 2 drone, flight altitude about 300 m (Sec. 4.1)"),
    R("sensors", "rgb", 8, "Sec. 4.1", "We utilized a DJI Mavic Air 2 drone, which is outfitted with a 48-megapixel visible", note="48-megapixel visible light camera"),
    R("annotation_type", "bounding_box", 9, "Sec. 4.1", "In order to perform image analysis and applications, we use the image annotation tool LabelImg and", note="boxes of diseased trees drawn under the guidance of forestry experts"),
    R("dataset_name", "own UAV RGB patches (640 x 640 pixels) of pine wilt disease trees, Zhuanshanzi forest field (300 drone images)", 9, "Sec. 4.1", "we opted to extract image patches of 640 × 640 pixels from 300 drone forestry images", item="dataset1"),
    R("dataset_key", "own_p124_zhuanshanzi_pwd", 9, "Sec. 4.1", "we opted to extract image patches of 640 × 640 pixels from 300 drone forestry images", item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 4.1", "training set, while 1200 image patches were utilized for the validation set", item="dataset1", note="10000 training and 1200 validation patches from 300 images"),
    D("dataset_public", "on_request", "Data Availability Statement (p.15): available on request", item="dataset1"),
    D("n_images", "11200", "sum of the 10000 training patches and the 1200 validation patches of 640 x 640 pixels (Sec. 4.1), cut from 300 UAV images of 6000 x 4000 pixels", item="dataset1", unit="images"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="training settings not stated in the text")
ROWS += model("model1", "Light-ViTeYOLO (EfficientViT-YOLOv5 with CACSNet neck and EIoU loss, proposed)", "proposed", "cnn_transformer_hybrid", "YOLOv5 with an EfficientViT backbone (MBConv and lightweight multi-scale attention), the CACSNet content-aware cross-scale bidirectional fusion neck and the EIoU loss",
              None, None, 6, "Sec. 3.1", "In this paper, the above-designed EfficientViT model replaces the feature extraction network", cat_note="convolutional blocks with lightweight multi-scale attention (EfficientViT) in the YOLOv5 backbone", **NPRE)
BASE = (("model2", "Faster R-CNN", "cnn_baseline", "cnn", "Faster-RCNN 75.4 75.2 66.2", "Faster-RCNN 41.1 78.1 15.5", "75.4", "75.2", "66.2", "41.1", "78.1", "15.5"),
        ("model3", "RetinaNet", "cnn_baseline", "cnn", "RetinaNET 96.6 95.9 92.5", "RetinaNET 36.1 81.6 12.3", "96.6", "95.9", "92.5", "36.1", "81.6", "12.3"),
        ("model4", "YOLOv6", "cnn_baseline", "cnn", "YOLOV6 96.7 95.9 80.8", "YOLOV6 17.1 21.8 26.3", "96.7", "95.9", "80.8", "17.1", "21.8", "26.3"),
        ("model5", "YOLOv7", "cnn_baseline", "cnn", "YOLOv7 93.9 82.5 55.9", "YOLOv7 6.5 13.9 39.5", "93.9", "82.5", "55.9", "6.5", "13.9", "39.5"),
        ("model6", "YOLOX", "cnn_baseline", "cnn", "YOLOX 96.9 96.0 84.3", "YOLOX 8.9 13.3 46.5", "96.9", "96.0", "84.3", "8.9", "13.3", "46.5"),
        ("model7", "YOLOv5", "cnn_baseline", "cnn", "YOLOv5 96.1 97.6 90.8", "YOLOv5 7.1 15.8 67.0", "96.1", "97.6", "90.8", "7.1", "15.8", "67.0"))
for item, name, role, cat, ev3, ev4, rec, m50, m5095, prm, fl, fps in BASE:
    ROWS += model(item, name, role, cat, name + " detector", None, None, 11, T3, ev3, cat_note="convolutional detector (the paper compares it as a baseline algorithm)", **NPRE)
res = Counter("res")
RID = {}
ev_l3, ev_l4 = "Light-ViTeYOLO 95.7 97.2 94.3", "Light-ViTeYOLO 3.89 7.4 57.9"
ST3 = "Table 3: validation patches (1200), one diseased class"
RID["m50"] = res()
ROWS += result(RID["m50"], "ap50", "model1", "dataset1", "97.2", 11, T3, ev_l3, unit="%", setting=ST3, note="mAP@0.5", **TH, primary=True)
ROWS += result(res(), "ap50_95", "model1", "dataset1", "94.3", 11, T3, ev_l3, unit="%", setting=ST3, note="mAP@0.5:0.95", **TH, primary=True, headline=True)
ROWS += [R("total_params", "3.89", 11, T4, ev_l4, item="model1", unit="M", note="the abstract prints 3.89 MFLOPs, which is the parameter count (Table 4)"), R("flops", "7.4", 11, T4, ev_l4, item="model1", unit="GFLOPs"),
         R("inference_speed", "57.9 FPS (Tesla V100 server of Table 2; input size not stated)", 11, T4, ev_l4, item="model1")]
RID["yolo5_params"] = None
for item, name, role, cat, ev3, ev4, rec, m50, m5095, prm, fl, fps in BASE:
    ROWS += result(res(), "ap50", item, "dataset1", m50, 11, T3, ev3, unit="%", setting=ST3, note="mAP@0.5", **TH, primary=True)
    ROWS += result(res(), "ap50_95", item, "dataset1", m5095, 11, T3, ev3, unit="%", setting=ST3, note="mAP@0.5:0.95", **TH, primary=True)
    ROWS += [R("total_params", prm, 11, T4, ev4, item=item, unit="M"), R("flops", fl, 11, T4, ev4, item=item, unit="GFLOPs"), R("inference_speed", f"{fps} FPS (Tesla V100 server of Table 2; input size not stated)", 11, T4, ev4, item=item)]
# Table 6: backbones in YOLOv5 (AP@0.5, AP@0.5:0.95, parameters, GFLOPs); not primary
T6ROWS = (("YOLOv5 + ViT", "cnn_transformer_hybrid", "transformer", "YOLOv5 +ViT 7.02 15.6 97.2 95.8", "7.02", "15.6", "97.2", "95.8"),
          ("YOLOv5 + BoTNet", "cnn_transformer_hybrid", "transformer", "YOLOv5 +BoTNet 6.69 15.5 97.2 95.9", "6.69", "15.5", "97.2", "95.9"),
          ("YOLOv5 + CoNet", "cnn_transformer_hybrid", "transformer", "YOLOv5 +CoNet 8.19 16.8 97.2 95.4", "8.19", "16.8", "97.2", "95.4"),
          ("YOLOv5 + ShuffleNetV2", "cnn", "lightweight", "YOLOv5 +Shufflenetv2 3.79 7.9 96.07 87.87", "3.79", "7.9", "96.07", "87.87"),
          ("YOLOv5 + MobileNetV3", "cnn", "lightweight", "YOLOv5 +Mobilenetv3 3.19 5.9 94.19 86.83", "3.19", "5.9", "94.19", "86.83"),
          ("YOLOv5 + RepVGG", "cnn", "lightweight", "YOLOv5 +RepVGG 7.19 16.3 97.13 85.93", "7.19", "16.3", "97.13", "85.93"),
          ("YOLOv5 + GhostNet", "cnn", "lightweight", "YOLOv5 +GhostNet 3.68 8.1 96.49 86.06", "3.68", "8.1", "96.49", "86.06"),
          ("YOLOv5 + EfficientViT", "cnn_transformer_hybrid", "efficientvit", "YOLOv5 +EfficientViT 3.74 6.8 97.22 93.6", "3.74", "6.8", "97.22", "93.6"))
for k, (name, cat, kind, ev, prm, fl, ap50, ap5095) in enumerate(T6ROWS):
    item = f"model{8 + k}"
    cnote = {"transformer": "vision transformer type backbone inside YOLOv5 (the paper calls ViT, BoTNet and CoNet typical ViT models)", "lightweight": "classic lightweight convolutional feature extraction network inside YOLOv5",
             "efficientvit": "EfficientViT backbone only (before the CACSNet neck and the EIoU loss)"}[kind]
    ROWS += model(item, name, "transformer_baseline" if kind == "transformer" else ("other_baseline" if kind == "lightweight" else "other_baseline"), cat, name.replace("YOLOv5 + ", "") + " feature extraction network in YOLOv5", None, None, 13, T6, ev, cat_note=cnote, **NPRE)
    ROWS += [R("total_params", prm, 13, T6, ev, item=item, unit="M"), R("flops", fl, 13, T6, ev, item=item, unit="GFLOPs")]
    ROWS += result(res(), "ap50", item, "dataset1", ap50, 13, T6, ev, unit="%", setting="Table 6: backbone comparison in YOLOv5, validation patches", note="AP@0.5", **TH)
    ROWS += result(res(), "ap50_95", item, "dataset1", ap5095, 13, T6, ev, unit="%", setting="Table 6: backbone comparison in YOLOv5, validation patches", note="AP@0.5:0.95", **TH)
ROWS += conflict("cf1", "model7.total_params", "7.02", 12, "Table 5", "baseline 7.02 15.8 97.6 90.8", "V1b",
                 "Table 4 (the cost table of the main comparison) gives 7.1 M parameters for YOLOv5 with 15.8 GFLOPs and 67.0 FPS; the ablation Table 5 gives 7.02 M", unit="M")
ROWS += conflict("cf2", f"{RID['m50']}.value", "97.27", 12, "Table 5", "+EIOU 3.89 7.4 97.27 94.3", "V1a",
                 "Table 3, the main comparison, gives mAP@0.5 97.2 for Light-ViTeYOLO; the ablation Table 5 gives 97.27", unit="%")
ROWS += [R("limitations", "The method was only tested on a standard GPU platform and not deployed on a drone, and recall and mAP@0.5 are slightly below those of the best baselines", 15, "Sec. 5", "The method proposed in this paper has been experimentally verified on a standard"),
         D("code_available", "no", "No code repository or availability statement for code in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "on_request", 15, "Data Availability Statement", "Data Availability Statement: The data presented in this study are available on request from the")]
ROWS += [D("notes", "Table 3 compares Light-ViTeYOLO with Faster R-CNN, RetinaNet, YOLOv6, YOLOv7, YOLOX and YOLOv5 (recall, mAP@0.5, mAP@0.5:0.95) and Table 4 gives parameters, GFLOPs and FPS; mAP@0.5 and mAP@0.5:0.95 are recorded as primary (recall is not recorded because AP is given). Light-ViTeYOLO has the best mAP@0.5:0.95 of Table 3 and the fewest parameters and GFLOPs, but a lower mAP@0.5 than YOLOv5 (97.2 against 97.6) and a lower recall than RetinaNet, YOLOv6 and YOLOX. Table 6 (YOLOv5 with ViT, BoTNet, CoNet, ShuffleNetV2, MobileNetV3, RepVGG, GhostNet or EfficientViT) is recorded without primary; the ViT, BoTNet and CoNet variants reach a higher mAP@0.5:0.95 (95.8, 95.9, 95.4) than Light-ViTeYOLO (94.3) at about twice the parameters. Table 5 (ablation of EfficientViT, CACSNet and EIoU) is not recorded except as cf items. The evaluation set is the 1200-patch validation set; the paper has no separate test set. The abstract prints '3.89 MFLOPs' for the parameter count.",
           "Sec. 4 and Tables 3 to 6 (pp. 11 to 13)")]
