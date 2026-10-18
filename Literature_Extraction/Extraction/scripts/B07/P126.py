import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T6, T4 = "P126", "Table 6", "Table 4"
# Improved Faster R-CNN (Swin Transformer backbone with a feature pyramid, a balance feature enhancement module and dynamic training) detects standing dead trees (SDTs) in UAV RGB images. Table 6 is the main
# comparison (six models; mAP50, R50, F1, FPS) on the test split; its mAP50, F1 and FPS of the proposed model repeat Table 4. mAP50 and F1 are recorded (recall is not, because F1 and AP are given). Finding dead
# trees is mortality_detection. The ablation Table 5 gives an AP50 of 88.9 for the ResNet baseline against 88.7 in Table 6 (cf2).
TM = Tags(task="mortality_detection", level="object", split="test", agg="single")
OBJ_EV = "this paper proposes a multi-scale attention mechanism detection method for identifying SDTs in UAV RGB images"
SPLIT_EV = "training (70%), validation (10%), and testing (20%)"
ROWS = [
    R("task", "mortality_detection", 1, "Abstract", OBJ_EV, note="detection of standing dead trees (SDTs)"),
    R("output_unit", "tree_instance", 6, "Sec. 2.3", "The software Labelme was subsequently used to label the 3940 SDTs in the expanded", note="boxes of standing dead trees (the model is of the COCO type)"),
    D("tree_setting", "natural_forest", "forest of Changsha County; forest type is not described in the paper"),
    R("tree_taxa", "standing dead trees of broadleaved and coniferous trees (health status only)", 6, "Sec. 2.3", "two types: broadleaved and conifers, with only their health status"),
    R("study_area", "Changsha County, Changsha City, Hunan Province, China", 4, "Sec. 2.1", "The study area is located in Changsha County, Changsha City, Hunan Province in the People’s"),
    R("biome", "subtropical", 4, "Sec. 2.1", "It has a humid continental and subtropical monsoon climate"),
    R("platform", "uav", 4, "Sec. 2.2", "The DJI Matrice M300 RTK was used for data collection", note="flight height 150 m"),
    R("sensors", "rgb", 4, "Sec. 2.2", "with a DJI Zenmuse H20T multi-function digital camera", note="wide-angle visible light camera of the Zenmuse H20T"),
    R("annotation_type", "bounding_box", 6, "Sec. 2.3", "The software Labelme was subsequently used to label the 3940 SDTs in the expanded", note="marked on the orthophotos and checked in the field with the UAV"),
    R("dataset_name", "own UAV RGB images of standing dead trees in four test areas, Changsha County (SDTs dataset)", 6, "Sec. 2.3", "a total of 568 pictures representing 978 SDTs are finally finalized after an on-site inspection", item="dataset1"),
    R("dataset_key", "own_p126_sdts_changsha", 6, "Sec. 2.3", "a total of 568 pictures representing 978 SDTs are finally finalized after an on-site inspection", item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 2.3", SPLIT_EV, item="dataset1", note="70, 10 and 20 percent for training, validation and testing; the test data is not augmented"),
    R("dataset_public", "partial", 1, "Data Availability Statement", "we have released the minimum dataset that meets the requirements on GitHub", item="dataset1", note="a minimum dataset on GitHub (RY1213/SDTs)"),
    R("dataset_url", "https://github.com/RY1213/SDTs", 1, "Data Availability Statement", "we have released the minimum dataset that meets the requirements on GitHub", item="dataset1", note="the PDF text breaks the address after https://github.com/"),
    R("n_images", "568", 6, "Sec. 2.3", "a total of 568 pictures representing 978 SDTs are finally finalized after an on-site inspection", item="dataset1", unit="images", note="2031 images after augmentation"),
    R("n_trees", "978", 6, "Sec. 2.3", "a total of 568 pictures representing 978 SDTs are finally finalized after an on-site inspection", item="dataset1", unit="trees", note="3940 labels after augmentation (Table 2: four test areas with 131, 179, 81 and 587 labels)"),
]
NPRE = dict(pre_page=13, pre_loc="Sec. 4.1", pre_ev="The pre-trained network model is", pre_note="a pre-trained model imported by transfer learning; the pre-training data is not stated", strat_note="AdamW, learning rate 0.0001, batch size 4, 100 epochs with early stopping")
ROWS += model("model1", "Improved Faster R-CNN (Swin Transformer backbone, balance feature enhancement and dynamic training, proposed)", "proposed", "hierarchical_vit", "Swin Transformer with a feature pyramid, a balance feature enhancement module (Gaussian non-local attention) and Faster R-CNN heads",
              None, None, 1, "Abstract", "This method takes Faster-RCNN as the basic framework and uses Swin-Transformer as the backbone network for feature extraction", cat_note="Faster R-CNN with a Swin Transformer backbone and FPN", pre_status="not_reported", **{k: v for k, v in NPRE.items() if k != "pre_page"} | dict(pre_page=13))
BASE = (("model2", "Grid R-CNN", "cnn_baseline", "cnn", "Grid R-CNN", "Grid-RCNN [45] 94.6 90.3 92.4 12.1", "94.6", "92.4", "12.1", "Grid R-CNN adopts a position-sensitive, fully convolutional network to accurately position the target (Sec. 5.2)"),
        ("model3", "RetinaNet", "cnn_baseline", "cnn", "RetinaNet", "RetinaNet [46] 93.0 85.1 88.9 13.9", "93.0", "88.9", "13.9", "single-stage detector with a feature pyramid (Sec. 5.2)"),
        ("model4", "YOLOv3", "cnn_baseline", "cnn", "YOLOv3 with a Darknet backbone", "YOLO v3[47] 89.5 85.3 87.3 37.5", "89.5", "87.3", "37.5", "YOLOv3 with Darknet as the backbone network (Sec. 5.2)"),
        ("model5", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "DETR (Transformer with a CNN)", "DETR [48] 88.3 92.4 90.3 5.5", "88.3", "90.3", "5.5", "DETR is the first end-to-end algorithm combining Transformer with CNN (Sec. 5.1)"),
        ("model6", "Faster R-CNN (original)", "cnn_baseline", "cnn", "Faster R-CNN with a ResNet backbone", "Faster-RCNN 88.7 90.1 89.4 10.8", "88.7", "89.4", "10.8", "Faster R-CNN with a ResNet backbone (Table 5)"))
for item, name, role, cat, bb, ev, ap, f1, fps, cnote in BASE:
    ROWS += model(item, name, role, cat, bb, None, None, 17, T6, ev, cat_note=cnote, pre_status="not_reported", pre_note="not stated", strat_note="trained with SGD, learning rate 0.0001, batch size 4, 100 epochs and early stopping (Sec. 5.1)")
res = Counter("res")
RID = {}
ST = "Table 6: test set (20 percent, not augmented), standing dead trees"
EVO = "Improved Faster-RCNN(Ours) 95.9 92.1 93.9 10.4"
RID["ap"] = res()
ROWS += result(RID["ap"], "ap50", "model1", "dataset1", "95.9", 17, T6, EVO, unit="%", setting=ST, note="mAP at IoU 0.5 (the same value in Table 4)", **TM, primary=True, headline=True)
ROWS += result(res(), "f1", "model1", "dataset1", "93.9", 17, T6, EVO, unit="%", setting=ST, note="F1-score (the same value in Table 4)", **TM, primary=True)
ROWS += [R("total_params", "45.01", 14, T4, "95.9 92.1 39.9 93.9 10.4 45.01", item="model1", unit="M"), R("inference_speed", "10.4 FPS (images per second; hardware not stated)", 17, T6, EVO, item="model1")]
for item, name, role, cat, bb, ev, ap, f1, fps, cnote in BASE:
    RID[(item, "ap")] = res()
    ROWS += result(RID[(item, "ap")], "ap50", item, "dataset1", ap, 17, T6, ev, unit="%", setting=ST, note="mAP at IoU 0.5", **TM, primary=True)
    RID[item] = res()
    ROWS += result(RID[item], "f1", item, "dataset1", f1, 17, T6, ev, unit="%", setting=ST, note="F1-score", **TM, primary=True)
    ROWS.append(R("inference_speed", f"{fps} FPS (images per second; hardware not stated)", 17, T6, ev, item=item))
ROWS += conflict("cf1", "model6.inference_speed", "10.4", 17, "Sec. 5.1", "with the FPS being 10.4. Such a speed, the same as that of the original Faster-RCNN", "V1a",
                 "Table 6 gives 10.8 FPS for the original Faster R-CNN; the text of Sec. 5.1 says the proposed model's 10.4 FPS is the same as that of the original Faster-RCNN", unit="FPS")
ROWS += conflict("cf2", f"{RID[('model6', 'ap')]}.value", "88.9", 15, "Table 5", "88.9", "V1a", "Table 6, the main comparison, gives mAP50 88.7 for the original Faster R-CNN; the ablation Table 5 gives 88.9 for the ResNet baseline", unit="%")
ROWS += [R("limitations", "The detection speed is low (10.4 FPS against 37.5 FPS for YOLOv3 in Table 6)", 17, "Sec. 5.1",
           "However, the detection speed is slower, with the FPS being 10.4"),
         D("code_available", "no", "No code repository for the detection model in the paper (only the minimum dataset is on GitHub)"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "partial", 1, "Data Availability Statement", "we have released the minimum dataset that meets the requirements on GitHub")]
ROWS += [D("notes", "Table 6 compares Grid R-CNN, RetinaNet, YOLOv3, DETR, the original Faster R-CNN and the improved Faster R-CNN with a Swin Transformer backbone on the test split (mAP50, R50, F1, FPS); mAP50 and F1 are recorded (R50 is not, because AP and F1 are given). Table 4 repeats the results of the proposed model (mAP50 95.9, R50 92.1, mAPs 39.9, F1 93.9, FPS 10.4, 45.01 M parameters). Table 5 (ablation of Swin, the balance module and dynamic training; AP50 88.9, 93.9, 94.1, 92.3, 95.9) is not recorded except as cf2. The proposed model has the best mAP50 and F1 (1.3 and 1.5 points above Grid R-CNN) but DETR has the highest recall (92.4) and YOLOv3 the highest speed. The paper states a pre-trained network model was imported but not its pre-training data.",
           "Sec. 4 to 5 and Tables 4 to 6 (pp. 14 to 17)")]
