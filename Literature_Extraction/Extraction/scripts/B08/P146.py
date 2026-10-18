import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T3, T4 = "P146", "Table 1", "Table 3", "Table 4"
# PWDViTNet: a lightweight single-stage detector for early-stage pine wilt disease (PWD) trees in UAV RGB images: ShuffleNetV2 with a MobileViT-style ViT block (ShuffleViT), FPN, PAN, RFB and ECA. Table 1 is the main
# comparison (seven models: AP_PWD = mAP@0.5 of the single early-PWD class, size in MB, GFLOPs, FPS): every row is recorded (AP50, GFLOPs and FPS; the model size in MB has no field) and primary. The paper does not name
# the split of Tables 1 to 3 (it has training, validation and an independent test set of 400 uncut images), so split is not_stated there. Table 4 gives TP, FP and FN of PWDViTNet on the test set; the printed
# accuracy rate of 73.1 percent is the share of the 633 labels that were detected (463 of 633), that is recall. Table 3 (ablation) adds one non-primary row, the baseline with only the ViT block.
TH = Tags(task="health_classification", level="object", split="not_stated", agg="single")
TASK_EV = "this paper proposes a lightweight single-stage detection model with PWDViTNet for detecting early PWD-infected pine trees"
BOX_EV = "LabelImg software was used to label the early infected areas"
DS_EV = "7000 high-quality PWD images are finally filtered"
SPLIT_EV = "the dataset is divided into training and validation sets in the ratio of 8:2, containing 5600 and 1400 samples, respectively"
ROWS = [
    R("task", "health_classification", 1, "Abstract", TASK_EV, note="detection of pine trees with early-stage pine wilt disease (needles lose luster and turn yellow)"),
    R("output_unit", "tree_instance", 4, "Sec. 2.2.3", BOX_EV, note="boxes of early infected areas; the test set counts 633 labels of infected trees"),
    R("tree_setting", "natural_forest", 3, "Sec. 2.2.1", "Laoshan National Forest Park in Nanjing, Jiangsu Province", note="forest park; forest type is not described"),
    R("tree_taxa", "pine trees infected by pine wilt disease (species not named)", 1, "Abstract", TASK_EV),
    R("study_area", "Laoshan National Forest Park, Nanjing, Jiangsu Province, China", 3, "Sec. 2.2.1", "Laoshan National Forest Park in Nanjing, Jiangsu Province"),
    D("biome", "subtropical", "Nanjing, Jiangsu; not stated in the paper"),
    R("platform", "uav", 3, "Sec. 2.2.1", "using a DJI Mini 4 Pro UAV", note="altitudes 40 to 150 m, 20 missions in September"),
    R("sensors", "rgb", 3, "Sec. 2.2.1", "a CMOS camera with 48 effective megapixels, which provides high-resolution RGB images"),
    R("annotation_type", "bounding_box", 4, "Sec. 2.2.3", BOX_EV),
    R("dataset_name", "own UAV RGB images of early-stage PWD, Laoshan National Forest Park, Nanjing (1333 x 1000 pixel crops of 4032 x 3024 images)", 3, "Sec. 2.2.2", DS_EV, item="dataset1"),
    R("dataset_key", "own_p146_laoshan_pwd", 3, "Sec. 2.2.2", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 2.2.3", SPLIT_EV, item="dataset1", note="5600 training and 1400 validation images plus an independent test set of 400 uncut original images"),
    R("dataset_public", "on_request", 15, "Data availability", "Data will be made available on request", item="dataset1"),
    R("n_images", "7000", 3, "Sec. 2.2.2", DS_EV, item="dataset1", unit="images", note="1333 x 1000 pixel images from more than 10,000 UAV images of 4032 x 3024 pixels (3 x 3 split); the independent test set has another 400 uncut images"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="AdamW, initial learning rate 0.01, decay 0.0005, batch size 4, 300 epochs on an RTX 3060 (Sec. 3.1); initialisation not stated")
M = [  # row regex, name, role, category, backbone, category note
    (r"Faster R-CNN", "Faster R-CNN", "cnn_baseline", "cnn", "Faster R-CNN", "convolutional two-stage detector"),
    (r"YOLOv5", "YOLOv5", "cnn_baseline", "cnn", "YOLOv5", "convolutional YOLO detector"),
    (r"YOLOv7-tiny", "YOLOv7-tiny", "cnn_baseline", "cnn", "YOLOv7-tiny", "convolutional YOLO detector"),
    (r"YOLOv8", "YOLOv8", "cnn_baseline", "cnn", "YOLOv8", "convolutional YOLO detector"),
    (r"ShuffleNetv2", "ShuffleNetv2 (baseline of PWDViTNet)", "cnn_baseline", "cnn", "ShuffleNetV2", "lightweight CNN; the baseline network that PWDViTNet is built on"),
    (r"MobileNetv3", "MobileNetv3", "cnn_baseline", "cnn", "MobileNetV3", "lightweight CNN"),
    (r"PWDViTNet \(Ours\)", "PWDViTNet (Ours)", "proposed", "cnn_transformer_hybrid", "ShuffleNetV2 fused with a ViT structure of MobileViT (ShuffleViT), FPN, PAN, RFB and ECA attention",
     "lightweight CNN backbone with ViT blocks added (protocol 4.5: hybrid); ECA is a generic channel attention module"),
]
res = Counter("res")
for k, (rx, name, role, cat, bb, cnote) in enumerate(M, start=1):
    item = f"model{k}"
    nums, line = row(PID, 10, rf"^\s*{rx}\s+(?=\d)", 0, cols=(0, 84))
    assert len(nums) == 4, (name, nums)  # AP_PWD, size (MB), GFLOPs, FPS
    ROWS += model(item, name, role, cat, bb, None, None, 10, T1, line, cat_note=cnote, **NP)
    ROWS += [R("flops", nums[2], 10, T1, line, item=item, unit="GFLOPs"),
             R("inference_speed", f"{nums[3]} FPS (experimental PC: NVIDIA RTX 3060; input size not stated)", 10, T1, line, item=item)]
    ROWS += result(res(), "ap50", item, "dataset1", nums[0], 10, T1, line, unit="%", setting="Table 1: early-stage PWD class; split not stated", note="AP_PWD, the mAP@0.5 of the early PWD class; model size in MB is not recorded", **TH, primary=True, headline=(rx == r"PWDViTNet \(Ours\)"))
# ablation row of Table 3: the baseline with only the ViT block (not primary)
nums, line = row(PID, 11, r"^\s*Baseline \+ ShuffleViT\s+(?=\d)", 0, cols=(0, 84))
assert nums[0] == "71.7", nums
ROWS += model("model8", "ShuffleNetv2 with ViT block only (ablation: Baseline + ShuffleViT)", "other_baseline", "cnn_transformer_hybrid", "ShuffleNetV2 with the ShuffleViT block, without FPN, PAN, RFB and ECA",
              None, None, 11, T3, line, cat_note="ablation variant: lightweight CNN with the ViT block (protocol 4.5: hybrid)", **NP)
ROWS += [R("flops", nums[2], 11, T3, line, item="model8", unit="GFLOPs"), R("inference_speed", f"{nums[3]} FPS (experimental PC: NVIDIA RTX 3060; input size not stated)", 11, T3, line, item="model8")]
ROWS += result(res(), "ap50", "model8", "dataset1", nums[0], 11, T3, line, unit="%", setting="Table 3: ablation, baseline with the ViT block only; split not stated", note="AP_PWD of the ablation row; model size in MB is not recorded", **TH)
# test set (400 uncut images, 633 labels): the accuracy rate of 73.1 percent is the share of labels detected (recall)
TT = TH(split="test")
ROWS += result(res(), "recall", "model7", "dataset1", "73.1", 11, T4, "resulting in an accuracy rate of 73.1 %", unit="%", setting="Table 4: independent test set of 400 uncut images (633 labels); TP 463, FP 94, FN 170",
               note="printed as accuracy rate; it equals the share of the 633 labels that were detected (463 of 633), that is recall by the paper's formula TP/(TP+FN); precision is not printed", **TT)
ROWS += [R("limitations", "Many RGB images are needed, early symptoms look like mid-stage symptoms (false detections), and drought, nutrient deficiency or other pests cause similar discoloration", 12, "Sec. 4",
           "the proposed method requires many RGB images to train the model, and the symptoms of early PWD infection are visually like those in the mid-stage, which may lead to false detection")]
ROWS += avail(code="no", weights="not_stated", data="on_request", data_page=15, data_loc="Data availability", data_ev="Data will be made available on request")
ROWS += [D("notes", "Table 1 compares Faster R-CNN, YOLOv5, YOLOv7-tiny, YOLOv8, ShuffleNetv2, MobileNetv3 and PWDViTNet (AP_PWD, size in MB, GFLOPs, FPS); PWDViTNet has 72.6, Faster R-CNN is highest with 73.3. All rows are recorded (AP50, GFLOPs, FPS) and primary, the headline is PWDViTNet. The split of Tables 1 to 3 is not named; the paper has an 8:2 training and validation split (5600 and 1400 images) and an independent test set of 400 uncut images, on which only PWDViTNet is scored (Table 4: TP 463, FP 94, FN 170 of 633 labels). The printed accuracy rate of 73.1 percent equals 463 of 633 and is recorded as recall. Table 3 (ablation, the baseline ShuffleNetv2 at 68.5 gains 3.2 points from the ViT block alone) is recorded for the ViT-only row; Table 2 (attention modules GAM, CBAM, SimAM, ECA) and the other ablation rows are not recorded. The model size in MB (Table 1) has no field and is not recorded.",
           "Sec. 3 and Tables 1 to 4 (pp. 10 to 11)")]
