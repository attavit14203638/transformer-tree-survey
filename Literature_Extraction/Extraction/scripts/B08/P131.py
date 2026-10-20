import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T4, T5, T8, T9, T6 = "P131", "Table 1", "Table 4", "Table 5", "Table 8", "Table 9 (image)", "Table 6 (image)"
# Semantic segmentation of pine wilt disease (PWD) infected trees in UAV RGB images: U-Net, DeepLabv3+ and FPN (all with a ResNet-152 backbone) on three input schemes (RGB, HSV, RGBHSV), compared with a random forest and
# with a fine-tuned SAM (a transformer, only run with RGB and HSV because it takes three channels). No model is proposed; the paper's best is U-Net with RGBHSV (abstract). The RGB scheme is the only default input that all four
# deep models share, so it is the primary setting; HSV and RGBHSV rows and the random forest are not primary (the RGBHSV row of U-Net carries the headline). No split is described, so split is not_stated everywhere. Pixel
# metrics (F1 and IoU of Tables 4, 6 and 9, precision not recorded) are level pixel. The "success rate" of Tables 1, 5 and 8 is the share of the 561 reference PWD-infected tree sample points that the predictions found
# (success + fail = 561), so it is recall at object level and is recorded as such.
TP = Tags(task="health_classification", level="pixel", split="not_stated", agg="single")
TO = TP(level="object")
OBJ_EV = "utilized semantic segmentation techniques in deep learning to detect individual PWD-infected trees"
DS_EV = "a total of 561 PWD-infected tree sample points were identified and vectorized through manual annotation"
MODELS_EV = "this study employed and compared three advanced semantic segmentation models to identify the PWD-infected trees, including U-Net, FPN, and DeepLabV3+"


def ev_of(label, line):
    """Evidence of a table row: the printed line, prefixed with the model label (verified separately) when the line omits it."""
    return line if line.startswith(label) else f"{label} ... {line}"


ROWS = [
    R("task", "health_classification", 1, "Abstract", OBJ_EV, note="detection of pine wilt disease infected trees by semantic segmentation of UAV RGB images"),
    R("output_unit", "crown_mask", 4, "Sec. 2.2", "The annotated results were binarized in ArcGIS Pro to create label rasters", note="semantic masks of PWD-infected trees (not separated into instances); the predictions are checked against 561 reference tree points"),
    NR("tree_setting", note="forest farms and mountain forest of Qianshan City (Sec. 2.2); the paper does not say whether the stands are natural or planted"),
    R("tree_taxa", "pine trees infected by pine wilt disease (species not named)", 1, "Abstract", "Pine wilt disease (PWD) is a severe forest disease caused by the invasion of"),
    R("study_area", "Qianshan City, Anhui Province, China (Tianzhushan Forest Farm, Tianzhushan Town and part of Tuoling Forest Farm)", 3, "Sec. 2.1", "was selected as the study area, which is in the southwest Anhui Province"),
    D("biome", "subtropical", "Qianshan (30 to 31 N), Anhui; not stated in the paper"),
    R("platform", "uav", 4, "Sec. 2.2", "The DJI M300 series drone"),
    R("sensors", "rgb", 4, "Sec. 2.2", "Zenmuse-P1", note="Zenmuse-P1 RGB camera (45 million effective pixels); the HSV and RGBHSV schemes are derived from the RGB bands"),
    R("gsd", "0.1", 4, "Sec. 2.2 (image)", "all RGB images were resampled with a spatial resolution of 0.1 m", unit="m", note="after resampling; flying altitude 350 m, November 2021; the text layer of this page is garbled, read from the rendered page"),
    R("annotation_type", "polygon_mask;point", 5, "Sec. 2.2", DS_EV, note="561 tree sample points vectorized and binarized into label rasters (1 diseased tree, 0 others); the points are also the reference of the success rate"),
    R("dataset_name", "own UAV RGB images (DJI M300 with Zenmuse-P1) of PWD trees, Qianshan City (561 sample points, 512 x 512 slices)", 5, "Sec. 2.2", DS_EV, item="dataset1"),
    R("dataset_key", "own_p131_qianshan_pwd", 5, "Sec. 2.2", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.2", "we selected a total of 2050 slices containing diseased trees", item="dataset1", note="the split into training and validation data is not described; the success rate is computed on the prediction of all UAV images"),
    D("dataset_public", "no", "Data Availability Statement (p.20): Data are contained within the article; no data link", item="dataset1"),
    D("n_images", "3184", "sum of the 2050 slices containing diseased trees and the 1134 samples of other land cover types (Sec. 2.2, p.4); 512 x 512 pixel slices", item="dataset1", unit="images"),
    R("n_trees", "561", 5, "Sec. 2.2", DS_EV, item="dataset1", unit="trees", note="PWD-infected tree sample points"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="batch size 4, 50 iterations, learning rate 0.0001, DiceLoss and Adam (Sec. 3.2.4); initialisation not stated")
CNN = (("model1", "U-Net", "U-Net", "67.156881", "U-Net (3 channels) 67,156,881 trainable parameters, Table 3 (6 channels: 67,166,289)"),
       ("model2", "DeepLabv3+", "DeepLabv3+", "61.313361", "DeepLabv3+ (3 channels) 61,313,361 trainable parameters, Table 3 (6 channels: 61,322,769)"),
       ("model3", "FPN", "FPN", "60.751809", "FPN (3 channels) 60,751,809 trainable parameters, Table 3 (6 channels: 60,761,217)"))
for item, name, lab, prm, basis in CNN:
    ROWS += model(item, name, "cnn_baseline", "cnn", "ResNet-152 backbone network (all three semantic segmentation models)", None, None, 6, "Sec. 3.2", MODELS_EV, cat_note="convolutional semantic segmentation network (Sec. 3.2)", **NP)
    ROWS.append(D("trainable_params", prm, basis + "; printed as a full count, divided by 1e6", item=item, unit="M"))
ROWS += model("model4", "Random forest (RF)", "other_baseline", "classical", "random forest with 500 trees on the six RGBHSV features of random points (Google Earth Engine)", None, None, 9, "Sec. 3.3", "implemented the random forest model using the Google Earth Engine",
              cat_note="pixel-wise random forest classifier", pre_status="not_reported", pre_note="not applicable", strat_note="trained on random PWD-infected and non-diseased tree points; the split is not stated")
ROWS += model("model5", "SAM (fine-tuned segment anything model)", "transformer_baseline", "foundation_model", "SAM of Meta with its image encoder and prompt encoder frozen; bounding boxes as prompts in training, random points at inference", "sa1b", "head_only", 9, "Sec. 3.4",
              "The segment anything model (SAM) is a semantic segmentation model developed by Meta", cat_note="segment anything model (protocol 4.5: foundation_model)", pre_page=10, pre_loc="Sec. 3.4", pre_ev="trained on a dataset of 11 million images and 1.1 billion masks",
              pre_note="pre-trained by Meta on SA-1B", strat_page=10, strat_loc="Sec. 3.4", strat_ev="we froze all VisionEncoder and PromptEncoder layers", strat_note="only the remaining (mask decoder) layers are trained; fine-tuned on RGB and on HSV images, not on RGBHSV (three-channel input only)")
ROWS.append(D("trainable_params", "93.735472", "SAM 93,735,472 trainable parameters, Table 7 (printed although the encoders are frozen); printed as a full count, divided by 1e6", item="model5", unit="M"))

res = Counter("res")
RID = {}
SCHEMES = ("RGB", "HSV", "RGBHSV")
RX4 = r"^\s*(?:U-Net\s+|DeepLabv3\+\s+|FPN\s+)?(?:RGBHSV|RGB|HSV)\s+(?=0\.\d{4}\s+0\.\d{4}\s+0\.\d{4}\s*$)"
RX1 = r"^\s*(?:U-Net\s+|DeepLabv3\+\s+|FPN\s+)?(?:RGBHSV|RGB|HSV)\s+(?=\d+\s+\d+\.\d+%\s+\d+\s+\d+\.\d+%\s*$)"
# Table 4 (pixel metrics) and Table 1 (object-level success rate) of the three CNNs: nine rows each, in the printed order
for k, (item, name, lab, prm, basis) in enumerate(CNN):
    for j, sch in enumerate(SCHEMES):
        n = 3 * k + j
        prim = sch == "RGB"
        nums, line = row(PID, 12, RX4, n)
        assert len(nums) == 3, (item, sch, nums)
        ev = ev_of(lab, line)
        st = f"Table 4: {name}, {sch} feature scheme; split not stated"
        head = item == "model1" and sch == "RGBHSV"
        RID[(item, sch, "f1")] = res()
        ROWS += result(RID[(item, sch, "f1")], "f1", item, "dataset1", nums[0], 12, T4, ev, unit="fraction", setting=st, note="F1-score of the PWD-infected tree class", **TP, primary=prim, headline=head)
        ROWS += result(res(), "iou", item, "dataset1", nums[1], 12, T4, ev, unit="fraction", setting=st, note="IoU of the PWD-infected tree class", **TP, primary=prim)
        nums1, line1 = row(PID, 11, RX1, n)
        assert len(nums1) == 4 and int(nums1[0]) + int(nums1[2]) == 561, (item, sch, nums1)
        ev1 = ev_of(lab, line1)
        ROWS += result(res(), "recall", item, "dataset1", nums1[1], 11, T1, ev1, unit="%", setting=f"Table 1: {name}, {sch} feature scheme; prediction of all UAV images against the 561 reference points",
                       note="success rate: PWD-infected tree sample points identified, of 561 (success plus fail); recall at object level", **TO, primary=prim)
# SAM (Tables 8 and 9) and the random forest (Tables 5 and 6)
for sch, ev9, f1, iou in (("RGB", "SAM RGB 0.7932 0.7153 0.6824", "0.7932", "0.7153"), ("HSV", "SAM HSV 0.6853 0.6201 0.5942", "0.6853", "0.6201")):
    st = f"Table 9: SAM, {sch} feature scheme; split not stated"
    ROWS += result(res(), "f1", "model5", "dataset1", f1, 16, T9, ev9, unit="fraction", setting=st, note="F1-score of the PWD-infected tree class", **TP, primary=(sch == "RGB"))
    ROWS += result(res(), "iou", "model5", "dataset1", iou, 16, T9, ev9, unit="fraction", setting=st, note="IoU of the PWD-infected tree class", **TP, primary=(sch == "RGB"))
nums, line = row(PID, 15, r"^\s*SAM\s+RGB\s+(?=\d)", 0)
assert nums == ["458", "81.64", "103", "18.36"], nums
ROWS += result(res(), "recall", "model5", "dataset1", nums[1], 15, T8, line, unit="%", setting="Table 8: SAM, RGB feature scheme; prediction of all UAV images against the 561 reference points",
               note="success rate: PWD-infected tree sample points identified, of 561 (success plus fail); recall at object level", **TO, primary=True)
nums, line = row(PID, 15, r"^\s*HSV\s+(?=402)", 0)
assert nums == ["402", "71.66", "159", "28.34"], nums
ROWS += result(res(), "recall", "model5", "dataset1", nums[1], 15, T8, "SAM ... " + line, unit="%", setting="Table 8: SAM, HSV feature scheme; prediction of all UAV images against the 561 reference points",
               note="success rate: PWD-infected tree sample points identified, of 561 (success plus fail); recall at object level", **TO)
nums, line = row(PID, 14, r"^\s*RF\s+(?=\d)", 0)
assert nums == ["526", "93.76", "35", "6.24"], nums
ROWS += result(res(), "recall", "model4", "dataset1", nums[1], 14, T5, line, unit="%", setting="Table 5: random forest on the RGBHSV features; prediction of all UAV images against the 561 reference points",
               note="success rate: PWD-infected tree sample points identified, of 561 (success plus fail); recall at object level", **TO)
for metric, v, nm in (("f1", "0.9332", "F1-score"), ("iou", "0.8745", "IoU")):
    ROWS += result(res(), metric, "model4", "dataset1", v, 15, T6, f"RF {v}", unit="fraction", setting="Table 6: random forest on the RGBHSV features; split not stated", note=f"{nm} of the PWD-infected tree class; precision 0.9299", **TP)
ROWS += [R("limitations", "Misclassifications and false positives remain (red roofs are confused with reddish-brown infected trees), and the infection stages are not distinguished", 20, "Sec. 6", "some limitations, such as instances of misclassification and false positives")]
ROWS += avail(code="no", weights="not_stated", data="no", note_d="Data Availability Statement (p.20): Data are contained within the article; no data link")
ROWS += [D("notes", "Compares U-Net, DeepLabv3+ and FPN (ResNet-152) on RGB, HSV and RGBHSV inputs (F1 and IoU in Table 4; success rates in Table 1), a random forest (Tables 5 and 6) and a fine-tuned SAM (Tables 8 and 9). RGB is primary because it is the default input that every deep model, including SAM (three channels only), has in common; the HSV and RGBHSV rows, the random forest (RGBHSV features) and the HSV row of SAM are not primary, and the headline is the abstract's best result (U-Net, RGBHSV, F1 0.9586). Precision is not recorded (F1 is given); the abstract's recall of 0.9553 for U-Net with RGBHSV is not recorded. The success rate is the share of the 561 reference tree points found in the prediction of all UAV images, recorded as recall at object level; it is not a held-out test (the split is not stated). The success plus fail counts add up to 561. The trainable parameters of the three CNNs are printed as full counts (Table 3: 3 and 6 channels) and converted to M; the training time per epoch (U-Net 5 min 55 s, DeepLabv3+ 5 min 48 s, FPN 5 min 4 s, SAM 10 min 56 s) is not recorded. The text layer of the Forests pages is duplicated; Table 6 and Table 9 values are read from the rendered pages.",
           "Sec. 4 and Tables 1 to 9 (pp. 10 to 16)")]
