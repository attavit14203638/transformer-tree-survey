import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1 = "P009", "Table 1"
URL = "https://github.com/NourO93/Palm-Tree-Dataset"
HDR = "Model AP (%) AR (%) AF (%)"
# Detection of palm trees with boxes on 595 test images. The paper's AP and AR are per-image averages of precision and recall (Eqs. 1 and 2), AF the average F1 (Eq. 3).
T = Tags(task="detection", level="object", split="test", agg="single")
DS_EV = "A total of 2,976 images containing 21,935 palm trees were compiled"

ROWS = [
    R("task", "detection", 1, "Abstract", "a novel High-Resolution (HR) remote sensing dataset designed for the autonomous detection of palm trees in the UAE"),
    R("output_unit", "tree_instance", 4, "Sec. 5", "exporting the prediction results to shapefile that shows a point feature in the center of each detected palm tree"),
    R("tree_setting", "plantation", 1, "Sec. 1", "given the vast size of palm plantations, which are difficult to monitor effectively from the ground",
      note="the images come from several municipalities; the setting of the images is not described beyond evenly spaced or densely packed palms with other vegetation"),
    R("tree_taxa", "date palm", 1, "Sec. 1", "The date palm tree is considered one of the most ancient trees in the Arabian Peninsula"),
    R("study_area", "United Arab Emirates (Ajman, Dubai, Khorfakkan and Al Ain)", 1, "Abstract", "covering various areas within UAE, including Ajman, Dubai, Khorfakkan, and Al-Ain"),
    D("biome", "arid", "United Arab Emirates; not stated in the paper"),
    R("platform", "uav", 1, "Abstract", "The dataset has been acquired using Unmanned Aerial Vehicles (UAVs)"),
    R("sensors", "rgb", 2, "Sec. 4", "aerial RGB images of different sizes with a spatial resolution of 10cm were collected from various municipalities across the UAE"),
    R("gsd", "10", 2, "Sec. 4", "aerial RGB images of different sizes with a spatial resolution of 10cm were collected from various municipalities across the UAE", unit="cm"),
    D("annotation_type", "bounding_box", "palm trees were annotated in Pascal VOC, COCO and YOLO formats (p.2), which are bounding-box formats; the paper does not use the words bounding box for the labels"),
    R("dataset_name", "own palm tree dataset, UAE (Ajman, Dubai, Khorfakkan, Al Ain), 256 × 256 patches", 2, "Sec. 3", DS_EV, item="dataset1"),
    R("dataset_key", "own_p009_uae_palm", 2, "Sec. 3", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 3", "13,938 palm trees were designated for model training ... 595 images containing 4,505 palm trees were used", item="dataset1"),
    R("dataset_public", "partial", 2, "Sec. 3", "A portion of the dataset for the Al Ain area is publicly available on GitHub", item="dataset1"),
    R("dataset_url", URL, 2, "Sec. 3", "publicly available on GitHub at https://github.com/NourO93/Palm-Tree-Dataset", item="dataset1"),
    R("n_images", "2976", 2, "Sec. 3", DS_EV, item="dataset1", unit="images", note="1,905 training, 476 validation and 595 test images; 256 × 256 patches"),
    R("n_trees", "21935", 2, "Sec. 3", DS_EV, item="dataset1", unit="trees", note="13,938 training, 3,492 validation and 4,505 test trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="initialisation not stated; trained for 200 epochs with Adam at a learning rate of 10-4 (Sec. 5)")
ROWS += model("model1", "YOLOv4", "cnn_baseline", "cnn", "CSPDarkNet53 with SPP and PANet", None, None, 3, "Sec. 4.1.1",
              "YOLOv4 is composed of four key components: the backbone network, CSPDarkNet53, for feature extraction", **NP)
ROWS += model("model2", "YOLOv5s", "cnn_baseline", "cnn", "YOLOv5s backbone (not described further)", None, None, 3, "Sec. 4.1.2", "In this paper, YOLOv5s is selected as the base model", **NP)
ROWS += model("model3", "FRCNN", "cnn_baseline", "cnn", "backbone not stated", None, None, 3, "Sec. 4.2",
              "Faster R-CNN (FRCNN) (Ren et al., 2016) was introduced as an enhanced version of Fast RCNN, featuring a Region Proposal Network (RPN)", **NP)
ROWS += model("model4", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "CNN backbone (such as ResNet) with an encoder-decoder transformer", None, None, 3, "Sec. 4.3",
              "DETR’s architecture is composed of three main components: a CNN backbone (such as ResNet) for initial feature extraction, an encoder-decoder Transformer",
              cat_note="generic description of DETR; the backbone actually used is not stated", **NP)
res = Counter("res")
for item, lab in (("model1", "YOLOv4"), ("model2", "YOLOv5s"), ("model3", "FRCNN"), ("model4", "DETR")):
    nums, ev = row(PID, 4, rf"^\s*{lab}\s+(?=\d)", 0, cols=(0, 68))
    assert len(nums) == 3, (lab, nums)  # AP, AR, AF
    ROWS += result(res(), "f1", item, "dataset1", nums[2], 4, T1, HDR + " ... " + ev, setting="595 test images; AF, the average per-image F1", note="AF column; AP and AR (per-image precision and recall) are in Table 1",
                   **T, primary=True, headline=(item == "model2"))
ROWS += avail(data="partial", data_page=2, data_loc="Sec. 3", data_ev="A portion of the dataset for the Al Ain area is publicly available on GitHub", note_d="only the Al Ain part is public")
ROWS += [D("notes", "Benchmark of four detectors on the authors' UAE palm dataset (Table 1). The paper's AP and AR average the per-image precision and recall (Eqs. 1 and 2), so only the F1 column (AF) is recorded as a standard metric. The abstract names YOLOv5s (AP 96.6 percent, AF 95.7 percent) as the best model; DETR has the highest recall. YOLOv4 also detected about 74 percent of the trees in a GIS monitoring test (p.4).",
           "Sec. 5 and Table 1 (p.4)")]
