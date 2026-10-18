import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1 = "P004", "Table I"
HDR = "mAP50(%) mAP50:95(%) GFLOP(G) Params(M)"
# Default tags of every result (protocol 6): bayberry tree detection with boxes; Table I does not say which split it is computed on.
T = Tags(task="detection", level="object", split="not_stated", agg="single")

ROWS = [
    R("task", "detection", 1, "Abstract", "low detection accuracy and high false-negative rates in bayberry tree detection"),
    R("output_unit", "tree_instance", 3, "Sec. III-C", "TP = number of correctly detected bayberry tree targets"),
    R("tree_setting", "orchard", 1, "Abstract", "To address challenges in orchard environments such as complex mountainous conditions and variable sunlight",
      note="the data are flights over Dayangshan Forest Park (Sec. III-A)"),
    R("tree_taxa", "bayberry (Yangmei)", 1, "Sec. I", "Bayberry is economically important in southern China"),
    R("study_area", "Dayangshan Forest Park, Yongjia County, Zhejiang, China", 3, "Sec. III-A", "Flights over Dayangshan Forest Park, Yongjia County, Zhejiang, on 23–24 Jan 2019"),
    D("biome", "subtropical", "Yongjia County, Zhejiang; not stated in the paper"),
    R("platform", "uav", 3, "Sec. III-A", "Data were sourced from the publicly released bayberry UAV imagery dataset"),
    D("sensors", "rgb", "imagery type not stated; the DJI Phantom 4 images in Fig. 5 and Fig. 6 (p.3, p.4) are colour photographs"),
    D("annotation_type", "bounding_box", "annotations were created in LabelImg (p.3), a bounding-box annotation tool; the paper does not use the words bounding box"),
    # dataset item: the key comes from Extraction/datasets.csv
    R("dataset_name", "publicly released bayberry UAV imagery dataset (Global Change Science Research Data Publishing and Repository), Dayangshan Forest Park", 3, "Sec. III-A",
      "Data were sourced from the publicly released bayberry UAV imagery dataset (Global Change Science Research Data Publishing and Repository)", item="dataset1"),
    R("dataset_key", "bayberry_uav_dayangshan", 3, "Sec. III-A",
      "Data were sourced from the publicly released bayberry UAV imagery dataset (Global Change Science Research Data Publishing and Repository)", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. III-A", "the data were split 8:1:1 into train/val/test (3,384/423/423)", item="dataset1"),
    R("dataset_public", "yes", 3, "Sec. III-A", "Data were sourced from the publicly released bayberry UAV imagery dataset (Global Change Science Research Data Publishing and Repository)",
      item="dataset1", note="no link printed"),
    R("n_images", "4230", 3, "Sec. III-A", "after removing invalid tiles,4,230 effective samples remained", item="dataset1", unit="tiles",
      note="tiles cut from 600 of the 3,690 original images (5472 × 3684), resized to 1499 × 999; split 3,384/423/423 (train/val/test)"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="initialisation not stated; trained for 300 epochs with AdamW (Sec. III-B)")

base, ev_b = row(PID, 4, r"^\s*1\.base\s+", 0, cols=(0, 85))
ours, ev_o = row(PID, 4, r"^\s*8\.ours\s+", 0, cols=(0, 85))
assert len(base) == 4 and len(ours) == 4, (base, ours)  # mAP50, mAP50:95, GFLOPs, params
ROWS += model("model1", "ADME-DETR", "proposed", "cnn_transformer_hybrid", "AGLNet (HCFA-based backbone with dynamic sparse self-attention), RT-DETR hybrid encoder and decoder", None, None, 4, T1,
              "The full ADME-DETR delivers the best results ... " + ev_o, cat_note="RT-DETR-based detector with self-attention in the backbone (protocol 4.5)", **NP)
ROWS += model("model2", "RT-DETR-R18", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-18 (RT-DETR-R18)", None, None, 4, T1,
              "Using RT-DETR-R18 as the baseline ... " + ev_b, cat_note="RT-DETR with a CNN backbone (protocol 4.5); the baseline of the ablation", **NP)

SET = "Table I ablation: full model against the RT-DETR-R18 baseline; split not stated (dataset split 3,384/423/423)"
res = Counter("res")
# Table I is the only results table; its baseline row and its full-model row are the compared configurations (rows 2 to 7 are partial module sets and are not recorded)
for item, nums, ev in (("model1", ours, ev_o), ("model2", base, ev_b)):
    ROWS += result(res(), "ap50", item, "dataset1", nums[0], 4, T1, HDR + " ... " + ev, setting=SET, note="mAP@0.5", **T, primary=True, headline=(item == "model1"))
    ROWS += result(res(), "ap50_95", item, "dataset1", nums[1], 4, T1, HDR + " ... " + ev, setting=SET, note="mAP@0.5:0.95", **T, primary=True)
    ROWS += [R("flops", nums[2], 4, T1, HDR + " ... " + ev, item=item, unit="GFLOPs"),
             R("total_params", nums[3], 4, T1, HDR + " ... " + ev, item=item, unit="M")]
ROWS += [R("inference_speed", "70.3 fps on an NVIDIA GeForce RTX 3090 GPU", 1, "Abstract", "inference speeds of 70.3 fps on RTX 3090 graphics cards", item="model1")]
ROWS += avail(data="yes", data_page=3, data_loc="Sec. III-A",
              data_ev="Data were sourced from the publicly released bayberry UAV imagery dataset (Global Change Science Research Data Publishing and Repository)")
ROWS += [D("notes", "Table I (p.4) is the only results table: an ablation of RT-DETR-R18 with the modules AGLNet, AMFN and HCRU; the baseline (row 1) and the full model (row 8) are recorded, the six partial combinations (rows 2 to 7) are not. The conclusion also compares with YOLO11m (full model higher by 3.8 mAP@0.5 and 4.4 mAP@0.5:0.95) but the YOLO11m values are not printed. The paper does not say whether Table I is computed on the validation or the test split.",
           "Sec. IV-A and Table I (p.4)")]
