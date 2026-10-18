import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P125", "Table 3"
# RCANet (lightweight U-Net with a refined VGG encoder, deformable convolution, CARAFE upsampling and scSE attention, all convolutional) segments the crowns of pine wood nematode affected pine trees in UAV orthophoto
# tiles. Table 3 compares it with UNet, DeepLabv3+, HRNet, PSPNet and the transformer models SegFormer-B0 and Swin-UNet on the validation set (class-specific metrics of the affected trees class). Discolored (affected) tree
# segmentation is health_classification at pixel level. All seven rows are primary; IoU and F1 are recorded (precision and recall are in the same table).
TH = Tags(task="health_classification", level="pixel", split="validation", agg="single")
OBJ_EV = "we proposed a novel lightweight model termed Refined and Deformable Carafe Attention Net (RCANet)"
ROWS = [
    R("task", "health_classification;crown_segmentation", 1, "Abstract", "To achieve more rapid and precise segmentation of trees affected by PWN", note="semantic segmentation of the crowns of pine trees affected by the pine wood nematode (one affected class)"),
    R("output_unit", "crown_mask", 4, "Data preprocessing", "the polygon tool is used to precisely fit the entire contour of the discolored trees", note="semantic masks of affected tree crowns (not separated into instances)"),
    R("tree_setting", "natural_forest", 3, "Study area", "The data in this study were collected using a DB-II fixedwing UAV in Qingdao, Shandong Province, China", note="forest type is not described; mountainous pine forest"),
    R("tree_taxa", "Pinus thunbergii and Pinus densiflora", 3, "Study area", "The primary vegetation consists of Pinus thunbergii and Pinus densiflora"),
    R("study_area", "Qingdao, Shandong Province, China", 3, "Study area", "The data in this study were collected using a DB-II fixedwing UAV in Qingdao, Shandong Province, China"),
    D("biome", "temperate", "Qingdao, Shandong, China; coded from the study area"),
    R("platform", "uav", 3, "Study area", "The data in this study were collected using a DB-II fixedwing UAV", note="fixed-wing UAV flying below 700 m"),
    R("sensors", "rgb", 3, "Study area", "camera with a 35 mm prime lens", note="Sony Alpha 7R II camera"),
    R("gsd", "8", 4, "Study area", "The resulting Digital Orthophoto Map (DOM) had a resolution of 8 cm", unit="cm"),
    R("annotation_type", "polygon_mask", 4, "Data preprocessing", "the polygon tool is used to precisely fit the entire contour of the discolored trees", note="annotation checked by nematode DNA tests of 45 sampled pine trees (43 positive)"),
    R("dataset_name", "own fixed-wing UAV orthophoto tiles (256 x 256 pixels, 8 cm) of pine wood nematode affected trees, Qingdao", 4, "Data preprocessing", "We obtained a total of 2,981", item="dataset1"),
    R("dataset_key", "own_p125_qingdao_pwd", 4, "Data preprocessing", "We obtained a total of 2,981", item="dataset1"),
    R("dataset_role", "train_test", 4, "Data preprocessing", "samples, each containing at least one affected tree lesion spot, which were divided into training set, validation set and testing set in an 8:1:1 ratio", item="dataset1",
      note="8:1:1 split; Table 3 reports the validation results"),
    D("dataset_public", "on_request", "Data availability (p.15): the datasets are available from the corresponding author on reasonable request", item="dataset1"),
    R("n_images", "2981", 4, "Data preprocessing", "We obtained a total of 2,981", item="dataset1", unit="tiles", note="256 x 256 pixel samples, each with at least one affected tree"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_page=10, strat_loc="Baseline model selection", strat_ev="a batch size of 16, 100 epochs, the adoption of",
            strat_note="batch size 16, 100 epochs, Adam; initialisation not stated")
ROWS += model("model1", "RCANet (Refined and Deformable Carafe Attention Net, proposed)", "proposed", "cnn", "UNet with a refined (channel-reduced) VGG encoder, deformable convolution (DCN), CARAFE upsampling and scSE attention", None, None, 1, "Abstract", OBJ_EV,
              cat_note="convolutional U-Net; scSE and CARAFE are generic attention and upsampling modules (protocol 4.5)", **NPRE)
BASE = (("model2", "UNet (VGG, baseline)", "cnn_baseline", "cnn", "UNet with a VGG backbone", "0.673 0.805 0.810 0.800 24.981 112.918", "0.673", "0.805", "24.981", "112.918", "UNet is the baseline model of the paper"),
        ("model3", "DeepLabv3+ (MobileNet)", "cnn_baseline", "cnn", "DeepLabv3+ with a MobileNet backbone", "0.347 0.515 0.502 0.528 5.818 53.026", "0.347", "0.515", "5.818", "53.026", "convolutional segmentation network"),
        ("model4", "HRNet (HRNetV2-W18)", "cnn_baseline", "cnn", "HRNetV2-W18", "0.546 0.706 0.810 0.626 9.642 32.948", "0.546", "0.706", "9.642", "32.948", "convolutional segmentation network"),
        ("model5", "PSPNet (MobileNet)", "cnn_baseline", "cnn", "PSPNet with a MobileNet backbone", "0.626 0.770 0.831 0.718 2.377 6.034", "0.626", "0.770", "2.377", "6.034", "convolutional segmentation network"),
        ("model6", "SegFormer (SegFormer-B0)", "transformer_baseline", "hierarchical_vit", "SegFormer-B0 (MiT encoder)", "0.659 0.794 0.825 0.766 3.720 13.696", "0.659", "0.794", "3.720", "13.696", "SegFormer with a hierarchical MiT transformer encoder (protocol 4.5)"),
        ("model7", "Swin-UNet", "transformer_baseline", "hierarchical_vit", "Swin-UNet (Swin Transformer U-Net)", "0.665 0.799 0.817 0.781 98.30 26.33", "0.665", "0.799", "98.30", "26.33", "Swin Transformer U-shaped network (protocol 4.5)"))
for item, name, role, cat, bb, ev, iou, f1, prm, fl, cnote in BASE:
    ROWS += model(item, name, role, cat, bb, None, None, 11, T3, ev, cat_note=cnote, **NPRE)
res = Counter("res")
RID = {}
ROWS_T3 = [("model1", "0.729 0.843 0.864 0.823 5.373 11.276", "0.729", "0.843", "5.373", "11.276")] + [(i, ev, iou, f1, prm, fl) for i, _n, _r, _c, _b, ev, iou, f1, prm, fl, _cn in BASE]
for item, ev, iou, f1, prm, fl in ROWS_T3:
    st = "Table 3: validation set, class of the affected trees"
    RID[item] = res()
    ROWS += result(RID[item], "iou", item, "dataset1", iou, 11, T3, ev, unit="fraction", setting=st, note="IoU of the affected trees class (class-specific metric)", **TH, primary=True, headline=(item == "model1"))
    ROWS += result(res(), "f1", item, "dataset1", f1, 11, T3, ev, unit="fraction", setting=st, note="F1-score of the affected trees class", **TH, primary=True)
    ROWS += [R("total_params", prm, 11, T3, ev, item=item, unit="M"), R("flops", fl, 11, T3, ev, item=item, unit="GFLOPs")]
ROWS += [R("inference_speed", "83.14 fps (hardware and input size not stated; training on an RTX 3090)", 1, "Abstract", "The number of parameters in RCANet is only 5.373 million, the segmentation speed reached 83.14 fps", item="model1")]
ROWS += [R("limitations", "The scSE and CARAFE improvements are shown for a single class (affected trees) segmentation task on one study area, and the real-time deployment on a UAV is left for the future", 1, "Abstract",
           "which is expected to be deployed on UAVs in the future for real-time recognition", note="the model is not yet deployed on a UAV (abstract)"),
         R("code_url", "https://github.com/ib00r/RCANet", 15, "Data availability", "the datasets used in current study available from the"), R("code_available", "yes", 15, "Data availability", "the datasets used in current study available from the"),
         D("weights_available", "not_stated", "No statement about trained weights"), R("data_available", "on_request", 15, "Data availability", "the datasets used in current study available from the")]
ROWS += [D("notes", "Table 3 compares RCANet with UNet (baseline), DeepLabv3+, HRNet, PSPNet, SegFormer-B0 and Swin-UNet on the validation set (IoU, F1, precision, recall of the affected trees class, parameters and GFLOPs); IoU and F1 are recorded. RCANet is convolutional (the attention modules scSE and CARAFE are generic); the two transformer baselines (SegFormer 0.659 and Swin-UNet 0.665 IoU) are below UNet (0.673) and RCANet (0.729). Table 4 (ablation of refined VGG, scSE, CARAFE and DCN) and Table 2 (augmentation) are not recorded. The test split of the 8:1:1 division is not reported separately; the paper's evaluation set is the validation set. The abstract's 5.6 percent IoU gain over UNet is 0.729 against 0.673.",
           "Sec. Experiments and results and Table 3 (pp. 10 to 11)")]
