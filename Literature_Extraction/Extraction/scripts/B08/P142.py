import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T3 = "P142", "Table 1", "Table 3"
# Discolored pine wilt disease (PWD) trees in 0.8 m Gaofen-2 and Gaofen-7 imagery: an FCN baseline (ResNet50v1c) against four single-head self-attention models (NLNet, CCNet, DANet, GCNet: attention modules as
# decoder heads) and three multi-head models (SegFormer-b0, SegFormer-b2, Swin Transformer-T), all initialised from ADE20K weights of MMSegmentation. Table 1 gives the mIoU of the validation set (mean of the last ten
# epochs) of the eight models; this mIoU is the mean of the IoU of the positive class and of the negative class (formula of Sec. 2.5), a score averaged over a class that is not trees, so it is never primary (protocol 6.3
# item 6). Table 3 counts correct, missed and wrong trees of the test set for three models: producer accuracy is recall and user accuracy is precision by their definitions (TP/(TP+FN) and TP/(TP+FP), Sec. 2.5) at object
# level, no F1 is given, so recall and precision are recorded and primary; the mean accuracy is the mean of the two and is not recorded. Single-head self-attention heads count as self-attention layers added to a CNN
# (protocol 4.5, cnn_transformer_hybrid), a judgement call listed in decisions.md.
TP = Tags(task="health_classification", level="pixel", split="validation", agg="single")
TO = Tags(task="health_classification", level="object", split="test", agg="single")
TASK_EV = "We constructed a dataset of discolored pine trees affected by pine wilt disease using imagery from the Gaofen-2 and Gaofen-7 satellites"
DS_EV = "comprising 4352 sample slices"
ROWS = [
    R("task", "health_classification", 1, "Abstract", TASK_EV, note="semantic segmentation of discolored (infected) pine trees at the single-tree scale"),
    R("output_unit", "crown_mask", 6, "Sec. 2.3", "sample labels drawn in ArcGIS Pro", note="binary masks of discolored trees; counts of correct, missed and wrong trees are made on the predictions of the test set"),
    R("tree_setting", "natural_forest", 3, "Sec. 2.1", "mixed coniferous and broad-leaved forests as the dominant forest type"),
    R("tree_taxa", "pine trees infected by pine wilt disease (species not named)", 1, "Abstract", TASK_EV),
    R("study_area", "Yichun and Fuzhou (Jiangxi Province) and Nanping (Fujian Province), southern China", 3, "Sec. 2.1", "The study area encompasses Yichun and Fuzhou in Jiangxi Province, and Nanping in Fujian Province, located in southern China"),
    D("biome", "subtropical", "Jiangxi and Fujian, southern China; not stated in the paper"),
    R("platform", "satellite", 1, "Abstract", TASK_EV, note="Gaofen-7 (Yichun and Fuzhou, August and September 2022) and Gaofen-2 (Nanping, September 2019)"),
    R("sensors", "multispectral", 5, "Sec. 2.2", "The imagery consists of four spectral bands", note="red, green, blue and near-infrared bands, fused to 0.8 m"),
    R("gsd", "0.8", 5, "Sec. 2.2", "all with a spatial resolution of 0.8 meters", unit="m"),
    R("annotation_type", "polygon_mask", 6, "Sec. 2.3", "sample labels drawn in ArcGIS Pro", note="visual interpretation labels (binary masks), supported by 119 ground survey points in Nanping"),
    R("dataset_name", "own Gaofen-2 and Gaofen-7 semantic segmentation dataset of discolored PWD trees (512 x 512 slices, four bands)", 6, "Sec. 2.3", DS_EV, item="dataset1"),
    R("dataset_key", "own_p142_gaofen_pwd", 6, "Sec. 2.3", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 2.3", "The dataset was randomly split into training, validation, and testing sets in a 6:2:2 ratio", item="dataset1",
      note="the validation set scores the models (Table 1); the test set is used for visual analysis and for the tree counts of Table 3"),
    D("dataset_public", "partial", "Data Availability Statement (p.18): Gaofen images can be requested from CRESDA and ground survey data from BDPC; the labelled dataset is not stated to be released", item="dataset1"),
    R("n_images", "4352", 6, "Sec. 2.3", DS_EV, item="dataset1", unit="slices", note="3948 slices with discolored trees and 404 typical object samples without positives; 512 x 512 pixels with four bands"),
]
NP = dict(pre_note="ADE20K pretrained weights officially provided by MMSegmentation (not remote sensing data)", strat_note="transfer learning from ADE20K weights; AdamW, learning rate 0.001, batch size 96, 100 epochs on four A100 GPUs; whether any layers are frozen is not stated")
PRE = dict(pre_page=11, pre_loc="Sec. 2.6", pre_ev="the ADE20K dataset pretrained weights officially provided by MMSegmentation were loaded for transfer learning",
           strat_page=11, strat_loc="Sec. 2.6", strat_ev="The AdamW optimizer was employed, with a learning rate set to 0.001, a batch size of 96, and a total of 100 training epochs")
HY = "convolutional FCN with a single-head self-attention module as the decoder head (the authors' single-head attention model); protocol 4.5: self-attention layers added to a CNN"
M = [  # table regex, name, role, category, backbone, category note
    (r"FCN \(baseline\)", "FCN (baseline)", "cnn_baseline", "cnn", "ResNet50v1c with a plain convolutional decoder head", "fully convolutional baseline without attention"),
    (r"NLNet", "NLNet (non-local module)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50v1c with a non-local self-attention head", HY),
    (r"CCNet", "CCNet (criss-cross attention)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50v1c with a criss-cross attention head", HY),
    (r"DANet", "DANet (dual attention)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50v1c with a dual attention head (position and channel self-attention)", HY),
    (r"GCNet", "GCNet (global context)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50v1c with a global context head (simplified non-local operation and squeeze-and-excitation)",
     HY + "; the global context block is a simplified non-local operation with squeeze-and-excitation reweighting"),
    (r"SegFormer-b0", "SegFormer-b0", "transformer_baseline", "hierarchical_vit", "MiT-b0 encoder", "SegFormer MiT encoder (protocol 4.5: hierarchical_vit)"),
    (r"SegFormer-b2", "SegFormer-b2", "transformer_baseline", "hierarchical_vit", "MiT-b2 encoder", "SegFormer MiT encoder (protocol 4.5: hierarchical_vit)"),
    (r"Swin Transformer-T", "Swin Transformer-T", "transformer_baseline", "hierarchical_vit", "Swin-Tiny backbone with a decoder head of MMSegmentation", "Swin Transformer backbone (protocol 4.5: hierarchical_vit)"),
]
res = Counter("res")
RID = {}
for k, (rx, name, role, cat, bb, cnote) in enumerate(M, start=1):
    item = f"model{k}"
    nums, ev = row(PID, 11, rf"^\s*{rx}\s+", 0)
    assert len(nums) == 4, (name, nums)  # params (M), learning speed, convergence speed, mIoU
    ROWS += model(item, name, role, cat, bb, "other", "full_fine_tuning", 11, T1, ev, cat_note=cnote, **NP, **PRE)
    ROWS.append(R("total_params", nums[0], 11, T1, ev, item=item, unit="M"))
    RID[item] = res()
    ROWS += result(RID[item], "miou", item, "dataset1", nums[3], 11, T1, ev, unit="%", setting="Table 1: validation set, mean of the last ten epochs",
                   note="mIoU of the positive (diseased tree) and negative class; the paper calls it recognition accuracy; mean of the final ten training epochs of one run", **TP, headline=(rx == "SegFormer-b2"))
# Table 3: test set, three models (FCN baseline, DANet, SegFormer-b2): tree counts, producer accuracy (recall) and user accuracy (precision)
for rx, item, lab in ((r"^\s*(?=1901\s+136)", "model1", "FCN"), (r"^\s*DANet\s+(?=1889)", "model4", "DANet"), (r"^\s*SegFormer-b2\s+(?=1858)", "model7", "SegFormer-b2")):
    nums, line = row(PID, 14, rx, 0)
    assert len(nums) == 6, (lab, nums)  # correct, missed, wrong, producer accuracy, user accuracy, mean accuracy
    ev = f"FCN ... {line}" if lab == "FCN" else line
    setting = f"Table 3: test set, {lab}; counts of correct ({nums[0]}), missed ({nums[1]}) and wrong ({nums[2]}) detections"
    ROWS += result(res(), "recall", item, "dataset1", nums[3], 14, T3, ev, unit="%", setting=setting, note="producer accuracy = TP/(TP+FN) of tree counts (Sec. 2.5), that is recall", **TO, primary=True)
    ROWS += result(res(), "precision", item, "dataset1", nums[4], 14, T3, ev, unit="%", setting=setting, note="user accuracy = TP/(TP+FP) of tree counts (Sec. 2.5), that is precision", **TO, primary=True)
ROWS += [R("limitations", "The 0.8 m imagery is coarse, and densely overlapping crowns in natural forest are hard to separate", 17, "Sec. 4.2",
           "In areas of natural forest where tree crowns are densely overlapped, it is challenging to distinguish individual infected trees")]
ROWS += avail(code="no", weights="not_stated", data="partial", data_page=18, data_loc="Data Availability Statement", data_ev="Gaofen images can be requested from China Centre for Resources Satellite Data and Application",
              note_d="the raw Gaofen images can be requested from CRESDA; the labelled dataset is not stated to be released")
ROWS += [D("notes", "Table 1 compares an FCN baseline with four single-head attention models and three multi-head models (mIoU of the validation set, parameters, learning and convergence speed in epochs, the last two are not recorded). The mIoU averages the IoU of the diseased and the negative class, so these values are not primary (protocol 6.3 item 6); SegFormer-b2 (76.39) is the abstract's headline. Table 3 gives tree counts on the test set for FCN, DANet and SegFormer-b2: producer accuracy (recall) 93.32, 92.73 and 91.21 and user accuracy (precision) 95.19, 96.13 and 96.51, primary at object level (the paper gives no F1; mean accuracy, the mean of the two, 94.25, 94.43 and 93.86, is not recorded). The single-head self-attention models (NLNet, CCNet, DANet, GCNet) are recorded as cnn_transformer_hybrid under the protocol's self-attention layers added; if they are not counted as transformer-based, DANet becomes the strongest non-transformer model of the study. Table 2 (ablation of the position and channel attention of DANet) is not recorded.",
           "Sec. 2.5, 3 and Tables 1 to 3 (pp. 10 to 14)")]
