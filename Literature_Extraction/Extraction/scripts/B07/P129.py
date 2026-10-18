import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T4 = "P129", "Table 3 (image)", "Table 4"  # the text layer of several Table 3 rows is interleaved, so the table is read from the rendered page
# EVitNet (lightweight U-Net-like segmentation network with a MobileViT encoder, MobileNetV2 blocks and an expanded-convolution decoder) segments pine wilt disease discolored trees in UAV orthophoto tiles.
# Dataset A (Qingdao, 9:1 training and test split) is primary: Table 3 compares seven models (IoU, F1, precision, recall, parameters, GFLOPs, FPS); the text layer of several pages is garbled, so the dataset
# facts of p.4 are read from the rendered page. Dataset B (Yantai, another UAV and sensor) is the transfer target: IoU_B of Table 4 gives the transfer pairs and the fine-tuned model (global fine-tuning). Dataset A is
# taken as the same data as in P125 (same authors, same equipment and area, 2981 samples with discolored trees).
TH = Tags(task="health_classification", level="pixel", split="test", agg="single")
OBJ_EV = "a lightweight semantic segmentation model based on a CNN-Transformer hybrid architecture was designed in this study, named EVitNet"
SPLIT_EV = "testing set at a 9:1 ratio"
ROWS = [
    R("task", "health_classification;crown_segmentation", 1, "Abstract", OBJ_EV, note="semantic segmentation of pine wilt disease discolored trees (one class)"),
    R("output_unit", "crown_mask", 4, "Sec. 2.2", "out based on the visual characteristics of discolored pine trees in remote sensing images", note="semantic masks of discolored tree crowns (not separated into instances)"),
    D("tree_setting", "natural_forest", "pine forests of Qingdao and Yantai; forest type is not described in the paper"),
    R("tree_taxa", "Pinus densiflora (Dataset A); Pinus densiflora and Pinus thunbergii (Dataset B)", 4, "Table 1", "Pinus densiflora"),
    R("study_area", "Qingdao and Yantai, Shandong Province, China", 4, "Table 1", "Yantai, Shandong"),
    D("biome", "temperate", "Shandong, China; coded from the study area"),
    D("platform", "uav", "DB-II fixed-wing UAV (Dataset A) and DJI Phantom 4 Pro (Dataset B), Table 1"),
    R("sensors", "rgb", 3, "Sec. 2.1", "Alpha 7 R II, Sony Group Corporation, Japan) was used due to the high flight altitude", note="Sony Alpha 7R II full-frame camera (A) and the DJI Phantom 4 Pro camera (B)"),
    R("annotation_type", "polygon_mask", 4, "Sec. 2.2", "Under the guidance of forest disease control experts, annotations were carried", note="polygon annotations converted to binary masks; 256 x 256 tiles"),
    R("dataset_name", "own DB-II fixed-wing UAV orthophoto tiles of discolored pine trees, Qingdao (Dataset A)", 4, "Table 1", "DB-II + Snoy alpha 7 R II", item="dataset1"),
    R("dataset_key", "own_p129_qingdao_pwd", 4, "Table 1", "DB-II + Snoy alpha 7 R II", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.2", SPLIT_EV, item="dataset1", note="9:1 split into training and testing sets"),
    D("dataset_public", "on_request", "Data availability (p.19): the datasets are available from the corresponding author upon reasonable request", item="dataset1"),
    R("n_images", "2981", 4, "Sec. 2.2 (image)", "A total of 31,500 samples were obtained from the data collected in Qingdao, Shandong, with 2981 of them containing at least one discolored tree lesion", item="dataset1", unit="tiles",
      note="256 x 256 tiles with at least one discolored tree, out of 31,500 tiles; the text layer is garbled, values read from the rendered page"),
    R("dataset_name", "own DJI Phantom 4 Pro UAV orthophoto tiles of discolored pine trees, Yantai (Dataset B)", 4, "Table 1", "DJI Phantom 4 Pro", item="dataset2"),
    R("dataset_key", "own_p129_yantai_pwd", 4, "Table 1", "DJI Phantom 4 Pro", item="dataset2"),
    R("dataset_role", "transfer_target", 13, "Sec. 3.3", "the IoU dropped to 0.321", item="dataset2", note="a different UAV, camera, height, terrain and species; also used for fine-tuning (9:1 split into training and testing sets)"),
    D("dataset_public", "on_request", "Data availability (p.19): the datasets are available from the corresponding author upon reasonable request", item="dataset2"),
    R("n_images", "524", 4, "Sec. 2.2", "524 had discolored tree lesions", item="dataset2", unit="tiles", note="of 3753 tiles of 256 x 256 pixels from Yantai (the sentence on p.4 reads 3753 samples, 524 with discolored trees)"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_page=11, strat_loc="Sec. 3.2", strat_ev="all hyperparameters were kept consistent: batch size = 16",
            strat_note="batch size 16, 100 epochs, Adam with cosine annealing, input 256 x 256")
ROWS += model("model1", "EVitNet (lightweight ViT encoder and CNN decoder, proposed)", "proposed", "cnn_transformer_hybrid", "U-Net like network with an EasyViT (MobileViT) encoder, MobileNetV2 blocks and an expanded convolution decoder", None, None, 1, "Abstract", OBJ_EV,
              cat_note="CNN-Transformer hybrid: a lightweight Vision Transformer encoder from the fourth encoder stage and a CNN decoder", **NPRE)
BASE = (("model2", "UNet (VGG, baseline)", "cnn_baseline", "cnn", "UNet with a VGG backbone", "UNet (Baseline) vgg 0.669 0.800 0.793 0.808 24.981 112.918 39.81", "0.669", "0.800", "24.981", "112.918", "39.81", "convolutional segmentation network (the paper's baseline)"),
        ("model3", "DeepLabv3+ (MobileNet)", "cnn_baseline", "cnn", "DeepLabv3+ with a MobileNet backbone", "DeepLabv3+ mobilenet 0.472 0.645 0.588 0.715 5.818 53.026 100.45", "0.472", "0.645", "5.818", "53.026", "100.45", "convolutional segmentation network"),
        ("model4", "HRNet (HRNetV2-W18)", "cnn_baseline", "cnn", "HRNetV2-W18", "HrNet hrnetv2_w18 0.561 0.711 0.783 0.651 9.642 32.948 211.19", "0.561", "0.711", "9.642", "32.948", "211.19", "convolutional segmentation network"),
        ("model5", "PSPNet (MobileNet)", "cnn_baseline", "cnn", "PSPNet with a MobileNet backbone", "PSPNet mobilenet 0.617 0.760 0.819 0.708 2.377 6.034 886.50", "0.617", "0.760", "2.377", "6.034", "886.50", "convolutional segmentation network"),
        ("model6", "SegFormer (SegFormer-B0)", "transformer_baseline", "hierarchical_vit", "SegFormer-B0 (MiT encoder)", "Segformer Segformer_b0 0.655 0.786 0.812 0.768 3.720 13.696 485.17", "0.655", "0.786", "3.720", "13.696", "485.17", "SegFormer with a hierarchical MiT transformer encoder (protocol 4.5)"),
        ("model7", "Swin-UNet", "transformer_baseline", "hierarchical_vit", "Swin-UNet (Swin Transformer U-Net)", "Swin-Unet Swin-Unet 0.665 0.799 0.817 0.781 98.30 26.33 162.43", "0.665", "0.799", "98.30", "26.33", "162.43", "Swin Transformer U-shaped network (protocol 4.5)"))
for item, name, role, cat, bb, ev, iou, f1, prm, fl, fps, cnote in BASE:
    ROWS += model(item, name, role, cat, bb, None, None, 12, T3, ev, cat_note=cnote, **NPRE)
ROWS += model("model8", "EVitNet fine-tuned on Dataset B (global fine-tuning)", "proposed", "cnn_transformer_hybrid", "EVitNet fine-tuned without freezing the backbone on a share of the Dataset B training samples", None, "full_fine_tuning", 13, "Sec. 3.3",
              "the final model achieved the highest accuracy when fine-tuning without", cat_note="EVitNet after fine-tuning", pre_status="not_reported", pre_note="initialised from the EVitNet trained on Dataset A", strat_page=13, strat_loc="Sec. 3.3",
              strat_ev="freezing the backbone network", strat_note="global fine-tuning (no frozen backbone) on a share of the Dataset B training samples; Fig. 9 compares three strategies")
res = Counter("res")
RID = {}
ROWS_T3 = [("model1", "EVitNet (Ours) EasyVit 0.713 0.833 0.853 0.814 1.195 1.636 850.43", "0.713", "0.833", "1.195", "1.636", "850.43")] + [(i, ev, iou, f1, prm, fl, fps) for i, _n, _r, _c, _b, ev, iou, f1, prm, fl, fps, _cn in BASE]
for item, ev, iou, f1, prm, fl, fps in ROWS_T3:
    st = "Table 3: Dataset A test set, discolored tree class"
    RID[item] = res()
    ROWS += result(RID[item], "iou", item, "dataset1", iou, 12, T3, ev, unit="fraction", setting=st, note="IoU of the discolored tree class", **TH, primary=True, headline=(item == "model1"))
    ROWS += result(res(), "f1", item, "dataset1", f1, 12, T3, ev, unit="fraction", setting=st, note="F1-score of the discolored tree class", **TH, primary=True)
    ROWS += [R("total_params", prm, 12, T3, ev, item=item, unit="M"), R("flops", fl, 12, T3, ev, item=item, unit="GFLOPs"), R("inference_speed", f"{fps} FPS (hardware not stated; input 256 x 256)", 12, T3, ev, item=item)]
# Table 4: IoU_B (Dataset B) of EVitNet and UNet before fine-tuning, and of the fine-tuned EVitNet
EVB = "0.713 0.321 0.833 0.853 0.814 1.195 1.636 850.43"
EVU = "Unet(baseline) 0.669 0.109 0.800 0.793 0.808 24.981 112.918 39.81"
EVF = "0.295 0.735 0. 470 0.314 0.831 1.195 1.636 845.69"
SB = "Table 4: Dataset B (IoU_B), discolored tree class"
RID["evb"] = res()
ROWS += result(RID["evb"], "iou", "model1", "dataset2", "0.321", 14, T4, EVB, unit="fraction", setting=SB + ", EVitNet trained on Dataset A without fine-tuning", note="IoU_B before fine-tuning", **TH(split="not_stated"))
ROWS += result(res(), "iou", "model2", "dataset2", "0.109", 14, T4, EVU, unit="fraction", setting=SB + ", UNet trained on Dataset A without fine-tuning", note="IoU_B before fine-tuning", **TH(split="not_stated"))
ROWS += result(res(), "iou", "model8", "dataset2", "0.735", 14, T4, EVF, unit="fraction", setting=SB + ", EVitNet after fine-tuning", note="IoU_B after global fine-tuning (the same row gives an IoU of 0.295 on Dataset A after fine-tuning)", **TH(split="not_stated"))
for k, (mdl, nm, ind, ood, ev_ood, ev_ind) in enumerate((("model1", "EVitNet", "0.713", "0.321", EVB, "EVitNet (Ours) EasyVit 0.713 0.833 0.853 0.814 1.195 1.636 850.43"), ("model2", "UNet", "0.669", "0.109", EVU, "UNet (Baseline) vgg 0.669 0.800 0.793 0.808 24.981 112.918 39.81")), start=1):
    t = f"tr{k}"
    ROWS += [R("transfer_source", "Dataset A (Qingdao; DB-II fixed-wing UAV with a Sony Alpha 7R II)", 13, "Sec. 3.3", "the IoU dropped to 0.321", item=t),
             R("transfer_target", "Dataset B (Yantai; DJI Phantom 4 Pro)", 13, "Sec. 3.3", "the IoU dropped to 0.321", item=t),
             R("shift_type", "site;sensor;platform;species", 13, "Sec. 3.3", "significant differences in sampling", item=t, note="differences in sampling sensors, height, terrain and tree species between the two datasets (Sec. 3.3, Sec. 2.2)"),
             R("transfer_model", mdl, 14, T4, ev_ood, item=t), R("transfer_metric", "iou", 14, T4, ev_ood, item=t),
             R("value_in_domain", ind, 12, T3, ev_ind, item=t, unit="fraction", note="IoU on the Dataset A test set (Table 3)"),
             R("value_out_of_domain", ood, 14, T4, ev_ood, item=t, unit="fraction", note="IoU_B on Dataset B without fine-tuning (Table 4)"),
             R("transfer_level", "pixel", 14, T4, ev_ood, item=t)]
ROWS += conflict("cf1", f"{RID['model1']}.value", "0.7134", 13, "Sec. 3.3", "On Dataset A, the single-category IoU for discolored trees segmentation was as high as 0.7134, while on Dataset", "V1a",
                 "Table 3, the main comparison, gives an IoU of 0.713 for EVitNet on Dataset A; the text of Sec. 3.3 gives 0.7134", unit="fraction")
ROWS += [R("limitations", "EVitNet may only reach a high accuracy on datasets with similar features, and it needs fine-tuning with a few new samples on data from another sensor and site", 19, "Sec. 5",
           "in the limitations, the EvitNet may only achieve high accuracy on feature-similar datasets"),
         R("code_url", "https://github.com/ib00r/EVitNet", 19, "Data Availability Statement", "The code can be accessed on GitHub via https://github.com/ib00r/"), R("code_available", "yes", 19, "Data Availability Statement", "The code can be accessed on GitHub via https://github.com/ib00r/"),
         D("weights_available", "not_stated", "No statement about trained weights"), R("data_available", "on_request", 19, "Data Availability Statement", "the datasets used in the current study are available from the")]
ROWS += [D("notes", "Table 3 compares EVitNet with UNet (baseline), DeepLabv3+, HRNet, PSPNet, SegFormer-B0 and Swin-UNet on the Dataset A test set (IoU, F1, precision, recall, parameters, GFLOPs and FPS); IoU and F1 of the discolored tree class are recorded as primary. The two transformer baselines (SegFormer 0.655 and Swin-UNet 0.665 IoU) are below UNet (0.669). Table 4 (ablation) gives the IoU on Dataset B (IoU_B) of UNet (0.109), UNet with MobileViT (0.211), with MobileNetV2 blocks (0.297), EVitNet (0.321) and the fine-tuned EVitNet (0.735, with an IoU of 0.295 on Dataset A afterwards); the first, fourth and last rows are recorded and give two transfer pairs and the fine-tuned result. The abstract's fine-tuned IoU of 0.735 matches the best global fine-tuning point of Fig. 9 (0.7349 at 40 percent of the new training data; the caption says the best amount is around 30 percent); Fig. 9 (three fine-tuning strategies against the share of new training data) is not recorded. Table 2 (filtered training sets and focal loss) is not recorded. Several pages have a garbled text layer; the Dataset A tile counts of p.4 and Table 3 and 4 values were checked on the rendered pages.",
           "Sec. 3 to 4 and Tables 3 and 4 (pp. 12 to 14)")]
