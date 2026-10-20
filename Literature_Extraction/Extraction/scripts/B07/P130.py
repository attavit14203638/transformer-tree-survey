import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T2 = "P130", "Table I (image)", "Table II (image)"
# Clusterformer (cluster transformer encoder and D-cluster transformer decoder) segments pine wilt disease (PWD) infected trees in UAV images of two datasets: Hubei (first listed: primary) and Hunan. Tables I and II
# (images) compare 18 models of three groups (CNN, transformer, hybrid CNN and transformer): recall, precision, F1, IoU of the PWD class, FLOPs and parameters. All 18 rows are recorded (IoU and F1; precision and
# recall are in the tables). The models' sizes are those of Table I; Table II prints other FLOPs and parameters for most models (other dataset), which are not recorded. The test split is evaluated (the paper cuts 512 x 512
# tiles and gives training and testing tile numbers).
TH = Tags(task="health_classification", level="pixel", split="test", agg="single")
OBJ_EV = "we propose a novel UAV remote sensing segmentation model called Clusterformer with a conventional encoder–decoder structure"
HU_EV = "The original dataset consists of 64 remote sensing images in the RGB bands with the size of 6336 × 9504"
ROWS = [
    R("task", "health_classification;crown_segmentation", 1, "Abstract", OBJ_EV, note="segmentation of pine wilt disease infected trees (the PWD class; evaluation on the PWD class only)"),
    R("output_unit", "crown_mask", 2, "Sec. I", "where pixel-level labeling was applied to PWD-infected trees", note="pixel masks of PWD-infected trees (all infection stages as one class)"),
    NR("tree_setting", note="UAV images over forest farms in Hubei and Hunan; the paper does not say whether the stands are natural or planted"),
    NR("tree_taxa", note="pine trees infected by pine wilt disease; the species are not named"),
    R("study_area", "forest farms in Hubei and Hunan provinces, China", 8, "Sec. IV-A", HU_EV),
    D("biome", "subtropical", "Hubei and Hunan, China; coded from the study areas"),
    D("platform", "uav", "UAV-mounted Sony camera (Hubei) and multispectral camera (Hunan), Sec. IV-A"),
    R("sensors", "rgb;multispectral", 8, "Sec. IV-A", "is 183 while that of testing images is 80", note="Hubei: RGB (Sony camera); Hunan: multispectral camera with RGB and infrared bands (the quote gives the Hunan test size)"),
    R("annotation_type", "polygon_mask", 8, "Sec. IV-A", HU_EV, note="pixel-level labels of PWD-infected trees annotated by domain experts"),
    R("dataset_name", "own UAV RGB images of PWD trees over a forest farm in Hubei (Hubei dataset)", 8, "Sec. IV-A", HU_EV, item="dataset1"),
    R("dataset_key", "own_p130_hubei_pwd", 8, "Sec. IV-A", HU_EV, item="dataset1"),
    R("dataset_role", "train_test", 8, "Sec. IV-A", "the number of training images is 12 844 while that of testing images is 2964", item="dataset1", note="512 x 512 tiles cut with stride 500 from 64 images"),
    D("dataset_public", "no", "No data link or availability statement found in the paper (only the code is released)", item="dataset1"),
    D("n_images", "15808", "sum of the 12844 training and 2964 testing tiles of 512 x 512 pixels (Sec. IV-A)", item="dataset1", unit="tiles"),
    R("dataset_name", "own UAV multispectral (RGB and infrared) image of PWD trees over a forest farm in Hunan (Hunan dataset)", 8, "Sec. IV-A", "is 183 while that of testing images is 80", item="dataset2"),
    R("dataset_key", "own_p130_hunan_pwd", 8, "Sec. IV-A", "is 183 while that of testing images is 80", item="dataset2"),
    R("dataset_role", "train_test", 8, "Sec. IV-A", "is 183 while that of testing images is 80", item="dataset2", note="one 14262 x 8306 image cut into 512 x 512 tiles with stride 500; 183 training and 80 testing tiles after discarding blank tiles"),
    D("dataset_public", "no", "No data link or availability statement found in the paper (only the code is released)", item="dataset2"),
    D("n_images", "263", "sum of the 183 training and 80 testing tiles of 512 x 512 pixels (Sec. IV-A)", item="dataset2", unit="tiles"),
]
# key, name, role, category, backbone, group note, Hubei row (R, P, F1, IoU, FLOPs, Params), Hunan row
MD = [
    ("UNet", "UNet", "cnn_baseline", "cnn", "UNet", "CNN group of Table I", ("84.23", "89.51", "86.79", "76.66", "160.74", "17.26"), ("75.65", "71.92", "73.74", "58.40")),
    ("ResUNet++", "ResUNet++", "cnn_baseline", "cnn", "ResUNet++", "CNN group of Table I", ("84.78", "88.12", "86.42", "76.08", "63.26", "4.06"), ("83.96", "38.41", "52.71", "35.78")),
    ("PSPNet", "PSPNet", "cnn_baseline", "cnn", "ResNet50 encoder with a PSP head", "CNN group of Table I", ("87.13", "87.35", "87.24", "77.37", "184.73", "46.71"), ("71.40", "68.88", "70.12", "54.00")),
    ("PSANet", "PSANet", "cnn_baseline", "cnn", "ResNet50 encoder (decoder head listed as psphead in Table I and PSAhead in Table II)", "CNN group of Table I", ("85.4", "89.54", "87.42", "77.65", "199.44", "53.16"), ("70.72", "67.41", "69.03", "52.70")),
    ("Deeplabv3+", "Deeplabv3+", "cnn_baseline", "cnn", "ResNet50 encoder", "CNN group of Table I", ("84.97", "89.82", "87.32", "77.51", "173.40", "39.76"), ("75.59", "68.68", "71.97", "56.22")),
    ("UperNet", "UperNet", "cnn_baseline", "cnn", "ResNet50 encoder with an UPerHead", "CNN group of Table I", ("85.84", "88.83", "87.31", "77.49", "42.52", "31.34"), ("75.31", "74.53", "74.91", "59.89")),
    ("PIDNet", "PIDNet", "cnn_baseline", "cnn", "PIDNet", "CNN group of Table I", ("85.48", "88.61", "87.02", "77.02", "22.25", "28.54"), ("57.10", "68.92", "62.45", "45.40")),
    ("SCANet", "SCANet", "cnn_baseline", "cnn", "SCANet", "CNN group of Table I", ("84.27", "89.89", "86.99", "76.97", "99.20", "15.87"), ("72.74", "77.44", "75.02", "60.02")),
    ("MSSCN", "MSSCN", "cnn_baseline", "cnn", "MSSCN", "CNN group of Table I", ("84.28", "88.84", "86.50", "76.22", "13.05", "20.40"), ("67.53", "72.11", "69.75", "53.55")),
    ("SETR", "SETR", "transformer_baseline", "pure_vit", "ViT-base encoder with the setr_up_head decoder", "Transformer group of Table I (ViT-base encoder)", ("81.96", "84.52", "84.52", "71.26", "91.76", "86.66"), ("68.00", "66.97", "67.49", "50.93")),
    ("Segformer", "Segformer", "transformer_baseline", "pure_vit", "ViT-base encoder with the segformerhead decoder (as listed in Table I)", "Transformer group of Table I; the table lists a ViT-base encoder although SegFormer normally has a hierarchical MiT encoder", ("84.50", "87.95", "86.19", "75.73", "13.16", "7.72"), ("72.23", "71.98", "72.10", "56.38")),
    ("SwinUNet", "SwinUNet", "transformer_baseline", "hierarchical_vit", "Swin Transformer encoder and decoder", "Transformer group of Table I (Swin Transformer encoder and decoder)", ("81.63", "88.59", "84.97", "73.87", "40.93", "41.39"), ("74.00", "72.18", "73.07", "57.57")),
    ("DAE-former", "DAE-former", "transformer_baseline", None, "DAE-former", "listed in the Transformer group of Table I; its architecture is not described", ("85.63", "89.38", "87.46", "77.72", "105.59", "48.07"), ("76.47", "73.63", "75.02", "60.03")),
    ("TransUNet", "TransUNet", "transformer_baseline", "cnn_transformer_hybrid", "ResNet50 and ViT encoder", "Hybrid CNN and Transformer group of Table I", ("84.88", "89.38", "87.08", "77.11", "25.47", "105.91"), ("73.42", "70.72", "72.05", "56.31")),
    ("iformer", "iformer", "transformer_baseline", "cnn_transformer_hybrid", "iformer-base encoder with an uperhead", "Hybrid CNN and Transformer group of Table I", ("84.32", "89.39", "86.78", "76.65", "230.77", "40.80"), ("76.99", "71.83", "74.32", "59.14")),
    ("MISSFormer", "MISSFormer", "transformer_baseline", "cnn_transformer_hybrid", "MISSFormer (ViT-base encoder as listed in Table I)", "Hybrid CNN and Transformer group of Table I", ("84.65", "88.76", "86.65", "76.46", "36.77", "42.46"), ("65.41", "74.25", "69.55", "53.32")),
    ("AFFormer", "AFFormer", "transformer_baseline", "cnn_transformer_hybrid", "AFFormer", "Hybrid CNN and Transformer group of Table I", ("86.41", "87.35", "86.88", "76.80", "4.79", "2.90"), ("68.84", "66.00", "67.38", "50.81")),
    ("Clusterformer", "Clusterformer (proposed)", "proposed", "cnn_transformer_hybrid", "encoder of cluster transformers (cluster token mixer and SC-FFN with depth-wise convolutions) and a decoder of D-cluster transformers", "Hybrid CNN and Transformer group of Table I", ("87.06", "88.67", "87.85", "78.35", "17.69", "11.34"), ("76.04", "78.80", "77.39", "63.12")),
]
IT = {}
res = Counter("res")
for i, (key, name, role, cat, bb, grp, hub, hun) in enumerate(MD, start=1):
    item = f"model{i}"
    IT[key] = item
    ev = f"{name.replace(' (proposed)', '')} {' '.join(hub)}"
    ROWS += model(item, name, role, cat, bb, None, None, 9, T1, ev, cat_note=grp, pre_status="not_reported", pre_note="not stated", strat_note="training settings not stated in the text")
    ROWS += [R("total_params", hub[5], 9, T1, ev, item=item, unit="M"), R("flops", hub[4], 9, T1, ev, item=item, unit="GFLOPs")]
RID = {}
for ds, T, pg, col, prim in (("dataset1", T1, 9, 0, True), ("dataset2", T2, 10, 1, False)):
    for key, name, role, cat, bb, grp, hub, hun in MD:
        item = IT[key]
        if col == 0:
            r_, p_, f1, iou = hub[0], hub[1], hub[2], hub[3]
            ev = f"{name.replace(' (proposed)', '')} {' '.join(hub)}"
        else:
            r_, p_, f1, iou = hun
            ev = f"{name.replace(' (proposed)', '')} {' '.join(hun)}"
        st = f"{T.split(' (')[0]}: {'Hubei' if col == 0 else 'Hunan'} test set, PWD class"
        RID[(ds, key)] = res()
        ROWS += result(RID[(ds, key)], "iou", item, ds, iou, pg, T, ev, unit="%", setting=st, note="IoU of the PWD class (evaluation on the PWD class only)", **TH, primary=prim)
        ROWS += result(res(), "f1", item, ds, f1, pg, T, ev, unit="%", setting=st, note="F1-score of the PWD class", **TH, primary=prim)
ROWS += [R("limitations", "The proposed decoder with the D-cluster transformer shows no significant advantage over an uperhead on the Hubei dataset when the cluster token mixer is replaced by efficient or factorized attention", 13, "Sec. IV-D",
           "However, it is noted that the proposed decoder structure"),
         R("code_url", "https://github.com/huanliu233/Clusterformer", 1, "Abstract", "The code is available at https://github.com/huanliu233/Clusterformer"), R("code_available", "yes", 1, "Abstract", "The code is available at https://github.com/huanliu233/Clusterformer"),
         D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data link or availability statement found in the paper")]
ROWS += [D("notes", "Tables I (Hubei) and II (Hunan) compare 18 models of three groups on the PWD class (recall, precision, F1, IoU, FLOPs and parameters); IoU and F1 are recorded for all of them (Hubei primary, Hunan not), and the FLOPs and parameters of Table I. Table II prints other FLOPs and parameters for most models (for example UNet 160.90 against 160.74 GFLOPs, DAE-former 29.70 against 48.07 M parameters), which are not recorded. The text says that on the Hubei dataset CNN models generally outperform the transformer models, that DAE-former is the best transformer model and that Clusterformer beats it by 0.63 IoU and 0.39 F1 (Hunan: 3.09 and 2.37). The models are grouped by the paper (CNN, transformer, hybrid); the paper lists a ViT-base encoder for SegFormer and MISSFormer, and the architecture of DAE-former is not described. The ablation Tables III to VI are not recorded.",
           "Sec. IV and Tables I and II (pp. 8 to 10)")]
