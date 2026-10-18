import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P039", "Table 2 (image)"
URL = "https://pan.baidu.com/s/1xUp1Davs2-58i_oJxGUh7w?pwd=ngg3"
# Pixel-wise street tree segmentation: crown segmentation at pixel level. The paper does not say which split Table 2 uses; results are averages of five runs (Sec. 3.5).
T = Tags(task="crown_segmentation", level="pixel", split="not_stated", agg="mean")
SEG_EV = "marked all the street trees as the label"
SPLIT_EV = "Total 8000 1000 1000 10,000"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "Quick and accurate segmentation of street trees from high-resolution remote sensing images"),
    R("output_unit", "crown_mask", 4, "Sec. 3.1", SEG_EV, note="pixel mask of street trees; no individual tree is separated"),
    R("tree_setting", "urban", 1, "Sec. 1", "Street trees refer to the trees planted on both sides of a road for shade and noise reduction"),
    NR("tree_taxa", note="street trees; species not given"),
    R("study_area", "Lhasa, Tibet Autonomous Region, China", 4, "Sec. 3.1", "These two sets of products were both taken over Lhasa, Tibet Autonomous Region, China"),
    NR("biome", note="not stated"),
    R("platform", "satellite", 4, "Sec. 3.1", "The experimental data comprised RGB images and NDVI products from the Beijing-3 International Cooperative (BJ-3N) Satellite"),
    R("sensors", "rgb;multispectral", 4, "Sec. 3.1", "Each set contained one RGB image without a vegetation index and one NDVI", note="RGB image and an NDVI product (NDVI needs a near-infrared band)"),
    NR("gsd", note="the satellite's panchromatic images are 0.3 m and its multispectral 1.2 m (p.4); the resolution of the RGB and NDVI products used is not stated"),
    R("annotation_type", "polygon_mask", 4, "Sec. 3.1", SEG_EV, note="street tree pixels labelled in 12 street blocks"),
    R("dataset_name", "BJ-3N street tree dataset, Lhasa (12 street blocks, 256 × 256 tiles)", 4, "Sec. 3.1", SPLIT_EV, item="dataset1"),
    R("dataset_key", "own_p039_lhasa_street_trees", 4, "Sec. 3.1", SPLIT_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 3.1", "the dataset was divided into the training set, validation set, and test set in the ratio of 8:1:1", item="dataset1"),
    R("dataset_public", "yes", 4, "Sec. 3.1", "The dataset is publicly available on " + URL, item="dataset1"),
    R("dataset_url", URL, 4, "Sec. 3.1", "The dataset is publicly available on " + URL, item="dataset1"),
    R("n_images", "10000", 4, "Table 1", SPLIT_EV, item="dataset1", unit="tiles", note="8000 training, 1000 validation and 1000 test tiles; 8885 contain street trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="Adam (beta1 0.9, beta2 0.999), learning rate 0.001, 100 epochs, batch size 5, cross entropy loss (Sec. 3.5)")
M = [  # item, table label, name, role, category, backbone, page, evidence
    ("model1", "DB-MSC Net (Ours)", "DB-MSC Net", "proposed", "cnn_transformer_hybrid", "double-branch encoder (RGB and NDVI) with multi-scale contextual blocks of parallel dilated convolutions and transformer blocks, channel attention in the decoder", 1,
     "The MSC block combines parallel dilated convolutional layers and transformer blocks"),
    ("model2", "SETR", "SETR", "transformer_baseline", "pure_vit", "pure transformer encoder", 3, "Segmentation Transformer (SETR) [27] first replaced ... encoders with a pure transformer"),
    ("model3", "FCN-8s", "FCN-8s", "cnn_baseline", "cnn", "fully convolutional network", 2, "Since fully convolutional networks (FCNs) first achieved the aim of image segmentation [11]"),
    ("model4", "PSPNet", "PSPNet", "cnn_baseline", "cnn", "pyramid pooling module", 3, "Zhao et al. [22] proposed a pyramid pooling module (PPM)"),
    ("model5", "Deeplab V3+", "DeepLab V3+", "cnn_baseline", "cnn", "atrous spatial pyramid pooling", 3, "In Deeplab V3 and V3+ [20,21], they extended the ASPP module with different dilation rates of atrous convolution"),
    ("model6", "UNet", "UNet", "cnn_baseline", "cnn", "encoder-decoder with skip connections", 2, "such as Unet [12], the Deeplab series [13], HRnet [14]"),
    ("model7", "HRNet", "HRNet", "cnn_baseline", "cnn", "high-resolution network", 2, "such as Unet [12], the Deeplab series [13], HRnet [14]"),
    ("model8", "MSCSA-Net", "MSCSA-Net", "cnn_baseline", "cnn", "multi-scale channel spatial attention network", 3, "this network only used convolutional layers to build the channel and spatial attention module"),
]
# Table 2 (p.10, read from the rendered page): OA, mIoU, F1, Kappa (all %)
T2V = {
    "FCN-8s": "95.05 63.81 71.13 42.69", "PSPNet": "95.07 64.62 72.05 44.48", "Deeplab V3+": "95.81 69.21 75.98 52.28", "UNet": "95.98 69.78 76.05 52.43",
    "HRNet": "95.44 66.67 73.51 47.50", "SETR": "94.19 65.84 73.09 46.78", "MSCSA-Net": "95.87 69.82 76.24 53.16", "DB-MSC Net (Ours)": "96.14 70.95 77.66 55.63",
}
res = Counter("res")
for item, lab, name, role, cat, bb, page, ev in M:
    ROWS += model(item, name, role, cat, bb, None, None, page, "Sec. 3" if page > 1 else "Abstract", ev, cat_note={"model1": "CNN encoder with transformer blocks in the MSC block", "model2": "SETR, pure transformer encoder (Sec. 2)"}.get(item, "convolutional network"), **NP)
    oa, miou, f1, kappa = T2V[lab].split()
    e = f"{lab} {T2V[lab]}"
    st = "Table 2: street tree segmentation, mean of five runs"
    ROWS += result(res(), "iou", item, "dataset1", miou, 10, T2, e, unit="%", setting=st, note="labelled mIoU; its printed formula (Eq. 20) is TP/(TP+FP+FN) of street trees", **T, primary=True)
    ROWS += result(res(), "f1", item, "dataset1", f1, 10, T2, e, unit="%", setting=st, note="F1 of street trees; the printed formula (Eq. 19) omits the factor 2 of the numerator", **T, primary=True)
ROWS += [R("limitations", "Street trees in shadow-covered areas, and in areas with dense buildings (shadows), are segmented poorly", 12, "Sec. 5", "the method we proposed in this paper did not show a good effect for some street trees in shadow-covered areas")]
ROWS += avail(data="yes", data_page=4, data_loc="Sec. 3.1", data_ev="The dataset is publicly available on " + URL)
ROWS += [D("notes", "Table 2 compares DB-MSC Net with seven networks (OA, mIoU, F1, Kappa); iou and F1 are recorded for all eight. The table's mIoU is recorded as iou because the paper's formula (Eq. 20) is the IoU of street trees; its values are not consistent with the printed formulas (for the DB-MSC Net row, mIoU 70.95 would give a street-tree F1 of 83.0, while 77.66 is printed), so the table values may not follow the printed formulas. Table 3 (ablation of the double branch and the channel attention) is not recorded.",
           "Sec. 3.5 to 4.3 and Tables 2 and 3 (pp. 8 to 11)")]
