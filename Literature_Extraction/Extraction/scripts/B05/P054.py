import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T5, T7 = "P054", "Table II (image)", "Table V (image)", "Table VII (image)"
# Pixel-wise segmentation of citrus canopies: crown segmentation at pixel level on the test fifth of the patches. F1 is the canopy-class F1 (precision and recall of canopy pixels, Sec. 3.5);
# the tables' mIoU is recorded as miou and not primary because the paper's IoU formula is the canopy-class IoU while the tables call it mIoU. Primary: the 3-D input (RGB plus canopy height
# model) of the proposed network, the setting of Table V (3-D rows) and Table VII, whose Proposed row equals the 3-D result.
T = Tags(task="crown_segmentation", level="pixel", split="test", agg="single")
OBJ_EV = "we propose a parallel fusion neural network considering both local and global semantic information for citrus tree canopy segmentation from 3-D data"
SMP_EV = "generating a total of 25 000 samples with a patch size of 256×256"
SPLIT_EV = "One-fifth of these samples are test datasets"
GSD_EV = "DOM and DSM with a spatial resolution of 4 cm/pix"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", OBJ_EV),
    R("output_unit", "crown_mask", 1, "Abstract", OBJ_EV, note="binary canopy masks"),
    R("tree_setting", "orchard", 7, "Sec. 4.A", "the study area is suitable for planting citrus trees, such as navel orange trees"),
    R("tree_taxa", "citrus (navel orange)", 7, "Sec. 4.A", "the study area is suitable for planting citrus trees, such as navel orange trees"),
    R("study_area", "Xinfeng County, Jiangxi Province, China", 7, "Sec. 4.A", "the study area (located in Xinfeng County, Jiangxi Province, China)"),
    R("biome", "subtropical", 7, "Sec. 4.A", "the typical subtropical monsoon humid climate with abundant rainfall and sufficient sunlight"),
    R("platform", "uav", 7, "Sec. 4.B", "a small quadcopter UAV (DJI Phantom 4 RTK, DJI, Shenzhen, China)"),
    R("sensors", "rgb;dsm", 1, "Abstract", "3-D data incorporating RGB imagery and canopy height model derived by UAV photogrammetry", note="canopy height model = DSM minus DTM"),
    R("gsd", "4", 8, "Sec. 4.B", GSD_EV, unit="cm", note="DOM and DSM, the inputs of the networks; the images themselves are about 3 cm/pix (p.7)"),
    R("annotation_type", "polygon_mask", 8, "Sec. 4.C", "we manually delineated samples from UAV photogrammetry"),
    R("dataset_name", "own UAV citrus canopy dataset, Xinfeng (four plots, 25,000 patches of 256 × 256 after augmentation)", 8, "Sec. 4.C", SMP_EV, item="dataset1"),
    R("dataset_key", "own_p054_xinfeng_citrus", 8, "Sec. 4.C", SMP_EV, item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 4.C", SPLIT_EV, item="dataset1", note="one fifth test, the rest training and validation"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_images", "25000", 8, "Sec. 4.C", SMP_EV, item="dataset1", unit="tiles", note="after four-fold augmentation by rotation, flip and affine transformation"),
]
# Rows of the image tables as printed: name -> (OA, F1, mIoU, parameters in M, FLOPs in G) for Table II (2-D)
T2V = {
    "FCN": ("FCN [7]", "0.9508", "0.8723", "0.8641", "15.90", "160.97"), "BiseNet-V2": ("BiseNet-V2 [47]", "0.9607", "0.8768", "0.8695", "3.62", "25.75"),
    "U-Net": ("U-Net [8]", "0.9693", "0.8982", "0.8870", "13.40", "248.23"), "PSPNet": ("PSPNet [34]", "0.9691", "0.9078", "0.8973", "72.46", "347.40"),
    "DANet": ("DANet [11]", "0.9694", "0.9084", "0.8981", "70.94", "405.02"), "FPN": ("FPN [46]", "0.9696", "0.9096", "0.8991", "53.71", "105.11"),
    "EfficientNet-V2-S": ("EfficientNet-V2-S [40]", "0.9690", "0.9102", "0.9005", "26.76", "222.62"), "HRCNet_W48": ("HRCNet_W48 [35]", "0.9702", "0.9111", "0.9008", "62.71", "187.38"),
    "DeepLab-V3": ("DeepLab-V3 [9]", "0.9712", "0.9125", "0.9025", "65.50", "347.04"), "ResT-Tiny": ("ResT-Tiny [17]", "0.9677", "0.9041", "0.8935", "18.50", "211.02"),
    "Swin-Tiny": ("Swin-Tiny [19]", "0.9680", "0.9047", "0.8941", "36.80", "243.52"), "BeiT-Base": ("BeiT-Base [18]", "0.9685", "0.9066", "0.8960", "102.99", "424.93"),
    "CSwin-Tiny": ("CSwin-Tiny [20]", "0.9708", "0.9136", "0.9033", "30.31", "240.75"),
}
# Table V: the same networks and the proposed network with 3-D data (OA, F1, mIoU), and the proposed network with 2-D data
T5_3D = {"FCN": ("0.9537", "0.9046", "0.8816"), "U-Net": ("0.9591", "0.9241", "0.9106"), "EfficientNet-V2-S": ("0.9632", "0.9341", "0.9159"), "HRCNet_W48": ("0.9668", "0.9409", "0.9243"),
         "DeepLab-V3": ("0.9685", "0.9394", "0.9225"), "CSwin-Tiny": ("0.9711", "0.9407", "0.9231"), "Proposed": ("0.9735", "0.9498", "0.9346")}
T5_2D_PROP = ("0.9710", "0.9237", "0.9116")
# Table VII: state-of-the-art networks (OA, F1, mIoU, parameters in M, FLOPs in G); the Proposed row is the 3-D result
T7V = {"SETR_PUP": ("SETR_PUP [16]", "0.9386", "0.8842", "0.8561", "86.07", "192.39"), "TransUNet": ("TransUNet [36]", "0.9701", "0.9404", "0.9236", "37.39", "112.22"),
       "TransFuse": ("TransFuse [38]", "0.9690", "0.9384", "0.9211", "70.34", "191.46"), "CCTNet": ("CCTNet [48]", "0.9706", "0.9309", "0.9171", "62.95", "178.92")}
PROP_PARAMS = ("50.25", "193.16")
M = [  # item, key, name, role, category, backbone, pre kind, note
    ("model1", "Proposed", "Proposed parallel fusion network (EfficientNet-V2 and CSwin)", "proposed", "cnn_transformer_hybrid", "EfficientNet-V2-S and CSwin-Tiny in parallel with a coordinate attention fusion module (CAFM) and FPNHead decoder", "tl",
     "parallel CNN and transformer branches (Abstract)"),
    ("model2", "CSwin-Tiny", "CSwin-Tiny", "transformer_baseline", "hierarchical_vit", "CSwin-Tiny with FPNHead decoder", "ft", "CSwin transformer (the transformer branch of the proposed network)"),
    ("model3", "Swin-Tiny", "Swin-Tiny", "transformer_baseline", "hierarchical_vit", "Swin-Tiny with FPNHead decoder", "ft", "transformer model with multiscale feature fusion (Sec. 4.F)"),
    ("model4", "ResT-Tiny", "ResT-Tiny", "transformer_baseline", "hierarchical_vit", "ResT-Tiny with FPNHead decoder", "ft", "transformer model with multiscale feature fusion (Sec. 4.F)"),
    ("model5", "BeiT-Base", "BeiT-Base", "transformer_baseline", "pure_vit", "BeiT-Base with FPNHead decoder", "ft", "transformer model (Sec. 4.F)"),
    ("model6", "EfficientNet-V2-S", "EfficientNet-V2-S", "cnn_baseline", "cnn", "EfficientNet-V2-S with FPNHead decoder", "ft", "the CNN branch of the proposed network"),
    ("model7", "DeepLab-V3", "DeepLab-V3 (ResNet-50)", "cnn_baseline", "cnn", "ResNet-50", "ft", "best CNN in Table II on OA, F1 and mIoU"),
    ("model8", "HRCNet_W48", "HRCNet_W48", "cnn_baseline", "cnn", "HRCNet_W48", "ft", "best CNN on F1 and mIoU with 3-D data (Table V)"),
    ("model9", "FCN", "FCN", "cnn_baseline", "cnn", "fully convolutional network", "ft", "CNN"),
    ("model10", "U-Net", "U-Net", "cnn_baseline", "cnn", "U-Net", "ft", "CNN"),
    ("model11", "SETR_PUP", "SETR_PUP", "transformer_baseline", "pure_vit", "SETR with progressive upsampling decoder", "ft", "attention in the encoder, convolution in the decoder (Sec. 4.G)"),
    ("model12", "TransUNet", "TransUNet", "transformer_baseline", "cnn_transformer_hybrid", "serial CNN and transformer with skip connections", "ft", "serial structure fusing CNN and transformer (Sec. 4.G)"),
    ("model13", "TransFuse", "TransFuse", "transformer_baseline", "cnn_transformer_hybrid", "parallel CNN and transformer branches", "ft", "parallel structure fusing CNN and transformer (Sec. 4.G)"),
    ("model14", "CCTNet", "CCTNet", "transformer_baseline", "cnn_transformer_hybrid", "parallel CNN and transformer branches", "ft", "parallel structure fusing CNN and transformer (Sec. 4.G)"),
]
FT = dict(pre_status="not_reported", pre_note="pretrained network modules (transfer learning); the pre-training data are not named", strat_page=9, strat_loc="Sec. 4.D",
          strat_ev="these networks performed a fine-tuning operation using the training samples", strat_note="fine-tuned on the training samples (Sec. 4.D)")
for item, key, name, role, cat, bb, pk, note in M:
    if key in T2V:
        page, loc, ev = 10, T2, " ".join(T2V[key][:4])
    elif key in T7V:
        page, loc, ev = 13, T7, " ".join(T7V[key][:4])
    else:
        page, loc, ev = 1, "Abstract", "a parallel architecture, concatenated by EfficientNet-V2 and CSwin transformer, is used to extract local and global information of citrus trees"
    kw = dict(FT)
    if pk == "tl":
        kw.update(strat_ev="given the use of several pretrained network modules through transfer learning", strat_page=8, strat_loc="Sec. 4.C", strat_note="transfer learning of EfficientNet-V2 and CSwin modules; AdamW with a polynomial learning rate decay (Sec. 4.D)")
    ROWS += model(item, name, role, cat, bb, None, "full_fine_tuning", page, loc, ev, cat_note=note, **kw)
    if key in T2V:
        full = " ".join(T2V[key])
        ROWS += [R("total_params", T2V[key][4], 10, T2, full, item=item, unit="M"), R("flops", T2V[key][5], 10, T2, full, item=item, unit="GFLOPs")]
    elif key in T7V:
        full = " ".join(T7V[key])
        ROWS += [R("total_params", T7V[key][4], 13, T7, full, item=item, unit="M"), R("flops", T7V[key][5], 13, T7, full, item=item, unit="GFLOPs")]
PROP_ROW = "Proposed 0.9735 0.9498 0.9346 " + PROP_PARAMS[0] + " " + PROP_PARAMS[1]
ROWS += [R("total_params", PROP_PARAMS[0], 13, T7, PROP_ROW, item="model1", unit="M"), R("flops", PROP_PARAMS[1], 13, T7, PROP_ROW, item="model1", unit="GFLOPs")]
ITEM = {m[1]: m[0] for m in M}
res = Counter("res")
def add(key, oa, f1, miou, page, tab, ev, setting, prim):
    global ROWS
    ROWS += result(res(), "f1", ITEM[key], "dataset1", f1, page, tab, ev, unit="fraction", setting=setting, note="F1-score of the canopy class", **T, primary=prim)
    ROWS += result(res(), "miou", ITEM[key], "dataset1", miou, page, tab, ev, unit="fraction", setting=setting, note="labelled mIoU; the paper's IoU formula (Sec. 3.5) is the canopy-class IoU, but the values match the mean of the canopy and background IoU implied by the printed OA and F1", **T,
                   headline=(key == "Proposed" and "3-D" in setting and page == 11))
# 2-D (RGB) results: Table II networks and the proposed network of Table V
for key, v in T2V.items():
    if key in ITEM:
        add(key, v[1], v[2], v[3], 10, T2, " ".join(v[:4]), "Table II: 2-D RGB imagery", False)
add("Proposed", *T5_2D_PROP, 11, T5, "Proposed " + " ".join(T5_2D_PROP), "Table V: 2-D RGB imagery", False)
# 3-D (RGB and canopy height model) results: Table V networks and Table VII state-of-the-art networks
for key, v in T5_3D.items():
    add(key, v[0], v[1], v[2], 11, T5, f"{key} " + " ".join(v), "Table V: 3-D data (RGB and canopy height model)", True)
for key, v in T7V.items():
    add(key, v[1], v[2], v[3], 13, T7, " ".join(v[:4]), "Table VII: comparison with state-of-the-art methods (the Proposed row equals the 3-D result)", True)
ROWS += [D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table II compares 13 CNN and transformer networks on 2-D RGB imagery (OA, F1, mIoU, parameters, FLOPs); Table V repeats seven networks with 2-D and 3-D (RGB plus canopy height model) data; Table VII compares four CNN-transformer networks with the proposed network. Recorded: the proposed model, every transformer-based network, the CNN branch (EfficientNet-V2-S), the best CNNs (DeepLab-V3, HRCNet_W48), FCN and U-Net (Table V); BiseNet-V2, PSPNet, DANet and FPN of Table II are not recorded. Tables III (backbone sizes) and IV (ablation) and the four-plot results of Table VI are not recorded. Primary results are F1 with the 3-D input. The tables' mIoU is not primary: the Sec. 3.5 formula is the canopy-class IoU, but the printed values match the mean of the canopy IoU (from F1) and the background IoU (from OA and F1), e.g. 0.9345 against 0.9346 for the proposed 3-D row.",
           "Sec. 4 and Tables II to VII (pp. 10 to 13)")]
ROWS += conflict("cf1", "gsd", "3", 7, "Sec. 4.B", "which led to capturing UAV remote sensing images with a spatial resolution of approximately 3 cm/pix", "V1c",
                 "The orthophoto and surface model that the networks use are given at 4 cm/pix in the data section (p.8); the raw images are about 3 cm/pix (p.7)", unit="cm")
