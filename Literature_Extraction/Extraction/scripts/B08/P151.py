import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T2, T4, T5 = "P151", "Table 1", "Table 2", "Table 4", "Table 5"
# TreeFinder: a benchmark dataset of individual tree mortality (pixel labels from hand-delineated dead tree polygons) in NAIP imagery (0.6 m, four bands) of 1,000 sites in the contiguous United States, with six
# segmentation models: U-Net and DeepLabV3+ (CNN), ViT, SegFormer, Mask2Former (transformers) and DOFA (remote sensing foundation model). The paper has no single main result: Table 1 (random 80/20 split, mean of three runs)
# is the baseline scenario and gives the dead tree class and the "All" metrics (the mean of the dead tree class and the background class, so never primary). The dead tree F1 and IoU are recorded and primary for Table 1;
# the scenarios of generalisation (Tables 2, 4 and 5; Tables 4 and 5 repeat the "Climate", "Climate-hard", "Forest" and "Forest-hard" rows of the main Table 3) are recorded as results without primary and as transfer pairs
# without an in-domain value (no source-domain score of the same trained model is printed). The "All" columns, precision, recall and accuracy are not recorded (F1 and IoU are given). Fig. 3 (F1 against the training size, no values) is not recorded.
TP = Tags(task="mortality_detection", level="pixel", split="test", agg="mean")
TS = Tags(task="mortality_detection", level="pixel", split="test", agg="single")
ABS_EV = "we introduce TreeFinder, the first high-resolution remote sensing benchmark dataset designed for individual-level tree mortality mapping across the Contiguous United States (CONUS)"
DS_EV = "In total, we annotate NAIP images at 1,000 unique sites across CONUS"
PATCH_EV = "N = 15, 489 image patches"
PUB_EV = "Our dataset and code are publicly available on Kaggle and GitHub"
ROWS = [
    R("task", "mortality_detection", 1, "Abstract", ABS_EV, note="pixel-level mapping of dead trees (scattered individual tree deaths), scored per pixel of the dead tree class"),
    R("output_unit", "crown_mask", 5, "Sec. 3.1", "we delineate the individual dead trees using polygonbased tools through Google Earth Engine", note="binary dead tree masks from hand-drawn polygons of individual dead trees"),
    R("tree_setting", "natural_forest", 1, "Abstract", "TreeFinder contains images with pixel-level labels generated via extensive manual annotation that covers forested areas in 48 states with over 23,000 hectares", note="forested areas defined by a national forest cover basemap; forest types are given only as primary tree types"),
    R("tree_taxa", "mixed forests of the contiguous United States (primary tree types include maple, pine, oak, Douglas fir and cottonwood)", 6, "Sec. 3.3", "the top five most frequent primary tree types", note="primary tree type per image from the USDA Forest Service tree species parameter maps"),
    R("study_area", "1,000 sites in 48 states of the contiguous United States (over 23,000 hectares)", 5, "Sec. 3.1", DS_EV),
    D("biome", "multiple", "sites across the contiguous United States with several Koppen-Geiger climate zones (Sec. 3.1, p.5)"),
    R("platform", "airborne", 1, "Abstract", "the dataset uses NAIP imagery at 0.6m resolution that provides wall-to-wall coverage of the entire CONUS", note="NAIP aerial imagery"),
    R("sensors", "rgb;multispectral", 4, "Sec. 3.1", "NAIP provides 4-band aerial images including RGB and near-infrared (NIR) channels"),
    R("gsd", "0.6", 5, "Sec. 3.1", "we resampled these images to 0.6 m to maintain a consistent resolution across all samples", unit="m", note="imagery after 2021; the few 0.3 m images are resampled to 0.6 m"),
    R("annotation_type", "polygon_mask", 5, "Sec. 3.1", "we delineate the individual dead trees using polygonbased tools through Google Earth Engine", note="cross-annotator agreement above 95 percent on a random 20 percent of the sites"),
    R("dataset_name", "TreeFinder: NAIP individual tree mortality benchmark, 1,000 sites, patches of 224 x 224 pixels with four bands", 1, "Abstract", ABS_EV, item="dataset1"),
    R("dataset_key", "treefinder", 1, "Abstract", ABS_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 3.3", "20% of the labeled patches are held out as a fixed test set", item="dataset1", note="random 80/20 split plus the generalisation scenarios by region, climate zone and primary tree type"),
    R("dataset_public", "yes", 1, "Abstract", PUB_EV, item="dataset1"),
    R("dataset_url", "https://www.kaggle.com/datasets/zhihaow/tree-finder", 1, "Abstract", "https://www.kaggle.com/datasets/zhihaow/tree-finder", item="dataset1"),
    R("n_images", "15489", 5, "Sec. 3.2", PATCH_EV, item="dataset1", unit="patches", note="non-overlapping patches of 224 x 224 pixels with 4 bands, printed as N = 15, 489"),
]
MODELS = [  # table label, name, role, category, backbone, pretraining, strategy, pretraining note, evidence (p.7)
    ("U-Net", "U-Net", "cnn_baseline", "cnn", "U-Net encoder-decoder with skip connections", "none", "from_scratch", "trained from scratch",
     "Our U-Net is trained from scratch to provide a baseline with localized spatial modeling and no reliance on pretraining"),
    ("DeepLabV3+", "DeepLabV3+", "cnn_baseline", "cnn", "ResNet-50 with atrous spatial pyramid pooling", "imagenet", "full_fine_tuning", "ImageNet-pretrained ResNet-50, input and output customised",
     "A CNN-based model with a ResNet-50 backbone, leveraging atrous spatial pyramid pooling to aggregate multi-scale contextual features"),
    ("ViT", "ViT (segmentation)", "transformer_baseline", "pure_vit", "patch-based Vision Transformer with a lightweight transposed convolution decoder", "imagenet", "full_fine_tuning", "ImageNet-pretrained, segmentation head added",
     "A patch-based transformer with a lightweight transposed convolution decoder that upsamples hidden features back to full resolution"),
    ("SegFormer", "SegFormer", "transformer_baseline", "hierarchical_vit", "SegFormer hierarchical transformer encoder with a lightweight decoder", "other", "full_fine_tuning", "ADE20K-pretrained",
     "A hierarchical transformer architecture that is designed for semantic segmentation"),
    ("Mask2Former", "Mask2Former", "transformer_baseline", "hierarchical_vit", "Swin-Tiny backbone with multiscale deformable attention and a class-agnostic mask head", "other", "full_fine_tuning", "ADE20K-pretrained weights",
     "A transformer-based framework that combines a Swin-Tiny backbone with multiscale deformable attention and a class-agnostic mask prediction head"),
    ("DOFA", "DOFA", "transformer_baseline", "foundation_model", "DOFA multimodal foundation model for remote sensing (wavelength-conditioned spectral band embedding)", "domain_specific", "full_fine_tuning",
     "pre-trained on multi-sensor remote sensing imagery including Sentinel-1/2 and NAIP", "A multimodal foundation model specifically designed for remote sensing images"),
]
NP = dict(strat_note="batch size 32, initial learning rate 1e-4 with Adam, up to 100 epochs, binary cross-entropy plus Dice loss, early stopping on the validation loss (Sec. 4.1); fine-tuned when pretrained weights exist")
MID = {}
for k, (lab, name, role, cat, bb, pre, strat, prenote, ev) in enumerate(MODELS, start=1):
    item = f"model{k}"
    MID[lab] = item
    ROWS += model(item, name, role, cat, bb, pre, strat, 7, "Sec. 4.1", ev, cat_note={"foundation_model": "remote sensing foundation model (protocol 4.5)", "pure_vit": "patch-based plain ViT encoder (protocol 4.5)",
                  "hierarchical_vit": "hierarchical transformer (SegFormer MiT, or Mask2Former with a Swin-Tiny backbone; protocol 4.5)"}.get(cat, "convolutional segmentation network"), pre_note=prenote, **NP)
res = Counter("res")
RESULT_NOTE = "F1 or IoU of the dead tree class (the All columns average the dead tree class and the background class and are not recorded)"
# Table 1: random 80/20 split, mean of three runs (standard deviation in the note)
for lab, name, *_ in MODELS:
    mm = re.escape(lab)
    nums, line = row(PID, 8, rf"^\s*{mm}\s+(?=\d)", 0)
    allv, _ = row(PID, 8, rf"^\s*{mm}\s+(?=\d)", 0, drop_sd=False)
    assert len(nums) == 9 and len(allv) == 18, (lab, nums)
    for metric, v, sd, nm in (("f1", nums[0], allv[1], "F1"), ("iou", nums[6], allv[13], "IoU")):
        ROWS += result(res(), metric, MID[lab], "dataset1", v, 8, T1, line, unit="%", setting="Table 1: random 80/20 split, fixed test set; dead tree class", note=f"{nm} of the dead tree class, mean of three runs (SD {sd}); the All columns are not recorded",
                       **TP, primary=True)
# generalisation scenarios (Tables 2, 4 and 5): dead tree F1 and IoU as results without primary, and the F1 as a transfer pair without an in-domain value
SC = [  # page, table, scenario, description, source, target, shift
    (9, T2, "E-W", "train on eastern states, test on western states", "eastern states of the contiguous United States", "western states", "region"),
    (9, T2, "W-E", "train on western states, test on eastern states", "western states of the contiguous United States", "eastern states", "region"),
    (9, T2, "CO", "train on Colorado, test on all other states", "Colorado", "all other states", "region"),
    (23, T4, "Climate", "train on a group of climate zones (about half of the data), test on the remaining zones", "group of climate zones with about 50 percent of the data", "the remaining climate zones", "biome"),
    (23, T4, "Climate-swap", "the two climate zone groups of Climate swapped", "the second climate zone group", "the first climate zone group", "biome"),
    (23, T4, "Climate-hard", "train on the humid continental climate zone only, test on all other zones", "humid continental climate zone", "all other climate zones", "biome"),
    (24, T5, "Forest", "train on the five most frequent primary tree types (about half of the data), test on the other types", "five most frequent primary tree types (maple, pine, oak, Douglas fir, cottonwood)", "all other primary tree types", "species"),
    (24, T5, "Forest-swap", "the two primary tree type groups of Forest swapped", "the other primary tree types", "the five most frequent primary tree types", "species"),
    (24, T5, "Forest-hard", "train on maple only, test on all other primary tree types", "maple as primary tree type", "all other primary tree types", "species"),
]
tr = Counter("tr")
for pg, tb, sc, desc, src, tgt, shift in SC:
    for lab, name, *_ in MODELS:
        nums, line = row(PID, pg, rf"^\s*{re.escape(lab)}\s+{re.escape(sc)}\s+(?=\d)", 0)
        assert len(nums) == 9, (sc, lab, nums)
        setting = f"{tb}: scenario {sc} ({desc}); dead tree class"
        same = " (the same row appears in the main Table 3)" if sc in ("Climate", "Climate-hard", "Forest", "Forest-hard") else ""
        for metric, v, nm in (("f1", nums[0], "F1"), ("iou", nums[6], "IoU")):
            ROWS += result(res(), metric, MID[lab], "dataset1", v, pg, tb, line, unit="%", setting=setting, note=f"{nm} of the dead tree class on the unseen test domain{same}; the All columns are not recorded", **TS)
        t = tr()
        ROWS += [R("transfer_source", src, pg, tb, line, item=t), R("transfer_target", tgt, pg, tb, line, item=t),
                 R("shift_type", shift, pg, tb, line, item=t, note=f"scenario {sc}: {desc}"),
                 R("transfer_model", MID[lab], pg, tb, line, item=t), R("transfer_metric", "f1", pg, tb, line, item=t),
                 NR("value_in_domain", item=t, note="no score of the same trained model on its source domain is printed; the random split of Table 1 is another training set"),
                 R("value_out_of_domain", nums[0], pg, tb, line, item=t, unit="%", note="F1 of the dead tree class"), R("transfer_level", "pixel", pg, tb, line, item=t)]
ROWS += [R("limitations", "NAIP covers only the contiguous United States, the change of NAIP resolution over time is not considered, and training strategies such as active or meta-learning are not evaluated", 10, "Sec. 5",
           "First, NAIP offers near wall-to-wall coverage across CONUS, but is not available at the global scale")]
ROWS += avail(code="yes", weights="not_stated", data="yes", code_url="https://github.com/zhwang0/treefinder", page=1, loc="Abstract", ev=PUB_EV, data_page=1, data_loc="Abstract", data_ev=PUB_EV)
ROWS += [D("notes", "A benchmark dataset paper: six segmentation models (U-Net, DeepLabV3+, ViT, SegFormer, Mask2Former, DOFA) on TreeFinder with a random 80/20 split (Table 1, mean of three runs with standard deviations) and scenarios of generalisation by region (Table 2: E-W, W-E, CO), climate zone (Tables 3 and 4: Climate, Climate-swap, Climate-hard) and primary tree type (Tables 3 and 5: Forest, Forest-swap, Forest-hard). The dead tree F1 and IoU of Table 1 are primary (the All columns average the dead tree and the background class and are not recorded, nor are precision, recall and accuracy); the scenario values are recorded as results and as F1 transfer pairs without an in-domain value, because no source-domain score of the same trained model is printed (the random split model is trained on another set). Tables 4 and 5 repeat the Climate, Climate-hard, Forest and Forest-hard rows of the main Table 3 and add the swap rows. The paper has no single main result (no headline). Mask2Former has the best dead tree F1 in the random split (51.9) and SegFormer in most generalisation scenarios; DOFA is the lowest in the random split (29.2). Fig. 3 (F1 against the training data size from 10 to 80 percent) shows no values and is not recorded. Labels are pixel masks of individual dead trees; the scored unit is the pixel.",
           "Sec. 3 and 4, Tables 1 to 5 (pp. 5 to 10 and 23 to 24)")]
