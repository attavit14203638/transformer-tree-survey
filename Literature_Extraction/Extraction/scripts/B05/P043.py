import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T5 = "P043", "Table III (image)", "Table V (image)"
# Binary pixel-wise tree segmentation of street-view and smartphone photos: crown segmentation at pixel level; IoU is the tree-class IoU averaged over the test images (Table I).
T = Tags(task="crown_segmentation", level="pixel", split="test", agg="single")
TASK_EV = "the present work inspects whether or not reconstructed spectral images add relevant information to segmentation networks for improving urban tree identification"
JEK_EV = "The Jekyll dataset samples are grouped in three subsets of 3168, 395, and 386 samples for training, validation, and testing, respectively"
ARB_EV = "A total of 317 images were annotated and splited in 253, 32, and 32 subsets for training, evaluating, and testing segmentation algorithms, correspondingly"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", TASK_EV),
    R("output_unit", "crown_mask", 3, "Sec. 2.B", "tree pixel-wise labeling was performed by a team of professional image annotators using LabelMe software", note="binary tree masks; the tree is the only class (Table I)"),
    R("tree_setting", "urban", 3, "Sec. 2.B", "Dataset is a comprehensive tree dataset of urban street view images", note="Jekyll; Arbocensus are smartphone images of urban trees"),
    R("tree_taxa", "urban trees (Jekyll 22 species, Arbocensus about 100 species)", 3, "Sec. 2.B", "covering over three thousand images of 22 tree species ... capture about 3000 images of 100 species"),
    R("study_area", "ten cities in China (Jekyll) and Las Condes and La Reina, Santiago, Chile (Arbocensus)", 3, "Sec. 2.B",
      "images acquired with mobile devices in China’s ten cities ... mapped from Las Condes, and La Reina communes under the project Arbocensus in Santiago metropolitan region, Chile"),
    D("biome", "multiple", "ten Chinese cities and Santiago, Chile; not stated in the paper"),
    R("platform", "ground_camera", 3, "Sec. 2.B", "images acquired with mobile devices", note="Jekyll mobile devices; Arbocensus smartphones of volunteers"),
    R("sensors", "rgb", 3, "Sec. 2.B", "RGB images that are captured by volunteers using smartphones", note="ReD and NIR are not captured: they are reconstructed from RGB by a cGAN trained on the HSICityV2 hyperspectral dataset"),
    R("annotation_type", "polygon_mask", 3, "Sec. 2.B", "using polygons to outline individual objects", note="Arbocensus polygons; Jekyll pixel-wise labels (LabelMe)"),
    R("dataset_name", "Jekyll street-view urban tree dataset (3949 images, ten Chinese cities)", 3, "Sec. 2.B", JEK_EV, item="dataset1"),
    R("dataset_key", "jekyll_street_view_trees", 3, "Sec. 2.B", JEK_EV, item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.B", JEK_EV, item="dataset1", note="3168 training, 395 validation, 386 test images"),
    D("dataset_public", "no", "Published by Yang et al. [37]; this paper gives no link or availability statement", item="dataset1"),
    R("n_images", "3949", 1, "Abstract", "Jekyll, 3949 samples", item="dataset1", unit="images"),
    R("dataset_name", "Arbocensus smartphone urban tree images, Santiago (317 annotated images)", 3, "Sec. 2.B", ARB_EV, item="dataset2"),
    R("dataset_key", "own_p043_arbocensus", 3, "Sec. 2.B", ARB_EV, item="dataset2"),
    R("dataset_role", "train_test", 3, "Sec. 2.B", ARB_EV, item="dataset2", note="253 training, 32 validation, 32 test images"),
    D("dataset_public", "no", "No data link or availability statement in the paper", item="dataset2"),
    R("n_images", "317", 1, "Abstract", "Arbocensus, 317 samples", item="dataset2", unit="images"),
]
PRE = dict(pre_page=4, pre_loc="Sec. 3", pre_ev="deep neural networks (segmentation models) are pre-trained with two urban datasets (ADE20k [40] and CityScapes [41])", pre_note="ADE20k or Cityscapes (two pre-training sets, Table III)",
           strat_page=5, strat_loc="Sec. 3.C", strat_ev="pretrained on ADE20 k and CityScapes dataset are fine-tuned with the new multispectral datasets",
           strat_note="MMSegmentation default parameters: DeepLabV3 80 thousand and SegFormer 160 thousand iterations, batch size one (Sec. 3.D)")
SEL = "we select the DeepLabV3 with R-50-D8 backbone and SegFormer with MIT-B0 backbone from the MMSegmentation model zoo"
ROWS += model("model1", "DeepLabV3 (R-50-D8)", "cnn_baseline", "cnn", "ResNet-50-D8", "other", "full_fine_tuning", 5, "Sec. 3.B", SEL, cat_note="DeepLabV3 is a convolutional network (Sec. 3.B)", **PRE)
ROWS += model("model2", "SegFormer (MiT-B0)", "transformer_baseline", "hierarchical_vit", "MiT-B0", "other", "full_fine_tuning", 5, "Sec. 3.B", SEL, cat_note="SegFormer unifies transformers with lightweight MLP decoders (Sec. 3.B)", **PRE)
ROWS += model("model3", "SegFormer (MiT-B0) without pre-training", "transformer_baseline", "hierarchical_vit", "MiT-B0", "none", "from_scratch", 8, "Table V (image)", "datasets are not used as pretraining sets",
              cat_note="SegFormer; Table V footnote: the ADE20k and Cityscapes datasets are not used as pre-training sets",
              pre_page=8, pre_loc="Table V (image)", pre_ev="datasets are not used as pretraining sets", pre_note="Table V footnote", strat_page=8, strat_loc="Table V (image)", strat_ev="datasets are not used as pretraining sets", strat_note="no pre-training (Table V)")

# Table III (p.7, read from the rendered page at 260 dpi): average IoU of the tree class per input, for each pre-training set and network
IOU3 = {  # (pretrain, network): {dataset: (RGB, RGB+ReD, RGB+NIR)}
    ("ADE20k", "model1"): {"dataset1": ("0.90", "0.90", "0.90"), "dataset2": ("0.79", "0.78", "0.79")},
    ("ADE20k", "model2"): {"dataset1": ("0.89", "0.89", "0.88"), "dataset2": ("0.8", "0.82", "0.76")},
    ("Cityscapes", "model1"): {"dataset1": ("0.90", "0.90", "0.90"), "dataset2": ("0.80", "0.80", "0.79")},
    ("Cityscapes", "model2"): {"dataset1": ("0.89", "0.88", "0.89"), "dataset2": ("0.82", "0.83", "0.77")},
}
IOU5 = {"dataset1": ("0.85", "0.85", "0.84"), "dataset2": ("0.75", "0.73", "0.72")}  # Table V (p.8), SegFormer without pre-training
INP = ("RGB", "RGB+ReD", "RGB+NIR")
NM = {"model1": "DeepLabV3", "model2": "SegFormer", "model3": "SegFormer"}
DS = {"dataset1": "Jekyll", "dataset2": "Arbocensus"}
res = Counter("res")
for (pre, item), per in IOU3.items():
    for ds, vals in per.items():
        for k, v in enumerate(vals):
            prim = pre == "ADE20k" and ds == "dataset1" and k == 0
            ev = f"{pre} {NM[item]} IoU avg {' '.join(vals)}"
            ROWS += result(res(), "iou", item, ds, v, 7, T3, ev, unit="fraction", setting=f"Table III: {DS[ds]}, pre-trained on {pre}, input {INP[k]}", note="average tree-class IoU over the test images (Table I: a/(a+b+c))", **T, primary=prim)
for ds, vals in IOU5.items():
    for k, v in enumerate(vals):
        ROWS += result(res(), "iou", "model3", ds, v, 8, T5, f"SegFormer IoU avg {' '.join(vals)}", unit="fraction", setting=f"Table V: {DS[ds]}, no pre-training, input {INP[k]}", note="average tree-class IoU over the test images", **T)
ROWS += [R("limitations", "Only reconstructed reflectance in the red-edge and NIR bands was tested, and illumination and shadow effects on the cGAN were not explored", 9, "Sec. 5", "only reconstructed reflectance in the ReD and NIR bands was tested in the current work")]
ROWS += avail()
ROWS += [D("notes", "Table III gives average IoU, precision and recall (with std) of DeepLabV3 and SegFormer on Jekyll and Arbocensus for RGB, RGB+ReD and RGB+NIR with ADE20k or Cityscapes pre-training; Table V repeats SegFormer without pre-training. Average IoU is recorded for all configurations (primary: Jekyll, ADE20k, RGB). Table II (cGAN reconstruction quality on HSICityV2: SSIM, PSNR, R) and Table IV (t-test p-values) are not tree results and are not recorded.",
           "Sec. 4 and Tables III to V (pp. 7 to 8)")]
