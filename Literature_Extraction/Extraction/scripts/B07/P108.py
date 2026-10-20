import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P108"
# FBAN (hyperspectral individual tree segmentation and species recognition; Mask2Former-style decoder with a boundary-aware module and an adaptive crown-aware module that couples Transformer layers with dilated
# convolutions) against eight instance segmentation methods on two campus datasets. Tables V and VII (AP and AP50) and the Average rows of Tables VI and VIII (F1 averaged over the species) are images and are read from
# the rendered pages. The abstract's headline is on SZU-South (AP 4.5 points above Mask2former), so SZU-South is the primary dataset and SZU-North, listed first, is not (rule 6.3: the headline dataset).
# Species-wise segmentation and recognition scores averaged over the species classes: species_classification, instance_mask.
TS = Tags(task="species_classification", level="instance_mask", split="test", agg="single")
OBJ_EV = "this article proposes a fuzzy boundary-aware network (FBAN) for hyperspectral individual tree segmentation and recognition in dense forests"
N_EV = "In this experiment, 256 images were cropped from the captured"
S_EV = "97 images cropped from the acquired images"
CMP_EV = "are strictly set according to their original papers"

ROWS = [
    R("task", "crown_segmentation;species_classification", 1, "Abstract", OBJ_EV, note="individual tree segmentation and species recognition (instance masks with species classes)"),
    R("output_unit", "crown_mask;tree_label", 1, "Abstract", OBJ_EV),
    D("tree_setting", "urban", "trees of the north and south campuses of Shenzhen University (campus trees of 30 and 13 species); not stated as urban in the paper"),
    R("tree_taxa", "30 tree species on the north campus (for example Litchi chinensis, Araucaria cunninghamii, Ficus microcarpa) and 13 on the south campus of Shenzhen University", 7, "Sec. IV-A", "total of 5103 tree instances across 30 different tree species"),
    R("study_area", "north and south campuses of Shenzhen University, Shenzhen, China", 7, "Sec. IV-A", "using a UAV equipped with a Specim FX10 hyperspectral"),
    D("biome", "subtropical", "Shenzhen, China; coded from the study area"),
    R("platform", "uav", 7, "Sec. IV-A", "using a UAV equipped with a Specim FX10 hyperspectral"),
    R("sensors", "hyperspectral", 7, "Sec. IV-A", "112 continuous bands after preprocessing", note="Specim FX10 line-scanning camera, 400 to 1000 nm, 112 bands, spectral resolution 5.5 nm"),
    R("gsd", "0.1", 7, "Sec. IV-A", "resolution of 0.1 m per pixel", unit="m"),
    R("annotation_type", "polygon_mask", 7, "Sec. IV-A", "total of 5103 tree instances across 30 different tree species", note="instance masks of individual trees with species labels, from field surveys of species and positions"),
    R("dataset_name", "own SZU-North hyperspectral UAV dataset (north campus, 30 species)", 7, "Sec. IV-A", N_EV, item="dataset1"),
    R("dataset_key", "own_p108_szu_north", 7, "Sec. IV-A", N_EV, item="dataset1"),
    R("dataset_role", "train_test", 8, "Sec. IV-B", "In the SZU-North dataset, 192 images are randomly selected, which contain 3778 annotated masks for training", item="dataset1", note="192 training images with 3778 masks and 64 test images with 1325 masks"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_images", "256", 7, "Sec. IV-A", N_EV, item="dataset1", unit="images", note="512 x 512 pixel images cropped from the hyperspectral data"),
    R("n_trees", "5103", 7, "Sec. IV-A", "total of 5103 tree instances across 30 different tree species", item="dataset1", unit="trees"),
    R("dataset_name", "own SZU-South hyperspectral UAV dataset (south campus, 13 species)", 7, "Sec. IV-A", S_EV, item="dataset2"),
    R("dataset_key", "own_p108_szu_south", 7, "Sec. IV-A", S_EV, item="dataset2"),
    R("dataset_role", "train_test", 8, "Sec. IV-B", "selected, containing a total of 1659 masks for training", item="dataset2", note="75 training images with 1659 masks and 22 test images with 523 masks"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset2"),
    R("n_images", "97", 7, "Sec. IV-A", S_EV, item="dataset2", unit="images", note="images cropped from the hyperspectral data (532 x 532 pixels in Sec. IV-A; training images are 512 x 512)"),
    R("n_trees", "2182", 7, "Sec. IV-A", "includes 2182 tree", item="dataset2", unit="trees", note="instances of 13 different tree species"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="initialisation not stated")
ROWS += model("model1", "FBAN (proposed)", "proposed", "cnn_transformer_hybrid", "two-stage backbone feature extractor (64 and 128 channels) with a boundary-aware module and an adaptive crown-aware module (Transformer layers coupled with dilated convolutions), pixel decoder and Mask2former-style Transformer decoder",
              None, None, 1, "Abstract", "coupling Transformer layers with dilated convolution layers", cat_note="convolutional backbone with channel, spatial and spectral attention, a Transformer encoder in ACAM and a Transformer decoder as in Mask2former",
              pre_status="not_reported", pre_note="not stated", strat_page=8, strat_loc="Sec. IV-B", strat_ev="the adamw optimizer",
              strat_note="trained with AdamW (learning rate 0.0001, weight decay 0.05) for 100000 steps with batch size 8; initialisation not stated")
BASE = [("model2", "Mask R-CNN", "cnn_baseline", "cnn", 3, "Mask region-based convolutional neural network (Mask R-CNN) [22] is a two-stage instance segmentation method", "two-stage convolutional instance segmentation (Sec. II-A)"),
        ("model3", "Cascade Mask R-CNN", "cnn_baseline", "cnn", 3, "Subsequently, the Cascade Mask R-CNN [24] further improves the performance by introducing a multistage cascade structure", "multistage cascade of Mask R-CNN (Sec. II-A)"),
        ("model4", "CondInst", "cnn_baseline", "cnn", 11, "The CondInst model, which employs conditional convolutions", "conditional convolutions (Sec. IV-D)"),
        ("model5", "SOLOv2", "other_baseline", None, 9, CMP_EV, "the paper does not describe the architecture of SOLOv2"),
        ("model6", "BoxInst", "other_baseline", None, 9, CMP_EV, "the paper does not describe the architecture of BoxInst"),
        ("model7", "Transfiner", "transformer_baseline", "cnn_transformer_hybrid", 12, "Transformer-based architecture of Transfiner", "Transformer-based architecture (Sec. IV-D); the backbone is not stated"),
        ("model8", "Maskformer", "transformer_baseline", "cnn_transformer_hybrid", 3, "Maskformer [30] uses Transformer cross-attention to unify semantic and instance segmentation", "Transformer cross-attention with queries (Sec. II-A); the backbone is not stated"),
        ("model9", "Mask2former", "transformer_baseline", "cnn_transformer_hybrid", 3, "Building on Maskformer, Mask2former [31]", "Transformer decoder with masked attention on top of Maskformer (Sec. II-A); the backbone is not stated")]
for item, name, role, cat, pg, ev, cnote in BASE:
    ROWS += model(item, name, role, cat, "not stated (parameters set according to the original paper)", None, None, pg, "Sec. II-A and IV-D", ev, cat_note=cnote, **NPRE)
res = Counter("res")
RID = {}
# Table V (SZU-North, p.9), Table VII (SZU-South, p.11): AP, AP50, AP75, APs, APm, Confidence. Tables VI (p.10) and VIII (p.12): Average row of Precision, Recall and F1-score of the nine methods.
ORDER = ("model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model1")
NAMES = {"model2": "Mask R-CNN", "model3": "Cascade Mask R-CNN", "model4": "Condinst", "model5": "Solov2", "model6": "BoxInst", "model7": "Transfiner", "model8": "Maskformer", "model9": "Mask2former", "model1": "FBAN"}
T5 = {"model2": ("28.29", "46.67", "29.38", "13.60", "32.05", "91.36"), "model3": ("29.50", "46.79", "32.69", "14.36", "31.50", "91.00"), "model4": ("30.25", "48.10", "30.68", "14.73", "33.97", "79.98"),
      "model5": ("19.36", "28.74", "20.28", "11.01", "21.54", "84.70"), "model6": ("9.05", "35.23", "1.58", "6.12", "11.01", "87.09"), "model7": ("30.86", "49.20", "32.80", "15.52", "34.96", "95.05"),
      "model8": ("17.77", "26.02", "19.50", "9.07", "19.84", "87.67"), "model9": ("40.57", "61.24", "46.08", "20.03", "44.70", "94.68"), "model1": ("44.08", "65.29", "50.49", "22.01", "47.98", "95.57")}
T7 = {"model2": ("41.64", "67.72", "44.35", "23.18", "55.59", "96.16"), "model3": ("43.80", "71.18", "47.32", "24.24", "57.98", "94.84"), "model4": ("41.20", "71.64", "37.93", "22.74", "58.47", "84.32"),
      "model5": ("33.46", "57.49", "36.45", "20.17", "43.93", "91.56"), "model6": ("19.64", "59.10", "10.95", "14.50", "23.30", "86.97"), "model7": ("29.95", "69.15", "19.52", "16.22", "44.23", "97.27"),
      "model8": ("11.46", "18.84", "11.88", "6.37", "16.85", "87.56"), "model9": ("49.82", "74.21", "58.21", "34.25", "68.66", "97.04"), "model1": ("54.32", "78.36", "62.69", "39.59", "72.03", "98.02")}
F6 = {"model2": ("61.51", "46.04", "51.05"), "model3": ("70.30", "41.58", "51.08"), "model4": ("71.00", "46.76", "55.22"), "model5": ("52.12", "28.86", "35.05"), "model6": ("56.56", "36.82", "43.46"),
      "model7": ("76.62", "48.01", "57.34"), "model8": ("37.78", "25.53", "29.30"), "model9": ("83.08", "62.53", "70.25"), "model1": ("87.20", "65.60", "73.47")}
F8 = {"model2": ("71.92", "68.21", "67.42"), "model3": ("81.38", "70.37", "73.60"), "model4": ("85.27", "68.93", "71.89"), "model5": ("75.01", "54.69", "62.42"), "model6": ("71.01", "59.70", "63.88"),
      "model7": ("82.44", "72.24", "76.45"), "model8": ("52.85", "18.46", "25.94"), "model9": ("86.96", "73.92", "79.31"), "model1": ("89.28", "79.32", "83.26")}
for ds, dn, pgAP, tAP, pgF, tF, AP, FF in (("dataset2", "SZU-South", 11, "Table VII (image)", 12, "Table VIII (image)", T7, F8), ("dataset1", "SZU-North", 9, "Table V (image)", 10, "Table VI (image)", T5, F6)):
    prim = ds == "dataset2"
    for item in ORDER:
        v = AP[item]
        evA = f"{NAMES[item]} " + " ".join(v)
        st = f"{tAP.split(' (')[0]}: {dn} test set"
        head = prim and item == "model1"
        ROWS += result(res(), "map", item, ds, v[0], pgAP, tAP, evA, unit="%", setting=st, note="AP as printed, not defined in the paper, of the instance masks with species classes", **TS, primary=prim, headline=head)
        ROWS += result(res(), "ap50", item, ds, v[1], pgAP, tAP, evA, unit="%", setting=st, note="AP50 of the instance masks with species classes", **TS, primary=prim)
        f = FF[item]
        evF = f"Average {' '.join(f)}"
        ROWS += result(res(), "f1", item, ds, f[2], pgF, tF, evF, unit="%", setting=f"{tF.split(' (')[0]}: {dn} test set, Average row", note="F1-score averaged over the tree species (Average row of the per-class table; precision and recall are in the same table)", **TS, primary=prim)
ROWS += [R("limitations", "In extremely complex scenes, with complex lighting or severe occlusion, the segmentation and identification of individual trees still have limitations", 15, "Sec. V", "the segmentation and identification of individual trees still have certain limitations"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Tables V and VII (AP, AP50, AP75, APs, APm and a confidence score of eight comparison methods and FBAN on SZU-North and SZU-South) and the Average rows of Tables VI and VIII (precision, recall and F1-score averaged over the species) are images. AP and AP50 are recorded from Tables V and VII and the average F1-score from Tables VI and VIII; AP75, APs, APm, the confidence score, precision and recall are not. SZU-South is primary because the abstract's headline (AP 4.5 points above Mask2former) is on it; SZU-North, listed first, is not primary. The ablations of Tables III and IV (module removal) and Table IX (FBAN backbone with the Maskformer or Mask2former decoder) are not recorded. The comparison methods use the parameters of their original papers; their backbones are not stated, so SOLOv2 and BoxInst have no architecture category. Efficiency is shown only in Fig. 12 (time per image and FPS).",
           "Sec. IV and Tables V to VIII (pp. 9 to 13)")]
