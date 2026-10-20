import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P059"
# TreePseCo produces masks but is scored with box metrics (the paper says so in Sec. 5.3): detection at object level, mAP and mAP at IoU 0.5 on the NEON and VdA test sets. The NEON test set
# is the first listed, so it carries the primary results; the VdA results (with the pretrained and the fine-tuned DeepForest) are not primary.
T = Tags(task="detection", level="object", split="test", agg="single")
NEON_EV = "The latter achieves in fact a mAP of 18.04, while our TreePseCo model attains 15.43"
NEON50_EV = "where DeepForest reaches 49.89 while TreePseco sits at 41.68"
VDA_EV = "our approach often outperforms both the original DeepForest (6.54 mAP) and its finetuned version (9.10 mAP), reaching a mAP of 14.76"
VDA50_EV = "where our solution achieves 37.55, compared to 19.19 and 26.37 for the original and finetuned DeepForest models, respectively"
SITES_EV = "Each NEON site represents different forest ecosystems, from temperate deciduous forests of the eastern United States to coniferous forests of the Pacific Northwest"
NEON_DS_EV = "we used high-resolution RGB aerial imagery (0.15 m/pixel) acquired by NEON's Airborne Observation Platform (AOP) across 22 sites"
VDA_DS_EV = "we selected 80 high-resolution crops, capturing forest landscapes in a mountainous alpine environment"

ROWS = [
    R("task", "detection;crown_segmentation", 1, "Abstract", "(1) tree center detection using a modified Segment Anything Model (SAM) decoder that generates probability heatmaps, (2) instance mask generation through prompt-guided segmentation"),
    R("output_unit", "tree_instance;crown_mask", 3, "Sec. 4.1", "producing both pixel-wise masks and corresponding axis-aligned bounding boxes"),
    R("tree_setting", "natural_forest", 3, "Sec. 3", SITES_EV, note="NEON forest sites and the alpine forests of Valle d'Aosta"),
    R("tree_taxa", "temperate deciduous and coniferous forests (NEON sites), mixed alpine forests (Valle d'Aosta); species not given", 3, "Sec. 3", SITES_EV),
    R("study_area", "22 NEON sites in the United States and the Valle d'Aosta region, northwestern Italy", 3, "Sec. 3", NEON_DS_EV),
    D("biome", "multiple", "22 US sites and the Italian Alps; not coded in the paper"),
    R("platform", "airborne", 3, "Sec. 3", NEON_DS_EV, note="NEON Airborne Observation Platform; the Valle d'Aosta data are aerial orthophotos"),
    R("sensors", "rgb", 3, "Sec. 3", NEON_DS_EV, note="canopy height models from NEON LiDAR are used only to make the pretraining targets of the heatmap decoder"),
    R("gsd", "0.15", 3, "Sec. 3", NEON_DS_EV, unit="m", note="NEON; the resolution number of the Valle d'Aosta orthophotos is missing from the printed text (Sec. 3)"),
    R("annotation_type", "bounding_box", 3, "Sec. 3", "Each 600×600 pixel image patch was manually annotated with bounding boxes for individual trees", note="NEON also has bounding box crowns"),
    R("dataset_name", "NEON airborne RGB tree crown boxes (22 sites, over 30,000 crowns)", 3, "Sec. 3", NEON_DS_EV, item="dataset1"),
    R("dataset_key", "neon_aop_rgb_p059", 3, "Sec. 3", NEON_DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 5.1", "we implement a two-stage training approach, beginning with pretraining on the NEON dataset", item="dataset1"),
    D("dataset_public", "yes", "NEON data are public; no link is printed in this paper", item="dataset1"),
    R("dataset_name", "own Valle d'Aosta RGB aerial tree dataset (80 crops of 600 × 600 pixels)", 3, "Sec. 3", VDA_DS_EV, item="dataset2"),
    R("dataset_key", "own_p059_vda", 3, "Sec. 3", VDA_DS_EV, item="dataset2"),
    R("dataset_role", "train_test", 5, "Sec. 5.1", "we perform two finetuning passes on the NEON dataset annotations and on our VdA annotations respectively", item="dataset2", note="TreePseCo and DF (FT) are fine-tuned on it; the original DeepForest is applied without training"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset2"),
    R("n_images", "80", 3, "Sec. 3", VDA_DS_EV, item="dataset2", unit="tiles", note="crops of 600 × 600 pixels from 22 large orthophotos"),
]
ROWS += model("model1", "TreePseCo", "proposed", "foundation_model", "SAM ViT-H encoder (frozen) with a fine-tuned heatmap mask decoder and a Faster R-CNN ResNet50-FPN refinement head", "sa1b", "head_only", 4, "Sec. 4.4",
              "Our framework consists of two primary components: a modified SAM model for generating bounding box proposals (referred to as point decoder) and a classification network for refinement and final detection",
              cat_note="SAM foundation model with a trained decoder and a convolutional refinement head (Sec. 4.4)",
              pre_page=3, pre_loc="Sec. 4.2", pre_ev="This iterative approach yielded a large-scale dataset comprising over one billion masks across 11 million diverse images (SA1B)", pre_note="SAM trained on SA-1B; the decoder is then trained on NEON and on VdA",
              strat_page=5, strat_loc="Sec. 5.1", strat_ev="freezing it during the training phase to preserve its robust representation capabilities while exclusively updating the mask decoder",
              strat_note="encoder frozen, mask decoder and the classification head trained per dataset")
ROWS += model("model2", "DeepForest (pretrained)", "cnn_baseline", "cnn", "RetinaNet with a ResNet50 backbone and Feature Pyramid Network", "domain_specific", "zero_shot", 5, "Sec. 5.2",
              "The model employs RetinaNet with a ResNet50 backbone and Feature Pyramid Network to generate bounding boxes around individual trees", cat_note="RetinaNet, a convolutional detector (Sec. 5.2)",
              pre_page=5, pre_loc="Sec. 5.2", pre_ev="which has been trained on over 30 million algorithmically generated crowns from 22 National Ecological Observatory Network (NEON) sites",
              pre_note="NEON crowns and 10,000 hand-labeled crowns", strat_page=5, strat_loc="Sec. 5.2", strat_ev="perform well across diverse forest types without additional training",
              strat_note="the released pretrained model is applied as is")
ROWS += model("model3", "DeepForest fine-tuned on VdA (DF FT)", "cnn_baseline", "cnn", "RetinaNet with a ResNet50 backbone and Feature Pyramid Network", "domain_specific", "full_fine_tuning", 5, "Sec. 5.3",
              "comparing DeepForest (DF), DeepForest finetuned (DF FT) and TreePseCo (TP) respectively", cat_note="RetinaNet, a convolutional detector (Sec. 5.2)",
              pre_page=5, pre_loc="Sec. 5.2", pre_ev="which has been trained on over 30 million algorithmically generated crowns from 22 National Ecological Observatory Network (NEON) sites",
              pre_note="NEON crowns and 10,000 hand-labeled crowns", strat_page=5, strat_loc="Sec. 5.2",
              strat_ev="enabling users to fine-tune the pretrained model with relatively small amounts of local data", strat_note="fine-tuned with local VdA annotations")
res = Counter("res")
UN = "%"
def add(item, ds, metric, val, ev, setting, prim=False, headline=False):
    global ROWS
    ROWS += result(res(), metric, item, ds, val, 5, "Table 1 / Sec. 5.3", ev, unit=UN, setting=setting, note=("mAP as printed; the IoU thresholds are not stated" if metric == "map" else "mAP at IoU 0.50, printed as mAP@IoU=50") + "; box metrics, printed as a percentage scale",
                   **T, primary=prim, headline=headline)
RN = {}
for item, ds, metric, val, ev, st, prim, hl in [
    ("model2", "dataset1", "map", "18.04", NEON_EV, "NEON test set", True, False), ("model1", "dataset1", "map", "15.43", NEON_EV, "NEON test set", True, False),
    ("model2", "dataset1", "ap50", "49.89", NEON50_EV, "NEON test set", True, False), ("model1", "dataset1", "ap50", "41.68", NEON50_EV, "NEON test set", True, False),
    ("model2", "dataset2", "map", "6.54", VDA_EV, "VdA test annotations", False, False), ("model3", "dataset2", "map", "9.10", VDA_EV, "VdA test annotations", False, False),
    ("model1", "dataset2", "map", "14.76", VDA_EV, "VdA test annotations", False, True),
    ("model2", "dataset2", "ap50", "19.19", VDA50_EV, "VdA test annotations", False, False), ("model3", "dataset2", "ap50", "26.37", VDA50_EV, "VdA test annotations", False, False),
    ("model1", "dataset2", "ap50", "37.55", VDA50_EV, "VdA test annotations", False, False)]:
    add(item, ds, metric, val, ev, st, prim, hl)
# Transfer pairs (V8): the original DeepForest is trained on NEON and applied to the Valle d'Aosta data unseen; TreePseCo and DF (FT) are trained on VdA, so they are no transfer pairs
tr = Counter("tr")
for metric, vin, vout, evin, evout in (("map", "18.04", "6.54", NEON_EV, VDA_EV), ("ap50", "49.89", "19.19", NEON50_EV, VDA50_EV)):
    t = tr()
    ROWS += [R("transfer_source", "NEON airborne RGB crowns (22 US sites), the training data of the pretrained DeepForest", 5, "Sec. 5.2", "which has been trained on over 30 million algorithmically generated crowns from 22 National Ecological Observatory Network (NEON) sites", item=t),
             R("transfer_target", "Valle d'Aosta RGB aerial images, northwestern Italy", 3, "Sec. 3", VDA_DS_EV, item=t),
             R("shift_type", "region", 5, "Sec. 5.3", "TreePseCo demonstrates superior generalization capabilities when applied to new geographical contexts", item=t),
             R("transfer_model", "model2", 5, "Sec. 5.2", "The model employs RetinaNet with a ResNet50 backbone and Feature Pyramid Network to generate bounding boxes around individual trees", item=t),
             R("transfer_metric", metric, 5, "Sec. 5.3", evin, item=t, note="box metrics"),
             R("value_in_domain", vin, 5, "Sec. 5.3", evin, item=t, unit=UN),
             R("value_out_of_domain", vout, 5, "Sec. 5.3", evout, item=t, unit=UN),
             R("transfer_level", "object", 5, "Sec. 5.3", evout, item=t)]
ROWS += [R("limitations", "TreePseCo needs more computation than DeepForest and the VdA dataset is small", 6, "Sec. 6", "TreePseCo introduces a higher computational overhead compared to frameworks like DeepForest"),
         D("code_available", "no", "No code repository of the paper; the pycrown link in a footnote is a third-party library"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "no", "No data link or availability statement in the paper")]
ROWS += [D("notes", "Table 1 gives mAP, mAP at IoU 0.50 and 0.75, mAP by size and mAR for the pretrained DeepForest and TreePseCo on the NEON test set and for DeepForest, its fine-tuned version and TreePseCo on the VdA test annotations (printed with decimal commas); mAP and mAP at IoU 0.50 are recorded from the results text, which prints the same values with decimal points. The masks of TreePseCo are scored with box metrics. Primary results are the NEON ones (first listed test set). Tables 2 and 3 (oracle classifier, losses, upsampling) are hyperparameter studies and are not recorded.",
           "Sec. 5.3 and Table 1 (pp. 5 to 7)")]
