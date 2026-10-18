import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T5, T8, T9, T10 = "P038", "Table 5 (image)", "Table 8 (image)", "Table 9", "Table 10 (image)"
# Urban tree crown delineation in 5 cm UAV orthophotos of Cimahi City with four Mask R-CNN variants (ResNet50-FPN or ResNet101-FPN, pre-trained on COCO, OAM-TCD or Detectree2) and SAM-HQ (ViT-H Quant)
# as a semi-automatic annotation tool. Extracted at the author's decision of 2026-10-14 (kept after the full-text re-check): SAM-HQ is tested quantitatively (IoU of prompt strategies against manual
# annotations, Fig. 9 and p.12) and the Mask R-CNN crowns are scored. Table 8 (independent test set, polygons matched at IoU 0.5) is primary (F1; precision, recall and OA are not recorded); Table 5
# (5-fold cross-validation, mean of the folds) and the external validation in Bandung City (Table 9, two transfer pairs) are not primary. Tables 5, 8 and 10 are images and were typed from the rendered pages.
TM = Tags(task="crown_segmentation", level="instance_mask", split="test", agg="single")
TCV = TM(split="cross_validation", agg="mean")
TS = TM(split="not_stated")
ABS_EV = "This study proposes an enhanced Mask R-CNN approach, supported by semi-automatic annotation using the Segment Anything Model-HQ (ViT-H Quant), to improve tree crown delineation"
DS_EV = "The annotated dataset comprised 2,762 tree crowns across 135 images"
TILE_EV = "These areas were divided into non-overlapping 1024×1024-pixel tiles (0.262 hectares, 5 cm GSD)"
SAM_EV = "the IoU increased from 0.27 (single point) to 0.82 (multiple points) in dense canopy, and from 0.68 to 0.89 in shadowed areas"
ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", ABS_EV, note="instance segmentation of individual urban tree crowns"),
    R("output_unit", "tree_instance;crown_mask", 1, "Abstract", ABS_EV, note="a mask and a box for each tree crown (Mask R-CNN)"),
    R("tree_setting", "urban", 1, "Abstract", ABS_EV, note="urban trees of Cimahi City"),
    NR("tree_taxa", note="species are not described"),
    R("study_area", "Cimahi City, West Java, Indonesia (external validation in Bandung City)", 3, "Sec. II.A", "The study was conducted in Cimahi City, West Java"),
    D("biome", "tropical", "West Java, Indonesia; not stated in the paper"),
    R("platform", "uav", 4, "Sec. II.C", TILE_EV, note="UAV orthophoto mosaic"),
    R("sensors", "rgb", 1, "Abstract", "Results demonstrate that the model with a ResNet50 backbone pre-trained on OAM-TCD achieved the best performance (AP 49.56%, F1-score 0.722, precision 0.720, recall 0.723)",
      note="UAV RGB orthophotos (index terms: UAV RGB imagery)"),
    R("gsd", "5", 4, "Sec. II.C", TILE_EV, unit="cm"),
    R("annotation_type", "polygon_mask", 4, "Sec. II.C", "Crown annotation was performed on PNG files using X-Anylabeling with SAM-HQ (ViT-H Quant) through point-based prompting",
      note="SAM-HQ masks from point prompts, checked visually against 1.5 cm imagery and LiDAR tree points"),
    R("dataset_name", "own UAV RGB orthophoto tiles of urban tree crowns, Cimahi City (1024 x 1024 pixels, 5 cm)", 5, "Sec. II.C", DS_EV, item="dataset1"),
    R("dataset_key", "own_p038_cimahi", 5, "Sec. II.C", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. II.C", "with 2,501 instances) and testing (10%, 14 tiles with 261 instances) subsets", item="dataset1",
      note="121 training tiles (2,501 crowns, 5-fold cross-validation) and 14 test tiles (261 crowns)"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset1"),
    R("n_images", "135", 5, "Sec. II.C", DS_EV, item="dataset1", unit="tiles"),
    R("n_trees", "2762", 5, "Sec. II.C", DS_EV, item="dataset1", unit="crowns"),
    R("dataset_name", "own UAV RGB orthophoto of urban tree crowns, Bandung City (external validation)", 9, "Sec. III.D", "area (Bandung City)", item="dataset2"),
    R("dataset_key", "own_p038_bandung", 9, "Sec. III.D", "area (Bandung City)", item="dataset2"),
    R("dataset_role", "transfer_target", 9, "Sec. III.D", "external validation using UAV imagery from another urban", item="dataset2"),
    D("dataset_public", "no", "No data link or availability statement found in the paper", item="dataset2"),
]
FT = dict(strat_page=7, strat_loc="Sec. II.D", strat_ev="In Detectron2, a partial freezing approach (freeze_at=2)", strat_note="Detectron2 with freeze_at=2: the first two ResNet stages are frozen, stages 3 to 5 and the heads are trained; Tesla T4")
M = [  # item, label, name, backbone, pretraining, pre evidence (page, quote), pre note
    ("model1", "M1", "Mask R-CNN M1 (ResNet50-FPN, COCO)", "ResNet50-FPN", "other", (6, "1) COCO: ResNet50-FPN and ResNet101-FPN architectures were initialized with weights pre-trained on the"), "Microsoft COCO weights"),
    ("model2", "M2", "Mask R-CNN M2 (ResNet101-FPN, COCO)", "ResNet101-FPN", "other", (6, "1) COCO: ResNet50-FPN and ResNet101-FPN architectures were initialized with weights pre-trained on the"), "Microsoft COCO weights"),
    ("model3", "M3", "Mask R-CNN M3 (ResNet50-FPN, OAM-TCD)", "ResNet50-FPN", "domain_specific", (7, "(OAM-TCD) [46], which contains 280,000 individual"), "pre-trained on the OAM-TCD tree crown dataset"),
    ("model4", "M4", "Mask R-CNN M4 (ResNet101-FPN, Detectree2)", "ResNet101-FPN", "domain_specific", (7, "3) Detectree2: A ResNet101-FPN model trained using"), "Detectree2 tree crown model weights"),
]
for item, lab, name, bb, pre, (pp, pev), pnote in M:
    ROWS += model(item, name, "cnn_baseline" if lab in ("M1", "M2") else "proposed", "cnn", bb, pre, "full_fine_tuning", 1, "Abstract", ABS_EV,
                  cat_note="convolutional Mask R-CNN", pre_page=pp, pre_loc="Sec. II.D", pre_ev=pev, pre_note=pnote, **FT)
ROWS += model("model5", "SAM-HQ (ViT-H Quant)", "other_baseline", "foundation_model", "ViT-H Quant", None, "zero_shot", 4, "Sec. II.C",
              "Crown annotation was performed on PNG files using X-Anylabeling with SAM-HQ (ViT-H Quant) through point-based prompting",
              cat_note="Segment Anything foundation model used as a prompt-based annotation tool (protocol 4.5: foundation_model)", pre_note="pre-training data not stated", strat_note="prompted with points or boxes, not trained")
res = Counter("res")
T8ROWS = {"M1": ("187", "76", "80", "0.545", "0.711", "0.700", "0.706"), "M2": ("186", "75", "76", "0.552", "0.713", "0.710", "0.711"),
          "M3": ("188", "73", "72", "0.565", "0.720", "0.723", "0.722"), "M4": ("181", "80", "55", "0.573", "0.693", "0.767", "0.728")}
T5ROWS = {"M1": ("45.45", "2.27", "69.66", "1.48"), "M2": ("45.94", "1.85", "70.14", "2.31"), "M3": ("49.56", "2.16", "72.89", "2.24"), "M4": ("47.04", "2.24", "70.47", "1.99")}
SPEED = {"M1": "0.123", "M2": "0.129", "M3": "0.111", "M4": "0.128"}
TEST_F1 = {}
for item, lab, *_ in M:
    tp, fp, fn, oa, p, r, f1 = T8ROWS[lab]
    ev8 = f"{lab} {tp} {fp} {fn} {oa} {p} {r} {f1}"
    TEST_F1[item] = (f1, ev8)
    ROWS += result(res(), "f1", item, "dataset1", f1, 9, T8, ev8, unit="fraction", setting=f"Table 8: independent test set (14 tiles, 261 crowns); TP {tp}, FP {fp}, FN {fn}",
                   note="F1 of crowns matched at IoU 0.5; precision, recall and OA of the row are not recorded", **TM, primary=True, headline=(lab == "M3"))
    ap, apsd, ap50, ap50sd = T5ROWS[lab]
    ev5 = f"{lab} {ap} ± {apsd} {ap50} ± {ap50sd}"
    ROWS += result(res(), "ap50_95", item, "dataset1", ap, 8, T5, ev5, unit="%", setting="Table 5: 5-fold cross-validation on the training tiles", note=f"COCO mask AP, mean of five folds (SD {apsd})", **TCV)
    ROWS += result(res(), "ap50", item, "dataset1", ap50, 8, T5, ev5, unit="%", setting="Table 5: 5-fold cross-validation on the training tiles", note=f"COCO mask AP50, mean of five folds (SD {ap50sd})", **TCV)
    sp = SPEED[lab]
    ROWS.append(R("inference_speed", f"{sp} s per image (Tesla T4)", 9, T10, f"{lab} {sp}", item=item))
# external validation (Bandung City), two transfer pairs from the Cimahi test set
EXT = {"model3": ("0.824", "0.800, F1-score = 0.824), while Model M4 obtained com-"), "model4": ("0.782", "in a lower F1-score (0.782)")}
tr = Counter("tr")
for item, (f1, ev) in EXT.items():
    ROWS += result(res(), "f1", item, "dataset2", f1, 9, T9, ev, unit="fraction", setting="Table 9: external validation, Bandung City", note="F1 of crowns on another urban area; precision and recall are not recorded", **TM)
    t = tr()
    ROWS += [R("transfer_source", "Cimahi City (training and test tiles)", 5, "Sec. II.C", DS_EV, item=t), R("transfer_target", "Bandung City (another urban area)", 9, "Sec. III.D", "area (Bandung City)", item=t),
             R("shift_type", "site", 9, "Sec. III.D", "external validation using UAV imagery from another urban", item=t), R("transfer_model", item, 9, T9, ev, item=t),
             R("transfer_metric", "f1", 9, T9, ev, item=t), R("value_in_domain", TEST_F1[item][0], 9, T8, TEST_F1[item][1], item=t, unit="fraction", note="Table 8, independent Cimahi test set"),
             R("value_out_of_domain", f1, 9, T9, ev, item=t, unit="fraction"), R("transfer_level", "instance_mask", 9, T9, ev, item=t)]
# SAM-HQ prompt strategies on example crowns (Fig. 9): IoU against the manual annotations
for setting, v in (("Fig. 9 example, dense canopy, single point prompt", "0.27"), ("Fig. 9 example, dense canopy, multiple point prompts", "0.82"),
                   ("Fig. 9 example, shadowed area, single point prompt", "0.68"), ("Fig. 9 example, shadowed area, multiple point prompts", "0.89")):
    ROWS += result(res(), "iou", "model5", "dataset1", v, 12, "Sec. IV.C and Fig. 9", SAM_EV, unit="fraction", setting=setting,
                   note="IoU of the SAM-HQ crown mask against the manual annotation for one example scene (not a test set)", **TS)
ROWS += [R("limitations", "SAM-based annotation struggles to separate crowns with unclear boundaries and still needs manual correction; the Wilcoxon tests show no significant differences between the variants", 13, "Sec. IV.C",
           "Although SAM substantially accelerates the annotation process, manual correction remains essential")]
ROWS += avail(code="no", weights="not_stated", data="no")
ROWS += [D("notes", "Four Mask R-CNN variants (M1 and M2: ResNet50-FPN and ResNet101-FPN with COCO weights; M3: ResNet50-FPN with OAM-TCD weights; M4: ResNet101-FPN with Detectree2 weights) for urban tree crown delineation in Cimahi City; SAM-HQ (ViT-H Quant) is the annotation tool. Kept after the full-text re-check of 2026-10-14 because SAM-HQ is tested quantitatively: the IoU of its crown masks against the manual annotations rises from 0.27 (single point) to 0.82 (multiple points) in a dense canopy example and from 0.68 to 0.89 in a shadowed example (Fig. 9, p.12; examples, not a test set). Table 8 (independent test set) F1 is primary: M1 0.706, M2 0.711, M3 0.722 (headline with the abstract's AP 49.56), M4 0.728. Table 5 (5-fold cross-validation, mean and SD of the folds) gives COCO mask AP and AP50 (AP75 and the size-wise APs of Table 6 are not recorded). Table 9 (Bandung City) gives the two transfer pairs of M3 (0.722 to 0.824) and M4 (0.728 to 0.782). Inference times are from Table 10. Tables 5, 8 and 10 are images typed from the rendered pages; the Table 9 values are also in the text. No transformer is trained; the only transformer-based model is SAM-HQ.",
           "Sec. II to IV, Tables 5 and 8 to 10 (pp. 4 to 13)")]
