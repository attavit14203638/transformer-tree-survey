import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3 = "P143", "Table III"
# MARS: a multimodal foundation model (Gemini 2.5 Flash Lite, fine-tuned) routes each 224 x 224 NAIP patch to a scenario-specific segmentation model (U-Net, DeepLabV3+, SegFormer or Mask2Former) for dead tree mapping;
# the baseline is one All-Scenario model per backbone. Each (backbone, pipeline) pair is one model item: 4 backbones x 2 pipelines. The category follows the segmentation backbone (the router only selects the model; it
# is not scored for trees). The scores are polygon-level precision, recall and F1: a predicted polygon (a connected part of the mask) is a true positive if it overlaps any ground-truth polygon, so a polygon can be a
# group of dead trees (level instance_mask). Table III gives them per scenario with no overall value, so the first listed scenario (BigTreeForest, one of the two scenarios the authors focus on) is primary
# (protocol 6.3 item 5); only F1 is recorded. Table II (accuracy of the routers in naming the scenario of an image) is not a tree-level result and is not recorded. Fig. 5 (average bars, no labels) is not recorded.
TM = Tags(task="mortality_detection", level="instance_mask", split="not_stated", agg="single")
ABS_EV = "We evaluate our method using highresolution tree mortality mapping, which has been recognized as a critical gap in large-scale carbon monitoring"
NAIP_EV = "We use the NAIP imagery at 0.6m spatial resolution for tree mortality event mapping"
SPLIT_EV = "The data are split into fixed train/validation/test sets using"
POLY_EV = "A predicted polygon is considered a true positive if it overlaps with any groundtruth polygon; unmatched predictions or ground truths count as false positives or false negatives"
ROWS = [
    R("task", "mortality_detection", 1, "Abstract", ABS_EV, note="tree mortality (dead tree) mapping as segmentation, scored on polygons; scattered tree deaths and continuous wipe-outs"),
    R("output_unit", "crown_mask", 5, "Sec. IV.C", "we convert the masks into polygons and evaluate the scores at the polygon level", note="binary dead-tree masks; polygons are connected mask parts and can be several dead trees"),
    R("tree_setting", "natural_forest", 3, "Sec. IV.A", "we sampled 75 sites over forested regions across the contiguous United States", note="forested regions; forest types are not described"),
    R("tree_taxa", "not specified (forests of the contiguous United States)", 3, "Sec. IV.A", "we sampled 75 sites over forested regions across the contiguous United States"),
    R("study_area", "75 forested sites across the contiguous United States", 3, "Sec. IV.A", "we sampled 75 sites over forested regions across the contiguous United States"),
    D("biome", "multiple", "75 sites across the contiguous United States; not stated in the paper"),
    R("platform", "airborne", 3, "Sec. IV.A", NAIP_EV, note="NAIP aerial imagery"),
    R("sensors", "rgb;multispectral", 3, "Sec. IV.A", "Each patch comprises RGB, NIR, and NDVI bands", note="RGB, NIR and the NDVI computed from them (five channels)"),
    R("gsd", "0.6", 3, "Sec. IV.A", NAIP_EV, unit="m"),
    R("annotation_type", "polygon_mask", 3, "Sec. IV.A", "We manually annotate the dead trees using Google Earth Engine", note="binary dead-tree pixel labels drawn on NAIP imagery; a tree was labelled dead if visually distinct, with bare branches or discoloration"),
    R("dataset_name", "own NAIP tree mortality dataset: 75 sites, about 400 patches of 224 x 224 pixels, three scenarios (BigTreeForest, SmallTreeForest, Continuous)", 3, "Sec. IV.A", "In total, we extracted about 400 patches with an even split over the 3 scenarios", item="dataset1"),
    R("dataset_key", "own_p143_naip_mortality", 3, "Sec. IV.A", "In total, we extracted about 400 patches with an even split over the 3 scenarios", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. IV.A", SPLIT_EV, item="dataset1", note="fixed 70, 10 and 20 percent train, validation and test split across all scenarios"),
    D("dataset_public", "no", "No data or code link or availability statement found in the paper", item="dataset1"),
    R("n_images", "400", 3, "Sec. IV.A", "In total, we extracted about 400 patches with an even split over the 3 scenarios", item="dataset1", unit="patches", note="printed as about 400 patches of 224 x 224 pixels"),
]
NP0 = dict(pre_status="not_reported", pre_note="not stated")
SEG = [  # backbone label on the table, name, category, backbone, pretraining, strategy, evidence, evidence page, note
    ("U-Net", "U-Net", "cnn", "U-Net encoder-decoder with skip connections", "none", "from_scratch", "U-Net: Encoder-decoder with skip connections; trained from scratch as a localized, non-pretrained baseline", "convolutional encoder-decoder"),
    ("Deeplabv3+", "DeepLabV3+", "cnn", "ResNet-50 with atrous spatial pyramid pooling", "imagenet", "full_fine_tuning", "ResNet-50 backbone with atrous spa", "convolutional ResNet-50 DeepLabV3+"),
    ("Mask2Former", "Mask2Former", "hierarchical_vit", "Swin-Tiny backbone with masked attention and a class-agnostic mask head", "other", "full_fine_tuning", "Mask2Former: Transformer framework with a Swin-Tiny backbone and masked attention", "Mask2Former with a Swin-Tiny backbone (protocol 4.5: hierarchical_vit)"),
    ("SegFormer", "SegFormer", "hierarchical_vit", "SegFormer hierarchical transformer encoder with a lightweight decoder (multi-band projection, single-logit head)", "other", "full_fine_tuning", "SegFormer: Hierarchical Transformer with a lightweight decoder; initialized from ADE20K", "SegFormer MiT encoder (protocol 4.5: hierarchical_vit)"),
]
PRE_EV = {"U-Net": "trained from scratch as a localized, non-pretrained baseline", "Deeplabv3+": "ImageNetpretrained", "Mask2Former": "ADE20Kpretrained", "SegFormer": "initialized from ADE20K"}
PRE_NOTE = {"none": "trained from scratch", "imagenet": "ImageNet-pretrained ResNet-50, input and output layers adapted", "other": "ADE20K-pretrained weights (not remote sensing data)"}
res = Counter("res")
MID = {}
k = 0
for lab, name, cat, bb, pre, strat, ev, cnote in SEG:
    for pipe in ("All-Scenario", "Gemini MARS"):
        k += 1
        item = f"model{k}"
        MID[(lab, pipe)] = item
        nums, line = row(PID, 4, rf"^\s*{re.escape(lab)}\s+All-Scenario\s+(?=\d)" if pipe == "All-Scenario" else r"^\s*Gemini MARS\s+(?=\d)", 0 if pipe == "All-Scenario" else 0 * 4 + SEG.index((lab, name, cat, bb, pre, strat, ev, cnote)))
        full = f"{name} ({'All-Scenario baseline' if pipe == 'All-Scenario' else 'Gemini MARS'})"
        role = {"U-Net": "cnn_baseline", "Deeplabv3+": "cnn_baseline"}.get(lab, "transformer_baseline") if pipe == "All-Scenario" else "proposed"
        tev = line if pipe == "All-Scenario" else f"{lab} ... {line}"
        pre_page = 3
        bbt = bb if pipe == "All-Scenario" else bb + "; scenario-specific models selected by the fine-tuned Gemini 2.5 Flash Lite router"
        snote = ("one model trained on all scenarios; the best of three runs is selected on the validation set (p.3)" if pipe == "All-Scenario"
                 else "scenario-based training (a global model fine-tuned per scenario or separate models, p.2); initialisation as the backbone; router: Gemini 2.5 Flash Lite fine-tuned on scenario labels")
        ROWS += model(item, full, role, cat, bbt, pre, strat, 3, "Sec. III.C", ev, cat_note=cnote + ("" if pipe == "All-Scenario" else "; the category follows the segmentation backbone, the multimodal router only selects the model"),
                      pre_page=3, pre_loc="Sec. III.C", pre_ev=PRE_EV[lab], pre_note=PRE_NOTE[pre], strat_note=snote)
# F1 of all (scenario, backbone, pipeline) cells; BigTreeForest is primary
SCEN = (("BigTreeForest", 0), ("SmallTreeForest", 1), ("Continuous", 2))
for sname, si in SCEN:
    for bi, (lab, name, cat, bb, pre, strat, ev, cnote) in enumerate(SEG):
        for pipe in ("All-Scenario", "Gemini MARS"):
            item = MID[(lab, pipe)]
            if pipe == "All-Scenario":
                nums, line = row(PID, 4, rf"^\s*{re.escape(lab)}\s+All-Scenario\s+(?=\d)", si)
                assert len(nums) == 3, (lab, sname, nums)
                f1, tev = nums[0], line
            else:
                nums, line = row(PID, 4, r"^\s*Gemini MARS\s+(?=\d)", si * 4 + bi)
                assert len(nums) == 6, (lab, sname, nums)  # F1 and its difference, precision and its difference, recall and its difference
                f1, tev = nums[0], f"{lab} ... {line}"
            ROWS += result(res(), "f1", item, "dataset1", f1, 4, T3, tev, unit="%", setting=f"Table III: {sname} scenario; {pipe}; polygon-level F1; split not stated",
                           note="polygon-level F1 (precision and recall of the row are not recorded); a polygon is a connected part of the mask that overlaps a ground-truth polygon", **TM, primary=(sname == "BigTreeForest"))
ROWS += avail(code="no", weights="not_stated", data="no")
ROWS += [D("notes", "MARS (a fine-tuned Gemini 2.5 Flash Lite router choosing a scenario-specific segmentation model) against one All-Scenario model per backbone (U-Net, DeepLabV3+, Mask2Former, SegFormer) for dead tree segmentation in NAIP patches; polygon-level F1 per scenario in Table III. All 24 F1 values (4 backbones, 2 pipelines, 3 scenarios) are recorded, BigTreeForest (the first listed scenario and one of the two focus scenarios) is primary; the split is not stated (a fixed 70, 10 and 20 percent split exists). The All-Scenario DeepLabV3+ and SegFormer score 0.0 in BigTreeForest and SmallTreeForest, so MARS gains up to 74 points; in the Continuous scenario MARS is about equal or lower. Each model item is a backbone with its pipeline; the multimodal router (GPT-4o or Gemini, accuracy of scenario naming 33.3 to 100.0 percent in Table II) is not a tree-level result and is not recorded. The polygon match is by any overlap, so a polygon can be several trees, hence level instance_mask. Fig. 1 and Fig. 5 have no printed values.",
           "Sec. IV and Tables II and III (pp. 3 to 5)")]
