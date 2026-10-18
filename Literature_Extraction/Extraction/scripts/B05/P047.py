import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T4, T5 = "P047", "Table 3", "Table 4", "Table 5"
URL = "https://figshare.com/articles/dataset/High-resolution_UAV_desert_vegetation_dataset/28387949"
# Pixel-wise segmentation of desert vegetation (Haloxylon trees and shrubs) against the background: crown segmentation at pixel level. OA, mIoU and kappa are overall scores of the two
# classes (vegetation and non-vegetation), so none of them is a tree-only score and no result is primary (protocol 6.3).
T = Tags(task="crown_segmentation", level="pixel", split="test", agg="single")
SPLIT_EV = "a training group comprising 700 images, a validation group with 350 images, and a testing group containing 1050 images"
URL_EV = "The dataset can be accessed at https://figshare.com/articles/dataset/High-resolution_UAV_desert_vegetation_dataset/28387949"

ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "designed for the segmentation of desert vegetation from UAV-derived remote sensing imagery, thereby facilitating the rapid inventory of tree saplings counts"),
    R("output_unit", "crown_mask", 5, "Sec. 3.1", "ultimately predicts precise single-tree masks for desert vegetation"),
    R("tree_setting", "mixed", 1, "Sec. 1", "aiming to plant one hundred million Haloxylon ammodendron and other sand-fixing plants in the key ecological areas of Alxa", note="artificial Haloxylon plantations and wild desert vegetation"),
    R("tree_taxa", "Haloxylon ammodendron (also tamarisk, Populus euphratica, Nitraria)", 1, "Abstract", "survival rate of vegetation (such as Haloxylon ammodendron)"),
    R("study_area", "Suhaitu Gacha, Alxa League, Inner Mongolia, China", 3, "Sec. 2.1", "The research focuses on Suhaitu Gacha, a village within the Alxa League, located in the western region of the Inner Mongolia Autonomous Region, China"),
    R("biome", "arid", 3, "Sec. 2.1", "typified by its typical arid desert climate"),
    R("platform", "uav", 3, "Sec. 1", "We collect UAV imagery data from the Suhaitu Gacha of Alxa League in Inner Mongolia, China"),
    D("sensors", "rgb", "three-band UAV imagery from a DJI FC6310R camera; the vegetation indices of Table 2 use the R, G and B bands"),
    R("gsd", "2.68", 4, "Sec. 2.2", "with a calculated theoretical spatial resolution of 2.68 cm", unit="cm", note="theoretical value (Sec. 2.2)"),
    R("annotation_type", "polygon_mask", 3, "Sec. 1", "We create and publicly release a high-quality pixel-level fine segmentation dataset of desert vegetation", note="pixel-level masks; also used as a COCO dataset for the instance models"),
    R("dataset_name", "desert vegetation UAV dataset, Suhaitu Gacha (2100 images of 512 × 512 pixels)", 5, "Sec. 2.2", SPLIT_EV, item="dataset1"),
    R("dataset_key", "own_p047_alxa_desert", 5, "Sec. 2.2", SPLIT_EV, item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.2", SPLIT_EV, item="dataset1", note="GDPGO-SAM needs no training; the supervised models train on 700 images (or 70 in Table 3)"),
    R("dataset_public", "yes", 21, "Data Availability Statement", URL_EV, item="dataset1"),
    R("dataset_url", URL, 21, "Data Availability Statement", URL_EV, item="dataset1"),
    R("n_images", "2100", 4, "Sec. 2.2", "yielding 2100 sample images", item="dataset1", unit="tiles", note="700 training, 350 validation and 1050 test images"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained for 20k iterations in MMSegmentation with SGD, learning rate 0.0025, batch size 2 (Sec. 4.1)")
ROWS += model("model1", "GDPGO-SAM (Grounding DINO prompt generation and optimization with SAM)", "proposed", "foundation_model", "Grounding DINO detector with prompt optimization, SAM segmenter", None, "zero_shot", 1, "Abstract",
              "The framework combines the Grounding DINO object detector and the pre-trained visual model SAM", cat_note="foundation models (Grounding DINO and SAM) used without training (Sec. 3.1)",
              pre_status="not_reported", pre_note="pre-trained SAM and Grounding DINO; the pre-training data are not named", strat_page=5, strat_loc="Sec. 3.1", strat_ev="This study aims to achieve zero-shot precise segmentation of desert vegetation",
              strat_note="no training samples are needed")
ROWS += model("model2", "SegFormer", "transformer_baseline", None, "backbone not stated", None, None, 8, "Sec. 3.2.2", "four representative classic semantic segmentation models", cat_note="the paper names SegFormer only (reference [49], the SegFormer transformer paper) and describes no architecture", **NPRE)
ROWS += model("model3", "Deeplabv3plus", "cnn_baseline", None, "backbone not stated", None, None, 8, "Sec. 3.2.2", "four representative classic semantic segmentation models", cat_note="named only; the best supervised model per metric in Tables 3 and 5; no architecture is described", **NPRE)
ROWS += model("model4", "Grounding DINO boxes as SAM prompts (ablation baseline)", "transformer_baseline", "foundation_model", "Grounding DINO detector and SAM without the prompt optimization", None, "zero_shot", 14, T4, "Baseline",
              cat_note="the Baseline row of the ablation (Table 4): detector boxes passed to SAM without ART, NMS or NBE", pre_status="not_reported", pre_note="pre-trained SAM and Grounding DINO; the pre-training data are not named",
              strat_page=14, strat_loc=T4, strat_ev="Baseline", strat_note="no training")
ROWS += [R("total_params", "156.28", 16, T5, row(PID, 16, r"^\s*GDPGO-SAM \(Ours\)\s+(?=\d)", 0)[1], item="model1", unit="M", note="input 512 × 512 × 3"),
         R("flops", "308.42", 16, T5, row(PID, 16, r"^\s*GDPGO-SAM \(Ours\)\s+(?=\d)", 0)[1], item="model1", unit="GFLOPs", note="input 512 × 512 × 3")]
res = Counter("res")
RES = {}
ROWS3 = [("model1", r"GDPGO-SAM \(Ours\)", T3, 10, "Table 3: trained supervised methods use 70 training images; GDPGO-SAM needs no training"),
         ("model2", r"SegFormer", T3, 10, "Table 3: 70 training images"), ("model3", r"Deeplabv3plus", T3, 10, "Table 3: 70 training images")]
for item, rx, tab, page, st in ROWS3:
    nums, ev = row(PID, page, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(nums) == 3, (item, nums)
    for metric, v, nm in (("oa", nums[0], "OA"), ("miou", nums[1], "mIoU"), ("kappa", nums[2], "kappa")):
        RES[(item, 70, metric)] = res()
        ROWS += result(RES[(item, 70, metric)], metric, item, "dataset1", v, page, tab, ev, unit=("fraction" if metric == "kappa" else "%"), setting=st, note=f"{nm}; overall over vegetation and non-vegetation", **T,
                       headline=(item == "model1" and metric == "miou"))
for item, rx in (("model2", r"SegFormer"), ("model3", r"Deeplabv3plus")):
    nums, ev = row(PID, 16, rf"^\s*{rx}\s+(?=\d)", 0)
    assert len(nums) == 5, (item, nums)  # OA, mIoU, kappa, parameters, GFLOPs
    for metric, v, nm in (("oa", nums[0], "OA"), ("miou", nums[1], "mIoU"), ("kappa", nums[2], "kappa")):
        RES[(item, 700, metric)] = res()
        ROWS += result(RES[(item, 700, metric)], metric, item, "dataset1", v, 16, T5, ev, unit=("fraction" if metric == "kappa" else "%"), setting="Table 5: 700 training images", note=f"{nm}; overall over vegetation and non-vegetation", **T)
    ROWS += [R("total_params", nums[3], 16, T5, ev, item=item, unit="M", note="input 512 × 512 × 3"), R("flops", nums[4], 16, T5, ev, item=item, unit="GFLOPs", note="input 512 × 512 × 3")]
nb, eb = row(PID, 14, r"^\s*Baseline\s+(?=[\d−-])", 0)
assert len(nb) == 3
for metric, v, nm in (("oa", nb[0], "OA"), ("miou", nb[1], "mIoU"), ("kappa", nb[2].replace("\u2212", "-"), "kappa")):  # the printed kappa is negative (-0.003)
    ROWS += result(res(), metric, "model4", "dataset1", v, 14, T4, eb, unit=("fraction" if metric == "kappa" else "%"), setting="Table 4: ablation baseline", note=f"{nm}; overall over vegetation and non-vegetation", **T)
# Data efficiency: the supervised models trained on 70 (10 percent) and on 700 training images (mIoU)
de = Counter("de")
for item in ("model2", "model3"):
    for amount in (70, 700):
        rid = RES[(item, amount, "miou")]
        n_, e_ = row(PID, 10 if amount == 70 else 16, rf"^\s*{'SegFormer' if item == 'model2' else 'Deeplabv3plus'}\s+(?=\d)", 0)
        d = de()
        ev_amt = "the training set is randomly sampled, with 10% of the images (70 images) selected as a new training set" if amount == 70 else SPLIT_EV
        ROWS += [R("label_model", item, 10 if amount == 70 else 16, T3 if amount == 70 else T5, e_, item=d),
                 R("label_amount", str(amount), 5, "Sec. 2.2", ev_amt, item=d, unit="images", note="700 images is the full training group, 70 images 10 percent of it"),
                 R("label_unit", "images", 5, "Sec. 2.2", ev_amt, item=d),
                 R("label_metric", "miou", 10 if amount == 70 else 16, T3 if amount == 70 else T5, e_, item=d),
                 R("label_value", n_[1], 10 if amount == 70 else 16, T3 if amount == 70 else T5, e_, item=d, unit="%")]
ROWS += [R("limitations", "GDPGO-SAM does not reach the best overall accuracy of the supervised models, and SAM also segments shadows and background objects as vegetation", 18, "Sec. 5.2", "did not achieve the highest overall accuracy compared to supervised"),
         D("code_available", "no", "No code repository in the paper; only the dataset link is given"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "yes", 21, "Data Availability Statement", URL_EV)]
ROWS += [D("notes", "Table 3 compares five vegetation indices, nine supervised methods trained on 70 images and GDPGO-SAM (OA, mIoU, kappa); Table 5 repeats the supervised methods with 700 training images (with parameters and FLOPs). Following protocol 4.1 the proposed model, the transformer model (SegFormer), the best supervised model per metric (Deeplabv3plus) and the ablation baseline (Table 4) are recorded. OA, mIoU and kappa are overall scores that include the background, so there is no primary result. The 70 and 700 image runs are recorded as data-efficiency points. The ablation baseline's kappa is printed as -0.003 (Table 4, agreement below chance) and is recorded as printed.",
           "Sec. 4 to 5 and Tables 3 to 5 (pp. 10 to 16)")]
