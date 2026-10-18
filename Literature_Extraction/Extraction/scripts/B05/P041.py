import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T2, F4 = "P041", "Table 1", "Table 2", "Fig. 4 (image)"
URL = "https://drive.google.com/file/d/1bIbmHL-_br4SWwl0g7AaYPI4kEWLUBtm"
# Boxes: detection at object level. Masks: instance segmentation of palms, crown_segmentation at instance-mask level. One run per model. The validation set is the
# held-out part of the training estates; challenge set 1 (older palms, two other estates) is the first listed test set and carries the primary results (protocol 6.3).
TBV, TBT = Tags(task="detection", level="object", split="validation", agg="single"), Tags(task="detection", level="object", split="test", agg="single")
TMV, TMT = Tags(task="crown_segmentation", level="instance_mask", split="validation", agg="single"), Tags(task="crown_segmentation", level="instance_mask", split="test", agg="single")
DETR_EV = "The palm detection and segmentation models were built with Detection Transformer (DETR) [22] on top of the Detectron 2 framework [23]"
SEG_EV = "segmentation is specifically referred to as instance segmentation, focusing on pixels representing individual palms"
EST_EV = "eight estates owned by SD Guthrie across Malaysia were selected using a stratified random sampling method"
SPLIT_EV = "5168 tiles were randomly assigned to the training set, while the remaining 1331 tiles were used as the validation set"
CH1_EV = "a total of 100 tiles were selected and designated as test/challenge set 1"
CH2_EV = "a separate challenge set 2 was assembled from an estate impacted by a destructive storm, resulting in the generation of 100 tiles for evaluation"
DATA_EV = "The image dataset has been uploaded to https://drive.google.com/file/d/1bIbmHL-_"

ROWS = [
    R("task", "detection;crown_segmentation", 3, "Sec. 2", DETR_EV),
    R("output_unit", "tree_instance;crown_mask", 7, "Sec. 4", SEG_EV),
    R("tree_setting", "plantation", 2, "Sec. 2", EST_EV, note="oil palm estates"),
    R("tree_taxa", "oil palm (young palms, under 5 years; challenge set 1 older than 5 years)", 1, "Abstract", "drone images of young palms (< 5 year-old) from eight different estates were collected"),
    R("study_area", "eight estates of SD Guthrie across Malaysia", 2, "Sec. 2", EST_EV),
    D("biome", "tropical", "Malaysia; not stated in the paper"),
    R("platform", "uav", 2, "Sec. 2", "For this study, the DJI Mavic 2 Pro drone"),
    D("sensors", "rgb", "consumer-grade drone camera (Hasselblad on the DJI Mavic 2 Pro); the bands are not named in the paper"),
    NR("gsd", note="flight altitude 80 m; no ground sampling distance is printed"),
    D("annotation_type", "polygon_mask", "tiles annotated in LabelMe (p.3); the models are trained for boxes and instance masks"),
    R("dataset_name", "own UAV young oil palm tiles, training and validation estates (6499 tiles of 640 × 640 pixels)", 3, "Sec. 2", SPLIT_EV, item="dataset1"),
    R("dataset_key", "own_p041_palm_estates", 3, "Sec. 2", SPLIT_EV, item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2", SPLIT_EV, item="dataset1"),
    R("dataset_public", "yes", 12, "Data availability", DATA_EV, item="dataset1", note="Google Drive link; the URL continues on p.13"),
    R("dataset_url", URL, 12, "Data availability", DATA_EV, item="dataset1", note="the URL is split across the page break (p.12 to p.13)"),
    R("n_images", "6499", 3, "Sec. 2", "After manual inspection and quality checking, 6499 high-quality tiles were selected", item="dataset1", unit="tiles", note="5168 training and 1331 validation tiles"),
    R("dataset_name", "challenge set 1: palms older than 5 years from two additional estates (100 tiles)", 3, "Sec. 2", CH1_EV, item="dataset2"),
    R("dataset_key", "own_p041_palm_estates", 3, "Sec. 2", CH1_EV, item="dataset2", note="same study data; two other estates"),
    R("dataset_role", "transfer_target", 3, "Sec. 2", "Two additional independent estates were chosen", item="dataset2"),
    R("dataset_public", "yes", 12, "Data availability", DATA_EV, item="dataset2", note="Google Drive link; the URL continues on p.13"),
    R("n_images", "100", 3, "Sec. 2", CH1_EV, item="dataset2", unit="tiles"),
    R("dataset_name", "challenge set 2: palms of an estate impacted by a destructive storm (100 tiles)", 3, "Sec. 2", CH2_EV, item="dataset3"),
    R("dataset_key", "own_p041_palm_estates", 3, "Sec. 2", CH2_EV, item="dataset3", note="same study data; storm-affected estate"),
    R("dataset_role", "transfer_target", 3, "Sec. 2", CH2_EV, item="dataset3"),
    R("dataset_public", "yes", 12, "Data availability", DATA_EV, item="dataset3", note="Google Drive link; the URL continues on p.13"),
    R("n_images", "100", 3, "Sec. 2", CH2_EV, item="dataset3", unit="tiles"),
]
NP = dict(pre_status="not_reported", pre_note="no pre-training is mentioned", strat_note="initial learning rate 1e-4, batch size 16, weight decay 1e-4; the mask head is trained separately on frozen detection weights (p.3 to p.4)")
BB = "ResNeXt50_32 × 4D backbone with a transformer encoder-decoder (6 + 6 layers, 8 heads, 100 queries)"
ROWS += model("model1", "DETR baseline palm model", "transformer_baseline", "cnn_transformer_hybrid", BB, None, None, 3, "Sec. 2", DETR_EV, cat_note="DETR with a CNN backbone (Sec. 2)", **NP)
ROWS += model("model2", "DETR GAN palm model", "proposed", "cnn_transformer_hybrid", BB, None, None, 5, "Sec. 2",
              "33,746 good quality transformed tiles were combined with the original 5,168 tiles and used as the new training set to build the new palm detection and segmentation models", cat_note="same DETR architecture; trained with StyleGAN2 and CycleGAN synthetic tiles added", **NP)

res = Counter("res")
COLS = [("dataset1", "validation set", 0), ("dataset1", "validation set", 1), ("dataset2", "challenge set 1 (test)", 2), ("dataset2", "challenge set 1 (test)", 3),
        ("dataset3", "challenge set 2 (test)", 4), ("dataset3", "challenge set 2 (test)", 5)]
RES = {}
for page, tab, kind, tbv, tbt in ((6, T1, "box", TBV, TBT), (7, T2, "mask", TMV, TMT)):
    for metric, rx, nm in (("ap50_95", r"^\s*Mean Average Precision\s+(?=\d)", "mAP, COCO AP over IoU 0.50 to 0.95"), ("ap50", r"^\s*Average Precision \(AP\)\s+(?=\d)", "AP at IoU 0.50")):
        nums, ev = row(PID, page, rx, 0)
        assert len(nums) == 6, (tab, metric, nums)  # validation, test set 1, test set 2 for the baseline and the GAN model
        for ci, (ds, lab, k) in enumerate(COLS):
            item = "model1" if ci % 2 == 0 else "model2"
            tags = tbv if ds == "dataset1" else tbt
            if kind == "mask":
                tags = TMV if ds == "dataset1" else TMT
            prim = ds == "dataset2"
            RES[(kind, metric, ds, item)] = res()
            ROWS += result(RES[(kind, metric, ds, item)], metric, item, ds, nums[ci], page, tab, ev, unit="fraction", setting=f"{tab}: {lab}; {kind} {nm}", note=f"{nm} of the {kind} output", **tags, primary=prim)
# Count-based F1 of the detections at score 0.9 (Fig. 4, labels read from the figure): baseline and GAN model on the validation set and both challenge sets
F1V = {"dataset1": ("0.96", "0.99"), "dataset2": ("0.96", "0.98"), "dataset3": ("0.23", "0.97")}
for ds, lab in (("dataset1", "validation set"), ("dataset2", "challenge set 1 (test)"), ("dataset3", "challenge set 2 (test)")):
    for item, nm, v in (("model1", "Baseline", F1V[ds][0]), ("model2", "GAN", F1V[ds][1])):
        tags = TBV if ds == "dataset1" else TBT
        RES[("box", "f1", ds, item)] = res()
        ROWS += result(RES[("box", "f1", ds, item)], "f1", item, ds, v, 7, F4, f"{nm}, {v}", unit="fraction", setting=f"Fig. 4: {lab}; palm count F1 at detection score 0.9", note="F1 of the palm count at score 0.9; the figure also gives precision, recall and 95% CIs", **tags, primary=(ds == "dataset2"))
# Transfer pairs (V8): the validation set of the training estates is the in-domain value; challenge sets 1 and 2 are unseen estates (older palms; storm damage)
tr = Counter("tr")
for item, nm in (("model1", "baseline"), ("model2", "GAN")):
    for ds, tgt, note in (("dataset2", "challenge set 1: two additional estates with palms older than 5 years", "different estates and older palms"),
                          ("dataset3", "challenge set 2: an estate hit by a destructive storm", "storm-damaged canopies")):
        for kind, metric, lvl in (("box", "f1", "object"), ("box", "ap50", "object"), ("mask", "ap50", "instance_mask")):
            r_in, r_out = RES[(kind, metric, "dataset1", item)], RES[(kind, metric, ds, item)]
            if metric == "f1":
                loc, p = F4, 7
                v_in = F1V["dataset1"][0 if item == "model1" else 1]
                v_out = F1V[ds][0 if item == "model1" else 1]
                e_in, e_out = f"{'Baseline' if item == 'model1' else 'GAN'}, {v_in}", f"{'Baseline' if item == 'model1' else 'GAN'}, {v_out}"
            else:
                tab, p = (T1, 6) if kind == "box" else (T2, 7)
                loc = tab
                nums, ev = row(PID, p, r"^\s*Average Precision \(AP\)\s+(?=\d)", 0)
                ci_in, ci_out = (0 if item == "model1" else 1), ((2 if ds == "dataset2" else 4) + (0 if item == "model1" else 1))
                v_in, v_out, e_in, e_out = nums[ci_in], nums[ci_out], ev, ev
            t = tr()
            ROWS += [R("transfer_source", "own UAV young palm tiles of the eight training estates (validation set as the in-domain reference)", 3, "Sec. 2", SPLIT_EV, item=t),
                     R("transfer_target", tgt, 3, "Sec. 2", CH1_EV if ds == "dataset2" else CH2_EV, item=t),
                     R("shift_type", "site", 3, "Sec. 2", CH1_EV if ds == "dataset2" else CH2_EV, item=t, note=note),
                     R("transfer_model", item, 3, "Sec. 2", DETR_EV, item=t),
                     R("transfer_metric", metric, p, loc, e_in, item=t, note=f"{kind} output"),
                     R("value_in_domain", v_in, p, loc, e_in, item=t, unit="fraction"),
                     R("value_out_of_domain", v_out, p, loc, e_out, item=t, unit="fraction"),
                     R("transfer_level", lvl, p, loc, e_out, item=t)]
ROWS += [R("limitations", "Only one drone, one flight altitude (80 m), one planting density and one camera setting were tested", 11, "Sec. 5", "did not explore several potentially influential factors"),
         D("code_available", "no", "no code repository is named; only the StyleGAN2 palm generator is deployed on Hugging Face (p.12)"),
         D("weights_available", "not_stated", "only the StyleGAN2 palm generator is deployed on Hugging Face (p.12); nothing is said about the DETR weights"),
         R("data_available", "yes", 12, "Data availability", DATA_EV, note="Google Drive link; other information on reasonable request")]
ROWS += [D("notes", "Tables 1 and 2 (COCO evaluation of boxes and masks) give mAP (IoU 0.50 to 0.95), AP (IoU 0.50) and mAR for the baseline and GAN DETR models on the validation set and two challenge sets; mAP and AP are recorded (mAR is not). Fig. 4 labels give the palm-count precision, recall and F1 with CIs; F1 is recorded (precision and recall are in the figure and in the abstract). Primary results are challenge set 1, the first listed test set; validation values are in-domain references for the 12 transfer pairs.",
           "Sec. 3 to 4 and Tables 1 and 2, Fig. 4 (pp. 4 to 7)")]
