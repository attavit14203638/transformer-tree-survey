import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4 = "P154", "Table 4"
URL = "https://github.com/RolnickLab/FoMo-Bench"
# Only the individual-tree task of FoMo-Bench is extracted: object detection of tree crowns (Table 4: NeonTree and ReforesTree). The classification, grid segmentation and point cloud
# tasks are area-level or not scored per tree. Table 4 does not say which split it is computed on; values are means of three runs (Sec. 5).
T = Tags(task="detection", level="object", split="not_stated", agg="mean")
OD_EV = "The object detection task is useful in forest monitoring to count and precisely locate trees"
DS_EV = "We use the NeonTree and ReforesTree datasets to test binary and multi-class object detection respectively"
REPO = "https://github.com/RolnickLab/FoMo-Bench"

ROWS = [
    R("task", "detection", 7, "Sec. 5", OD_EV),
    R("output_unit", "tree_instance", 7, "Sec. 5", OD_EV),
    NR("tree_setting", note="not described for the two detection datasets"),
    NR("tree_taxa", note="not described for the two detection datasets (ReforesTree has 6 classes)"),
    NR("study_area", note="the locations of NeonTree and ReforesTree are not described in the text (only Fig. 2a)"),
    NR("biome", note="not stated"),
    R("platform", "multiple", 4, "Sec. 3", "NeonTree ... recorded from airborne systems ... ReforesTree ... recorded from an UAV", note="NeonTree: airborne; ReforesTree: UAV (Table 4: Airborne and Aerial)"),
    R("sensors", "rgb", 4, "Sec. 3", "RGB images recorded from an UAV", note="NeonTree RGB imagery at 0.1 m; ReforesTree RGB"),
    R("gsd", "0.1", 4, "Sec. 3", "FoMo-Bench includes the NeonTree high resolution RGB imagery (0.1m resolution)", unit="m", note="NeonTree; the ReforesTree GSD is not stated"),
    D("annotation_type", "bounding_box", "tree crown detection with Faster R-CNN, RetinaNet and YOLOS (box detectors); the annotation form is not described in the paper"),
    R("dataset_name", "NeonTree (Weinstein et al. 2021): airborne RGB tree crown detection, binary", 7, "Sec. 5", DS_EV, item="dataset1"),
    R("dataset_key", "neon_treeevaluation", 7, "Sec. 5", DS_EV, item="dataset1", note="the NeonTree benchmark of Weinstein et al. (2021)"),
    R("dataset_role", "train_test", 3, "Sec. 3", "We have curated all the datasets in FoMoBench, spatially split them to train machine learning models", item="dataset1"),
    R("dataset_public", "yes", 1, "Abstract", REPO, item="dataset1", note="public benchmark datasets curated in FoMo-Bench"),
    R("dataset_url", URL, 1, "Abstract", REPO, item="dataset1"),
    R("dataset_name", "ReforesTree (Reiersen et al. 2022): UAV RGB tree crown detection, 6 classes", 7, "Sec. 5", DS_EV, item="dataset2"),
    R("dataset_key", "reforestree", 7, "Sec. 5", DS_EV, item="dataset2"),
    R("dataset_role", "train_test", 3, "Sec. 3", "We have curated all the datasets in FoMoBench, spatially split them to train machine learning models", item="dataset2"),
    R("dataset_public", "yes", 1, "Abstract", REPO, item="dataset2", note="public benchmark datasets curated in FoMo-Bench"),
    R("dataset_url", URL, 1, "Abstract", REPO, item="dataset2"),
]
CO = dict(pre_page=7, pre_loc="Sec. 5", pre_ev="The models include ResNet-50 and DeiT-S ... backbones pre-trained on the COCO dataset", pre_note="COCO", strat_note="trained in a supervised fashion on the single task; which layers are updated is not stated")
M = [  # item, label (regex) in Table 4, name, role, category, backbone, pretraining kwargs, strategy, note
    ("model1", r"F\. R-CNN\s+ResNet-50", "Faster R-CNN (ResNet-50)", "cnn_baseline", "cnn", "ResNet-50", "coco", None, "supervised baseline"),
    ("model2", r"RetinaNet\s+ResNet-50", "RetinaNet (ResNet-50)", "cnn_baseline", "cnn", "ResNet-50", "coco", None, "supervised baseline"),
    ("model3", r"YOLOS\s+DeiT-S", "YOLOS (DeiT-S)", "transformer_baseline", "pure_vit", "DeiT-S", "coco", None, "YOLOS detector with a DeiT-S backbone"),
    ("model4", r"F\. R-CNN\s+PolyNetin", "Faster R-CNN (PolyNet, in-domain)", "cnn_baseline", "cnn", "PolyNet (ResNet-50 pretrained on several datasets in a supervised fashion)", "polynet", "head_only",
     "PolyNet = ResNet-50 pretrained on NeonTree and ReforesTree simultaneously; evaluated by linear probing on a dataset included in pre-training"),
    ("model5", r"F\. R-CNN\s+PolyNetout", "Faster R-CNN (PolyNet, out-of-domain)", "cnn_baseline", "cnn", "PolyNet (ResNet-50 pretrained on the other dataset only)", "polynet", "head_only",
     "PolyNet backbone pretrained on one dataset and evaluated on the other (linear probing)"),
    ("model6", r"F\. R-CNN\s+DINOv2", "Faster R-CNN (DINOv2)", "transformer_baseline", "foundation_model", "DINOv2", "dino", "full_fine_tuning", "DINOv2 backbone, fully finetuned"),
    ("model7", r"F\. R-CNN\s+FoMo-Net1", "Faster R-CNN (FoMo-Net1)", "proposed", "foundation_model", "FoMo-Net ViT encoder (12 layers, 12 heads), single linear projection for all bands", "self_supervised",
     "full_fine_tuning", "FoMo-Net foundation model with a randomly initialised Faster R-CNN head"),
    ("model8", r"F\. R-CNN\s+FoMo-Netm", "Faster R-CNN (FoMo-Netm)", "proposed", "foundation_model", "FoMo-Net ViT encoder (12 layers, 12 heads), one linear projection per band", "self_supervised",
     "full_fine_tuning", "FoMo-Net foundation model with a randomly initialised Faster R-CNN head"),
]
res = Counter("res")
PARSED = {}
for item, rx, name, role, cat, bb, pre, strat, note in M:
    n1, e1 = row(PID, 9, rf"^\s*{rx}\s+(?=\d)", 0)
    n2, e2 = row(PID, 9, rf"^\s*{rx}\s+(?=\d)", 1)
    assert len(n1) == 4 and len(n2) == 4, (item, n1, n2)  # trainable parameters, mIoU, mAP50, mAP50:95
    PARSED[item] = ((n1, e1), (n2, e2))
    if pre == "coco":
        pk = dict(CO)
    elif pre == "polynet":
        pk = dict(pre_page=6, pre_loc="Sec. 5", pre_ev="PolyNet denotes a ResNet-50 model pretrained on several datasets simultaneously", pre_note="supervised pre-training on NeonTree and ReforesTree (in-domain) or on the other dataset only (out-of-domain)", strat_note="")
    elif pre == "self_supervised":
        pk = dict(pre_page=6, pre_loc="Fig. 3", pre_ev="The masked autoencoder, fθ (·) parameterized by θ, reconstructs the partially masked inputs", pre_note="masked autoencoder pre-training on remote sensing data (FoMo-Net)",
                  strat_note="")
    else:
        pk = dict(pre_status="not_reported", pre_note="DINOv2 weights; the pre-training data and scheme are not described", strat_note="")
    if strat == "head_only":
        pk.update(strat_page=6, strat_loc="Sec. 5", strat_ev="we evaluate it via linear probing either in-domain (denoted PolyNetin ), i.e. on a dataset included in pre-training, or out-of-domain (denoted PolyNetout )",
                  strat_note="linear probing with a task-specific detection head")
    elif strat == "full_fine_tuning" and item == "model6":
        pk.update(strat_page=7, strat_loc="Sec. 5", strat_ev="We also include a Faster R-CNN with a DINOv2 backbone (Oquab et al. 2024) fully finetuned", strat_note="fully finetuned")
    elif strat == "full_fine_tuning":
        pk.update(strat_page=7, strat_loc="Sec. 5", strat_ev="FoMo-Net is augmented with a randomly initialized Faster R-CNN head for finetuning", strat_note="finetuned with a randomly initialised Faster R-CNN head")
    ROWS += model(item, name, role, cat, bb, {"coco": "other", "polynet": "other", "dino": None}.get(pre, pre), strat, 9, T4, e1, cat_note=note, **pk)
    ROWS += [R("trainable_params", n1[0], 9, T4, e1, item=item, unit="M", note=f"NeonTree; ReforesTree: {n2[0]} M; trainable parameters with the detection head")]
    for ds, (nums, ev), lab in (("dataset1", (n1, e1), "NeonTree (1 class)"), ("dataset2", (n2, e2), "ReforesTree (6 classes)")):
        st = f"Table 4: {lab}; mean of three runs"
        prim = ds == "dataset1"
        ROWS += result(res(), "miou", item, ds, nums[1], 9, T4, ev, unit="%", setting=st, note="mIoU of the detections", **T, primary=prim)
        ROWS += result(res(), "ap50", item, ds, nums[2], 9, T4, ev, unit="%", setting=st, note="mAP50", **T, primary=prim)
        ROWS += result(res(), "ap50_95", item, ds, nums[3], 9, T4, ev, unit="%", setting=st, note="mAP50:95", **T, primary=prim)
ROWS += [R("limitations", "FoMo-Net tokenizes each spectral band separately, so the token count grows with the number of bands (memory and quadratic attention cost)", 9, "Sec. 7",
           "as FoMo-Net tokenizes spectral bands independently the amount of tokens per sample increases proportionally to the number of spectral bands")]
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=REPO, data="yes", data_page=1, data_loc="Abstract", data_ev=REPO)
ROWS += [D("notes", "FoMo-Bench groups 15 datasets; only its object detection task (Table 4, NeonTree and ReforesTree: Faster R-CNN, RetinaNet, YOLOS, DINOv2, PolyNet, FoMo-Net) scores individual trees and is recorded. Land cover and genus-level classification (Table 2: BigEarthNet, Sen12MS, TalloS, RapidAI4EO, TreeSatAI), grid semantic segmentation (Table 3) and the point cloud experiments (Appendix E, not in this PDF) are area-level or not scored per tree. PolyNet in and out are different models (backbones pretrained on both datasets or on the other dataset), so they are not recorded as transfer pairs. The paper does not say which split Table 4 uses (split not_stated) and gives no single main result (no headline).",
           "Sec. 5 and Table 4 (pp. 7 to 9)")]
