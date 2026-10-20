import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, URL = "Table 1", "https://github.com/siddsach/distshift_agroforestry/"
ROWS = [
    R("task", "detection;counting", 3, "Sec. 3", "We report Average Precision ... We also measure R2 between predicted and true tree counts"),
    R("output_unit", "tree_instance", 2, "Sec. 2", "the model predicts bounding boxes for each individual tree"),
    R("tree_setting", "agroforestry", 1, "Abstract", "remote sensing tree detection in agroforestry farms in India"),
    NR("tree_taxa", note="trees on agroforestry farms; species not given"),
    R("study_area", "Rajasthan, India (8 agro-climatic zones)", 2, "Sec. 2", "a ground truth annotated dataset of satellite images from agricultural land in Rajasthan, India"),
    R("biome", "multiple", 8, "Supplementary Table 1", "Hyper-Arid Western Plains ... Sub-humid Southern Plains ... Humid Southeastern Plains"),
    R("platform", "satellite", 2, "Sec. 2", "We source RGB Skysat images (50cm spatial resolution) from the Planet Labs API"),
    R("sensors", "rgb", 2, "Sec. 2", "We source RGB Skysat images"),
    R("gsd", "50", 2, "Sec. 2", "Skysat images (50cm spatial resolution)", unit="cm"),
    R("annotation_type", "polygon_mask", 2, "Sec. 2", "annotate individual segmentation masks for each green tree crown in the image"),
    R("dataset_name", "Rajasthan agroforestry distribution-shift benchmark (SkySat, 400 x 400 px tiles)", 2, "Sec. 2", "we break these scenes into 400 x 400 pixel RGB images for annotation", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 3", "we split our dataset of images for each of Rajasthan's 8 agro-climatic zones into train and test images", item="dataset1"),
    R("dataset_public", "yes", 1, "Footnote", f"Data, code: {URL}", item="dataset1"),
    R("dataset_url", URL, 1, "Footnote", f"Data, code: {URL}", item="dataset1"),
    NR("n_images", item="dataset1", note="number of tiles not given; 400 images were double annotated for agreement"),
]
ROWS += model("model1", "Faster R-CNN", "cnn_baseline", "cnn", "ResNet-101 with FPN", "other", "full_fine_tuning", 8, "Supplementary Sec. 2",
              "we use a Faster-RCNN with a resnet-101 backbone and feature pyramid network pre-trained on COCO", pre_note="COCO",
              strat_note="trained on the benchmark from COCO weights; freezing not mentioned")
SAM_PRE = dict(pre_status="reported", pre_note="SAM weights are pre-trained on SA-1B; the paper does not name the pre-training data")
for item, label, strat, note in (("model2", "Full Finetune", "full_fine_tuning", ""), ("model3", "Finetune Head", "head_only", "frozen backbone"),
                                 ("model4", "Finetune Head then Full", "full_fine_tuning", "head fine-tuning first, then full fine-tuning")):
    ROWS += model(item, f"SAM encoder with Mask R-CNN head ({label})", "transformer_baseline", "foundation_model", "SAM image encoder (size not stated)",
                  "sa1b", strat, 5, T1, f"SAM Finetune {label}", strat_page=3, strat_loc="Sec. 3",
                  strat_ev="full fine-tuning, head fine-tuning with a frozen backbone, and head fine-tuning then full fine-tuning", strat_note=note,
                  pre_page=3, pre_loc="Sec. 3", pre_ev="we add a Mask-RCNN head on the SAM encoder", **SAM_PRE)
GD_PRE = dict(pre_page=5, pre_loc="Sec. 5", pre_ev="they were not pre-trained on remote sensing data", pre_note="general-domain pre-training; data not named")
for item, label, strat, note in (("model5", "Full Finetune", "full_fine_tuning", ""), ("model6", "Finetune Head", "head_only", "frozen backbone"),
                                 ("model7", "Finetune Head then Full", "full_fine_tuning", "head fine-tuning first, then full fine-tuning")):
    ROWS += model(item, f"Grounding DINO ({label})", "transformer_baseline", "vision_language_model", "not stated", "other", strat, 5, T1,
                  f"Grounding DINO {label}", strat_page=3, strat_loc="Sec. 3", strat_ev="We can directly fine tune Grounding DINO", strat_note=note, **GD_PRE)
ROWS += model("model8", "Grounding DINO (conventional and few-shot runs)", "transformer_baseline", "vision_language_model", "not stated", "other", None,
              5, T1, "Grounding DINO Conventional Eval", strat_note="fine-tuning setting not stated for these runs", **GD_PRE)
res = Counter("res")
ROWS += result(res(), "map", "model1", "dataset1", "0.778", 5, T1, "Faster-RCNN Conventional Eval 0.778", unit="fraction",
               setting="conventional evaluation (trained on all zones); ID-zone test", note="AP as printed in Table 1; its IoU threshold is not stated (Figure 3 labels the few-shot values AP50)")
ROWS += result(res(), "map", "model1", "dataset1", "0.631", 5, T1, "Faster-RCNN Conventional Eval ... 0.631", unit="fraction",
               setting="conventional evaluation (trained on all zones); OOD-zone test", note="AP as printed in Table 1")
ROWS += result(res(), "map", "model8", "dataset1", "0.821", 5, T1, "Grounding DINO Conventional Eval 0.821", unit="fraction",
               setting="conventional evaluation (trained on all zones); ID-zone test", note="AP as printed in Table 1")
ROWS += result(res(), "map", "model8", "dataset1", "0.667", 5, T1, "Grounding DINO Conventional Eval ... 0.667", unit="fraction",
               setting="conventional evaluation (trained on all zones); OOD-zone test", note="AP as printed in Table 1")
ROWS += result(res(), "r2", "model1", "dataset1", "0.922", 4, "Fig. 2 (image)", "R^2 = 0.922, y = 0.7999 x + 0.2001", unit="fraction",
               setting="conventional evaluation; ID test; per-image tree count", note="text gives 0.92 (Sec. 4)")
tr = Counter("tr")
for m, label, idv, oodv in (("model1", "Faster-RCNN", "0.778", "0.441"), ("model2", "SAM Finetune Full Finetune", "0.781", "0.485"),
                            ("model3", "SAM Finetune Head", "0.777", "0.483"), ("model4", "SAM Finetune Head then Full", "0.592", "0.417"),
                            ("model5", "Grounding DINO Full Finetune", "0.814", "0.495"), ("model6", "Grounding DINO Finetune Head", "0.810", "0.505"),
                            ("model7", "Grounding DINO Finetune Head then Full", "0.808", "0.487")):
    i = tr()
    ev = f"{label} Dist shift eval {idv} {oodv}"
    ROWS += [R("transfer_source", "train splits of 4 in-distribution agro-climatic zones", 3, "Sec. 3", "Train on train splits for ID Zones", item=i),
             R("transfer_target", "test splits of 4 out-of-distribution agro-climatic zones", 3, "Sec. 3", "Test on test splits for ID Zones vs. OOD Zones", item=i),
             R("shift_type", "region", 3, "Sec. 3", "we split the zones into four ID and four OOD zones", item=i),
             R("transfer_model", m, 5, T1, ev, item=i),
             R("transfer_metric", "map", 5, T1, ev, item=i, note="AP as printed in Table 1; its IoU threshold is not stated"),
             R("value_in_domain", idv, 5, T1, ev, item=i, unit="fraction"),
             R("value_out_of_domain", oodv, 5, T1, ev, item=i, unit="fraction")]
de = Counter("de")
for m, pts in (("model8", (("1", "51.1"), ("10", "51.2"), ("100", "60.0"))), ("model1", (("1", "44.6"), ("10", "49.5"), ("100", "57.8")))):
    for k, v in pts:
        i = de()
        lab = "Grounding DINO (blue)" if m == "model8" else "Faster-RCNN (orange)"
        ROWS += [R("label_model", m, 4, "Fig. 3", lab, item=i),
                 R("label_amount", k, 3, "Sec. 3", "Train on train splits for ID Zones, plus k={1,10,100,ALL} examples from the training splits of each of the OOD zones", item=i,
                   note="k images per OOD zone, added to all ID-zone training data"),
                 R("label_unit", "images", 3, "Sec. 3", "k={1,10,100,ALL} examples from the training splits of each of the OOD zones", item=i),
                 R("label_metric", "ap50", 4, "Fig. 3", "AP50 Out-of-Distribution", item=i),
                 R("label_value", v, 4, "Fig. 3", f"{lab} at {k} shots", item=i, unit="%", status="approximate", note="read off Figure 3; about +/- 0.5")]
ROWS += avail(code_url=URL, page=1, loc="Footnote", ev=f"Data, code: {URL}", data="yes", data_page=1, data_loc="Footnote", data_ev=f"Data, code: {URL}")
ROWS += [D("notes", "Inter-annotator agreement 86 AP on 400 double-annotated images; 94% of field-tagged trees in 38 plots were detected. Supplementary Table 1 gives per-zone tree-count R2 (0.759 to 0.970) for Grounding DINO, although the text says the per-zone values are for Faster R-CNN.", "Sec. 2, Sec. 4 and Supplementary Table 1 (pp. 2, 4, 8)")]
