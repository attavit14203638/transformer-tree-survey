import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2, T3 = "P044", "Table 2", "Table 3"
ROWS = [
    R("task", "crown_segmentation", 1, "Abstract", "accurate individual tree crown (ITC) segmentation has become a key topic of forestry"),
    R("output_unit", "crown_mask", 2, "Sec. 2.1", "more than 600,00 ITC masks in the datasets", note="instance masks of individual crowns"),
    R("tree_setting", "mixed", 2, "Sec. 2.1", "The study sites cover a wide range around the world", note="forest types from 11 sites in 9 countries"),
    NR("tree_taxa", note="all trees; species not given"),
    R("study_area", "11 sites in 9 countries: Canada, Malaysia, Panama, China, USA, Kenya, Norway, Germany, Australia", 2, "Table 1",
      "1 Canada ... 2 Malaysia ... 3 Panama ... 4 China ... 7 America ... 8 Kenya ... 9 Norway ... 10 German ... 11 Australia"),
    R("biome", "multiple", 2, "Table 1", "Canada ... Malaysia ... Panama ... Kenya ... Norway ... Australia", note="temperate, boreal and tropical sites"),
    R("platform", "multiple", 2, "Sec. 2.1", "data from different study are collected under different conditions, e.g., platform, sensor, flight height, date, and light"),
    D("sensors", "rgb", "imagery type not stated; the examples in Figure 3 (p.4) are colour images"),
    R("gsd", "2", 2, "Sec. 2.1", "ground sample distances (GSDs) of the images range from 2 to 10 cm", unit="cm", note="2 to 10 cm"),
    R("annotation_type", "polygon_mask", 2, "Sec. 2.1", "the ITC masks were labelled manually based on visual interpretation and checked by forest experts"),
    R("dataset_name", "ISPRS Individual Tree Crown Segmentation contest dataset (11 sub-datasets)", 2, "Table 1",
      "The detailed information of the dataset released by the ISPRS Individual Tree Crown Segmentation", item="dataset1"),
    R("dataset_role", "train_test", 2, "Sec. 2.1", "The training, validation, and testing sets contain data from dataset 1-9, 3-5, and 3-11, respectively and without overlap", item="dataset1",
      note="datasets 10 (Germany) and 11 (Australia) appear only in the test set"),
    R("dataset_public", "yes", 2, "Sec. 2.1", "The dataset used for this research was released by the ISPRS", item="dataset1", note="released for an ISPRS contest"),
    NR("n_images", item="dataset1", note="'more than 1,100' images in the text; Table 1 totals 7267 training, 816 validation and 3000 test samples (unit not stated)"),
]
M = [  # item, name, role, category, backbone, regex for the Table 2 line, occurrence
    ("model1", "SAM (automatic mask generation)", "transformer_baseline", "foundation_model", "ViT (size not stated)", r"^\s*SAM\s+ViT", 0),
    ("model2", "Cascade Mask R-CNN", "cnn_baseline", "cnn", "ResNet-50", r"^\s*CM R-CNN", 0),
    ("model3", "Mask DINO", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-50", r"^\s*Mask DINO\s+ResNet50", 0),
    ("model4", "HTC", "cnn_baseline", "cnn", "ResNet-50", r"^\s{20,}(?=36\.81)", 0),
    ("model5", "HTC", "proposed", "cnn", "CB-ResNet50", r"^\s*HTC\s+CB-ResNet50", 0),
    ("model6", "HTC", "transformer_baseline", "hierarchical_vit", "CB-Swin-B", r"^\s*CB-Swin-b", 0),
]
res = Counter("res")
for item, name, role, cat, bb, rx, n in M:
    nums, ev = row(PID, 5, rx, n, cols=(0, 68))
    assert len(nums) == 2, (item, nums, ev)
    if item == "model1":
        ROWS += model(item, name, role, cat, bb, "sa1b", "zero_shot", 4, "Sec. 3.1",
                      "Since SAM is a heavy-pretrained network, it is estimated by the fully automatic mask generation function based on the original pre-trained model")
    elif item in ("model5", "model6"):
        ROWS += model(item, f"{name} with {bb} backbone", role, cat, bb, None, "full_fine_tuning", 5, T2, ev, pre_status="not_reported", pre_note="not stated",
                      strat_page=5, strat_loc="Sec. 3.5", strat_ev="Model fine-tuning based on early-stop training scheme helps the networks perform the best",
                      strat_note="best checkpoints at epoch 4 (CB-ResNet50) and 25 (CB-Swin-b)",
                      cat_note="model chosen as the paper's ITC segmentation method" if item == "model5" else "")
    else:
        ROWS += model(item, f"{name} with {bb} backbone", role, cat, bb, None, None, 5, T2, ev, pre_status="not_reported", pre_note="not stated", strat_note="not stated",
                      cat_note="ResNet-50 is printed once for Cascade Mask R-CNN, Mask DINO and the first HTC row" if item in ("model2", "model4") else "")
    ROWS += result(res(), "ap50", item, "dataset1", nums[0], 5, T2, ev, setting="test set (datasets 3 to 11); masks", headline=(item == "model5"),
                   note="Table 2 also gives AP75")
INDOM = [("3", "59.82"), ("4", "20.22"), ("5", "49.94"), ("6", "65.40"), ("7", "64.65"), ("8", "36.03"), ("9", "47.41")]
tr = Counter("tr")
for ds, country in (("10", "Germany"), ("11", "Australia")):
    nums, ev = row(PID, 5, rf"^\s+{ds}\s+(?=\d)", 0, cols=(0, 68))
    i = tr()
    ROWS += [R("transfer_source", "ISPRS ITC sub-datasets 1 to 9 (training sites)", 2, "Sec. 2.1", "The training, validation, and testing sets contain data from dataset 1-9, 3-5, and 3-11", item=i),
             R("transfer_target", f"sub-dataset {ds} ({country}), not in the training set", 4, "Sec. 3.3", "The data in Dataset 10 and 11 is not included in the training set", item=i),
             R("shift_type", "site;region", 4, "Sec. 3.3", "the performances on these two datasets represent the transferability of the networks", item=i),
             R("transfer_model", "model5", 5, T3, "The average precision of the inference masks of each dataset in testing set that generated by the network with HTC baseline and CB-ResNet50", item=i),
             R("transfer_metric", "ap50", 5, T3, "AP50", item=i, note="masks"),
             D("value_in_domain", "49.07", "mean AP50 over the test parts of the training-site datasets 3 to 9 in Table 3 (p.5): (59.82 + 20.22 + 49.94 + 65.40 + 64.65 + 36.03 + 47.41) / 7 = 49.07", item=i, unit="%"),
             R("value_out_of_domain", nums[0], 5, T3, ev, item=i, unit="%")]
ROWS += avail()
ROWS += [D("notes", "Benchmark of instance segmentation networks on the ISPRS ITC contest data; the chosen model (HTC with CB-ResNet50) is convolutional, while Mask DINO, CB-Swin-b and SAM are the transformer-based entries. Table 3 gives AP50 per test sub-dataset for the chosen model (mean 40.98).", "Sec. 3 and Tables 2 to 4 (pp. 4 to 5)")]
