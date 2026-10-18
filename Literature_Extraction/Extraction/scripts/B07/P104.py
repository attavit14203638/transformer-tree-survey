import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4, T5 = "P104", "Table 4", "Table 5"
# SAM (box prompts) localises oil palm, coconut and banana trees in drone images and an EfficientNetB3 classifies the localised trees; ViT, MobileNetV2 (Carvalho et al.) and a YOLOv3 method (Putra et al.) are the
# comparison models. Tables 4 and 5 give precision, recall, F and accuracy per tree type and no overall value, so the first listed subset (oil palm) is primary (rule 6.3 item 5). The per-class "Acc" is a one-vs-rest
# accuracy and is not recorded; F is recorded as f1. The EfficientNet variants B0 to B2 of Table 3 are not recorded.
TD = Tags(task="detection", level="object", split="test", agg="single")
TC = Tags(task="species_classification", level="tree", split="test", agg="single")
OBJ_EV = "the proposed work aims to detect and classify multiple tree types"
DS_EV = "For experimentation, we constructed our own dataset"
SPLIT_EV = "25 original images were reserved as testing set"

ROWS = [
    R("task", "detection;species_classification", 1, "Abstract", OBJ_EV, note="tree localisation with SAM and classification of oil palm, coconut and banana trees"),
    R("output_unit", "tree_instance;tree_label", 1, "Abstract", OBJ_EV),
    R("tree_setting", "mixed", 6, "Sec. 4.1", "It encompasses both natural habitats and agricultural plantations"),
    R("tree_taxa", "oil palm, coconut and banana", 1, "Abstract", "we adapted SAM for detecting and localizing oil palm, coconut, and banana trees in drone images"),
    R("study_area", "Malaysia", 6, "Sec. 4.1", "all possible cases of real situations in Malaysia", note="frames from diverse locations, collected by a Malaysian drone services company"),
    D("biome", "tropical", "Malaysia; coded from the study area"),
    R("platform", "uav", 1, "Abstract", "this work considers images captured by drones"),
    D("sensors", "rgb", "frames of drone video (visible images); the sensor is not named in the paper"),
    R("annotation_type", "image_label", 6, "Sec. 4.1", "The augmented dataset for each species was expanded to 450 images", note="one tree type per image; how the detection reference was made is not described"),
    R("dataset_name", "own drone images of oil palm, coconut and banana trees, Malaysia (450 images per tree type after augmentation)", 5, "Sec. 4", DS_EV, item="dataset1"),
    R("dataset_key", "own_p104_malaysia_trees", 5, "Sec. 4", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 4.1", SPLIT_EV, item="dataset1", note="per tree type: 315 training, 135 validation and 25 original test images"),
    R("dataset_public", "no", 10, "Data availability", "The authors do not have permission to share data", item="dataset1"),
    R("n_images", "450", 6, "Sec. 4.1", "The augmented dataset for each species was expanded to 450 images", item="dataset1", unit="images", note="per tree type after augmentation (three tree types)"),
]
ROWS += model("model1", "Proposed SAM (box-prompted tree localisation)", "proposed", "foundation_model", "Segment Anything Model with bounding box prompts", None, None, 4, "Sec. 3.1", "we adapt the Segment Anything Model (SAM) in this work",
              cat_note="SAM, a promptable segmentation foundation model; the paper does not say how it is adapted", pre_status="not_reported", pre_note="not stated", strat_note="adapted SAM; no training of SAM is described")
ROWS += model("model2", "Proposed EfficientNetB3 classifier on SAM-localised trees", "proposed", "cnn", "EfficientNetB3 with global max pooling, dropout 0.2 and a three-class dense layer", "imagenet", None, 4, "Sec. 3.2",
              "This work involves transfer learning using a pre-trained EfficientNetB3 model, originally trained on the ImageNet benchmark dataset", cat_note="EfficientNet, a convolutional network",
              strat_note="transfer learning from ImageNet weights, 30 epochs with RMSprop; which layers are trained is not stated")
ROWS += model("model3", "ViT (16 x 16 patches, 12 layers)", "transformer_baseline", "pure_vit", "ViT with patch size 16, embedding dimension 768, 12 encoder layers, 12 heads and an MLP size of 3072", None, None, 5, "Table 1",
              "ViT utilizes a transformer-based architecture", cat_note="plain vision transformer on image patches", pre_status="not_reported", pre_note="not stated", strat_note="30 epochs with Adam; initialisation not stated")
ROWS += model("model4", "Carvalho et al. (MobileNetV2 classifier)", "cnn_baseline", "cnn", "MobileNetV2", None, None, 5, "Sec. 4", "MobileNet V2 [21] is a lightweight convolutional neural", cat_note="lightweight convolutional network",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated")
ROWS += model("model5", "Putra et al. (YOLOv3 based oil palm detection and counting)", "cnn_baseline", "cnn", "YOLOv3", None, None, 6, "Sec. 4", "we employed an object-based deep learning method with YOLOv3 [8]", cat_note="YOLOv3 convolutional detector",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated")
res = Counter("res")
CLS = (("palm tree", "oil palm"), ("coconut tree", "coconut"), ("banana tree", "banana"))
# Table 4: detection P, R, F per tree type (segments of the row quoted per tree type)
T4V = {"model1": (("Proposed SAM 0.98 0.97 0.97", "0.97"), ("0.89 0.88 0.89", "0.89"), ("0.92 0.92 0.91", "0.91")),
       "model5": (("Putra et al. [8] 0.90 0.95 0.93", "0.93"), ("0.75 0.77 0.76", "0.76"), ("0.78 0.82 0.80", "0.80"))}
for item, cells in T4V.items():
    for k, (ev, v) in enumerate(cells):
        ROWS += result(res(), "f1", item, "dataset1", v, 7, T4, ev, unit="fraction", setting=f"Table 4: detection of {CLS[k][1]} trees, test", note=f"F of the {CLS[k][1]} tree detection (no overall value is printed)", **TD, primary=(k == 0))
# Table 5: classification P, R, F, Acc per tree type
T5V = {"model2": ("Proposed Method 0.88 0.96 0.92 0.94 0.87 0.89 0.88 0.92 0.95 0.88 0.91 0.94", ("0.92", "0.88", "0.91")),
       "model3": ("ViT 0.91 0.92 0.92 0.94 0.83 0.84 0.83 0.89 0.87 0.88 0.87 0.92", ("0.92", "0.83", "0.87")),
       "model4": ("Carvalho et al. [21] 0.85 0.90 0.87 0.92 0.83 0.84 0.83 0.82 0.82 0.85 0.83 0.88", ("0.87", "0.83", "0.83"))}
for item, (ev, fs) in T5V.items():
    for k, v in enumerate(fs):
        ROWS += result(res(), "f1", item, "dataset1", v, 7, T5, ev, unit="fraction", setting=f"Table 5: classification of {CLS[k][1]} trees, test", note=f"F of the {CLS[k][1]} tree classification (no overall value is printed)", **TC, primary=(k == 0))
ROWS += [R("limitations", "The method fails when the drone images lose the crown shape and trunk texture, so the detection step misses regions and the trees are misclassified", 9, "Sec. 5", "when the images captured by drone lost vital information"),
         D("code_available", "no", "No code repository or availability statement for code in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "no", 10, "Data availability", "The authors do not have permission to share data")]
ROWS += [D("notes", "Table 4 compares the proposed SAM with the YOLOv3 method of Putra et al. for detection (P, R and F per tree type) and Table 5 compares the proposed EfficientNetB3 with ViT and the MobileNetV2 method of Carvalho et al. for classification (P, R, F and Acc per tree type). Neither table gives an overall value, so the oil palm column is primary (the first listed subset) and the F of every column is recorded; the per-class Acc (one against the rest) is not recorded. Table 3 (EfficientNet variants B0 to B3, the B3 row repeats the proposed classifier) is not recorded. ViT is the only transformer classifier; it ties the proposed model on oil palm (F 0.92). The test set is 25 original images per tree type (Sec. 4.1); the paper does not say whether the detection results use the same images.",
           "Sec. 4 and Tables 3 to 5 (pp. 6 to 9)")]
