import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T5, T4 = "P103", "Table 5", "Table 4"
# Ginkgo identification from UAV crowns: segment-then-classify (SAM with tuned mask generator parameters, then a ResNet-101 classifier) against YOLOv8n, in the green-leaf and yellow-leaf periods (separate data
# sets and models). Finding the crowns of one species is scored as species_classification at object level (F1, mAP50). The yellow-leaf period holds the abstract's first result (F1 92.96), so it is the primary dataset
# (rule 6.3: the headline dataset); the green-leaf results are not primary. Table 4 gives the accuracy of the ResNet-101 stage alone (non-primary).
TO = Tags(task="species_classification", level="object", split="test", agg="single")
TC = Tags(task="species_classification", level="tree", split="test", agg="single")
OBJ_EV = "a novel two-stage segment-then-classify (STC) strategy"
SPLIT_EV = "partitioned into training, validation, and test sets using a 5:2:2 ratio"
SRC_EV = "Among them, there were 531 images in the green-leaf stage and 137 images in the yellow-leaf stage"

ROWS = [
    R("task", "species_classification", 1, "Abstract", OBJ_EV, note="identification of Ginkgo crowns against other tree species, with crown segmentation by SAM"),
    R("output_unit", "tree_label;crown_mask", 1, "Abstract", "First, the Segment Anything Model (SAM) was fine-tuned for canopy segmentation across the green-leaf stage and the yellow-leaf stage"),
    R("tree_setting", "natural_forest", 3, "Sec. 2.1", "and contains many naturally distributed Ginkgo populations"),
    R("tree_taxa", "Ginkgo biloba against other broad-leaved species of a subtropical mixed forest", 3, "Sec. 2.1", "This reserve is a subtropical mixed forest of deciduous and evergreen broad-leaved trees"),
    R("study_area", "Tianmu Mountain National Nature Reserve, Lin'an District, Hangzhou, Zhejiang, China", 3, "Sec. 2.1", "Tianmu Mountain National Nature Reserve"),
    R("biome", "subtropical", 3, "Sec. 2.1", "This reserve is a subtropical mixed forest of deciduous and evergreen broad-leaved trees"),
    R("platform", "uav", 4, "Sec. 2.2", "the DJI Mavic 3E, equipped with a standard RGB sensor"),
    R("sensors", "rgb", 4, "Sec. 2.2", "only the RGB sensor data from both UAV platforms were utilized"),
    R("annotation_type", "bounding_box;image_label", 7, "Sec. 2.5", "were manually annotated with bounding boxes using the LabelImg software", note="Ginkgo boxes (positive) and non-Ginkgo crowns (negative); ground-surveyed Ginkgo coordinates from GinkgoDB"),
]
for ds, key, nm, n, ntr, per in (("dataset1", "own_p103_ginkgo_green", "own UAV RGB sub-images of Ginkgo crowns, green-leaf period (2000 x 2000 pixels, Tianmu Mountain)", "531", "2655 training, 1062 validation and 1062 test images after augmentation", "green-leaf"),
                                 ("dataset2", "own_p103_ginkgo_yellow", "own UAV RGB sub-images of Ginkgo crowns, yellow-leaf period (2000 x 2000 pixels, Tianmu Mountain)", "137", "685 training, 274 validation and 274 test images after augmentation", "yellow-leaf")):
    ROWS += [R("dataset_name", nm, 6, "Sec. 2.4", SRC_EV, item=ds), R("dataset_key", key, 6, "Sec. 2.4", SRC_EV, item=ds),
             R("dataset_role", "train_test", 6, "Sec. 2.4", SPLIT_EV, item=ds, note=f"{ntr}; no Ginkgo tree of the validation or test set is in the training set"),
             R("dataset_public", "on_request", 21, "Data Availability Statement", "The data presented in this study is available on request from the", item=ds, note="not public because of privacy"),
             R("n_images", n, 6, "Sec. 2.4", SRC_EV, item=ds, unit="images", note=f"sub-images of the {per} period before augmentation")]
ROWS += [D("n_trees", "137", "sum of the printed numbers of Ginkgo trees of the three plots (44, 45 and 48; Sec. 2.1)", item="dataset1", unit="trees"),
         D("n_trees", "137", "sum of the printed numbers of Ginkgo trees of the three plots (44, 45 and 48; Sec. 2.1)", item="dataset2", unit="trees")]
ROWS += model("model1", "STC (SAM canopy segmentation then ResNet-101 classification, proposed)", "proposed", "foundation_model", "SAM (ViT image encoder) with tuned mask generator parameters and mask post-processing, then a ResNet-101 classifier", "sa1b", "full_fine_tuning", 8, "Sec. 2.6",
              "The image encoder, a high-capacity Vision Transformer (ViT) pre-trained with Masked Autoencoders (MAEs)", cat_note="SAM (ViT encoder) segments the crowns and a ResNet-101 classifies them",
              pre_page=8, pre_loc="Sec. 2.6", pre_ev="Its training on an expansive dataset of 11 million images and over 1 billion masks", pre_note="SAM weights from SA-1B; the ResNet-101 classifier starts from ImageNet weights",
              strat_page=9, strat_loc="Sec. 2.6", strat_ev="We employed a transfer learning approach by initializing the model with ResNet-101 weights pre-trained on the ImageNet dataset",
              strat_note="the ResNet-101 stage is fine-tuned end to end; the 'fine-tuning' of SAM is the tuning of five mask generator parameters (Sec. 2.6), no weight training is described")
ROWS += model("model2", "YOLOv8n", "cnn_baseline", "cnn", "YOLOv8n with a decoupled head", "other", "full_fine_tuning", 7, "Sec. 2.5", "the lightweight YOLOv8n model was adopted for all subsequent experiments", cat_note="convolutional detector",
              pre_page=7, pre_loc="Sec. 2.5", pre_ev="initialized with weights pre-trained on the COCO dataset", pre_note="COCO weights", strat_page=7, strat_loc="Sec. 2.5", strat_ev="initialized with weights pre-trained on the COCO dataset",
              strat_note="trained for up to 500 epochs with early stopping, one model per period")
ROWS += model("model3", "ResNet-101 classifier of the STC second stage", "proposed", "cnn", "ResNet-101 with a new fully connected layer", "imagenet", "full_fine_tuning", 9, "Sec. 2.6",
              "We employed a transfer learning approach by initializing the model with ResNet-101 weights pre-trained on the ImageNet dataset", cat_note="convolutional classifier (stage 2 of STC)")
res = Counter("res")
RID = {}
n_g, ev_g = row(PID, 17, r"Green-leaf\s+YOLOv8\s+(?=\d)", 0)
n_gs, ev_gs = row(PID, 17, r"period\s+STC\s+(?=\d)", 0)
n_y, ev_y = row(PID, 17, r"Yellow-leaf\s+YOLOv8\s+(?=\d)", 0)
n_ys, ev_ys = row(PID, 17, r"period\s+STC\s+(?=\d)", 1)
assert all(len(x) == 4 for x in (n_g, n_gs, n_y, n_ys))  # P, R, F1, mAP50
for item, ds, nums, ev, per in (("model2", "dataset1", n_g, "Green-leaf YOLOv8 " + " ".join(n_g), "green-leaf"), ("model1", "dataset1", n_gs, "STC " + " ".join(n_gs), "green-leaf"),
                                ("model2", "dataset2", n_y, "Yellow-leaf YOLOv8 " + " ".join(n_y), "yellow-leaf"), ("model1", "dataset2", n_ys, "STC " + " ".join(n_ys), "yellow-leaf")):
    prim = ds == "dataset2"
    st = f"Table 5: Ginkgo identification, {per} period, test set"
    head = prim and item == "model1"
    ROWS += result(res(), "f1", item, ds, nums[2], 17, T5, ev, unit="%", setting=st, note="F1 of the Ginkgo identification (precision and recall are also in Table 5)", **TO, primary=prim, headline=head)
    ROWS += result(res(), "ap50", item, ds, nums[3], 17, T5, ev, unit="%", setting=st, note="mAP at IoU 0.50 (the paper's mAP50)", **TO, primary=prim)
# Table 4: accuracy of the ResNet-101 stage (test set)
a_g, ev_a = row(PID, 17, r"^\s*Green-leaf period\s+(?=\d)", 0)
a_y, ev_b = row(PID, 17, r"^\s*Yellow-leaf period\s+(?=\d)", 0)
assert len(a_g) == len(a_y) == 3
ROWS += result(res(), "oa", "model3", "dataset1", a_g[2], 17, T4, ev_a, unit="%", setting="Table 4: ResNet-101 stage, green-leaf period, test set", note="accuracy of the Ginkgo classification stage", **TC)
ROWS += result(res(), "oa", "model3", "dataset2", a_y[2], 17, T4, ev_b, unit="%", setting="Table 4: ResNet-101 stage, yellow-leaf period, test set", note="accuracy of the Ginkgo classification stage", **TC)
ROWS += [R("limitations", "The two-stage workflow (mask generation, post-processing, classification) is more complex and costly than a detector, and the study uses RGB images of one region and two phenological stages", 20, "Sec. 4.4",
           "STC demands a more complex workflow, including mask generation, post-processing, and classification"),
         D("code_available", "no", "No code repository or availability statement for code in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "on_request", 21, "Data Availability Statement", "The data presented in this study is available on request from the")]
ROWS += [D("notes", "Table 5 compares YOLOv8n and the STC strategy on the test sets of the green-leaf and yellow-leaf periods (precision, recall, F1, mAP50; F1 and mAP50 recorded; the YOLOv8 rows repeat the test rows of Table 2). The yellow-leaf period is primary because the abstract's first result (STC F1 92.96) is there; the green-leaf results are recorded without primary. Table 4 gives the accuracy of the ResNet-101 stage on the training, validation and test sets (test recorded). The 5:2:2 split was made on augmented sub-images (Sec. 2.4), and 531 and 137 are the sub-images before augmentation. Table 3 (SAM mask generator settings) and the Table 2 training and validation rows are not recorded.",
           "Sec. 3 and Tables 2, 4 and 5 (pp. 12 to 17)")]
