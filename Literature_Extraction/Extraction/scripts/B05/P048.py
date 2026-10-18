import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID = "P048"
# Stage 1: YOLOv7 detects mango canopies (detection at object level, scored on the validation split at the selected checkpoint). Stage 2: SAM delineates each canopy; the pipeline is scored by
# the agreement of its canopy areas with 175 manually digitized canopies of another orchard (attribute estimation, crown area, one value per tree).
TD = Tags(task="detection", level="object", split="validation", agg="single")
TA = Tags(task="attribute_estimation", level="tree", split="test", agg="single", attribute="crown_area")
OBJ_EV = "to detect individual mango trees. The resultant bounding boxes were used as prompts for the segment anything model (SAM) for precise delineation of canopy boundaries"
CK_EV = "was observed at epoch 905, with ζ 2 = 0.8235 and ζ 3 = 0.3524"
GT_EV = "A study was conducted on 175 mango trees in another mango orchard, whose canopies were manually digitized"

ROWS = [
    R("task", "detection;crown_segmentation", 1, "Abstract", "The You Only Look Once (YOLOv7) framework was trained on an annotated dataset to detect individual mango trees"),
    R("output_unit", "tree_instance;crown_mask", 1, "Abstract", "The resultant bounding boxes were used as prompts for the segment anything model (SAM) for precise delineation of canopy boundaries"),
    R("tree_setting", "orchard", 3, "Sec. 2.1", "A 65-acre mango orchard"),
    R("tree_taxa", "mango", 1, "Abstract", "UAV imagery was collected over a 65-acre mango orchard in Multan, Pakistan"),
    R("study_area", "Multan, Pakistan", 1, "Abstract", "UAV imagery was collected over a 65-acre mango orchard in Multan, Pakistan"),
    D("biome", "subtropical", "Multan, Pakistan; not stated in the paper"),
    R("platform", "uav", 5, "Sec. 2.1", "UAV and Realtime Kinematics (RTK) Mobile Station: The DJI Mavic 3M drone"),
    R("sensors", "rgb;multispectral", 5, "Sec. 2.1", "It features a 4/3 CMOS RGB camera with 20 MP", note="the RGB orthomosaic is used for detection and delineation; NDVI and NDRE from multispectral imagery are shown for health mapping (p.17)"),
    R("gsd", "3", 1, "Abstract", "processed into an RGB orthomosaic with a 3 cm ground sampling distance", unit="cm"),
    R("annotation_type", "bounding_box;polygon_mask", 6, "Sec. 2.2.1", "Composite image annotation aimed to enclose each of the 1871 mango tree canopies in a bounding box", note="boxes for training; the 175 evaluation canopies are manually digitized polygons (p.14)"),
    R("dataset_name", "own UAV mango orchard, Multan (RGB orthomosaic, 3 cm, 1871 annotated trees)", 6, "Sec. 2.2.1", "enclose each of the 1871 mango tree canopies in a bounding box", item="dataset1"),
    R("dataset_key", "own_p048_multan_mango", 6, "Sec. 2.2.1", "enclose each of the 1871 mango tree canopies in a bounding box", item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 2.2.2", "This ensured that 87.5% of individual trees (1637) were included in the training sample and 12.5% (234) of trees were kept for validation of the output", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement beyond further inquiries to the corresponding author", item="dataset1"),
    R("n_images", "16", 7, "Table 2", "Total Tiles 16", item="dataset1", unit="tiles", note="14 training tiles and 2 validation or test tiles of the orthomosaic; detection tiles are 800 × 800 pixels"),
    R("n_trees", "1871", 6, "Sec. 2.2.1", "enclose each of the 1871 mango tree canopies in a bounding box", item="dataset1", unit="trees", note="1637 training and 234 validation trees"),
    R("dataset_name", "second mango orchard, 175 manually digitized canopies (evaluation of the pipeline)", 14, "Sec. 3.3.1", GT_EV, item="dataset2"),
    R("dataset_key", "own_p048_multan_mango", 14, "Sec. 3.3.1", GT_EV, item="dataset2", note="same study; another orchard"),
    R("dataset_role", "test_only", 14, "Sec. 3.3.1", GT_EV, item="dataset2", note="called validation in the paper; not used for training or checkpoint selection"),
    D("dataset_public", "no", "No dataset link or availability statement beyond further inquiries to the corresponding author", item="dataset2"),
    R("n_trees", "175", 14, "Sec. 3.3.1", GT_EV, item="dataset2", unit="trees"),
]
ROWS += model("model1", "YOLOv7 with SAM (ViT-L) pipeline", "proposed", "foundation_model", "YOLOv7 detector prompting SAM (large ViT)", None, "zero_shot", 13, "Sec. 3.2",
              "To optimize performance, the large version of the visual transformer within SAM was selected", cat_note="YOLO boxes prompt a pre-trained SAM, as in other YOLO and SAM pipelines (protocol 4.5)",
              pre_status="not_reported", pre_note="pre-trained SAM; the pre-training data are not named", strat_page=7, strat_loc="Sec. 2.3", strat_ev="a pre-trained object segmentation model is used for mask generation for each tree",
              strat_note="SAM is not fine-tuned; YOLOv7 is fine-tuned on the annotated tiles")
ROWS += model("model2", "YOLOv7 (mango canopy detector)", "proposed", "cnn", "YOLOv7", None, "full_fine_tuning", 8, "Sec. 2.3.1", "The variant YOLOv7, chosen for its balance between speed and accuracy, is utilized in this process",
              cat_note="YOLO formulated as a regression with convolutional networks (Sec. 2.3.1)", pre_status="not_reported", pre_note="initial weights are not stated (a transfer learning cycle is mentioned, p.2)",
              strat_page=2, strat_loc="Sec. 1", strat_ev="weights at the end of the transfer learning cycle", strat_note="1000 epochs, batch size 2, lr0 0.01; the checkpoint of epoch 905 is used (Table 3 and Table 4)")
res = Counter("res")
ROWS += result(res(), "ap50", "model2", "dataset1", "0.8235", 12, "Table 4", CK_EV, unit="fraction", setting="Table 4: checkpoint of epoch 905 (minimum validation loss), validation tiles", note="mAP at IoU 0.5 (criterion ζ2)", **TD, primary=True)
ROWS += result(res(), "ap50_95", "model2", "dataset1", "0.3524", 12, "Table 4", CK_EV, unit="fraction", setting="Table 4: checkpoint of epoch 905 (minimum validation loss), validation tiles", note="mAP at IoU 0.5 to 0.95 (criterion ζ3)", **TD, primary=True)
r2 = res()
ROWS += result(r2, "r2", "model1", "dataset2", "0.97", 15, "Sec. 3.3.2", "with an R2 = 0.97, indicates that 97% of the variance in the automated measurements", unit="fraction", setting="canopy areas of the pipeline against 175 manually digitized canopies", note="R2 of the automatic against the manual canopy area (Fig. 8a)", **TA, primary=True, headline=True)
ROWS += result(res(), "rmse", "model1", "dataset2", "80.23", 15, "Sec. 3.3.2", "The RMSE of 80.23 sq ft against the average canopy size of 1290.14 sq ft", unit="sq ft", setting="canopy areas of the pipeline against 175 manually digitized canopies", note="average manual canopy area 1290.14 sq ft", **TA, primary=True)
mp = res()
ROWS += result(mp, "mape", "model1", "dataset2", "4.94", 15, "Fig. 8", "MAPE = 4.94%", unit="%", setting="canopy areas of the pipeline against 175 manually digitized canopies", note="MAPE of the canopy area", **TA, primary=True)
ROWS += [R("limitations", "Validation uses 175 trees of one orchard; a larger and more diverse dataset, lighting and shadow handling are still needed", 16, "Sec. 3.3.3", "further validation with a larger and more diverse dataset would certainly strengthen the reliability of the results")]
ROWS += avail(data="on_request", data_page=18, data_loc="Data Availability Statement", data_ev="further inquiries can be directed to the corresponding author")
ROWS += [D("notes", "Stage 1 (YOLOv7) is scored by mAP at the selected checkpoint (Table 4, validation tiles; precision, recall and F1 only as training curves). Stage 2 (SAM prompted by the boxes) is scored by crown area agreement with 175 manually digitized canopies of another orchard (R2, RMSE in sq ft, MAPE); the segmentation overlap (IoU or F1) is not reported. Fig. 8 also gives Bland-Altman limits of agreement (162.30 and -151.46 in the figure, plus or minus 155.23 in the text) which are not recorded.",
           "Sec. 3 and Fig. 8, Table 4 (pp. 12 to 16)")]
ROWS += conflict("cf1", f"{mp}.value", "4.8", 15, "Sec. 3.3.2", "The MAPE of 4.8%", "V1a",
                 "The abstract and the validation panel of Fig. 8 give 4.94%; the text of Sec. 3.3.2 gives 4.8%", unit="%")
