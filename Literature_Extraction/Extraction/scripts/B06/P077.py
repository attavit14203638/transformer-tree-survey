import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T5, T6 = "P077", "Table 5", "Table 6"
# Zero-shot instance segmentation of individual poplar seedlings in UAV images (Grounding DINO with SAM2) against three trained instance models (Table 5, test split 6:2:2), and biomass regression
# on the image features of the masks (Table 6; four regression models, three inputs). The paper presents two main results (precision 0.943 and R2 0.823), so there is no headline.
TS = Tags(task="crown_segmentation", level="instance_mask", split="test", agg="single")
TB = Tags(task="attribute_estimation", level="tree", split="test", agg="not_stated", attribute="biomass")
OBJ_EV = "assess the efficiency of using Grounding DINO-SAM2 for zero-shot instance segmentation of individual poplar seedlings across multiple genotypes"
DS_EV = "we collected 200 RGB images and multispectral images (80 in 2023 and 120 in 2024)"
REG_EV = "traditional regression modelling methods support vector machine (SVM) and random forest (RF), as well as deep learning regression modelling methods convolutional neural network (CNN) and multilayer perceptron (MLP)"

ROWS = [
    R("task", "crown_segmentation;attribute_estimation", 1, "Abstract", OBJ_EV, note="seedling instance masks, then biomass estimated from image features of the masks"),
    R("output_unit", "crown_mask;tree_attribute", 1, "Abstract", OBJ_EV),
    R("tree_setting", "plantation", 2, "Sec. 2.1", "was conducted in 2023 and 2024 at the Baguazhou Qinghao Seedling Breeding Base in Qixia District", note="poplar seedling nursery (pots in 2023, field rows in 2024)"),
    R("tree_taxa", "poplar (Populus) seedlings, six genotypes", 2, "Sec. 2.1", "Six poplar seedlings with"),
    R("study_area", "Qixia and Lishui Districts, Nanjing, Jiangsu Province, China", 2, "Sec. 2.1", "both in Nanjing, Jiangsu Province"),
    D("biome", "subtropical", "Nanjing, China; coded from the study area"),
    R("platform", "uav", 2, "Sec. 2.2", "UAV equipped with a multispectral camera", note="DJI M350 RTK with a MicaSense RedEdge-MX at 12 m; DJI Mini 2 RGB camera at 10 m"),
    R("sensors", "rgb;multispectral", 3, "Sec. 2.2", "The RGB images were acquired by DJI Mini 2", note="RGB and five-band multispectral (B, G, R, NIR, RedEdge)"),
    NR("gsd", note="flight altitudes of 12 m (multispectral) and 10 m (RGB) are given, the GSD is not"),
    R("annotation_type", "polygon_mask;regression_target", 3, "Sec. 2.4", "Four hundred images were manually annotated using the EIseg", note="masks only for evaluation, never for training; destructive dry mass of the seedlings as the regression target"),
    R("dataset_name", "own UAV RGB and multispectral images of poplar seedlings, Nanjing (2023 and 2024)", 3, "Sec. 2.4", DS_EV, item="dataset1"),
    R("dataset_key", "own_p077_poplar_uav", 3, "Sec. 2.4", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 3.1", "The images were split at a ratio of 6:2:2 for training", item="dataset1", note="the split is for the trained baselines; Grounding DINO-SAM2 needs no training"),
    R("dataset_public", "on_request", 10, "Data availability", "Data will be made available on request", item="dataset1"),
    R("n_images", "400", 3, "Sec. 2.4", "In this study, we selected NIR band images with the largest ground feature difference for the segmentation network, and the total number of model input images was 400", item="dataset1", unit="images", note="200 RGB and 200 near-infrared images of culture groups"),
    R("n_trees", "3772", 3, "Sec. 2.4", "the number of images of single plants was 3772", item="dataset1", unit="trees", note="1280 in 2023 and 2492 in 2024"),
]
ROWS += model("model1", "Grounding DINO-SAM2 (zero-shot, proposed)", "proposed", "foundation_model", "Grounding DINO text-prompted detector and SAM2 mask decoder", None, "zero_shot", 1, "Abstract", OBJ_EV,
              cat_note="Grounding DINO and SAM2 are transformer-based foundation models used without training", pre_status="not_reported", pre_note="pre-trained models used as released; the pre-training data are not stated",
              strat_page=1, strat_loc="Abstract", strat_ev="zero-shot instance segmentation of individual poplar seedlings", strat_note="no training required (Table 1); the labels are checked by hand in less than 15 minutes")
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained on the 60 percent training images of the 6:2:2 split")
for item, nm, cat, bb in (("model2", "YOLOv8-Seg", "cnn", "YOLOv8-Seg"), ("model3", "YOLACT", "cnn", "YOLACT"), ("model4", "Mask R-CNN", "cnn", "Mask R-CNN")):
    ROWS += model(item, nm, "cnn_baseline", cat, bb, None, None, 7, "Sec. 3.1", "used instance segmentation models (namely, YOLOv8-Seg, YOLACT,", cat_note="convolutional instance segmentation", **NPRE)
RGM = dict(pre_status="not_reported", pre_note="not applicable: regression on image features", strat_note="hyperparameter optimisation with 5-fold cross-validation (Sec. 3.2)")
ROWS += model("model5", "SVM biomass regression", "other_baseline", "classical", "support vector machine on image features of the masks", None, None, 7, "Sec. 3.2", REG_EV, cat_note="classical regression", **RGM)
ROWS += model("model6", "Random forest biomass regression", "other_baseline", "classical", "random forest on image features of the masks", None, None, 7, "Sec. 3.2", REG_EV, cat_note="classical regression", **RGM)
ROWS += model("model7", "CNN biomass regression", "cnn_baseline", "cnn", "convolutional regression network on image features of the masks", None, None, 7, "Sec. 3.2", REG_EV, cat_note="convolutional regression network", **RGM)
ROWS += model("model8", "MLP biomass regression", "other_baseline", "other", "multilayer perceptron on image features of the masks", None, None, 7, "Sec. 3.2", REG_EV, cat_note="multilayer perceptron (not a transformer or CNN)", **RGM)
res = Counter("res")
for item, lab in (("model2", "YOLOv8-Seg"), ("model3", "YOLACT"), ("model4", "Mask R-CNN"), ("model1", "Grounding DINO-SAM2")):
    nums, ev = row(PID, 8, rf"^\s*{lab}\s+(?=\d)", 0, cols=(0, 90))
    assert len(nums) == 5, (lab, nums)  # precision, recall, mAP, training time, test time
    ROWS += result(res(), "map", item, "dataset1", nums[2], 8, T5, ev, unit="%", setting="Table 5: test results of the segmentation models", note="mAP of the seedling masks; the IoU threshold is not stated (precision and recall are in the table)", **TS, primary=True)
RID = {}
for item, lab in (("model5", "SVM"), ("model6", "RF"), ("model7", "CNN"), ("model8", "MLP")):
    nums, ev = row(PID, 8, rf"^\s*{lab}\s+(?=\d)", 0, cols=(0, 95))
    assert len(nums) == 6, (lab, nums)  # RGB R2, RMSE, multispectral R2, RMSE, fusion R2, RMSE
    for k, inp in enumerate(("RGB features", "multispectral features", "RGB and multispectral feature fusion")):
        prim = inp.endswith("fusion")
        st = f"Table 6: biomass regression on {inp} of the Grounding DINO-SAM2 masks"
        RID[(item, k, "r2")] = res()
        ROWS += result(RID[(item, k, "r2")], "r2", item, "dataset1", nums[2 * k], 8, T6, ev, unit="fraction", setting=st, note="R2 of the biomass estimate (test set)", **TB, primary=prim)
        RID[(item, k, "rmse")] = res()
        ROWS += result(RID[(item, k, "rmse")], "rmse", item, "dataset1", nums[2 * k + 1], 8, T6, ev, unit="g/m2", setting=st, note="RMSE of the biomass estimate as printed (g/m2)", **TB, primary=prim)
ROWS += [R("limitations", "The 2D optical images capture the canopy but not the vertical organs, and the biomass models rest on two years of data from one region", 8, "Sec. 3.2", "structural information stored", note="the authors name the missing 3D structure (Sec. 3.2); two years of data (abstract)"),
         D("code_available", "no", "No code repository or availability statement in the paper; the introduction cites https://gitee.com/njtqf/iiisss/tree/master after SAM2 without saying that it is the authors' repository"),
         D("weights_available", "not_stated", "No statement about trained weights"), R("data_available", "on_request", 10, "Data availability", "Data will be made available on request")]
ROWS += [D("notes", "Table 5 compares Grounding DINO-SAM2 (zero-shot) with YOLOv8-Seg, YOLACT and Mask R-CNN (precision, recall, mAP, training and test hours); mAP is recorded, and the abstract's precision of 0.943 is the Table 5 value 94.3. Table 6 gives R2 and RMSE of four biomass regression models on RGB, multispectral and fused features of the masks (all 24 cells recorded; fusion is primary). Table 7 (label-free checking against label checking, and the segmentation and biomass accuracy for each input) is not recorded. The paper presents two main results (the segmentation precision and the R2 of 0.823), so there is no headline.",
           "Sec. 3 and Tables 5 to 7 (pp. 7 to 8)")]
ROWS += conflict("cf1", f"{RID[('model7', 2, 'rmse')]}.value", "0.132", 7, "Sec. 3.2", "after the fusion of multispectral and RGB features into the model, R2 can reach 0.823 and RMSE is 0.132 g/m2", "V1a",
                 "Table 6, the results table, gives 0.089 for the CNN fusion RMSE; the text of Sec. 3.2 gives 0.132, which is the MLP fusion value of Table 6", unit="g/m2")
