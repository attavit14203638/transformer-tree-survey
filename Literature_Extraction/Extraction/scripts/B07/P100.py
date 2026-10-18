import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T2 = "P100", "Table 1", "Table 2"
# VLM-TMN (LLaVA-7B with a depth projector and a magnifying glass strategy) estimates DBH and height of roadside trees from street-view images with a given tree box. Table 1 is the main comparison: Wang et al.
# (manual interpretation) and Choi et al. (detection and segmentation) are copied from their papers (their own images, not the benchmark), so they are not primary; Table 2 holds the ablation variants.
# The split of Tables 1 and 2 is not stated (an 8:1:1 split is described), the rounded Table 1 values are used for VLM-TMN (V1a) and the Table 2 values are the cf alternatives.
def tag(attr, **kw):
    return Tags(task="attribute_estimation", level="tree", split="not_stated", agg="single", attribute=attr)(**kw)
OBJ_EV = "are key morphological parameters of roadside tree"
DS_EV = "of point cloud data and 12,447 panoramic images collected"
SPLIT_EV = "First, we split the dataset into training, validation, and test sets"
ROWS = [
    R("task", "attribute_estimation", 1, "Abstract", OBJ_EV, note="DBH and tree height of roadside trees from street view images"),
    R("output_unit", "tree_attribute", 1, "Abstract", OBJ_EV),
    R("tree_setting", "urban", 1, "Sec. 1", "Roadside trees are an important part of the urban ecosystem"),
    NR("tree_taxa", note="roadside trees of Nanjing; the paper does not name species"),
    R("study_area", "Nanjing, China", 5, "Sec. 4.1", "from Nanjing, China. It includes annotations for more than"),
    D("biome", "subtropical", "Nanjing, China; coded from the study area"),
    R("platform", "ground_camera", 1, "Abstract", "using low-cost street view images as an alternative data source", note="12,447 street panoramic images"),
    R("sensors", "rgb", 1, "Abstract", "using low-cost street view images as an alternative data source"),
    R("annotation_type", "bounding_box;regression_target", 5, "Sec. 4.1", "images, object detection boxes, and corresponding tree morpho-", note="tree boxes and morphological parameters (DBH and height) of each tree"),
    R("dataset_name", "WHU-RSTree based benchmark of roadside tree images with boxes, DBH and height (Nanjing)", 5, "Sec. 4.1", DS_EV, item="dataset1"),
    R("dataset_key", "whu_stree_nj", 5, "Sec. 4.1", DS_EV, item="dataset1", note="the paper calls the source dataset WHU-RSTree (Dong et al. 2023); taken as the WHU-STree Nanjing data"),
    R("dataset_role", "train_test", 5, "Sec. 4.2", SPLIT_EV, item="dataset1", note="split 8:1:1 into training, validation and test sets"),
    D("dataset_public", "partial", "built from the WHU-RSTree data; the paper gives no link or release statement for the 19,613-sample benchmark", item="dataset1"),
    R("n_images", "12447", 5, "Sec. 4.1", DS_EV, item="dataset1", unit="images", note="panoramic images of the source dataset"),
    R("n_trees", "19613", 5, "Sec. 4.1", "processing, we constructed a total of 19,613 datasets containing", item="dataset1", unit="trees", note="samples (an image with a tree box and its parameters) after filtering; the paper calls them datasets"),
]
ROWS += model("model1", "VLM-TMN (LLaVA-7B with Depth Projector Module and Magnifying Glass Strategy, proposed)", "proposed", "vision_language_model", "LLaVA-7B with a cross-attention depth projector and a magnifying glass strategy on enlarged tree crops", "other", None, 1, "Abstract",
              "Built upon LLaVA-7B", cat_note="vision language model (LLaVA-7B, a CLIP image encoder with a language model)", pre_page=1, pre_loc="Sec. 1", pre_ev="We use Llava-7B (Liu et al., 2023b) as our baseline", pre_note="LLaVA-7B weights; supervised fine-tuning for the task",
              strat_note="supervised fine-tuning; the paper does not say which parts are trainable (Fig. 1 marks trainable and frozen parts)")
ROWS += model("model2", "LLaVA-7B (original, no module and no fine-tuning)", "transformer_baseline", "vision_language_model", "LLaVA-7B", "other", "zero_shot", 5, "Sec. 4.5", "generate into the original Llava-7B model", cat_note="vision language model",
              pre_page=1, pre_loc="Sec. 1", pre_ev="We use Llava-7B (Liu et al., 2023b) as our baseline", pre_note="LLaVA-7B weights", strat_page=5, strat_loc="Sec. 4.5", strat_ev="removed the depth projector module",
              strat_note="the row W/o All of Table 2: all modules and the supervised fine-tuning removed")
ROWS += model("model3", "LLaVA-7B with supervised fine-tuning only (W/o Depth & Visual)", "transformer_baseline", "vision_language_model", "LLaVA-7B fine-tuned on the tree measurement data", "other", None, 5, "Sec. 4.5", "removed the depth projector module", cat_note="vision language model",
              pre_page=1, pre_loc="Sec. 1", pre_ev="We use Llava-7B (Liu et al., 2023b) as our baseline", pre_note="LLaVA-7B weights", strat_note="supervised fine-tuning (SFT); the paper does not say which parts are trainable")
ROWS += model("model4", "VLM-TMN without the Depth Projector Module (W/o Depth)", "proposed", "vision_language_model", "LLaVA-7B with the magnifying glass strategy only", "other", None, 5, "Sec. 4.5", "removed the depth projector module", cat_note="vision language model",
              pre_page=1, pre_loc="Sec. 1", pre_ev="We use Llava-7B (Liu et al., 2023b) as our baseline", pre_note="LLaVA-7B weights", strat_note="supervised fine-tuning; the paper does not say which parts are trainable")
ROWS += model("model5", "Wang et al. (manual interpretation with reference objects)", "other_baseline", "classical", "manual measurement against standard urban objects of known size", None, None, 5, T1, "Wang et al. Manual interpretation 1.9 cm 0.5 m / / 0.67 h",
              cat_note="manual interpretation of images, no neural network", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model6", "Choi et al. (detection and segmentation of trees, measurement from pixel counts)", "other_baseline", None, "detection and segmentation pipeline", None, None, 5, T1, "Choi et al. Detection + Segmentation 15 cm 3.28 m 0.44 0.24 /",
              cat_note="the paper names the pipeline (detection and segmentation) but does not describe the networks", pre_status="not_reported", pre_note="not stated", strat_note="not stated")
res = Counter("res")
RID = {}
EV_OURS = "Ours Detection + VLM-TMN 7.1 cm 2.3 m 0.23 0.17 0.27 s"
EV_WANG = "Wang et al. Manual interpretation 1.9 cm 0.5 m / / 0.67 h"
EV_CHOI = "Choi et al. Detection + Segmentation 15 cm 3.28 m 0.44 0.24 /"
SET1 = "Table 1: comparison with image-based methods, "
# Table 1: RMSE (and NRMSE) of DBH and height
RID["dbh"] = res()
ROWS += result(RID["dbh"], "rmse", "model1", "dataset1", "7.1", 5, T1, EV_OURS, unit="cm", setting=SET1 + "DBH, benchmark data", note="RMSE of the DBH", **tag("dbh"), primary=True, headline=True)
RID["height"] = res()
ROWS += result(RID["height"], "rmse", "model1", "dataset1", "2.3", 5, T1, EV_OURS, unit="m", setting=SET1 + "height, benchmark data", note="RMSE of the tree height", **tag("height"), primary=True)
ROWS += result(res(), "other", "model1", "dataset1", "0.23", 5, T1, EV_OURS, unit="ratio", setting=SET1 + "DBH, benchmark data", note="NRMSE of the DBH (normalised RMSE, definition not given)", **tag("dbh"))
ROWS += result(res(), "other", "model1", "dataset1", "0.17", 5, T1, EV_OURS, unit="ratio", setting=SET1 + "height, benchmark data", note="NRMSE of the tree height (normalised RMSE, definition not given)", **tag("height"))
ROWS += result(res(), "rmse", "model5", "dataset1", "1.9", 5, T1, EV_WANG, unit="cm", setting=SET1 + "DBH, values of the cited paper", note="copied from the cited paper; measured on its own images, not on the benchmark", copied_from="Wang et al. (2018)", **tag("dbh"))
ROWS += result(res(), "rmse", "model5", "dataset1", "0.5", 5, T1, EV_WANG, unit="m", setting=SET1 + "height, values of the cited paper", note="copied from the cited paper; measured on its own images, not on the benchmark", copied_from="Wang et al. (2018)", **tag("height"))
ROWS += result(res(), "rmse", "model6", "dataset1", "15", 5, T1, EV_CHOI, unit="cm", setting=SET1 + "DBH, values of the cited paper", note="copied from the cited paper; measured on its own images, not on the benchmark", copied_from="Choi et al. (2022)", **tag("dbh"))
ROWS += result(res(), "rmse", "model6", "dataset1", "3.28", 5, T1, EV_CHOI, unit="m", setting=SET1 + "height, values of the cited paper", note="copied from the cited paper; measured on its own images, not on the benchmark", copied_from="Choi et al. (2022)", **tag("height"))
ROWS += result(res(), "other", "model6", "dataset1", "0.44", 5, T1, EV_CHOI, unit="ratio", setting=SET1 + "DBH, values of the cited paper", note="NRMSE of the DBH; copied from the cited paper", copied_from="Choi et al. (2022)", **tag("dbh"))
ROWS += result(res(), "other", "model6", "dataset1", "0.24", 5, T1, EV_CHOI, unit="ratio", setting=SET1 + "height, values of the cited paper", note="NRMSE of the tree height; copied from the cited paper", copied_from="Choi et al. (2022)", **tag("height"))
# Table 2: ablation variants (the Whole Pipeline row is VLM-TMN, whose Table 1 values are used; see cf1 and cf2)
for item, ev, d, h in (("model2", "W/o All 0.24s 24.39cm 9.37m 0.80 0.73 1.12 0.82", "24.39", "9.37"), ("model3", "W/o Depth & Visual 0.24s 7.66cm 2.41m 0.25 0.19 0.22 0.17", "7.66", "2.41"),
                       ("model4", "W/o Depth 0.26s 7.47cm 2.37m 0.24 0.18 0.22 0.18", "7.47", "2.37")):
    ROWS += result(res(), "rmse", item, "dataset1", d, 5, T2, ev, unit="cm", setting="Table 2: ablation, DBH", note="RMSE of the DBH; ablation variant", **tag("dbh"))
    ROWS += result(res(), "rmse", item, "dataset1", h, 5, T2, ev, unit="m", setting="Table 2: ablation, height", note="RMSE of the tree height; ablation variant", **tag("height"))
ROWS += [R("inference_speed", "0.27 s per tree (hardware not stated; training on one RTX 4090)", 5, T1, EV_OURS, item="model1"),
         R("inference_speed", "0.24 s per tree (hardware not stated)", 5, T2, "W/o All 0.24s 24.39cm 9.37m 0.80 0.73 1.12 0.82", item="model2"),
         R("inference_speed", "0.24 s per tree (hardware not stated)", 5, T2, "W/o Depth & Visual 0.24s 7.66cm 2.41m 0.25 0.19 0.22 0.17", item="model3"),
         R("inference_speed", "0.26 s per tree (hardware not stated)", 5, T2, "W/o Depth 0.26s 7.47cm 2.37m 0.24 0.18 0.22 0.18", item="model4"),
         R("inference_speed", "0.67 h (manual interpretation; unit of time cost as printed)", 5, T1, EV_WANG, item="model5")]
ROWS += conflict("cf1", f"{RID['dbh']}.value", "7.08", 5, T2, "Whole Pipeline 0.27s 7.08cm 2.25m 0.23 0.17 0.21 0.16", "V1a",
                 "Table 1, the main comparison, gives 7.1 cm for the DBH RMSE of VLM-TMN; Table 2 (ablation) and the abstract give 7.08 cm", unit="cm")
ROWS += conflict("cf2", f"{RID['height']}.value", "2.25", 5, T2, "Whole Pipeline 0.27s 7.08cm 2.25m 0.23 0.17 0.21 0.16", "V1a",
                 "Table 1, the main comparison, gives 2.3 m for the height RMSE of VLM-TMN; Table 2 (ablation) gives 2.25 m", unit="m")
ROWS += [R("limitations", "The performance may degrade in unseen scenarios, and the compared methods were evaluated on ideal unobstructed photographs with their own data", 5, "Sec. 4.4 and 5", "the performance may be degraded when facing some unseen scenarios"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         D("data_available", "partial", "built from the WHU-RSTree data; no statement about the released benchmark")]
ROWS += [D("notes", "Table 1 compares VLM-TMN with Wang et al. (manual interpretation) and Choi et al. (detection and segmentation): RMSE and NRMSE of DBH and height and the time cost. The values of the two cited methods are copied from their papers (the paper says they have no open code) and were measured on their own images; the comparison methods used ideal unobstructed photographs, the benchmark has many occlusions. Table 2 is the ablation (original LLaVA-7B, supervised fine-tuning only, without the depth projector, whole pipeline; RMSE, RRMSE, REL and time); RMSE is recorded for the three ablation variants, RRMSE and REL are not. The whole-pipeline row repeats Table 1 with more digits (cf items). The split of the results is not stated; the abstract's 7.57 percent is the RMSE gain over supervised fine-tuning alone (7.66 to 7.08 cm).",
           "Sec. 4 and Tables 1 and 2 (p. 5)")]
