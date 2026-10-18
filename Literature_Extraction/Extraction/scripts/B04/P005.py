import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T8 = "P005", "Table 2", "Table 8"
URL = "https://github.com/HAAClassic/AdaTreeFormer"
# Default tags (protocol 6): counts per image (E_MAE, E_RMSE, E_R2) are counting at image level; E_F1 of the detected tree locations is detection at object level.
# The paper never says that runs are repeated or averaged, so agg is single.
T = Tags(task="counting", level="image", split="test", agg="single")
TF1 = T(task="detection", level="object")
SRC_TGT = "In each scenario, one dataset is set as the source domain, and the other as the target domain"

ROWS = [
    R("task", "counting;detection", 1, "Abstract", "The process of estimating and counting tree density using only a single aerial or satellite image",
      note="tree counting from density maps; the locations of the detected trees are scored with F1 (Sec. 4.2)"),
    R("output_unit", "tree_count", 2, "Sec. 1", "a transformer-based decoder with domain adaptation heads to produce the final tree density maps", note="density maps give the count"),
    R("tree_setting", "mixed", 8, "Table 1", "Urban, residential area, dense park ... Cropland, suburban, urban, residential area ... Wooded mountainous"),
    NR("tree_taxa", note="all trees; species not given"),
    R("study_area", "London, United Kingdom; Jiangsu Province, China; Yosemite National Park, California, United States", 8, "Sec. 4.1",
      "captured in Jiangsu Province, China ... revolves around Yosemite National Park, located in California ... from London, United Kingdom"),
    D("biome", "multiple", "London (temperate), Jiangsu (subtropical) and Yosemite (temperate); not stated in the paper"),
    R("platform", "multiple", 8, "Sec. 4.1", "encompasses 24 satellite images taken by the GaofenII satellite ... a rectangular image from Google Maps",
      note="Jiangsu: Gaofen-II satellite; Yosemite: Google Maps image; the platform of the London images is not stated"),
    R("sensors", "rgb", 8, "Fig. 5", "Some samples of RGB images and corresponding tree density maps of the Yosemite, London, and Jiangsu datasets"),
    R("gsd", "0.2", 8, "Sec. 4.1.2", "captured at 0.2 m GSD from London, United Kingdom", unit="m", note="London 0.2 m; Jiangsu 0.8 m; Yosemite 0.12 m"),
    R("annotation_type", "point", 9, "Sec. 4.3", "The ground truth consists of the coordinates of tree locations, indicated by annotation dots"),
    # datasets (keys shared with P014 and P030, same authors and repositories)
    R("dataset_name", "London dataset (high-resolution images, 0.2 m GSD)", 8, "Sec. 4.1.2",
      "This dataset consists of high-resolution images captured at 0.2 m GSD from London, United Kingdom for training and testing", item="dataset1"),
    R("dataset_key", "kcl_london", 8, "Sec. 4.1.2", "There are 95,067 trees that are manually annotated across 613 images", item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 4.4", SRC_TGT, item="dataset1", note="source domain in two scenarios, target domain (5 labelled images) in two"),
    R("dataset_public", "yes", 8, "Sec. 4.1.2", "available at https://github.com/HAAClassic/TreeFormer/tree/main", item="dataset1"),
    R("dataset_url", "https://github.com/HAAClassic/TreeFormer/tree/main", 8, "Sec. 4.1.2", "available at https://github.com/HAAClassic/TreeFormer/tree/main", item="dataset1"),
    R("n_images", "613", 8, "Sec. 4.1.2", "There are 95,067 trees that are manually annotated across 613 images", item="dataset1", unit="images",
      note="452 training and 161 test images (p.8); the same section later gives a training set of 450 images and a test set of 152 images"),
    R("n_trees", "95067", 8, "Sec. 4.1.2", "There are 95,067 trees that are manually annotated across 613 images", item="dataset1", unit="trees", note="Table 1 also gives 95,067"),
    R("dataset_name", "Jiangsu dataset (Gaofen-II satellite, 0.8 m GSD)", 8, "Sec. 4.1.1",
      "This dataset encompasses 24 satellite images taken by the GaofenII satellite with a ground sample distance (GSD) of 0.8 m", item="dataset2"),
    R("dataset_key", "jiangsu_gaofen2", 8, "Sec. 4.1.1", "There are 664,487 manually annotated trees spread across 2400 images", item="dataset2"),
    R("dataset_role", "train_test", 9, "Sec. 4.4", SRC_TGT, item="dataset2", note="source domain in two scenarios, target domain (5 labelled images) in two"),
    R("dataset_public", "yes", 8, "Sec. 4.1.1", "available at https://github.com/sddpltwanqiu/TreeCountNet/tree/main", item="dataset2"),
    R("dataset_url", "https://github.com/sddpltwanqiu/TreeCountNet/tree/main", 8, "Sec. 4.1.1", "available at https://github.com/sddpltwanqiu/TreeCountNet/tree/main", item="dataset2"),
    R("n_images", "2400", 8, "Sec. 4.1.1", "There are 664,487 manually annotated trees spread across 2400 images, 1920 images are used for training while 480 images for testing",
      item="dataset2", unit="images", note="cut from 24 Gaofen-II satellite images; 1920 training and 480 test images"),
    R("n_trees", "664487", 8, "Sec. 4.1.1", "There are 664,487 manually annotated trees spread across 2400 images", item="dataset2", unit="trees", note="Table 1 also gives 664,487"),
    R("dataset_name", "Yosemite dataset (Google Maps image of Yosemite National Park, 0.12 m GSD)", 8, "Sec. 4.1.3",
      "It comprises a rectangular image from Google Maps with dimensions of 19,200 × 38,400 pixels and a GSD of 0.12 m", item="dataset3"),
    R("dataset_key", "yosemite_trees", 8, "Sec. 4.1.3", "Within this image, 98,949 trees have been manually annotated", item="dataset3"),
    R("dataset_role", "train_test", 9, "Sec. 4.4", SRC_TGT, item="dataset3", note="source domain in two scenarios, target domain (5 labelled images) in two"),
    R("dataset_public", "yes", 8, "Sec. 4.1.3", "available at https://github.com/nightonion/yosemite-tree-dataset", item="dataset3"),
    R("dataset_url", "https://github.com/nightonion/yosemite-tree-dataset", 8, "Sec. 4.1.3", "available at https://github.com/nightonion/yosemite-tree-dataset", item="dataset3"),
    R("n_images", "2700", 8, "Sec. 4.1.3", "The image is split into 2700 smaller images, with 1350 chosen for training and the remaining 1350 selected for testing",
      item="dataset3", unit="images", note="1350 training and 1350 test images cut from one 19,200 × 38,400 pixel image"),
    R("n_trees", "98949", 8, "Sec. 4.1.3", "Within this image, 98,949 trees have been manually annotated", item="dataset3", unit="trees", note="Table 1 also gives 98,949"),
]

# Models: the transformer-based AdaTreeFormer and its two degraded versions (the baseline it is built on). Table 2 lists 12 methods; the other nine are not
# described architecturally in the paper, so no best-CNN row can be chosen (protocol 4.1) and they are left to the notes.
NPRE = dict(pre_status="not_reported", pre_note="not stated; the text only says the transformer parameters are set according to Liu et al. (2021b)")
ROWS += model("model1", "AdaTreeFormer", "proposed", "hierarchical_vit",
              "shared hierarchical tree feature extraction (HTFE) encoder of shifted-window transformer blocks (layers 2, 2, 18, 2) with transformer-based domain attention decoder",
              None, "few_shot", 1, "Abstract",
              "Our method, termed as AdaTreeFormer, contains one shared encoder with a hierarchical feature extraction scheme",
              strat_page=9, strat_loc="Sec. 4.3", strat_ev="We only select 5 labeled images (5-shot) from the target domain to evaluate the cross-domain capability of AdaTreeFormer by default",
              strat_note="5 labelled target-domain images by default (1 and 10 shots in Table 8)", cat_note="shifted-window (Swin-type, Liu et al. 2021b) hierarchical transformer encoder", **NPRE)
ROWS += model("model2", "Baseline", "transformer_baseline", "hierarchical_vit", "AdaTreeFormer degraded to one subnet, trained on the source data only", None, None, 9, "Sec. 4.4",
              "AdaTreeFormer can be degraded with one subnet for training on the source data only, which is indicated as the Baseline in the table",
              strat_note="trained on the source domain only (no target labels); initialisation not stated", cat_note="the baseline AdaTreeFormer is built on", **NPRE)
ROWS += model("model3", "Baseline+FT", "transformer_baseline", "hierarchical_vit", "AdaTreeFormer degraded to one subnet, fine-tuned on 5 target images", None, "few_shot", 9, "Sec. 4.4",
              "this trained baseline can be further fine-tuned (FT) using 5 shots of the target domain (Baseline+FT)",
              strat_note="fine-tuned with 5 labelled target-domain images; which parameters are updated is not stated", cat_note="the baseline AdaTreeFormer is built on", **NPRE)

# Table 2: six scenarios (source to target); each layout line holds the two scenarios of one row group (left block, right block), six values each
# in the order E_MAE, E_RMSE, E_R2, E_P, E_R, E_F1.
LINES = {"model1": r"^\s*AdaTreeFormer\s+✓\s+✓\s+", "model2": r"^\s*Baseline\s+✗\s+✗\s+", "model3": r"^\s*Baseline\+FT\s+✗\s+✓\s+"}
SCEN = [  # label, group, side, target dataset item, source name, target name
    ("S_L → S_J", 0, 0, "dataset2", "London", "Jiangsu"), ("S_L → S_Y", 0, 1, "dataset3", "London", "Yosemite"),
    ("S_J → S_L", 1, 0, "dataset1", "Jiangsu", "London"), ("S_J → S_Y", 1, 1, "dataset3", "Jiangsu", "Yosemite"),
    ("S_Y → S_L", 2, 0, "dataset1", "Yosemite", "London"), ("S_Y → S_J", 2, 1, "dataset2", "Yosemite", "Jiangsu"),
]
TRAIN = {"model1": "all {s} training images plus 5 labelled {t} images", "model2": "{s} training images only (no target labels)",
         "model3": "{s} Baseline fine-tuned on 5 labelled {t} images"}
PRIMARY = "S_Y → S_J"  # the cross-domain example of the abstract (the headline); every compared model is primary there (protocol 6.3)
LINE = {m: [row(PID, 10, rx, n) for n in range(3)] for m, rx in LINES.items()}
for m, parsed in LINE.items():
    assert all(len(nums) == 12 for nums, _ in parsed), (m, parsed)
res, tr, de = Counter("res"), Counter("tr"), Counter("de")
ITEM = {}
for lab, g, side, ds, sname, tname in SCEN:
    for m in ("model1", "model2", "model3"):
        nums, ev = LINE[m][g]
        v = nums[6 * side:6 * side + 6]  # E_MAE, E_RMSE, E_R2, E_P, E_R, E_F1
        setting = f"{lab}: {TRAIN[m].format(s=sname, t=tname)}; {tname} test set"
        prim = lab == PRIMARY
        for metric, val, unit, tags, nt in (("mae", v[0], "trees", T, "E_MAE, counting error in trees per image"), ("rmse", v[1], "trees", T, "E_RMSE, counting error in trees per image"),
                                            ("r2", v[2], "fraction", T, "E_R2"), ("f1", v[5], "%", TF1, "E_F1 of the tree locations (search circle of 15 pixels)")):
            it = res()
            ITEM[(lab, m, metric)] = it
            ROWS += result(it, metric, m, ds, val, 10, f"{T2} ({lab})", ev, unit=unit, setting=setting, note=nt, **tags, primary=prim,
                           headline=(m == "model1" and lab == PRIMARY and metric == "mae"))
# Transfer pairs (V8): the proposed model on every scenario; the source-domain test result of the same model is not printed, so value_in_domain is not_reported
for lab, g, side, ds, sname, tname in SCEN:
    nums, ev = LINE["model1"][g]
    t = tr()
    ROWS += [R("transfer_source", f"{sname} dataset (all training images) plus 5 labelled {tname} images", 9, "Sec. 4.4", SRC_TGT, item=t),
             R("transfer_target", f"{tname} dataset test set", 9, "Sec. 4.4", SRC_TGT, item=t),
             R("shift_type", "region", 7, "Sec. 4.1", "gathered from different domains, i.e. different countries that contain various tree shapes, sizes, and distributions", item=t,
               note="the datasets also differ in GSD (0.8, 0.2 and 0.12 m) and landscape"),
             R("transfer_model", "model1", 10, f"{T2} ({lab})", ev, item=t),
             R("transfer_metric", "mae", 10, f"{T2} ({lab})", ev, item=t, note="E_MAE"),
             NR("value_in_domain", item=t, note="the paper reports only target-domain results (Table 2); no source-domain test result of the same model"),
             R("value_out_of_domain", nums[6 * side], 10, f"{T2} ({lab})", ev, item=t, unit="trees"),
             R("transfer_level", "image", 10, f"{T2} ({lab})", ev, item=t)]
# Data efficiency (Table 8): AdaTreeFormer with 1, 5 and 10 labelled target images; columns are E_MAE and E_F1 for S_L → S_J, S_J → S_L, S_Y → S_L
for shots in ("1", "5", "10"):
    nums, ev = row(PID, 14, rf"^\s*{shots} shot\s+", 0, cols=(0, 96))
    assert len(nums) == 6, (shots, nums)
    for k, lab in enumerate(("S_L → S_J", "S_J → S_L", "S_Y → S_L")):
        for metric, val, unit in (("mae", nums[2 * k], "trees"), ("f1", nums[2 * k + 1], "%")):
            d = de()
            ROWS += [R("label_model", "model1", 14, T8, ev, item=d),
                     R("label_amount", shots, 14, T8, ev, item=d, note=f"{shots} labelled image(s) of the target domain; scenario {lab}"),
                     R("label_unit", "images", 14, "Sec. 4.5.3", "1, 5, and 10 labeled images of the target domain are utilized for training", item=d),
                     R("label_metric", metric, 14, T8, ev, item=d, note="E_MAE" if metric == "mae" else "E_F1 of the tree locations"),
                     R("label_value", val, 14, T8, ev, item=d, unit=unit, note=f"scenario {lab}")]
ROWS += [R("limitations", "Accuracy can fall when source and target images come from different seasons (leafless versus leafy trees) or when a sparse, low-diversity source meets a complex target",
           9, "Sec. 4.4", "when the source and target domain images pertain to distinct seasons, the performance of the model might deteriorate")]
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=f"The codes and datasets are available at {URL}", data="yes", data_page=1, data_loc="Abstract",
              data_ev=f"The codes and datasets are available at {URL}")
ROWS += [D("notes", "Table 2 (p.10) compares 12 methods in six source-to-target scenarios on the target test set; recorded are the transformer-based AdaTreeFormer, its source-only Baseline and Baseline+FT (protocol 4.1: the paper does not describe the architecture of AdaCrowd, CODA, CycleGAN, SE-CycleGAN, DGCC, NLT, FSCC, FusionCount+FT or CHSNet+FT, so no best CNN can be chosen). E_P and E_R are in Table 2 too. The abstract's example, Yosemite to Jiangsu (S_Y → S_J), is the headline and the primary setting. Tables 3 to 7 are ablations of AdaTreeFormer; Table 8 varies the number of target images (data-efficiency items). Only target-domain results are printed, so transfer pairs have no in-domain value.",
           "Sec. 4.4 to 4.5 and Tables 2 to 8 (pp. 9 to 14)")]
# Value choices (protocol 6.1): the text of Sec. 4.4 gives 15.9 as the AdaTreeFormer E_MAE in S_Y → S_J, Table 2 gives 122.8; Table 8 gives 15.5 for the source-only E_F1 in S_Y → S_L, Table 2 gives 15.3
ROWS += conflict("cf1", f"{ITEM[('S_Y → S_J', 'model1', 'mae')]}.value", "15.9", 9, "Sec. 4.4",
                 "The achieved ... using our model in the ... scenario is 15.9, much lower than that of the NLT", "V1a",
                 "Table 2, the main comparison, gives 122.8; the abstract uses 15.9 as the reduction in absolute counting error (138.7 of FusionCount+FT minus 122.8)", unit="trees")
ROWS += conflict("cf2", f"{ITEM[('S_Y → S_L', 'model2', 'f1')]}.value", "15.5", 14, T8, "w/o 273.0 5.2 124.7 36.6 122.6 15.5", "V1a",
                 "Table 2, the main comparison, gives 15.3 for the Baseline; the 'w/o' row of the ablation Table 8 (no target labels) matches the Baseline in its other five values", unit="%")
