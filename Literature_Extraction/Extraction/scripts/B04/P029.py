import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T1, T2, T3, T4 = "P029", "Table 1", "Table 2", "Table 3", "Table 4"
URL = "https://github.com/nightonion/yosemite-tree-dataset"
# Counting errors per block (Yosemite) or per test image (NEON): counting at image level, test split, one run.
T = Tags(task="counting", level="image", split="test", agg="single")
SPLIT_EV = "Region B and D are used as a training set and Region A and C as a test set"

ROWS = [
    R("task", "counting", 1, "Abstract", "we propose a new efficient deep learning model called density transformer or DENT for automatic tree counting from aerial images"),
    R("output_unit", "tree_count", 1, "Abstract", "a fast tree counter to estimate the number of trees in each input image"),
    R("tree_setting", "natural_forest", 9, "Sec. 4", "in the Yosemite National Park and build a benchmark dataset for tree counting", note="NeonTreeEvaluation covers 22 sites with varying forest types"),
    NR("tree_taxa", note="all trees; species not given"),
    R("study_area", "Yosemite National Park, California, USA; 22 NEON sites across the United States", 9, "Sec. 4", "in the Yosemite National Park ... collected from 22 sites across the United States"),
    D("biome", "multiple", "Yosemite (temperate) and 22 NEON sites across the United States; not stated in the paper"),
    R("platform", "airborne", 9, "Sec. 4.1", "build a benchmark dataset for tree counting based on RGB aerial images", note="Yosemite images are from Google Maps; NEON images by several sensor types"),
    R("sensors", "rgb", 9, "Sec. 4.1", "build a benchmark dataset for tree counting based on RGB aerial images"),
    R("gsd", "11.8", 9, "Sec. 4.1", "The images are collected via Google Maps at 11.8 cm ground sampling distance (GSD) and stitched together", unit="cm",
      note="Yosemite; NEON images are 400 × 400 pixels for 40 m × 40 m"),
    R("annotation_type", "point", 9, "Sec. 4.1", "the position of each individual tree is manually labeled"),
    R("dataset_name", "Yosemite Tree Dataset (Google Maps aerial images, 19,200 × 38,400 pixels)", 9, "Sec. 4.1", "The total number of labeled trees is 98,949", item="dataset1"),
    R("dataset_key", "yosemite_trees", 9, "Sec. 4.1", "The total number of labeled trees is 98,949", item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. 4.1", SPLIT_EV, item="dataset1"),
    R("dataset_public", "yes", 9, "Sec. 4.1", "The dataset is publicly available for download at https://github.com/nightonion/yosemite-tree-dataset", item="dataset1"),
    R("dataset_url", URL, 9, "Sec. 4.1", "The dataset is publicly available for download at https://github.com/nightonion/yosemite-tree-dataset", item="dataset1"),
    R("n_trees", "98949", 9, "Sec. 4.1", "The total number of labeled trees is 98,949", item="dataset1", unit="trees"),
    R("dataset_name", "NeonTreeEvaluation Dataset (fully labelled RGB data, 22 sites)", 9, "Sec. 4.2",
      "We also evaluate the models using NeonTreeEvaluation Dataset [16], which is collected from 22 sites across the United States", item="dataset2"),
    R("dataset_key", "neon_treeevaluation", 9, "Sec. 4.2", "We also evaluate the models using NeonTreeEvaluation Dataset [16], which is collected from 22 sites across the United States", item="dataset2"),
    R("dataset_role", "train_test", 10, "Sec. 4.2", "A training set including 15 much larger images, containing 17,790 annotated trees", item="dataset2", note="trained on its 3395 cropped training images, tested on 194 test images"),
    R("dataset_public", "yes", 13, "Sec. 6", "We also used an existing cross-site dataset to test the robustness of the methods", item="dataset2", note="existing public benchmark [16]; no link printed"),
    D("n_images", "3589", "sum of the printed counts: 3395 cropped training images (p.10) plus 194 test images (p.9)", item="dataset2", unit="images"),
    D("n_trees", "24424", "sum of the printed counts: 17,790 trees in the training images plus 6634 trees in the test images (p.9 and p.10)", item="dataset2", unit="trees"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="not stated")
DENT = dict(pre_page=12, pre_loc="Sec. 5.4", pre_ev="The ResNet in the Multi-RF network is pre-trained on the ImageNet dataset", pre_note="the ResNet is ImageNet-pretrained, all other components are learned from scratch",
            strat_page=12, strat_loc="Sec. 5.4", strat_ev="All the other components of DENT are learned from scratch", strat_note="end-to-end training of the ImageNet-pretrained ResNet and the transformer learned from scratch (Adam, 300 epochs)")
BB = "Multi-RF CNN (ResNet18 with two extra paths) and a transformer encoder (d_model 512, 8 heads)"
r30 = row(PID, 12, r"^\s*Multi-RF\s+0\s+(?=\d)", 0)
rd, rc, rn, rt, rf = [row(PID, 11, rx, 0) for rx in (r"^\s*DENT-DMG\s+(?=\d)", r"^\s*DENT-CNT\s+(?=\d)", r"^\s*CANNet \[23\]\s+(?=\d)", r"^\s*CenterNet-DLA34 \[10\]\s+(?=\d)", r"^\s*Faster-RCNN-ResNet50 \[6\]\s+(?=\d)")]
ROWS += model("model1", "DENT-DMG (density transformer, density map generator head)", "proposed", "cnn_transformer_hybrid", BB, "imagenet", "full_fine_tuning", 11, T1, rd[1],
              cat_note="CNN features with a transformer encoder on top (Abstract, Sec. 3)", **DENT)
ROWS += [R("total_params", "17.82", 11, "Sec. 5.3", "the DENT model has 17.82 M parameters", item="model1", unit="M", note="whole DENT network"),
         R("inference_speed", "47.8 s for the whole 19,200 × 38,400 pixel Yosemite study area on one NVIDIA Tesla V100 SXM2 GPU (batch size 1, FP32)", 13, T4, "DENT ResNet18 47.8", item="model1")]
ROWS += model("model2", "DENT-CNT (density transformer, tree counter head)", "proposed", "cnn_transformer_hybrid", BB, "imagenet", "full_fine_tuning", 11, T1, rc[1],
              cat_note="same network as DENT-DMG with the tree counter head (Sec. 3.4)", **DENT)
ROWS += [R("total_params", "17.82", 11, "Sec. 5.3", "the DENT model has 17.82 M parameters", item="model2", unit="M", note="whole DENT network, both heads")]
ROWS += model("model3", "CANNet", "cnn_baseline", "cnn", "CANNet (contextual density regressor; backbone not stated)", None, None, 11, T1, rn[1],
              cat_note="one of the fully convolutional density regressors (Abstract); best CNN in every column of Table 1", **NPRE)
ROWS += model("model4", "CenterNet-DLA34", "cnn_baseline", "cnn", "DLA34", None, None, 11, T2, rt[1], cat_note="anchor-free CNN detector; best CNN for MAE in Table 2", **NPRE)
ROWS += [R("inference_speed", "61.5 s for the whole 19,200 × 38,400 pixel Yosemite study area on one NVIDIA Tesla V100 SXM2 GPU (batch size 1, FP32)", 13, T4, "CenterNet [10] DLA34 61.5", item="model4")]
ROWS += model("model5", "Faster-RCNN-ResNet50", "cnn_baseline", "cnn", "ResNet50", None, None, 11, T2, rf[1], cat_note="two-stage CNN detector; best CNN for RMSE in Table 2", **NPRE)
ROWS += [R("inference_speed", "290.9 s for the whole 19,200 × 38,400 pixel Yosemite study area on one NVIDIA Tesla V100 SXM2 GPU (batch size 1, FP32)", 13, T4, "Faster-RCNN [6] ResNet50 290.9", item="model5")]
ROWS += model("model6", "Multi-RF CNN without transformer layers (DENT ablation)", "cnn_baseline", "cnn", "Multi-RF CNN (ResNet18 with two extra paths); features linearly projected to the density map", None, None, 12, T3,
              r30[1], cat_note="the CNN counterpart of DENT: zero transformer layers (Table 3 caption)", **NPRE)

res = Counter("res")
COLS = [(960, "A", "mae"), (960, "A", "rmse"), (960, "C", "mae"), (960, "C", "rmse"), (4800, "A", "mae"), (4800, "A", "rmse"), (4800, "C", "mae"), (4800, "C", "rmse")]
for item, (nums, ev) in (("model1", rd), ("model2", rc), ("model3", rn), ("model4", rt), ("model5", rf)):
    assert len(nums) == 8, (item, nums)
    for (blk, reg, metric), v in zip(COLS, nums):
        ROWS += result(res(), metric, item, "dataset1", v, 11, T1, ev, unit="trees", setting=f"Table 1: Yosemite Region {reg}, blocks of {blk} × {blk} pixels", note=f"{metric.upper()} of the tree counts per block",
                       **T, primary=(blk == 960 and reg == "A"))
neon = {"model1": r"^\s*DENT-DMG\s+(?=\d)", "model2": r"^\s*DENT-CNT\s+(?=\d)", "model3": r"^\s*CANNet \[23\]\s+(?=\d)", "model4": r"^\s*CenterNet-DLA34 \[10\]\s+(?=\d)",
        "model5": r"^\s*Faster-RCNN-ResNet50 \[6\]\s+(?=\d)"}
for item, rx in neon.items():
    nums, ev = row(PID, 11, rx, 1)
    assert len(nums) == 2, (item, nums)
    for metric, v in zip(("mae", "rmse"), nums):
        ROWS += result(res(), metric, item, "dataset2", v, 11, T2, ev, unit="trees", setting="Table 2: NeonTreeEvaluation test images (400 × 400 pixels)", note=f"{metric.upper()} of the tree counts per test image", **T)
# Table 3: ablation on the union of Regions A and C (blocks of 960 × 960): the full model (Multi-RF, two layers, DMG) and its CNN-only counterpart
for item, rx, nl in (("model1", r"^\s*Multi-RF\s+2\s+(?=\d)", 2), ("model6", r"^\s*Multi-RF\s+0\s+(?=\d)", 0)):
    nums, ev = row(PID, 12, rx, 0)
    assert len(nums) == 2, (item, nums)
    for metric, v in zip(("mae", "rmse"), nums):
        ROWS += result(res(), metric, item, "dataset1", v, 12, T3, ev, unit="trees",
                       setting=f"Table 3: union of Regions A and C, blocks of 960 × 960 pixels, {nl} transformer layers", note=f"{metric.upper()}; ablation variant", **T)
ROWS += avail(data="yes", data_page=9, data_loc="Sec. 4.1", data_ev="The dataset is publicly available for download at https://github.com/nightonion/yosemite-tree-dataset")
ROWS += [D("notes", "Table 1 compares 18 methods on the Yosemite test regions A and C at two block sizes (960 × 960 and 4800 × 4800 pixels) and Table 2 on NeonTreeEvaluation; following protocol 4.1 the proposed DENT (both heads) and the best CNN per column are recorded (CANNet in every column of Table 1, CenterNet-DLA34 for MAE and Faster-RCNN-ResNet50 for RMSE in Table 2), with all their values. Primary results are Region A with 960 × 960 blocks, the first listed subset. Table 3 ablates the CNN and the number of transformer layers on the union of Regions A and C (recorded: the two-layer DENT and the CNN-only counterpart). DENT-faster variants and the other inference times are in Table 4.",
           "Sec. 5 and Tables 1 to 4 (pp. 11 to 13)")]
