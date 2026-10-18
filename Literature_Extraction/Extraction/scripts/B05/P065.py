import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P065", "Table 2"
# Segmentation of the branch structure of apple trees in close-range hyperspectral images (component segmentation). Table 2 values are the average over a 10-fold cross-validation (Sec. 2.6.1).
# mAP@50 is an instance-mask score; the paper does not make the unit of precision, recall, accuracy and F1 clear, so F1 has no stated level.
TM = Tags(task="component_segmentation", level="instance_mask", split="cross_validation", agg="mean")
TF = Tags(task="component_segmentation", level="not_stated", split="cross_validation", agg="mean")
MODEL_EV = "The core design idea is to leverage the powerful modeling capability of the Transformer model, combining hyperspectral image data and the Path Attention Mechanism to precisely segment tree branches"
DS_EV = "data were collected from apple orchards in Qixia City, Shandong Province, and Linhe District, Bayannur City, Inner Mongolia Autonomous Region, from March 2023 to March 2024"
CV_EV = "the average performance metrics were reported as the final results"

ROWS = [
    R("task", "component_segmentation", 7, "Sec. 2.5", MODEL_EV, note="leaves, branches and background of apple trees; the pruning evaluation of Table 3 is a classification of pruning effect and is not tree extraction"),
    R("output_unit", "tree_component", 5, "Sec. 2.2", "leaves, branches, background, and other parts were gradually annotated"),
    R("tree_setting", "orchard", 4, "Sec. 2.1", DS_EV),
    R("tree_taxa", "apple", 1, "Title", "Apple Tree Pruning and Evaluation with Multi-Modal Data", note="apple orchards (Sec. 2.1)"),
    R("study_area", "Qixia City (Shandong) and Linhe District, Bayannur (Inner Mongolia), China", 4, "Sec. 2.1", DS_EV),
    D("biome", "temperate", "Shandong and Inner Mongolia, China; coded from the study area"),
    R("platform", "ground_camera", 5, "Sec. 2.1", "sensor mounted on a stand to capture the three-dimensional information of the tree", note="hyperspectral camera on a stand in the orchard"),
    R("sensors", "hyperspectral", 4, "Sec. 2.1", "which features an excellent spectral range from 400 nm to 1000 nm", note="three visible bands (blue, green, red) are used as model input"),
    R("gsd", "1.5", 4, "Sec. 2.1", "with a spatial resolution of 1.5 mm", unit="mm"),
    R("annotation_type", "polygon_mask", 5, "Sec. 2.2", "leaves, branches, background, and other parts were gradually annotated", note="annotated in ENVI with SVM and CNN pre-annotation"),
    R("dataset_name", "own close-range hyperspectral apple tree images, Shandong and Inner Mongolia", 4, "Sec. 2.1", DS_EV, item="dataset1"),
    R("dataset_key", "own_p065_apple_hyperspectral", 4, "Sec. 2.1", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 13, "Sec. 2.6.1", "The dataset was split into training and validation sets in a ratio of 7:3", item="dataset1", note="plus 10-fold cross-validation"),
    R("dataset_public", "on_request", 22, "Data Availability Statement", "The data presented in this study are available on request from the corresponding author", item="dataset1"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="AdamW, learning rate 1e-4 with cosine annealing, batch size 32 (Sec. 2.6.1); initial weights are not stated")
ROWS += model("model1", "Trim-SegTransformer (proposed)", "proposed", "pure_vit", "tokenizer, Path Attention and Transformer encoder on hyperspectral input", None, None, 7, "Sec. 2.5", MODEL_EV,
              cat_note="Transformer encoder with a path attention variant of self-attention (Sec. 2.5)", pre_status="not_reported", pre_note="not stated",
              strat_note="AdamW, learning rate 1e-4 with cosine annealing, batch size 32, dropout 0.1 (Sec. 2.6.1); initial weights are not stated")
ROWS += model("model2", "Mask R-CNN", "cnn_baseline", "cnn", "Mask R-CNN", None, None, 14, "Sec. 2.6.2", "Mask R-CNN combines Region Proposal Networks (RPNs) and Fully Convolutional Networks (FCNs)",
              cat_note="convolutional instance segmentation (Sec. 2.6.2)", **NPRE)
ROWS += model("model3", "SegNet", "cnn_baseline", "cnn", "SegNet", None, None, 14, "Sec. 2.6.2", "SegNet, CNN segmentation model, is suitable for semantic segmentation tasks", cat_note="described as a CNN (Sec. 2.6.2)", **NPRE)
ROWS += model("model4", "Tiny-Segformer", "other_baseline", None, "backbone not stated", None, None, 14, "Sec. 2.6.2", "Tiny-Segformer, a lightweight visual segmentation model, is designed for real-time disease detection",
              cat_note="named only; the paper describes it as a lightweight visual segmentation model without an architecture", **NPRE)
ROWS += model("model5", "CS-net", "transformer_baseline", "cnn_transformer_hybrid", "CNN with Simpleformer (Simple-Attention Block)", None, None, 14, "Sec. 2.6.2", "CS-Net [20], a hybrid architecture combining CNNs and Simpleformer",
              cat_note="hybrid of CNN and Simpleformer (Sec. 2.6.2)", **NPRE)
ROWS += model("model6", "Box2Mask", "other_baseline", None, "backbone not stated", None, None, 14, "Sec. 2.6.2", "Box2Mask [19], an instance segmentation model based on bounding box annotations",
              cat_note="named as a box-supervised instance segmentation model; no architecture is described", **NPRE)
res = Counter("res")
RX = {"model2": r"Mask R-CNN \[21\]", "model3": r"SegNet \[22\]", "model4": r"Tiny-Segformer \[23\]", "model5": r"CS-net \[20\]", "model6": r"Box2Mask \[19\]", "model1": r"Proposed Method"}
for item, rx in RX.items():
    n, ev = row(PID, 16, rf"{rx}\s+(?=\d)", 0)
    assert len(n) == 7, (item, n)  # precision, recall, accuracy, F1, mAP@50, mAP@75, FPS
    st = "Table 2: tree segmentation results, 10-fold cross-validation average"
    ROWS += result(res(), "f1", item, "dataset1", n[3], 16, T2, ev, unit="fraction", setting=st, note="F1 score of the tree segmentation task", **TF, primary=True, headline=(item == "model1"))
    ROWS += result(res(), "ap50", item, "dataset1", n[4], 16, T2, ev, unit="fraction", setting=st, note="mAP@50 of the tree segmentation task", **TM, primary=True)
    ROWS += [R("inference_speed", f"{n[6]} FPS", 16, T2, ev, item=item)]
ROWS += [R("limitations", "The data cover two regions and one sensor; the paper names extension to other fruit species and multi-season pruning as future work", 21, "Sec. 4", "extended to other fruit species such as pear and peach trees"),
         D("code_available", "no", "No code repository or availability statement in the paper"),
         D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "on_request", 22, "Data Availability Statement", "The data presented in this study are available on request from the corresponding author")]
ROWS += [D("notes", "Table 2 compares Trim-SegTransformer with five segmentation models (precision, recall, accuracy, F1, mAP@50, mAP@75, FPS); F1 and mAP@50 and the FPS are recorded. Table 3 (SVM, MLP, random forest and the proposed network for the pruning evaluation from sensor data), Table 4 (sensor ablation) and the online learning curves are a pruning evaluation task and are not recorded. The conclusion calls the 0.94 'accuracy', which is the precision of Table 2.",
           "Sec. 3 and Table 2 (pp. 15 to 16)")]
