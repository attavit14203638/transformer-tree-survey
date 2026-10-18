import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T3, T4 = "Table 3", "Table 4"
ROWS = [
    R("task", "detection", 1, "Abstract", "a pre-trained convolutional neural network to detect individual trees using high-resolution multispectral imagery"),
    R("output_unit", "tree_instance", 3, "Sec. 3.2.3", "The training and test reference data are provided in the form of enclosing bounding boxes"),
    R("tree_setting", "natural_forest", 2, "Sec. 3.1", "Most of the mixed forest is characterised by large groups of beeches surrounded by spruces", note="managed mixed forest (Bayerische Staatsforsten)"),
    R("tree_taxa", "beech; spruce", 2, "Sec. 3.1", "large groups of beeches surrounded by spruces", note="test plots: coniferous (#1), mixed (#2), deciduous (#3)"),
    R("study_area", "Kranzberg Forest Roof Experiment (KROOF), about 35 km northeast of Munich, Bavaria, Germany", 2, "Sec. 3.1", "located at 11°39'42\" E, 48°25'12\" N, approximately 35 km northeast of Munich"),
    D("biome", "temperate", "central European mixed forest (Bavaria); not stated in the paper"),
    R("platform", "uav", 2, "Sec. 3.2.1", "MX Dual camera (MicaSense, 2022) attached to a remotely piloted hexacopter (DJI M 600 Pro)"),
    R("sensors", "rgb", 7, "Conclusions", "Future experiments will focus on (i) usage of multispectral channels (e.g. NIR, NDVI, NDRE)",
      note="RGB true orthophotos from a ten-channel MicaSense RedEdge-MX Dual camera; multispectral channels left for future work"),
    R("gsd", "5", 2, "Sec. 3.2.1", "The exported TDOPs had a cell size of 5 cm", unit="cm", note="flight GSD 5.9 cm (2020) and 5.3 cm (2021)"),
    R("annotation_type", "bounding_box", 3, "Sec. 3.2.3", "The training and test reference data are provided in the form of enclosing bounding boxes"),
    R("dataset_name", "own UAV dataset, KROOF forest (training and validation areas, field-surveyed test plots #1 to #3)", 3, "Fig. 1",
      "Research site KROOF as RGB TDOP showing three test plots #1, #2, and #3. The remaining area was used for training", item="dataset1"),
    R("dataset_role", "train_test", 3, "Fig. 1", "showing three test plots #1, #2, and #3. The remaining area was used for training and validation", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_images", "505", "433 training + 72 validation tiles (Table 3, p.2); test plot tiles not counted", item="dataset1", unit="images"),
    D("n_trees", "1515", "1167 training + 223 validation trees labelled visually (Table 3) + 125 field-surveyed test trees (p.1, Abstract)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "DETR", "proposed", "cnn_transformer_hybrid", "ResNet-50", "other", "full_fine_tuning", 3, "Sec. 4.2",
              "First, the features are extracted using ResNet50 (He et al., 2015). Afterwards, the positional encodings are", pre_page=5, pre_loc="Sec. 5.2",
              pre_ev="A pre-trained model based on the common objects in context (COCO) detection dataset (Lin et al., 2014) was used", pre_note="COCO",
              strat_page=5, strat_loc="Sec. 5.2", strat_ev="adjusted learning rate of 1e-7, a backbone learning rate of 1e-6", strat_note="backbone trained with a lower learning rate")
ROWS += model("model2", "YOLOv4", "cnn_baseline", "cnn", "CSPDarknet-53 (YOLOv4 default)", "other", "full_fine_tuning", 5, "Sec. 5.3",
              "A pre-trained model based on the COCO detection data set (Lin et al., 2014) was selected for transfer learning", pre_note="COCO",
              cat_note="backbone not named in the paper; YOLOv4 default")
res = Counter("res")
for m, lab, vals in (("model1", "DETR", ("83", "86", "71")), ("model2", "YOLOv4", ("87", "65", "67"))):
    for plot, v, desc in zip(("#1", "#2", "#3"), vals, ("coniferous", "mixed", "deciduous")):
        ROWS += result(res(), "f1", m, "dataset1", v, 6, T4, f"{lab} {desc} plot {plot} F1-score {v}", setting=f"test plot {plot} ({desc}); IoU 0.5",
                       headline=(m == "model1" and plot == "#2"))
ROWS += avail()
ROWS += [R("limitations", "DETR detects small trees (crown diameter below 5 m) worse than YOLOv4", 5, "Sec. 6", "In conclusion, DETR has apparently problems to detect smaller trees"),
         D("notes", "Table 4 also reports accuracy, precision and recall. YOLOv4 results are means over repeated runs with different seeds; DETR varied little.", "Sec. 5.1 and Table 4 (pp. 5 to 6)")]
