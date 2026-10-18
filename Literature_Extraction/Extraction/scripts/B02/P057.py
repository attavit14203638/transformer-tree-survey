import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T6, F7 = "P057", "Table 6", "Fig. 7 (image)"
ROWS = [
    R("task", "crown_segmentation", 1, "Title", "Towards complete tree crown delineation by instance segmentation with Mask R–CNN and DETR"),
    R("output_unit", "crown_mask", 6, "Sec. 3.1", "The basic idea of our approach is the delineation of single trees by instance segmentation"),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "The mixed forest area, administered by the Bayerische Staatsforsten, is characterised by spruces and large groups of beeches",
      note="managed mixed forest"),
    R("tree_taxa", "spruce; beech", 4, "Sec. 2.1", "is characterised by spruces and large groups of beeches", note="test plots: coniferous, mixed (60% conifers) and deciduous (76% broadleaves)"),
    R("study_area", "near the Kranzberg Forest Roof Experiment (KROOF), 35 km northeast of Munich, Bavaria, Germany", 4, "Sec. 2.1",
      "Our study area was located 35 km northeast of Munich, near the Kranzberg Forest Roof Experiment (KROOF) research site"),
    D("biome", "temperate", "central European mixed forest (Bavaria); not stated in the paper"),
    R("platform", "uav", 5, "Table 2", "Platform DJI M 600 Pro"),
    R("sensors", "multispectral;lidar", 6, "Table 3", "RGB Red (612) ... CIR NIR (842) Red (612) Green (560) ... LIDAR CHMlidar P_DENSE M_INTEN",
      note="multispectral camera and RIEGL miniVUX-1UAV lidar; inputs RGB, CIR, lidar rasters or panchromatic plus lidar"),
    R("gsd", "5", 4, "Sec. 2.2", "TDOP with a cell size of 5 cm was exported", unit="cm"),
    R("point_density", "560", 5, "Table 2", "Point density (pts/m2) ca. 560", unit="pts/m2"),
    R("annotation_type", "polygon_mask", 5, "Sec. 2.4.2", "The reference data are provided in the form of individual tree segments as enclosing polygons"),
    R("dataset_name", "own UAV multispectral and lidar dataset, KROOF site (training and validation areas, test plots #1 to #3)", 5, "Table 4",
      "Tree parameters of areas for training and validation", item="dataset1"),
    R("dataset_role", "train_test", 8, "Sec. 4.1", "three areas were selected for testing. The remaining data set was subdivided into 80% training and 20% validation", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_trees", "1533", "1185 training + 223 validation trees (Table 4, p.6) + 55 + 36 + 34 test-plot trees (Table 5, p.6)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "Mask R-CNN", "cnn_baseline", "cnn", "ResNet-50", "imagenet", "full_fine_tuning", 6, "Sec. 3.1",
              "Mask R–CNN and DETR with ResNet-50 (He et al., 2015) as a backbone pre-trained on the ImageNet data set", strat_page=8, strat_loc="Sec. 3.4",
              strat_ev="methods achieved best results using an unfrozen ResNet-50 backbone", cat_note="comparative study of Mask R-CNN and DETR")
ROWS += model("model2", "DETR (panoptic segmentation head)", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-50", "other", "full_fine_tuning", 6, "Sec. 3.1",
              "Mask R–CNN and DETR with ResNet-50 (He et al., 2015) as a backbone pre-trained on the ImageNet data set", pre_page=8, pre_loc="Sec. 3.3",
              pre_ev="two pre-trained models were loaded, one for object detection and one for panoptic segmentation",
              pre_note="pre-trained DETR detection and panoptic models (object queries optimised for COCO, p.11)", strat_page=8, strat_loc="Sec. 3.4",
              strat_ev="methods achieved best results using an unfrozen ResNet-50 backbone")
# F1 per plot and channel set, read from the bars of Figure 7 (measured heights; rounded to whole percent)
F1 = {  # (plot, channel): (Mask R-CNN, DETR, measured MRCNN, measured DETR)
    ("coniferous plot #1", "RGB"): ("92", "90", 92.16, 90.15), ("coniferous plot #1", "CIR"): ("92", "90", 92.16, 90.15),
    ("coniferous plot #1", "LIDAR"): ("92", "88", 92.16, 87.99), ("coniferous plot #1", "P + LIDAR"): ("90", "87", 90.15, 87.22),
    ("mixed plot #2", "RGB"): ("84", "89", 84.29, 89.23), ("mixed plot #2", "CIR"): ("83", "84", 83.20, 84.29),
    ("mixed plot #2", "LIDAR"): ("81", "82", 80.89, 82.28), ("mixed plot #2", "P + LIDAR"): ("80", "82", 79.96, 82.28),
    ("deciduous plot #3", "RGB"): ("83", "77", 83.13, 77.10), ("deciduous plot #3", "CIR"): ("85", "81", 85.14, 80.81),
    ("deciduous plot #3", "LIDAR"): ("81", "80", 80.81, 79.88), ("deciduous plot #3", "P + LIDAR"): ("75", "80", 74.94, 79.88),
}
TEXT = {  # Mask R-CNN F1 values stated in the text (p.9)
    ("coniferous plot #1", "CIR"): "The best channel combination, CIR, achieved an F1 score of 92% in the coniferous plot, 83% in the mixed plot and 85% in the deciduous plot",
    ("mixed plot #2", "CIR"): "The best channel combination, CIR, achieved an F1 score of 92% in the coniferous plot, 83% in the mixed plot and 85% in the deciduous plot",
    ("deciduous plot #3", "CIR"): "The best channel combination, CIR, achieved an F1 score of 92% in the coniferous plot, 83% in the mixed plot and 85% in the deciduous plot",
    ("coniferous plot #1", "RGB"): "In the coniferous plot, using the channel combination RGB, we obtained an F1 score of 92%",
}
BARNOTE = "read from the bar heights in Figure 7 (no data labels; measured {:.2f}, gridlines every 10%); about ±0.5"
res = Counter("res")
for (plot, ch), (mr, dt, mr_m, dt_m) in F1.items():
    st = f"{plot}; input {ch}; IoU ≥ 0.5"
    if (plot, ch) in TEXT:
        ROWS += result(res(), "f1", "model1", "dataset1", mr, 9, "Sec. 4.3.1", TEXT[(plot, ch)], setting=st, headline=(plot.startswith("coniferous") and ch == "CIR"))
    else:
        note = BARNOTE.format(mr_m)
        if (plot, ch) in (("mixed plot #2", "RGB"), ("deciduous plot #3", "RGB")):
            note += "; the text gives '83% and 84%, respectively' for the mixed and deciduous plots, while the bars show 84 and 83"
        ROWS += result(res(), "f1", "model1", "dataset1", mr, 10, F7, f"Mask R–CNN panel, {plot}, {ch}: F1-score bar", setting=st, note=note, status="approximate")
    ROWS += result(res(), "f1", "model2", "dataset1", dt, 10, F7, f"DETR panel, {plot}, {ch}: F1-score bar", setting=st, note=BARNOTE.format(dt_m), status="approximate")
for item, lab in (("model1", r"Mask R–CNN \(CIR\)"), ("model2", r"DETR \(CIR\)")):
    nums, ev = row(PID, 11, rf"^\s*{lab}", 0, cols=(0, 90))
    for plot, v in zip(("coniferous plot #1", "mixed plot #2", "deciduous plot #3"), (nums[0], nums[2], nums[4])):
        ROWS += result(res(), "other", item, "dataset1", v, 11, T6, ev, setting=f"{plot}; input CIR",
                       note="segmentation quality: mean IoU of true-positive crowns with their reference (Eq. 7); true positives in brackets in Table 6")
ROWS += avail()
ROWS += [D("notes", "Lidar-based unsupervised baselines (NCut, Silva, Li, watershed) were also evaluated (Fig. 7c, f, i) and are not recorded. All F1 values except the Mask R-CNN values stated in the text come from bar charts.", "Sec. 4.3 and Fig. 7 (pp. 9 to 10)")]
