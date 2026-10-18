import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T3 = "Table 3"
ROWS = [
    R("task", "detection", 1, "Title", "Individual Tree Detection in Coal Mine Afforestation Area Based on Improved Faster RCNN in UAV RGB Images",
      note="detections are also used to compute stand density per area (Table 4)"),
    R("output_unit", "tree_instance", 4, "Sec. 2.3", "the algorithm determined the final object bounding box through proposal and feature map"),
    R("tree_setting", "plantation", 1, "Abstract", "In the Ulan Mulun River Basin (Ordos, China), afforestation is standard method for reclamation", note="afforestation on coal-mine reclamation land"),
    NR("tree_taxa", note="afforestation trees; species not given"),
    R("study_area", "Bulianta Coal Mine, Ulan Mulun River Basin, Ordos, Inner Mongolia, China (four flight areas A to D)", 4, "Sec. 2.2", "Area A located in the western region of the Bulianta Coal Mine"),
    R("biome", "arid", 3, "Sec. 2.1", "arid continental climate"),
    R("platform", "uav", 4, "Sec. 2.2", "DJI Phantom 4 Multispectral (P4M) UAV acquired images in August 2021"),
    R("sensors", "rgb", 4, "Sec. 2.2", "This study mainly used a color sensor lens of RGB"),
    R("gsd", "0.01", 4, "Table 1", "Spatial resolution 0.01 m", unit="m", note="Table 1 also lists 'Average ground sampling distance 5m', apparently an error"),
    R("annotation_type", "bounding_box", 4, "Sec. 2.2", "The expert made labels by LabelImage in PascalVOC format"),
    R("dataset_name", "own UAV dataset, Bulianta Coal Mine areas A to D", 4, "Sec. 2.2", "Then we cropped for 576 images by Python", item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 2.2", "The green box represents the training dataset region, and the gray box represents the validation dataset region", item="dataset1",
      note="Table 3 test set described as 'all labeled UAV images' (p.10)"),
    R("dataset_public", "on_request", 16, "Data Availability Statement", "The data presented in this study are available on reasonable from the corresponding author", item="dataset1"),
    R("n_images", "576", 4, "Sec. 2.2", "Then we cropped for 576 images", item="dataset1", unit="images"),
    NR("n_trees", item="dataset1", note="number of labelled trees not given"),
]
ROWS += model("model1", "Improved Faster R-CNN (multi-strategy augmentation, DE-FPN, Alpha-IoU)", "proposed", "cnn", "ResNet-50", "imagenet", "full_fine_tuning",
              10, T3, "Faster R-CNN ResNet50 89.89", pre_page=7, pre_loc="Sec. 2.7", pre_ev="The transfer learning dataset was the ImageNet dataset",
              strat_page=7, strat_loc="Sec. 2.7", strat_ev="the parameters of the first two layers frozen", strat_note="ImageNet weights with early layers frozen",
              cat_note="convolutional; the transformer models appear only as baselines")
ROWS += model("model2", "DETR", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-50", None, None, 10, T3, "DETR [89] ResNet50 86.06",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated")
ROWS += model("model3", "ViDT", "transformer_baseline", "pure_vit", "ViT (as listed in Table 3)", None, None, 10, T3, "ViDT [90] ViT 85.67",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated")
ROWS += model("model4", "EFLDet", "cnn_baseline", "cnn", "BRNet-ResNet50", None, None, 10, T3, "EFLDet [59] BRNet-ResNet50 86.08",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated", cat_note="best CNN on AP in Table 3")
res = Counter("res")
for m, lab, v in (("model1", "Faster R-CNN ResNet50", "89.89"), ("model2", "DETR [89] ResNet50", "86.06"), ("model3", "ViDT [90] ViT", "85.67"),
                  ("model4", "EFLDet [59] BRNet-ResNet50", "86.08")):
    ROWS += result(res(), "map", m, "dataset1", v, 10, T3, f"{lab} AP {v}", setting="labelled UAV images", headline=(m == "model1"),
                   note="'AP' as printed; Table 3 also gives AP at score thresholds 50, 60 and 70")
ROWS += avail(data="on_request", data_page=16, data_loc="Data Availability Statement",
              data_ev="The data presented in this study are available on reasonable from the corresponding author")
ROWS += [D("notes", "Table 3 compares 12 detectors; following protocol 4.1, recorded: the proposed model, both transformer models (DETR, ViDT) and the best CNN on AP (EFLDet). The other eight (Mask R-CNN, TridentNet, YOLOv3 to v5, SSD, FCOS, CenterNet511) are in Table 3 (p.10). The plain Faster R-CNN baseline is not in Table 3; module gains are in Table 2 (e.g. DE-FPN +2.82 AP, Alpha-IoU +2.60 AP). In AP50, DETR (94.74) exceeds all CNNs except the proposed model (98.93).", "Sec. 3.2 to 3.3 and Tables 2 to 3 (p.10)")]
