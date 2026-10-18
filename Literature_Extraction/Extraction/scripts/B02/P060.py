import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T6 = "P060", "Table 6"
URL = "https://github.com/soenke-sp/TreeSeg"
ROWS = [
    R("task", "crown_segmentation", 1, "Title", "TreeSeg—A Toolbox for Fully Automated Tree Crown Segmentation Based on High-Resolution Multispectral UAV Data"),
    R("output_unit", "crown_mask", 5, "Sec. 2.2", "We converted the shapefiles into binary masks with the extent of their corresponding multispectral images", note="instance crown masks (boxes for Faster R-CNN)"),
    R("tree_setting", "natural_forest", 4, "Sec. 2.1", "The forests are dominated by Picea abies (European spruce)"),
    R("tree_taxa", "Picea abies (dominant); Abies alba; Larix decidua; Fagus sylvatica; Acer pseudoplatanus; Acer platanoides; Fraxinus excelsior; Sorbus aucuparia", 5, "Table 1",
      "Picea abies (European spruce) 867 ... Total 1096"),
    R("study_area", "Black Forest east of Freiburg im Breisgau, Germany", 4, "Sec. 2.1", "The study area is located in the Black Forest region to the east of Freiburg im Breisgau"),
    D("biome", "temperate", "Black Forest, Germany; not stated in the paper"),
    R("platform", "uav", 4, "Sec. 2.1", "The multispectral UAV data were collected using a DJI Matrice 300 RTK drone system"),
    R("sensors", "rgb;multispectral", 4, "Sec. 2.1", "with the RedEdge-MX™ Dual Camera sensor ... consisted of five bands", note="model comparison on RGB; vegetation-index inputs tested with Faster R-CNN"),
    R("gsd", "1.3", 4, "Sec. 2.1", "with a resolution of approximately 1.3 to 2.3 cm per pixel", unit="cm", note="1.3 to 2.3 cm"),
    R("annotation_type", "polygon_mask", 4, "Sec. 2.1", "The segmentation labels were created manually at a very high level of detail based on the multispectral images in a GIS"),
    R("dataset_name", "own UAV multispectral dataset, Black Forest near Freiburg", 4, "Sec. 2.1", "In total, there are 1096 labeled trees in the study area", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 2.2", "all data were randomly split into training and test datasets (approximately 10:1)", item="dataset1",
      note="about 800 training trees (4.3 ha) and 80 test trees (0.4 ha)"),
    R("dataset_public", "no", 13, "Data Availability Statement", "Data are contained within the article", item="dataset1", note="no raw data link"),
    R("n_trees", "1096", 4, "Sec. 2.1", "In total, there are 1096 labeled trees in the study area", item="dataset1", unit="trees"),
]
FT = dict(strat_page=6, strat_loc="Sec. 2.3", strat_ev="Tables A1 and A2 in the Appendix A show the optimum hyperparameter settings for the respective models found during fine-tuning")
M = [("model1", "Faster R-CNN", "Faster R-CNN", "ResNet-101 with FPN", "We use ResNet-101 as a backbone during our experiments", 6),
     ("model2", "Mask R-CNN", "Mask R-CNN", "ResNet-50 with FPN", "with ResNet-50 serving as the backbone network in combination with FPN", 6),
     ("model3", "TensorMask", "TensorMask", "ResNet-50 with FPN", "Similar to Mask R-CNN, TensorMask utilizes ResNet-50 as the backbone in combination with FPN", 6)]
res = Counter("res")
for item, lab, name, bb, bb_ev, bb_p in M:
    nums, ev = row(PID, 9, rf"^\s*{lab}\s+(?=\d)")
    assert len(nums) == 4, (lab, nums)
    ROWS += model(item, name, "cnn_baseline", "cnn", bb, None, "full_fine_tuning", bb_p, "Sec. 2.3", bb_ev, pre_status="not_reported", pre_note="not stated",
                  cat_note="comparison on RGB images", **FT)
    ROWS += result(res(), "iou", item, "dataset1", nums[0], 9, T6, ev, setting="test set; RGB input")
    ROWS += result(res(), "f1", item, "dataset1", nums[3], 9, T6, ev, setting="test set; RGB input", headline=(item == "model1"))
nums, ev = row(PID, 9, r"^\s*SAM\s+(?=\d)")
ROWS += model("model4", "SAM (automatic mask generation)", "transformer_baseline", "foundation_model", "SAM image encoder (size not stated)", "sa1b", "zero_shot",
              7, "Sec. 2.3", "SAM (Segment Anything Model) was pre-trained on 1.1 million images and 1.1 billion masks", strat_page=14, strat_loc="Table A2",
              strat_ev="Optimal hyperparameter configuration for SAM. ... Points per Side ... Stability Score Threshold", strat_note="automatic mask generator settings tuned; no weight training")
ROWS += result(res(), "iou", "model4", "dataset1", nums[0], 9, T6, ev, setting="test set; RGB input")
ROWS += result(res(), "f1", "model4", "dataset1", nums[3], 9, T6, ev, setting="test set; RGB input")
ROWS += avail(code_url=URL, page=10, loc="Sec. 3.3", ev=f"The trained models, toolbox and user instructions are provided on GitHub ({URL})", weights="yes",
              data_page=13, data_loc="Data Availability Statement", data_ev="Data are contained within the article", note_d="no raw data link")
ROWS += [D("notes", "The vegetation-index and image-enhancement experiments (Table 7) use Faster R-CNN only and are not recorded. Faster R-CNN predicts boxes; the paper evaluates all models with IoU and F1 against the crown masks.", "Sec. 2.3, Tables 6 to 7 (pp. 6 to 9)")]
