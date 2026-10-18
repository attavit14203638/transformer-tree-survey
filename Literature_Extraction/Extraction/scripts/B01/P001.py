import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1 = "Table 1"
ROWS = [
    R("task", "detection;crown_segmentation", 1, "Abstract", "mapping Ghaf trees ... mean average precision values ... for detection and ... for segmentation"),
    R("output_unit", "tree_instance;crown_mask", 3, "Sec. 3.1", "Mask R-CNN introduces a pixel-level segmentation mask for each detected object"),
    R("tree_setting", "mixed", 2, "Sec. 2.1", "spans diverse urban and agricultural regions within the Fujairah and Sharjah emirates", note="urban and agricultural"),
    R("tree_taxa", "Ghaf (Prosopis cineraria)", 1, "Abstract", "The Prosopis cineraria, commonly known as the Ghaf tree"),
    R("study_area", "Fujairah and Sharjah emirates (Kalba, Fujairah city), United Arab Emirates; about 25 km2", 2, "Sec. 2.1", "The region covers an area of approximately 25 km²"),
    R("biome", "arid", 1, "Abstract", "supports biodiversity within arid ecosystems"),
    R("platform", "uav", 2, "Sec. 2.1", "imagery was collected using a Sensefly eBee X fixed-wing drone"),
    R("sensors", "rgb", 2, "Sec. 2.1", "equipped with a professional-grade Sensefly S.O.D.A RGB camera"),
    R("gsd", "2.5", 2, "Sec. 2.1", "ground sampling distance of the generated orthomosaic is 2.5 cm", unit="cm"),
    R("annotation_type", "polygon_mask", 2, "Sec. 2.2", "detailed manual delineation of each Ghaf tree within the imagery"),
    # dataset
    R("dataset_name", "own UAV Ghaf tree dataset (Fujairah and Sharjah)", 2, "Sec. 2.2", "the study area was segmented into three distinct regions, each providing UAV images and their corresponding labels", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 2.2", "2770 images ... for training ... 680 ... for validation ... 885 images ... testing", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_images", "4335", "2770 train + 680 validation + 885 test images (p.3, Sec. 2.2)", item="dataset1", unit="images"),
    D("n_trees", "6183", "3890 train + 1020 validation + 1273 test instances (p.3, Sec. 2.2)", item="dataset1", unit="trees"),
]
M = [("model1", "Mask R-CNN with Swin Transformer backbone", "hierarchical_vit", "Swin Transformer (size not stated)"),
     ("model2", "Mask DINO with Swin-tiny backbone", "hierarchical_vit", "Swin-T"),
     ("model3", "Mask R-CNN with MViTv2-tiny backbone", "hierarchical_vit", "MViTv2-T"),
     ("model4", "Mask R-CNN with ViT-base backbone", "pure_vit", "ViT-B")]
for item, name, cat, bb in M:
    ROWS += model(item, name, "transformer_baseline", cat, bb, None, "full_fine_tuning", 4, T1, name.replace(" with ", "-").replace(" backbone", ""),
                  pre_status="not_reported", pre_note="initialized with pre-trained weights; source not stated (p.3, Sec. 3.2)",
                  strat_page=3, strat_loc="Sec. 3.2", strat_ev="The models were initialized with pre-trained weights and trained over 100,000 iterations",
                  strat_note="freezing not mentioned")
res = Counter("res")
TEST = {  # model: (det mAP50, det F1, seg mAP50, seg F1), testing rows of Table 1
    "model1": ("84.2", "88.3", "85.1", "88.6", "Mask R-CNN-Swin transformer"),
    "model2": ("84.1", "87", "84.8", "87.6", "Mask DINO-Swin-tiny"),
    "model3": ("80.0", "83.55", "82.2", "85.5", "Mask R-CNN-MViTv2-tiny"),
    "model4": ("83.8", "86.4", "84.0", "86.49", "Mask R-CNN-ViT-base"),
}
for m, (dap, df1, sap, sf1, lab) in TEST.items():
    hl = "yes" if m == "model1" else "no"
    ROWS += result(res(), "ap50", m, "dataset1", dap, 4, T1, f"Testing, {lab}: detection mAP50 {dap}", headline=(hl == "yes"), setting="test set; detection (box)")
    ROWS += result(res(), "f1", m, "dataset1", df1, 4, T1, f"Testing, {lab}: detection F1-score {df1}", setting="test set; detection (box)")
    ROWS += result(res(), "ap50", m, "dataset1", sap, 4, T1, f"Testing, {lab}: segmentation mAP50 {sap}", setting="test set; segmentation (mask)")
    ROWS += result(res(), "f1", m, "dataset1", sf1, 4, T1, f"Testing, {lab}: segmentation F1-score {sf1}", setting="test set; segmentation (mask)")
ROWS += avail()
ROWS += [D("notes", "Comparative study of four transformer-based instance segmentation models; no CNN baseline and no proposed model, so model roles are recorded as baselines. Validation results are also in Table 1.", "coded from Sec. 3.1 and Table 1")]
