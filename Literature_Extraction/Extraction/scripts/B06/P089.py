import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T4 = "P089", "Table 4"
# Tree point cloud enhancement: a completion network with an Upsample Transformer (Tables 1 to 3, Chamfer distance and F-score of completed point clouds) and an adversarial sample augmentation network whose
# classifier has multi-head self-attention pooling (Sec. 3 and 4.4). Extracted at the author's decision of 2026-10-14 (kept after the full-text re-check). The tree-level task is the species classification
# of individual tree point clouds (Table 4: classification accuracy of the classifier trained on the original samples, with conventional augmentation and with the proposed augmentation; the Avg row is
# recorded as overall accuracy and is primary). The completion scores (CD and F-score of point sets) are not a task of the review and are not recorded (only
# tree-level scores are recorded). The robustness test (Table 6) and the ablations (Tables 3, 5 and 7) are not recorded.
TC = Tags(task="species_classification", level="tree", split="test", agg="single")
DS_EV = "In total, 1415 individual tree point clouds were collected, including 284 Larix gmelini,"
SPLIT_EV = "The training and test sets were then divided according to the configuration used in PointNet++"
AVG_EV = "Avg 87.2% 88.1% 90.1%"
ROWS = [
    R("task", "species_classification", 14, "Sec. 4.3", "achieved the highest classification accuracy (avgAP = 90.1%)", note="species classification of individual tree point clouds (six species), the tree-level task of the paper"),
    R("output_unit", "tree_label", 14, T4, AVG_EV, note="one species label per individual tree point cloud"),
    R("tree_setting", "urban", 3, "Sec. 2.1", "It is situated in the central urban area of Harbin, Heilongjiang Province, covering an area of 44 ha", note="urban forestry demonstration base planted with several tree species"),
    R("tree_taxa", "Larix gmelinii, Pinus sylvestris var. mongolica, Pinus tabuliformis, Fraxinus mandshurica, Quercus mongolica, Betula platyphylla", 4, "Sec. 2.2", DS_EV),
    R("study_area", "Urban forestry demonstration base of Northeast Forestry University, Harbin, Heilongjiang Province, China (44 ha)", 3, "Sec. 2.1",
      "The study area is located in the urban forestry demonstration base of the Northeast Forestry University"),
    R("biome", "temperate", 3, "Sec. 2.1", "temperate mixed conifer–broadleaf forests", note="cold-temperate coniferous and temperate mixed forests"),
    R("platform", "uav", 3, "Sec. 2.1", "The UAV platform used a Jingwei M300 RTK equipped with a DJI Zenmuse L1, which integrated a Livox LiDAR module", note="flown at 100 m on 4 May 2021"),
    R("sensors", "lidar", 3, "Sec. 2.1", "The UAV platform used a Jingwei M300 RTK equipped with a DJI Zenmuse L1, which integrated a Livox LiDAR module"),
    R("point_density", "200", 4, "Sec. 2.1", "with an average point density of 200 points per square meter", unit="pts/m2"),
    R("annotation_type", "image_label", 4, "Sec. 2.2", "the watershed algorithm based on markers, proposed by Chen et al. [30], was applied to segment individual trees",
      note="individual trees segmented by watershed and edited by hand, matched with field RTK positions; one species label per tree point cloud"),
    R("dataset_name", "own UAV LiDAR individual tree point clouds of six species, Harbin urban forestry demonstration base", 4, "Sec. 2.2", DS_EV, item="dataset1"),
    R("dataset_key", "own_p089_harbin_uls", 4, "Sec. 2.2", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 11, "Sec. 4.1", SPLIT_EV, item="dataset1"),
    D("dataset_public", "no", "The Data Availability Statement (p.17) says the data are contained within the article; no point cloud release", item="dataset1"),
    R("n_trees", "1415", 4, "Sec. 2.2", DS_EV, item="dataset1", unit="trees", note="284 Larix, 237 Pinus sylvestris, 185 Pinus tabuliformis, 209 Fraxinus, 307 Quercus, 193 Betula"),
]
CL_EV = "head self-attention pooling layers in the feature extraction modules of both the point"
NP = dict(pre_note="not stated", strat_page=11, strat_loc="Sec. 4.1", strat_ev="the point cloud augmentation network was trained for 250 epochs with a batch size of 8, also using the Adam optimizer with an initial learning rate of 0.001",
          strat_note="trained from the tree point clouds (no pre-training stated); Adam, learning rate 0.001, 250 epochs")
CAT = "point-based classification network with multi-head self-attention pooling in its feature extraction (Sec. 4.4); protocol 4.5: multi-head self-attention blocks, recorded as cnn_transformer_hybrid"
ROWS += model("model1", "Species classifier trained on the original samples", "other_baseline", "cnn_transformer_hybrid", "classification network of the augmentation system with multi-head self-attention pooling",
              None, "from_scratch", 15, "Sec. 4.4", CL_EV, cat_note=CAT, pre_status="not_reported", **NP)
ROWS += model("model2", "Species classifier trained with conventional data augmentation", "other_baseline", "cnn_transformer_hybrid", "classification network of the augmentation system with multi-head self-attention pooling",
              None, "from_scratch", 15, "Sec. 4.4", CL_EV, cat_note=CAT, pre_status="not_reported", **NP)
ROWS += model("model3", "Species classifier trained with the proposed completion and adversarial augmentation (ours)", "proposed", "cnn_transformer_hybrid",
              "classification network with multi-head self-attention pooling, trained on the original and augmented samples", None, "from_scratch", 15, "Sec. 4.4", CL_EV, cat_note=CAT, pre_status="not_reported", **NP)
res = Counter("res")
for item, v, lab, head in (("model1", "87.2", "Original", False), ("model2", "88.1", "Conventional DA", False), ("model3", "90.1", "Ours", True)):
    ROWS += result(res(), "oa", item, "dataset1", v, 14, T4, AVG_EV, setting=f"Table 4: Avg row, {lab}; test set of the PointNet++ configuration",
                   note="classification accuracy averaged over the six species (Avg row of Table 4); the per-species accuracies are not recorded", **TC, primary=True, headline=head)
ROWS += avail(code="no", weights="not_stated", data="no", note_d="the statement says that the data are contained within the article; no point cloud release")
ROWS += [D("notes", "Tree point cloud enhancement: a completion network with an Upsample Transformer (average CD x 1000 of 4.84 and F-score 0.90 at 50 percent missing points, against PCN, GRNet, PoinTr and SnowflakeNet in Table 1) and an adversarial sample augmentation network. Only the tree-level species classification is recorded: classification accuracy (Avg row of Table 4) of the classifier trained on the original samples 87.2, with conventional augmentation 88.1 and with the proposed augmentation 90.1 (primary, headline; the text calls it avgAP). The classifier and the completion network use multi-head self-attention pooling (Sec. 4.4), so the scored classifier is recorded as cnn_transformer_hybrid. The completion scores (Tables 1 to 3), the robustness test (Table 6) and the ablations (Tables 5 and 7) are not recorded.",
           "Sec. 2 to 4 and Table 4 (pp. 3 to 15)")]
