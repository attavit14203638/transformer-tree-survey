import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T3 = "P083", "Table 3"
ROWS = [
    R("task", "detection;attribute_estimation", 1, "Abstract", "All of the rubber trees are accurately identified and segmented to facilitate single-trunk localisation and feature extraction",
      note="3D trunk detection, then trunk radius and position by cylinder fitting"),
    R("output_unit", "tree_instance;tree_attribute", 5, "Sec. 2.2.2", "the hand-labelled 3D bounding box"),
    R("tree_setting", "plantation", 2, "Sec. 2.1", "The rubber trees were planted at 7 m × 3 m from 2012"),
    R("tree_taxa", "rubber tree (Hevea brasiliensis), variety Reyan 73397", 2, "Sec. 2.1", "The variety is Reyan 73397"),
    R("study_area", "Danzhou City, Hainan Province, China", 2, "Sec. 2.1", "in Danzhou City, Hainan Province, China"),
    R("biome", "tropical", 2, "Sec. 2.1", "Hainan Province is located in a tropical area with high typhoon occurrence"),
    R("platform", "mobile_lidar", 3, "Sec. 2.1", "Figure 2c illustrates the U-shaped walking route when collecting point cloud data", note="handheld VLP-16 LiDAR with IMU carried between rows"),
    R("sensors", "lidar", 3, "Sec. 2.1", "A Bluetooth IMU WT901BLE5.0C ... was fixed on top of the LiDAR VLP-16 (Velodyne"),
    R("annotation_type", "bounding_box", 5, "Sec. 2.2.2", "the 3D point cloud was labelled in the LiDAR Labeler toolbox in Matlab R2021b", note="3D boxes for trunks, branches and shrubs"),
    R("dataset_name", "own handheld LiDAR frames of a rubber plantation (Danzhou)", 5, "Sec. 2.2.2", "A sum of 500 frames of point clouds were labelled", item="dataset1"),
    NR("dataset_role", item="dataset1", note="training and test split not described"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    R("n_images", "500", 5, "Sec. 2.2.2", "A sum of 500 frames of point clouds were labelled", item="dataset1", unit="frames"),
    R("n_trees", "7360", 5, "Sec. 2.2.2", "a total of 7360 trunks, 2586 branches and 3367 shrubs objects were labelled", item="dataset1", unit="trunks",
      note="trunk objects across frames; the plot has 224 trees (p.2)"),
]
ROWS += model("model1", "Improved PointPillars with multi-scale pillar features and a Swin Transformer module", "proposed", "cnn_transformer_hybrid",
              "PointPillars with Swin Transformer module", None, None, 1, "Abstract", "The Swin Transformer module is employed in the backbone to enrich the contextual semantics",
              pre_status="not_reported", pre_note="not stated", strat_note="trained on the labelled frames; details not stated")
res = Counter("res")
ROWS += result(res(), "precision", "model1", "dataset1", "0.9613", 1, "Abstract", "the precision and recall of the target detection reach 0.9613 and 0.8754, respectively",
               unit="fraction", setting="trunk detection over the labelled frames; 3D boxes", headline=True, note="F1 and accuracy are only plotted (Fig. 8)")
ROWS += result(res(), "recall", "model1", "dataset1", "0.8754", 1, "Abstract", "the precision and recall of the target detection reach 0.9613 and 0.8754, respectively",
               unit="fraction", setting="trunk detection over the labelled frames; 3D boxes")
r2, ev_r2 = row(PID, 12, r"^\s*1\.0 m\s+R2\s+(?=\d)")
rm, ev_rm = row(PID, 12, r"^\s*RMSE/cm\s+(?=\d)", 3)  # occurrence 0 is Table 2; 1 to 3 are the 0.2, 0.6 and 1.0 m rows of Table 3
for k, fit in enumerate(("RANSAC", "IRTLS")):
    st = f"trunk radius at 1.0 m height, {fit} cylinder fitting, against field measurement"
    ROWS += result(res(), "rmse", "model1", "dataset1", rm[k], 12, T3, ev_rm, unit="cm", setting=st)
    ROWS += result(res(), "r2", "model1", "dataset1", r2[k], 12, T3, ev_r2, unit="fraction", setting=st)
ROWS += avail()
ROWS += [D("notes", "Baselines PointPillars, PointRCNN and VoxelNet are compared only in a radar chart (Fig. 8), so no exact baseline values exist. Trunk radius results depend on the cylinder fitting (Table 3 gives six heights); the 1.0 m row is recorded.", "Sec. 3 and Tables 2 to 3 (pp. 10 to 12)")]
