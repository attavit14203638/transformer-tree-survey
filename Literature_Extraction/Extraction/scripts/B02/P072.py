import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T5 = "P072", "Table 5"
ROWS = [
    R("task", "attribute_estimation", 15, "Sec. 3.3", "the enumeration of trees in the woodland and the measurement of the diameter at breast height (DBH) of individual trees",
      note="DBH (and tree height and counts) from reconstructed point clouds"),
    R("output_unit", "tree_attribute", 16, "Sec. 3.3.2", "we determined the breast diameter of trees in the three sample plots utilizing LIDAR360"),
    R("tree_setting", "mixed", 6, "Table 1", "1 9 poplar 21 planted forest Simple / 2 45 redpine 35 natural forests / 3 26 poplar 26 natural forests"),
    R("tree_taxa", "poplar; red pine", 6, "Table 1", "Principal Species ... poplar ... redpine ... poplar"),
    R("study_area", "Dongsheng Bajia Country Park, Jiufeng National Forest Park and Olympic Forest Park, Haidian District, Beijing, China; three 15 m x 15 m plots", 5, "Sec. 2.1.1",
      "This study selected Dongsheng Bajia Country Park, Jiufeng National Forest Park, and Olympic Forest Park in Haidian District, Beijing, China"),
    D("biome", "temperate", "Beijing (temperate); not stated in the paper"),
    R("platform", "ground_camera", 1, "Abstract", "took forest stand sequence photos with an iPhone for the research"),
    R("sensors", "rgb", 1, "Abstract", "took forest stand sequence photos with an iPhone"),
    R("annotation_type", "regression_target", 6, "Sec. 2.1.2", "this study employed a diameter tape to measure the DBH of all trees across the three sample plots"),
    R("dataset_name", "own iPhone image sequences of three plots (Beijing)", 6, "Sec. 2.2", "The study employs video recording and frame extraction techniques to generate a consistent image sequence", item="dataset1"),
    R("dataset_role", "test_only", 16, "Sec. 3.3.2", "we evaluated the accuracy of MVSFormer and CPH-Fmnet in determining the diameter at breast height of individual trees across several sample plots",
      item="dataset1", note="field-measured plots used to evaluate the reconstructions"),
    R("dataset_public", "on_request", 21, "Data Availability Statement", "The data presented in this study are available on request from the corresponding author", item="dataset1"),
    D("n_trees", "82", "21 + 35 + 26 trees in the three plots (Table 1, p.6)", item="dataset1", unit="trees"),
]
ROWS += model("model1", "CPH-Fmnet", "proposed", "cnn_transformer_hybrid", "pre-trained ViT with CPHFPN and 3D U-Net", "imagenet", None, 6, "Sec. 2.3",
              "The model comprises three components: the pre-trained ViT, the CPHFPN, and the 3D U-Net", pre_page=7, pre_loc="Sec. 2.3.1",
              pre_ev="This study leverages a pre-trained Vision Transformer (ViT) for robust feature extraction. Pre-trained on the large-scale ImageNet dataset",
              strat_note="the model was 'trained on multiple scales' (p.6); training data and freezing not stated")
ROWS += model("model2", "MVSFormer", "transformer_baseline", "cnn_transformer_hybrid", "pre-trained ViT with multi-view stereo network", None, None, 3, "Sec. 1",
              "Cao et al. proposed a multi-view stereo architecture integrated with the Vision Transformer, termed MVSFormer [30]", pre_status="not_reported",
              pre_note="not stated for this study", strat_note="not stated")
ROWS += model("model3", "SfM + MVS", "other_baseline", "classical", "structure from motion with multi-view stereo (no learning)", "none", None, 15, "Table 3", "SFM + MVS [47]",
              strat_note="not a trained model")
res = Counter("res")
LAB = {"model3": r"SFM \+ MVS \[47\]", "model2": r"MVSFormer \[30\]", "model1": r"CPH-Fmnet"}
for plot in range(3):
    for item, rx in LAB.items():
        nums, ev = row(PID, 17, rf"(?:^\s+[123]\s+)?{rx}\s+(?=\d)", plot + 2)  # Table 4 rows come first on this page
        assert len(nums) == 3, (plot, item, nums, ev)
        st = f"sample plot {plot + 1}; DBH of individual trees"
        ROWS += result(res(), "rmse", item, "dataset1", nums[0], 17, T5, ev, unit="unit not printed", setting=st, headline=(item == "model1" and plot == 0))
        ROWS += result(res(), "other", item, "dataset1", nums[1], 17, T5, ev, setting=st, note="rRMSE (relative RMSE)")
        ROWS += result(res(), "mae", item, "dataset1", nums[2], 17, T5, ev, unit="unit not printed", setting=st)
ROWS += avail(data="on_request", data_page=21, data_loc="Data Availability Statement", data_ev="The data presented in this study are available on request from the corresponding author")
ROWS += [D("notes", "Close-range photogrammetry: models reconstruct point clouds from iPhone image sequences and DBH is then measured in LiDAR360. Tables 3, 4 and 6 give tree-count reconstruction rates, average-height accuracy and point densities; the ablation (Table 7) shows RMSE rising from 1.874 to 2.145 when the pre-trained ViT is removed (plot 2).", "Sec. 3.3 to 3.5 and Tables 3 to 7 (pp. 15 to 18)")]
