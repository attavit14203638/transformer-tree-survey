import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

S = "Sec. IV-A"
ROWS = [
    R("task", "component_segmentation;attribute_estimation", 1, "Abstract", "enabling multilevel analysis from image segmentation to 3-D reconstruction and measurement",
      note="trunk segmentation in single photographs, then trunk height and volume"),
    R("output_unit", "tree_component;tree_attribute", 1, "Index Terms", "individual-tree attribute measurement, tree trunk identification"),
    R("tree_setting", "mixed", 1, "Abstract", "we collected images of 5192 individual trees across multiple scenarios (natural forests, afforested area, and urban parks) in Beijing"),
    R("tree_taxa", "S. japonica; G. biloba; Q. mongolica; R. pseudoacacia; B. platyphylla; P. tomentosa; P. tabulaeformis", 9, S,
      "For S. japonica and G. biloba ... For Q. mongolica ... For R. pseudoacacia and B. platyphylla ... For P. tomentosa ... For P. tabulaeformis"),
    R("study_area", "Beijing, China (natural forests, afforested areas and urban parks)", 1, "Abstract", "across multiple scenarios (natural forests, afforested area, and urban parks) in Beijing"),
    D("biome", "temperate", "Beijing (temperate); not stated in the paper"),
    R("platform", "ground_camera", 2, "Sec. II", "HUAWEI Mate 50 smartphone camera (50 Mpixels)", note="handheld mobile laser scanning (HMLS) used as the reference for height and volume"),
    R("sensors", "rgb", 2, "Sec. II", "HUAWEI Mate 50 smartphone camera (50 Mpixels)"),
    R("annotation_type", "polygon_mask;regression_target", 9, "Sec. III", "The trunk height and volume reconstructed via laser scanning were used as ground truth to validate the results obtained by SGS-3DRecon",
      note="trunk masks for segmentation; HMLS-derived height and volume for measurement"),
    R("dataset_name", "own smartphone images of 5192 trees (Beijing) with HMLS reference for 250 trees", 1, "Abstract",
      "we collected images of 5192 individual trees ... point cloud data of 250 trees were acquired using handheld mobile laser scanning (HMLS)", item="dataset1"),
    R("dataset_role", "train_test", 9, "Sec. III", "The dataset was divided into training, validation, and test sets at a ratio of 60%, 20%, and 20%, respectively", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    R("n_trees", "5192", 1, "Abstract", "we collected images of 5192 individual trees", item="dataset1", unit="trees"),
]
NP = dict(pre_status="not_reported", pre_note="not stated")
ROWS += model("model1", "SegFormer (first stage of SGS-3DRecon)", "transformer_baseline", "hierarchical_vit", "SegFormer (MiT variant not stated)", None, None, 9, S,
              "Segformer exhibits strong baseline performance across all metrics", strat_note="not stated", cat_note="ablation stage; no CNN comparison in the paper", **NP)
ROWS += model("model2", "SegFormer with GAN inversion (second stage)", "other_baseline", "hierarchical_vit", "SegFormer (MiT variant not stated)", None, None, 9, S,
              "After GAN inversion was applied, the mean IoU rose from 79.99% to 85.74%", strat_note="not stated", cat_note="ablation stage", **NP)
ROWS += model("model3", "SGS-3DRecon (SegFormer, GAN inversion and Sobel refinement)", "proposed", "hierarchical_vit", "SegFormer (MiT variant not stated)", None, None, 9, S,
              "After integrating the Sobel operator on top of the Segformer + GAN approach, the mean IoU increased further to 86.40%", strat_note="not stated", **NP)
res = Counter("res")
ST = "validation samples; trunk masks; mean over images"
for item, iou, dice, ev in (("model1", "79.99", "87.16", "After GAN inversion was applied, the mean IoU rose from 79.99% to 85.74% ... The mean Dice coefficient went from 87.16% to 91.78%"),
                            ("model2", "85.74", "91.78", "the mean IoU rose from 79.99% to 85.74%, an increase of 5.75%. The mean Dice coefficient went from 87.16% to 91.78%"),
                            ("model3", "86.40", "92.84", "the mean IoU increased further to 86.40% ... The mean Dice coefficient rose from 91.78% to 92.84%")):
    ROWS += result(res(), "iou", item, "dataset1", iou, 9, S, ev, setting=ST, headline=(item == "model3"), note="trunk IoU")
    ROWS += result(res(), "f1", item, "dataset1", dice, 9, S, ev, setting=ST, note="Dice coefficient")
ROWS += result(res(), "other", "model3", "dataset1", "17.91", 1, "Abstract", "the average relative root-mean-square error (rRMSE) was 17.91%, and the average rBias was −0.40%",
               setting="trunk volume against HMLS reference; average over species", note="rRMSE of volume, %")
ROWS += avail()
ROWS += [D("notes", "Single-photograph pipeline: SegFormer trunk segmentation refined by GAN inversion and a Sobel operator, then pixel-to-physical mapping for 3-D reconstruction. Species-level height and volume agreement with HMLS is reported in Figs. 11 to 12 (e.g. height R2 0.92 to 0.98).", "Sec. IV (pp. 9 to 12)")]
